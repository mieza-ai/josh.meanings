(ns mieza.meanings.initializations.afk
  "Fast and Provably Good Seedings for k-Means is a paper by Olivier Bachem, 
   Mario Lucic, S. Hamed Hassani, and Andreas Krause which introduces an 
   improvement to the monte carlo markov chain approximation of k-means++ 
   D^2 sampling. It accomplishes this by computing the D^2 sampling 
   distribution with respect to the first cluster. This has the practical 
   benefit of removing some of the assumptions, like choice of distance 
   metric, which were imposed in the former framing. As such the name of 
   this algorithm is assumption free k-mc^2. A savvy reader may note that 
   by computing the D^2 sampling distribution as part of the steps this 
   algorithm loses some of the theoretical advantages of the pure markov 
   chain formulation. The paper argues that this is acceptable, because 
   in practice computing the first D^2 sampling distribution ends up paying 
   for itself by reducing the chain length necessary to get convergence 
   guarantees."
  (:refer-clojure :exclude [assoc defn defn- fn get let loop])
  (:require
   [clojure.string :as string]
   [clojure.spec.alpha :as s]
   [fastmath.core]
   [ham-fisted.lazy-noncaching :as hfln]
   [mieza.meanings.advise :as advise]
   [mieza.meanings.distances :as distances]
   [mieza.meanings.initializations.core :refer [initialize-centroids]]
   [mieza.meanings.initializations.utils
    :as utils
    :refer [add-default-chain-length sample-one weighted-sample]]
   [mieza.meanings.persistence :as p]
   [progrock.core :as pr]
   [taoensso.timbre :as log]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.neanderthal :refer [dataset->dense]]
   [uncomplicate.commons.core :as uc]
   [uncomplicate.neanderthal.core :as ne :refer [axpy dim entry! sum]]
   [uncomplicate.neanderthal.native :refer [fv]]
   [uncomplicate.neanderthal.vect-math :as vm]
   [clojure.core :as c]
   [babashka.fs :as fs]
   [clj-fast.clojure.core :refer [assoc defn defn- fn get let loop]]))


(defn- points-sidecar-arrow-file
  [conf suffix]
  (let [file (:points conf)
        filename (str (fs/file-name file))
        basename (string/replace filename #"\.[^.]+$" "")]
    (str (fs/path (or (fs/parent file) ".")
                  (str basename "." suffix ".arrow")))))

(s/fdef qx-file :args (s/cat :conf :mieza.meanings.specs/configuration) :ret string?)
(defn qx-file
  "Returns the path to the file where the qx column is stored."
  [conf]
  (points-sidecar-arrow-file conf "qx"))


(defn afk-centroids-checkpoint-file
  "Path to the persisted partial-centroid set written incrementally during
   AFK-MC² sampling. AFK is the longest single step in the pipeline
   (river took ~9 h 25 m) and without a checkpoint a crash at sample
   N/k loses all N draws. We write this file every
   `afk-centroids-checkpoint-every` samples so resume only loses up to
   that many."
  [conf]
  (points-sidecar-arrow-file conf "afk-centroids"))


(def ^:const afk-centroids-checkpoint-every
  "Write a partial-centroid snapshot every N new draws. Each snapshot is
   a full arrow rewrite of the current cluster set (only tens of rows at
   200-centroid runs, so rewriting is cheap). 10 keeps the worst-case
   loss at 10 samples and the IO overhead under a percent of total AFK
   runtime."
  10)


(defn- afk-centroids-checkpoint-tmp
  "Tmp path for the atomic-rename checkpoint write. Inserts `.tmp` *before*
   the file extension so `persistence/filename->format` still routes to the
   correct writer (`.arrow` → arrow writer); appending `.tmp` after the
   extension leaves `.tmp` as the extension, no writer matches, and the
   checkpoint write NPEs mid-AFK."
  [conf]
  (let [dest (afk-centroids-checkpoint-file conf)
        ext  (re-find #"\.[^.]+$" dest)]
    (str (subs dest 0 (- (count dest) (count ext))) ".tmp" ext)))


(defn- write-afk-checkpoint!
  [conf clusters]
  (let [tmp  (afk-centroids-checkpoint-tmp conf)
        dest (afk-centroids-checkpoint-file conf)]
    (p/write-datasets tmp [clusters])
    (fs/move tmp dest {:replace-existing true :atomic-move true})
    (log/info "AFK-MC² checkpoint saved" {:centroids (ds/row-count clusters)
                                           :target (:k conf)
                                           :path dest})))


(defn- read-afk-checkpoint
  "Returns the persisted partial-centroid dataset, or nil if no
   checkpoint exists or the file can't be read."
  [conf]
  (c/let [path (afk-centroids-checkpoint-file conf)]
    (when (fs/exists? path)
      (try
        (c/let [ds-seq (p/read-dataset-seq path)
                concatenated (apply ds/concat-copying (c/first ds-seq) (c/rest ds-seq))]
          (log/info "AFK-MC² checkpoint found"
                    {:path path :centroids (ds/row-count concatenated)})
          concatenated)
        (catch Exception e
          (log/warn "AFK-MC² checkpoint unreadable; ignoring"
                    {:path path :error (ex-message e)})
          nil)))))

(defn- checkpoint-column-mismatch?
  [conf checkpoint]
  (not= (set (map name (:col-names conf)))
        (set (map name (ds/column-names checkpoint)))))

(defn- checkpoint-row-mismatch?
  [conf checkpoint]
  (let [row-count (ds/row-count checkpoint)]
    (or (zero? row-count)
        (> row-count (:k conf)))))

(defn- valid-afk-checkpoint
  [conf]
  (when-let [checkpoint (read-afk-checkpoint conf)]
    (cond
      (checkpoint-column-mismatch? conf checkpoint)
      (do (log/warn "Ignoring AFK-MC² checkpoint with incompatible columns"
                    {:path (afk-centroids-checkpoint-file conf)
                     :expected (:col-names conf)
                     :actual (vec (ds/column-names checkpoint))})
          nil)

      (checkpoint-row-mismatch? conf checkpoint)
      (do (log/warn "Ignoring AFK-MC² checkpoint with incompatible row count"
                    {:path (afk-centroids-checkpoint-file conf)
                     :k (:k conf)
                     :centroids (ds/row-count checkpoint)})
          nil)

      :else checkpoint)))


(def qx-column-name "qx")

(s/fdef load-datasets-with-qx
  :args (s/cat :conf :mieza.meanings.specs/configuration)
  :ret :mieza.meanings.specs/datasets)
(defn load-datasets-with-qx
  "Load points dataset with q(x)."
  [conf]
  (let [column-names (-> (:col-names conf)
                         (conj qx-column-name))]
    (-> (p/read-dataset-seq (qx-file conf))
        (p/select-columns-seq column-names))))


(s/fdef samples :args (s/cat :conf :mieza.meanings.specs/configuration) :ret :mieza.meanings.specs/dataset)
(defn samples
  "Get all the samples we'll need for the markov chain."
  ([conf]
   (ds/->dataset
    (weighted-sample (load-datasets-with-qx conf) qx-column-name (:m conf)))))


(s/fdef qx-denominator-accelerated
  :args (s/cat :device-context map?
               :conf :mieza.meanings.specs/configuration
               :cluster :mieza.meanings.specs/dataset)
  :ret number?)
(defn qx-denominator
  "Calculates the denominator of the q(x) distribution.
   Releases Neanderthal matrices per chunk to prevent native memory accumulation."
  [conf cluster]
  (reduce + 0.0
          (hfln/map (fn [ds]
                      (let [matrix (dataset->dense ds :row :float32)
                            dists  (distances/gpu-distance @distances/gpu-context matrix cluster)
                            _      (uc/release matrix)
                            v      (fv (seq dists))
                            v2     (vm/pow v 2)
                            result (sum v2)]
                        (uc/release v)
                        (uc/release v2)
                        result))
                    (p/read-dataset-seq conf :points))))


(s/fdef qx-regularizer :args (s/cat :conf :mieza.meanings.specs/configuration) :ret number?)
(defn qx-regularizer [conf] (/ 1.0 (* (:size-estimate conf) 2)))


(s/fdef q-of-x
  :args (s/cat :conf :mieza.meanings.specs/configuration
               :cluster :mieza.meanings.specs/dataset
               :denominator number?)
  :ret :mieza.meanings.specs/datasets)
(defn- q-of-x
  "Computes the q(x) distribution for all x in the dataset on the GPU.
   Implements Bachem et al. 2016 (NeurIPS) Eq. 4:
     q(x|c1) = (1/2) * d(x,c1)^2 / sum_x' d(x',c1)^2 + 1/(2n)
   The denominator passed in is sum_x' d(x',c1)^2 (computed in qx-denominator).
   Releases Neanderthal matrices per chunk to prevent native memory accumulation."
  ([conf cluster denominator]
   (let [regularizer (qx-regularizer conf)
         cluster-matrix (distances/dataset->matrix conf cluster)]
     (hfln/map (fn [ds]
                 (let [matrix (distances/dataset->matrix conf ds)
                       dists  (distances/gpu-distance @distances/gpu-context matrix cluster-matrix)
                       _      (uc/release matrix)
                       dist-v (fv (seq dists))
                       d2     (vm/pow dist-v 2)
                       reg-v  (entry! (fv (seq dists)) regularizer)
                       qx-val (axpy (/ 0.5 denominator) d2 reg-v)]
                   (uc/release dist-v)
                   (uc/release d2)
                   (assoc ds :qx qx-val)))
               (p/read-dataset-seq conf :points)))))



(s/fdef q-of-x!
  :args (s/cat :conf :mieza.meanings.specs/configuration
               :clusters :mieza.meanings.specs/dataset))
(defn q-of-x!
  "Computes and saves the q(x) distribution for all x in the dataset.
   After writing the per-points qx file, hints the kernel to drop :points
   (river) pages from the page cache — sampling only reads qx, so the ~86 GB of
   river mmap pages accumulated during qx-denominator + q-of-x passes
   can be evicted to make room for qx.arrow to stay resident."
  ([conf cluster]
   (p/write-datasets (qx-file conf)
                     (q-of-x conf cluster (qx-denominator conf (distances/dataset->matrix conf cluster))))
   (try (advise/dontneed! (:points conf))
        (catch Throwable t
          (log/warn t "page-cache drop hint failed for :points"
                    {:path (:points conf)})))))


(s/fdef mcmc-sample
  :args (s/cat :conf     :mieza.meanings.specs/configuration
               :points   :mieza.meanings.specs/dataset
               :clusters :mieza.meanings.specs/dataset)
  :ret :mieza.meanings.specs/dataset)
(defn mcmc-sample
  "Perform markov chain monte carlo sampling to approximate D^2 sampling.
   Implements Bachem et al. 2016 Algorithm 1 line 11. The MH acceptance ratio
   for AFK-MC^2 with proposal q(x) targeting p(x|C) ∝ d(x,C)^2 is:
       α = min( (d^2(y,C) · q(x)) / (d^2(x,C) · q(y)), 1 )
   Precompute w(i) = d^2(i,C) / q(i) so the ratio becomes w(y)/w(x).
   Releases all intermediate Neanderthal vectors to prevent native memory accumulation."
  [conf points clusters]
  (let [min-dists (distances/minimum-distance conf points clusters)
        d2        (vm/pow min-dists 2)
        qx-vec    (fv (get points qx-column-name))
        w         (vm/div d2 qx-vec)
        rand-buf  (utils/generate-random-buffer points)
        rands     (ne/view-vctr rand-buf)
        cluster-index (reduce
                       (fn [^long acc-index ^long index]
                         (let [acc  (ne/entry w acc-index)
                               wy   (ne/entry w index)
                               rand (ne/entry rands index)]
                           (if (or (zero? acc) (> (/ wy acc) rand))
                             index
                             acc-index)))
                       0
                       (range 0 (dim w)))
        result (->  (ds/select-rows points cluster-index)
                    (ds/select-columns (:col-names conf)))]
    (uc/release min-dists)
    (uc/release d2)
    (uc/release qx-vec)
    (uc/release w)
    (uc/release rand-buf)
    result))


(s/fdef find-next-cluster
  :args (s/cat :conf :mieza.meanings.specs/configuration
               :clusters :mieza.meanings.specs/dataset)
  :ret :mieza.meanings.specs/dataset)
(defn find-next-cluster
  "Performs markov chain monte carlo sampling with respect to the 
   distance from existing clusters on a sampling dataset sampled 
   from a q(x) distribution."
  [conf clusters]
  (mcmc-sample conf (samples conf) clusters))


(s/fdef k-means-assumption-free-mc-initialization :args (s/cat :conf :mieza.meanings.specs/configuration) :ret :mieza.meanings.specs/points)
(defn k-means-assumption-free-mc-initialization
  "Runs AFK-MC² to pick k centroids.

   Resume semantics: an incremental snapshot of the centroid set is
   written to `afk-centroids-checkpoint-file` every
   `afk-centroids-checkpoint-every` draws. On entry, if such a file
   exists and holds < k centroids, sampling resumes from that row count
   instead of from 1. q(x) is recomputed from qx.arrow (which itself is
   reused via its own checkpoint in `q-of-x!`), so resume only costs
   the newly-sampled centroids, not the entire AFK run."
  [conf]
  (distances/with-gpu-context conf
    (let [checkpoint   (valid-afk-checkpoint conf)
          initial      (cond
                         (and checkpoint (< (ds/row-count checkpoint) (:k conf)))
                         (do (log/info "Resuming AFK-MC² from checkpoint"
                                       {:centroids (ds/row-count checkpoint)
                                        :target (:k conf)})
                             checkpoint)

                         (and checkpoint (= (ds/row-count checkpoint) (:k conf)))
                         checkpoint

                         :else
                         (sample-one conf))
          k            (:k conf)
          _            (q-of-x! conf initial)
          progress-bar (pr/progress-bar k)
          _            (println "Sampling clusters...")
          final-clusters
          (loop [clusters initial]
            (let [centroid-count (ds/row-count clusters)]
              (pr/print (pr/tick progress-bar centroid-count))
              (if (< centroid-count k)
                (let [next-clusters (ds/concat clusters (find-next-cluster conf clusters))]
                  (when (zero? (mod (ds/row-count next-clusters)
                                    afk-centroids-checkpoint-every))
                    (write-afk-checkpoint! conf next-clusters))
                  (recur next-clusters))
                clusters)))]
      ;; Persist the completed set so a crash in downstream steps
      ;; (transform/load) doesn't force a full AFK redo on retry.
      (write-afk-checkpoint! conf final-clusters)
      (pr/print (pr/done (pr/tick progress-bar k)))
      final-clusters)))


(defmethod initialize-centroids
  :afk-mc
  [conf]
  (k-means-assumption-free-mc-initialization (add-default-chain-length conf)))
