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
  (:refer-clojure
   :exclude
   [get nth assoc get-in merge assoc-in update-in select-keys destructure let fn loop defn defn-])
  (:require
   [clojure.spec.alpha :as s]
   [fastmath.core]
   [ham-fisted.lazy-noncaching :as hfln]
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
   [uncomplicate.neanderthal.core :as ne :refer [axpy dim entry! sum]]
   [uncomplicate.neanderthal.native :refer [fv]]
   [uncomplicate.neanderthal.vect-math :as vm]
   [clojure.core :as c]
   [babashka.fs :as fs]
   [clj-fast.clojure.core :refer [get nth assoc get-in merge assoc-in update-in select-keys destructure let fn loop defn defn-]]))



(s/fdef qx-file :args (s/cat :conf :mieza.meanings.specs/configuration) :ret string?)
(defn qx-file
  "Returns the path to the file where the qx column is stored."
  [conf]
  (let [file (:points conf)] (str (fs/path (fs/parent file) "qx.arrow"))))


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
  "Calculates the denominator of the q(x) distribution."
  [conf cluster]
  (reduce + 0
          (->> (p/read-dataset-seq conf :points)
               (hfln/map (fn [ds] (dataset->dense ds :row :float32)))
               (hfln/map (fn [matrix] (fv (seq (distances/gpu-distance @distances/gpu-context matrix cluster)))))
               (hfln/map (fn [vector] (sum (vm/pow vector 2)))))))


(s/fdef qx-regularizer :args (s/cat :conf :mieza.meanings.specs/configuration) :ret number?)
(defn qx-regularizer ^double [conf]
  (/ 1.0 (* (double (:size-estimate conf)) 2.0)))


(s/fdef q-of-x
  :args (s/cat :conf :mieza.meanings.specs/configuration
               :cluster :mieza.meanings.specs/dataset
               :denominator number?)
  :ret :mieza.meanings.specs/datasets)
(defn- q-of-x
  "Computes the q(x) distribution for all x in the dataset on the GPU.
   Implements Bachem et al. 2016 (NeurIPS) Eq. 4:
     q(x|c1) = (1/2) * d(x,c1)^2 / sum_x' d(x',c1)^2 + 1/(2n)
   The denominator passed in is sum_x' d(x',c1)^2 (computed in qx-denominator)."
  ([conf cluster ^double denominator]
   (let [regularizer (qx-regularizer conf)
         cluster-matrix (distances/dataset->matrix conf cluster)
         qx (fn [matrix]
              ;; matrix is the raw distance vector d(x, c1) returned by gpu-distance.
              ;; Square it, then compute (1/2) * d^2 / denominator + regularizer.
              (let [d2 (vm/pow (fv (seq matrix)) 2)]
                (axpy
                 (/ 0.5 denominator)
                 d2
                 (entry! (fv (seq matrix)) regularizer))))]
     (hfln/map (fn [ds]
                 (assoc ds :qx
                        (->
                         (distances/gpu-distance
                          @distances/gpu-context (distances/dataset->matrix conf ds) cluster-matrix)
                         (qx))))
               (p/read-dataset-seq conf :points)))))



(s/fdef q-of-x!
  :args (s/cat :conf :mieza.meanings.specs/configuration
               :clusters :mieza.meanings.specs/dataset))
(defn q-of-x!
  "Computes and saves the q(x) distribution for all x in the dataset."
  ([conf cluster]
   (p/write-datasets (qx-file conf)
                     (q-of-x conf cluster (qx-denominator conf (distances/dataset->matrix conf cluster))))))


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
   Precompute w(i) = d^2(i,C) / q(i) so the ratio becomes w(y)/w(x)."
  [conf points clusters]
  (let [min-dists (distances/minimum-distance conf points clusters)
        d2        (vm/pow min-dists 2)
        qx-vec    (fv (get points qx-column-name))
        w         (vm/div d2 qx-vec)
        rands     (ne/view-vctr (utils/generate-random-buffer points))
        cluster-index (reduce
                       (fn [^long acc-index ^long index]
                         (let [acc  (double (ne/entry w acc-index))
                               wy   (double (ne/entry w index))
                               rand (double (ne/entry rands index))]
                           (if (or (zero? acc) (> (/ wy acc) rand))
                             index
                             acc-index)))
                       0
                       (range 0 (dim w)))]
    (->  (ds/select-rows points cluster-index)
         (ds/select-columns (:col-names conf)))))


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
  [conf]
  (distances/with-gpu-context conf
    (let [initial-cluster (sample-one conf)
          k (long (:k conf))
          _ (q-of-x! conf initial-cluster)
          progress-bar (pr/progress-bar k)
          _  (println "Sampling clusters...")
          final-clusters (loop [clusters initial-cluster]
                           (let [centroid-count (long (ds/row-count clusters))]
                             (pr/print (pr/tick progress-bar centroid-count))
                             (if (< centroid-count k)
                               (recur (ds/concat clusters (find-next-cluster conf clusters)))
                               clusters)))]
      (pr/print (pr/done (pr/tick progress-bar k)))
      final-clusters)))


(defmethod initialize-centroids
  :afk-mc
  [conf]
  (k-means-assumption-free-mc-initialization (add-default-chain-length conf)))
