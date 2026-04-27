(ns mieza.meanings.initializations.utils
  "Utilities to help with rapid k mean cluster initialization."
  (:refer-clojure :exclude [assoc defn fn get let])
  (:require [bigml.sampling.reservoir :as res-sample]
            [clojure.spec.alpha :as s]
            [ham-fisted.reduce :as hamfr]
            [ham-fisted.lazy-noncaching :as hfl]
            [mieza.meanings.persistence :as p]
            [mieza.meanings.records.clustering-state]
            [mieza.meanings.specs]
            [taoensso.timbre :as log]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.neanderthal :as ds-nean]
            [tech.v3.datatype :as dtype]
            [tech.v3.datatype.functional :as dfn]
            [uncomplicate.commons.core :as uc]
            [uncomplicate.neanderthal
             [random :refer [rand-uniform!]]
             [native :as native]]
            [clojure.core :as c]
            [clj-fast.clojure.core :refer [assoc defn fn get let]])
  (:import [mieza.meanings.records.clustering_state KMeansState]))




(set! *warn-on-reflection* true)
(set! *unchecked-math* true)

(def t-dataset :mieza.meanings.specs/dataset)
(def t-config  :mieza.meanings.specs/configuration)
(def t-points  :mieza.meanings.specs/points)


(defn vector->dataset
  "Converts a vector of points to a dataset with col-name column names."
  [data col-names]
  (log/info "Converting centroids vector to dataset.")
  (ds/->dataset (map (partial zipmap col-names) data)))

(s/fdef centroids->dataset :args (s/cat :conf t-config :results t-points) :ret t-dataset)
(defn centroids->dataset
  "Converts a vector of points to a dataset."
  [^KMeansState s results]
  {:pre [(= (count results) (:k s))]
   :post [(= (count results) (ds/row-count %))]}
  (vector->dataset results (.col_names s)))


(defn uniform-sample
  [ds-seq n & options]
  (log/debug "Getting uniform sample of size" n)
  (let [sample #(apply res-sample/sample (ds/rowvecs %) n options)]
    (apply res-sample/merge (map sample ds-seq))))




;; What is a good weighted sampling algorithm for sampling k items from a large collection?
;; 
;; Weighted reservoir sampling was researched by Pavlos Efraimidis and Paul Spirakis and a summary of 
;; their research can be found in Weighted Random Sampling [1][2]. 
;;
;; In pseudocode the algorithm can be described in only one line:
;;
;; heapq.nlargest(k, items, key=lambda item: math.pow(random.random(), 1/weight(item)))
;;
;; [1]: https://stackoverflow.com/questions/17117872/is-there-an-algorithm-for-weighted-reservoir-sampling
;; [2]: http://utopia.duth.gr/~pefraimi/research/data/2007EncOfAlg.pdf

;; Java's PriorityQueue class implements a priority heap.  By default it expects to a min heap so 
;; reversing the comparator produces a max heap."
  
(def reservoir-sampling-max-heap-comparator
  (reify java.util.Comparator
    (compare
     [_this item1 item2]
     (compare (get item2 :res-rank)  (get item1 :res-rank)))))


(defn max-heap
  "Returns a max heap."
  ^java.util.PriorityQueue
  [k]
  (new java.util.PriorityQueue k reservoir-sampling-max-heap-comparator))

(defn parallel-max-heap
  "Returns a max heap."
  ^java.util.concurrent.PriorityBlockingQueue
  [k]
  (new java.util.concurrent.PriorityBlockingQueue k reservoir-sampling-max-heap-comparator))


(s/fdef generate-random-buffer :args (s/cat :dataset t-dataset))
(defn generate-random-buffer
  "Generates a random buffer."
  [dataset]
  (let [row-count (ds/row-count dataset)
        buffer    (native/fge row-count 1)]
    (rand-uniform! 0.0 1.0 buffer)))


;; Experiments to run:
;; 
;; These tests might not be appropriate because it isn't where the 
;; profiling is showing issues.
;; 
;; 7. Test version which uses random column provided by tech.ml.dataset.
;; 8. Test version which does full res-rank calculation in neanderthal.
(defn reservoir-rank
  [dataset column-name]
  (let [rand-buf (generate-random-buffer dataset)
        rand-dataset (ds/rename-columns (ds-nean/dense->dataset rand-buf) [:random])
        result (assoc dataset :res-rank (dfn/pow (rand-dataset :random) (dfn// 1 (dataset column-name))))]
    (uc/release rand-buf)
    result))


(defn- peek-res-rank
  "Reads :res-rank on the heap's current top without paying per-call boxing
   overhead in the surrounding hot loop; still one boxed lookup per call
   but now only invoked when the threshold cache is refreshed."
  ^double [^java.util.concurrent.PriorityBlockingQueue acc]
  (if-let [top (.peek acc)]
    (double (get top :res-rank))
    Double/NEGATIVE_INFINITY))


(defn weighted-sample
  [ds-seq weight-col ^long k]
  (let [add-to-queue
        (fn ^java.util.concurrent.PriorityBlockingQueue
          [^java.util.concurrent.PriorityBlockingQueue acc ds]
          ;; Read :res-rank once as a primitive DoubleReader so the inner
          ;; loop compares raw doubles instead of calling Dataset.readObject
          ;; (which boxes every column value via Double/valueOf).
          (let [^tech.v3.datatype.Buffer rank-reader (dtype/->reader (ds :res-rank) :float64)
                ^java.util.List rows (ds/rows ds)
                n (.lsize rank-reader)
                ;; Cache the heap's smallest rank so we only refresh it when
                ;; the heap actually changes; avoids a per-row (get peek) box.
                threshold (double-array 1)
                _ (aset threshold 0 (if (< (.size acc) k)
                                      Double/NEGATIVE_INFINITY
                                      (peek-res-rank acc)))]
            (dotimes [i n]
              (let [rank (.readDouble rank-reader i)]
                (when (< (aget threshold 0) rank)
                  (if (< (.size acc) k)
                    (do (.add acc (.get rows (int i)))
                        (when (>= (.size acc) k)
                          (aset threshold 0 (peek-res-rank acc))))
                    (do (.poll acc)
                        (.add acc (.get rows (int i)))
                        (aset threshold 0 (peek-res-rank acc)))))))
            acc))
        merge-queues
        (fn ^java.util.concurrent.PriorityBlockingQueue
          [^java.util.concurrent.PriorityBlockingQueue acc
           ^java.util.concurrent.PriorityBlockingQueue rows]
          (doseq [row rows]
            (when (< (peek-res-rank acc) (double (get row :res-rank)))
              (.poll acc)
              (.add acc row)))
          (while (> (.size acc) k)
            (.poll acc))
          acc)
        ^java.util.concurrent.PriorityBlockingQueue q
        (->> ds-seq
             (hfl/map (fn [dataset] (reservoir-rank dataset weight-col)))
             (hamfr/preduce (partial parallel-max-heap k) add-to-queue merge-queues))]
    (->> q
         (.iterator)
         (iterator-seq)
         (into []))))



(s/fdef sample-one :args (s/cat :conf t-config) :ret t-dataset)
(defn sample-one
  "Returns a one-row dataset drawn from the :points collection. Picks
   uniformly within a record batch chosen uniformly among batches.

   Why not the obvious `rand-nth (concat-all-rows)`: concatenating all
   rows across ~10^5 Arrow batches of a 70 GB mmap'd file to pick one
   point was the dominant cost of a fresh stage start — observed at
   multiple hours on a 71 GB turn.arrow because each per-batch
   `ds/rand-nth` materialized a full row via `hamf/vec` across 200+
   Arrow-backed columns, one potential page fault per column read,
   times ~10^5 batches in parallel pmap. That is O(N) work for an
   O(1) question and it swamps real compute (qx-denominator, q-of-x!,
   Lloyd) that genuinely needs to touch every point.

   Realizing the seq of Dataset objects (vec) does not touch row data —
   each Dataset is a wrapper over column Buffers that is created
   lazily without paging in the underlying Arrow bytes. `ds/select-rows`
   with a one-element index vector is a zero-copy view projection.

   Bias: weighted by batch-size inverse rather than uniform across all
   rows. Acceptable because AFK-MC² only uses this for its first
   centroid, whose placement is compensated for by the q(x)
   distribution computed in the next step; the algorithm's theoretical
   guarantees do not require uniform first-centroid sampling."
  [conf]
  (let [batches (vec (p/read-dataset-seq conf :points))
        batch   (c/rand-nth batches)
        idx     (c/rand-int (ds/row-count batch))]
    (ds/select-rows batch [idx])))


(def ^:const d2-weight-col "__d2_weight")

(defn add-d2-weights
  "Adds a D² weight column to each dataset in ds-seq. Each row gets the squared
   distance to its nearest centroid. Returns a new lazy seq of datasets."
  [config ds-seq centroids]
  (let [distance-fn (:distance-fn config)]
    (hfl/map
     (fn [dataset]
       (let [weights (mapv (fn [row]
                             (let [point (vec (vals row))]
                               (reduce min (map #(Math/pow (double (distance-fn point %)) 2.0) centroids))))
                           (ds/rows dataset))]
         (assoc dataset d2-weight-col weights)))
     ds-seq)))

(defn shortest-distance-*
  "Denotes the shortest distance from a data point to a 
	 center. Which distance to use is decided by the k means 
	 configuration."
  [configuration]
  (let [distance-fn (:distance-fn configuration)]
    (fn [point centroids]
      (apply min (map #(distance-fn point %) centroids)))))

(defn shortest-distance-squared-*
  "Denotes the shortest distance from a data point to a 
	 center squared. Useful for computing a D^2 sampling 
	 distribution."
  [configuration centroids]
  (let [shortest-distance (shortest-distance-* configuration)]
    (fn [point]
      (Math/pow
       (shortest-distance point centroids)
       2))))


;; Helper methods to make setting up chain lengths less of a mental 
;; burden.  When chain lengths aren't provided this code will figure 
;; out a reasonable chain length and set it.

(s/fdef chain-length-warnings :args (s/cat :conf t-config :results t-config))
(defn chain-length-warnings
  "Analyzes the chain length and emits warnings if necessary."
  [config]
  (when (> (:m config) (:size-estimate config))
		;; the monte carlo sampling is intended to approximate the sampling distribution 
		;; computed relative to the entire dataset which is constructed during k means++ 
		;; computation. A larger sample size results in a better approximation, eventually 
		;; converging to the true sampling distribution - at which point the monte carlo simulation 
		;; is just overheard. We aren't eliminating the sampling distribution error, but 
		;; doing wasteful computations.
    (log/warn ":m, the chain length for mc sampling, is greater than the dataset size. You ought to be using k-means++ directly."))
  config)


(s/fdef should-update-chain-length? :args (s/cat :conf t-config :results boolean?))
(defn should-update-chain-length? [conf] (nil? (:m conf)))

(s/fdef update-chain-length :args (s/cat :conf t-config :results t-config))
(defn update-chain-length
  [conf]
	;; We choose to use a default chain length of k*log2(n)log(k)
	;; because this was the chain length used in Bachem's 2016 analysis 
	;; and so it has theoretical guarantees under some conditions.
  (let [n (:size-estimate conf)
        k (:k conf)
        proposed-chain-length (int (* k (/ (Math/log n) (Math/log 2)) (Math/log k)))
        m (min proposed-chain-length (dec n))]
    (assoc conf :m m)))


(s/fdef add-default-chain-length :args (s/cat :conf t-config :results t-config))
(defn add-default-chain-length
  "For monte carlo methods we need a chain length to use when 
	 doing sampling. Although callers can pass in a chain length 
	 there are some dangers when doing so - for example if the 
	 chain length is low it won't necessarily approximate k means 
	 plus plus. Meanwhile if the chain length is too low then 
	 there will be no point in doing sampling at all - we could 
	 just use k means plus plus rather than approximating it.

	 This function checks to see if a chain length is set and if 
	 one is then it does nothing, but it nothing is set it uses 
	 the formulas provided in the k means plus plus approximation 
	 papers to determine a reasonable chain length."
  [conf]
  (->
   (if (should-update-chain-length? conf)
     (update-chain-length conf)
     conf)
   chain-length-warnings))