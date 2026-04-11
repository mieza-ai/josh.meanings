(ns mieza.meanings.kmeans
  "K-Means clustering generates a specific number of disjoint, 
	 non-hierarchical clusters. It is well suited to generating globular
	 clusters. The K-Means method is numerical, unsupervised, 
	 non-deterministic and iterative. Every member of a cluster is closer 
	 to its cluster center than the center of any other cluster.

	 The choice of initial partition can greatly affect the final clusters 
	 that result, in terms of inter-cluster and intracluster distances and 
	 cohesion. As a result k means is best run multiple times in order to 
	 avoid the trap of a local minimum."
  (:refer-clojure
   :exclude
   [get nth assoc get-in merge assoc-in update update-in select-keys destructure let fn loop defn defn-])
  (:require [clojure.spec.alpha :as s]
            [clojure.string]
            [ham-fisted.lazy-noncaching :as hfln]
            [mieza.meanings.distances :as distances]
            [mieza.meanings.initializations
             [mc2 :as init-mc2]
             [afk :as init-afk]
             [plusplus :as init-plusplus]
             [parallel :as init-parallel]]
            [mieza.meanings.initializations.core :refer [initialize-centroids]]
            [mieza.meanings.persistence :as persist]
            [mieza.meanings.protocols.classifier :refer [assignments Classifier]]
            [mieza.meanings.records.cluster-result :refer [map->ClusterResult]]
            [mieza.meanings.records.clustering-state :refer [->KMeansState]]
            [progrock.core :as pr]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.reductions :as dsr]
            [uncomplicate.neanderthal.core :as ne]
            [uncomplicate.neanderthal.native :as ne-native]
            [clj-fast.clojure.core :refer [get nth assoc get-in merge assoc-in update-in select-keys destructure let fn loop defn defn-]])
  (:import [mieza.meanings.records.cluster_result ClusterResult]
           [mieza.meanings.records.clustering_state KMeansState]))





(declare k-means)

(def default-format       :arrow)
(def default-init         :afk-mc)
(def default-distance-key :emd)
(def default-chain-length  200)


(def default-options
  {:format       default-format
   :init         default-init
   :distance-key default-distance-key
   :chain-length default-chain-length
   :fused-assign false})


(defn estimate-size
  "Estimates the number of rows in the dataset at filepath."
  [filepath]
  (let [stats (dsr/aggregate {"n" (dsr/row-count)} (persist/read-dataset-seq filepath))]
    (first (get stats "n"))))


(defn column-names
  [filepath]
  (vec (remove #{"assignments" "q(x)" :assignments} (ds/column-names (first (persist/read-dataset-seq filepath))))))


(defn initialize-k-means-state
  "Sets initial configuration options for the k means calculation."
  [points-file k options]
  (let [{:keys [format init distance-key m fused-assign on-progress]} (merge default-options options)
        points-file (persist/convert-file points-file format)
        col-names (get options :columns (column-names points-file))
        state (assoc (->KMeansState
                      k
                      points-file
                      format
                      init
                      distance-key
                      m
                      k-means
                      (estimate-size points-file)
                      col-names
                      true
                      fused-assign)
                     :distance-fn (distances/get-distance-fn distance-key))]
    (cond-> state
      on-progress (assoc :on-progress on-progress))))


(defn assignments-api
  "Updates a sequence of assignment datasets with the new assignments."
  ([^KMeansState conf points-seq]
   (let [assign-fn (if (:fused-assign conf)
                     distances/fused-minimum-index
                     distances/minimum-index)
         assign (fn [ds] (assoc ds :assignments (assign-fn conf ds)))]
     (hfln/map assign points-seq))))



(extend-type KMeansState
  Classifier
  (assignments [this dataset-seq]
    (assignments-api this dataset-seq)))


(extend-type ClusterResult
  Classifier
  (assignments [this dataset-seq]
    (let [config (-> this
                     :configuration
                     (assoc :centroids (:centroids this)))]
      (assignments config dataset-seq))))


(s/fdef calculate-objective :args (s/cat :s :mieza.meanings.specs/configuration
                                         :centroids :mieza.meanings.specs/dataset) :ret number?)
(defn calculate-objective
  "Computes the total cost: sum of minimum distances from each point to its nearest centroid."
  [^KMeansState conf centroid-ds]
  (let [col-names (:col-names conf)
        centroid-ds (if (contains? (set (ds/column-names centroid-ds)) :assignments)
                      (ds/select-columns centroid-ds col-names)
                      centroid-ds)
        calculate-cost (fn [ds] (ne/sum (distances/minimum-distance conf ds centroid-ds)))]
    (reduce + 0.0 (hfln/map calculate-cost (persist/read-dataset-seq conf :points)))))


(defn stabilized?
  "K-means is said to be stabilized when performing an
	 iterative refinement (often called a lloyd iteration), 
	 does not result in any shifting of points between 
	 clusters. A stabilized k-means calculation can be 
	 stopped, because further refinement won't produce 
	 any changes."
  [centroids-1 centroids-2]
  ;; The bits of equality can be thought of as a selection of the clusters which have moved positions.  
  ;; When a centroid moves positions, that means its nearest neighbors need to be recalculated.  
  (= centroids-1 centroids-2))


(defn update-centroids
  [old-centroids new-centroids]
  (let [column-present (complement (set (get new-centroids :assignments)))]
    (ds/sort-by-column
     (ds/concat
      (ds/filter-column old-centroids :assignments column-present)
      new-centroids)
     :assignments)))



(s/fdef lloyd :args (s/cat :conf :mieza.meanings.specs/configuration
                           :initial-centroids :mieza.meanings.specs/dataset))
(defn lloyd ^ClusterResult [^KMeansState conf initial-centroids]
  (let [column-names (:col-names conf)
        max-iterations (long (get conf :iterations 100))
        progress-bar (pr/progress-bar max-iterations)]
    (println "Performing lloyd iteration...")
    (let [final-centroids
          (loop [centroids initial-centroids
                 iteration (long 0)]
            (if (< iteration max-iterations)
              (do
                (pr/print (pr/tick progress-bar iteration))
                (when-let [on-progress (:on-progress conf)]
                  (on-progress {:iteration iteration
                                :max-iterations max-iterations}))
                (let [new-centroids
                      (update-centroids centroids
                                        (distances/with-centroids centroids
                                          (dsr/group-by-column-agg
                                           :assignments
                                           (zipmap column-names (map dsr/mean column-names))
                                           (assignments conf (persist/read-dataset-seq conf :points)))))]
                  (if (stabilized? centroids new-centroids)
                    (do
                      (pr/print (pr/done (pr/tick progress-bar max-iterations)))
                      (when-let [on-progress (:on-progress conf)]
                        (on-progress {:iteration iteration
                                      :max-iterations max-iterations
                                      :converged true}))
                      new-centroids)
                    (recur new-centroids (inc iteration)))))
              (do
                (pr/print (pr/done (pr/tick progress-bar iteration)))
                centroids)))]
      (map->ClusterResult
       {:centroids final-centroids
        :cost (calculate-objective conf final-centroids)
        :configuration (.configuration conf)}))))


(defn- centroid-ds->float-array
  "Extracts centroid coordinates from a dataset as a flat row-major float array."
  ^floats [centroid-ds col-names]
  (let [matrix (distances/dataset->matrix {:col-names col-names} centroid-ds)]
    (distances/matrix->float-array matrix)))


(defn- float-array->centroid-ds
  "Builds a centroid dataset from a flat row-major float array [k, dims]."
  [^floats centroid-arr ^long k col-names]
  (let [dims (long (count col-names))
        rows (loop [c (long 0)
                    acc []]
               (if (< c k)
                 (let [offset (* c dims)
                       row-values (loop [d (long 0)
                                         values []]
                                    (if (< d dims)
                                      (recur (inc d)
                                             (conj values (aget centroid-arr (int (+ offset d)))))
                                      values))]
                   (recur (inc c)
                          (conj acc
                                (zipmap (conj col-names :assignments)
                                        (conj row-values c)))))
                 acc))]
    (ds/->dataset rows)))


(defn- assignment-at
  "Gets the assignment index from a polymorphic array (byte/short/int).
   Handles unsigned conversion for byte and short types."
  ^long [arr ^long i]
  (cond
    (instance? (Class/forName "[B") arr)
    (Byte/toUnsignedInt (aget ^bytes arr (int i)))

    (instance? (Class/forName "[S") arr)
    (Short/toUnsignedInt (aget ^shorts arr (int i)))

    :else
    (aget ^ints arr (int i))))


(defn- accumulate-chunk!
  "Given a chunk's raw point data (float array, row-major [n, dims]) and
   assignment indices (byte/short/int array), accumulate into centroid-sums
   (double array for precision) and centroid-counts arrays."
  [^floats points-arr assignments-arr ^doubles centroid-sums ^ints centroid-counts
   n dims]
  (let [n (int n)
        dims (int dims)]
    (clojure.core/loop [i (int 0)]
      (when (< i n)
        (let [c (int (assignment-at assignments-arr i))
              p-offset (int (* i dims))
              c-offset (int (* c dims))]
          (clojure.core/loop [d (int 0)]
            (when (< d dims)
              (let [idx (int (+ c-offset d))]
                (aset centroid-sums idx
                      (+ (aget centroid-sums idx)
                         (double (aget points-arr (int (+ p-offset d)))))))
              (recur (unchecked-inc-int d))))
          (aset centroid-counts c (int (inc (aget centroid-counts c)))))
        (recur (unchecked-inc-int i))))))


(defn- compute-centroids-from-sums
  "Divides accumulated sums by counts to get new centroid coordinates.
   Returns a flat float array [k, dims]."
  ^floats [^doubles centroid-sums ^ints centroid-counts ^long k ^long dims]
  (let [result (float-array (* k dims))]
    (dotimes [c k]
      (let [cnt (int (aget centroid-counts c))
            c-offset (* c dims)]
        (if (pos? cnt)
          (dotimes [d dims]
            (aset result (+ c-offset d)
                  (float (/ (aget centroid-sums (+ c-offset d)) (double cnt)))))
          ;; Empty cluster — will be filled from old centroids downstream
          (dotimes [d dims]
            (aset result (+ c-offset d) Float/NaN)))))
    result))


(defn- point-squared-distance
  [points-arr centroid-arr p-offset c-offset dims]
  (let [^floats points-arr points-arr
        ^floats centroid-arr centroid-arr
        p-offset (long p-offset)
        c-offset (long c-offset)
        dims (long dims)]
    (loop [d (long 0)
           dist (double 0.0)]
      (if (>= d dims)
        dist
        (let [diff (- (double (aget points-arr (int (+ p-offset d))))
                      (double (aget centroid-arr (int (+ c-offset d)))))]
          (recur (inc d) (+ dist (* diff diff))))))))


(defn- centroids-converged?
  "Checks if centroids have converged using mean relative squared shift.
   Returns true when the average (shift/value)² across coordinates is below tol."
  [^floats old-arr ^floats new-arr ^double tol]
  (let [len (alength old-arr)
        inv-len (/ 1.0 (double len))]
    (clojure.core/loop [i (int 0)
                        shift (double 0.0)]
      (if (>= i len)
        (< (* shift inv-len) tol)
        (let [o (double (aget old-arr i))
              n (double (aget new-arr i))
              scale (Math/max (Math/abs o) 1.0)
              d (/ (- n o) scale)]
          (recur (unchecked-inc-int i) (+ shift (* d d))))))))


(defn- preload-chunks
  "Preloads dataset chunks as [matrix float-array n] triples.
   Keeps Neanderthal matrices for GPU and raw float arrays for CPU accumulation."
  [^KMeansState conf]
  (doall
   (map (fn [ds]
          (let [matrix (distances/dataset->matrix conf ds)
                n (ne/mrows matrix)
                points-arr (distances/matrix->float-array matrix)]
            [matrix points-arr n]))
        (persist/read-dataset-seq conf :points))))


(defn- lloyd-fast-iteration
  "Runs one Lloyd iteration: GPU fused assign + CPU accumulation.
   Returns [new-centroid-arr inertia]."
  [chunks ^floats centroid-arr ^long k ^long dims]
  (let [centroid-matrix (ne-native/fge k dims centroid-arr {:layout :row})
        _ (distances/write-centroids-buffer! distances/gpu-context centroid-matrix)
        ctx @distances/gpu-context
        sums (double-array (* k dims))
        counts (int-array k)
        inertia-acc (atom 0.0)]
    (doseq [[matrix points-arr n] chunks]
      (let [assignments-arr (distances/gpu-fused-assign ctx matrix)
            n (long n)
            dims-i (long dims)]
        (accumulate-chunk! points-arr assignments-arr sums counts n dims-i)
        ;; Compute inertia contribution
        (clojure.core/loop [i (long 0) chunk-cost (double 0.0)]
          (if (>= i n)
            (swap! inertia-acc + chunk-cost)
            (let [c (int (assignment-at assignments-arr i))
                  p-off (* i dims-i)
                  c-off (* (long c) dims-i)
                  point-dist (double (point-squared-distance points-arr centroid-arr p-off c-off dims-i))]
              (recur (inc i) (+ chunk-cost point-dist)))))))
    (distances/release-centroids-buffer! distances/gpu-context)
    (let [^floats new-arr (compute-centroids-from-sums sums counts k dims)]
      ;; Fill empty clusters from old centroids
      (dotimes [c k]
        (when (Float/isNaN (aget new-arr (* c dims)))
          (System/arraycopy centroid-arr (* c dims) new-arr (* c dims) dims)))
      [new-arr @inertia-acc])))


(defn lloyd-fast
  "Fast Lloyd iteration using fused GPU assignment + Java array accumulation.
   Bypasses the dataset abstraction layer for centroid updates entirely."
  ^ClusterResult [^KMeansState conf initial-centroids]
  (let [col-names (:col-names conf)
        dims (long (count col-names))
        k (long (:k conf))
        max-iterations (long (get conf :iterations 100))
        progress-bar (pr/progress-bar max-iterations)]
    (println "Performing fast lloyd iteration...")
    (let [initial-arr (centroid-ds->float-array initial-centroids col-names)
          chunks (preload-chunks conf)
          [final-arr final-inertia]
          (loop [^floats centroid-arr initial-arr
                 iteration (long 0)
                 prev-inertia (double Double/MAX_VALUE)]
            (pr/print (pr/tick progress-bar iteration))
            (if (>= iteration max-iterations)
              (do (pr/print (pr/done (pr/tick progress-bar iteration)))
                  [centroid-arr prev-inertia])
              (let [iteration-result (lloyd-fast-iteration chunks centroid-arr k dims)
                    ^floats new-arr (nth iteration-result 0)
                    new-inertia (double (nth iteration-result 1))
                    rel-change (/ (Math/abs (- prev-inertia new-inertia))
                                  (Math/max (Math/abs new-inertia) 1.0))]
                (when-let [on-progress (:on-progress conf)]
                  (on-progress {:iteration iteration
                                :max-iterations max-iterations
                                :cost new-inertia}))
                (if (and (> iteration 0) (< rel-change 1e-4))
                  (do (pr/print (pr/done (pr/tick progress-bar max-iterations)))
                      (when-let [on-progress (:on-progress conf)]
                        (on-progress {:iteration iteration
                                      :max-iterations max-iterations
                                      :cost new-inertia
                                      :converged true}))
                      [new-arr new-inertia])
                  (recur new-arr (inc iteration) new-inertia)))))
          final-centroids (float-array->centroid-ds final-arr k col-names)]
      (map->ClusterResult
       {:centroids final-centroids
        :cost final-inertia
        :configuration (.configuration conf)}))))


(defn k-means-via-file
  [points-filepath k & options]
  (let [^KMeansState conf (initialize-k-means-state points-filepath k (apply hash-map options))
        centroids
        (assoc (initialize-centroids conf)
               :assignments
               (range 0 (:k conf)))]
    (distances/with-gpu-context conf
      (if (:fused-assign conf)
        (lloyd-fast conf centroids)
        (lloyd conf centroids)))))



;; We never want to rely on things fitting in memory, but in practice 
;; forcing a narrow calling convention on anyone wishing to do a k-means 
;; calculation makes the library harder to use.
;; 
;; To help get around that we are willing to allow anything that we can 
;; transform into a dataset. We handle the transformation step via 
;; multimethods which dispatch based on the datasets type.
(defmulti k-means
  "Runs Lloyd's algorithm to produce a `ClusterResult` record (see `mieza.meanings.records.cluster-result`).

  **Arguments**

  - `dataset` — `String` path to on-disk data, or a `clojure.lang.LazySeq` of datasets
    (written to a temporary file using `:format`).
  - `k` — number of clusters (positive integer).
  - `options` — optional keyword seq merged into `default-options`:

    | Key | Meaning |
    |-----|---------|
    | `:format` | Working-file format: `:arrow` (default), `:arrows`, `:parquet`, `:csv` |
    | `:init` | Initialization: `:afk-mc`, `:k-means-++`, `:k-means-parallel`, `:k-mc-squared`, `:naive` |
    | `:distance-key` | Distance: see `mieza.meanings.distances/distance-keys` ; default `:emd` |
    | `:m` | Chain length for sampling-based inits (see `default-chain-length`) |
    | `:columns` | Feature column names; default: all columns except assignment helpers |
    | `:iterations` | Max Lloyd iterations (default 100) |
    | `:fused-assign` | When true, use fused distance+argmin kernel (default false) |

  **Returns** a record with `:centroids`, `:cost`, and `:configuration`.

  Example: `(k-means \\\"data.parquet\\\" 10 :distance-key :euclidean)`"
  (fn [dataset _ & _] (class dataset)))


;; In the ideal we don't have any work to do. We've already got a 
;; reference to the file we were hoping for.
(defmethod k-means
  java.lang.String
  [points-filepath k & options]
  (apply k-means-via-file points-filepath k options))


;; If we don't get a reference to our file, we'll have to create it.
;; We don't want to support things that are too large to fit in memory
;; even still so we're accepting lazy seqs and not everything.
(defmethod k-means
  clojure.lang.LazySeq
  [lazy-seq k & options]
  (let [format (or (:format options) default-format)
        suffix (:suffix (format persist/formats))
        filename (str (java.util.UUID/randomUUID) suffix)]
    (persist/write-datasets filename lazy-seq)
    (apply k-means filename k options)))


(defn k-means-seq
  "Returns an infinite lazy sequence of independent `k-means` runs for the same
  `dataset` and `k`. Use `take` + `(apply min-key :cost ...)` to keep the best objective."
  [dataset k & options]
  (repeatedly #(apply k-means dataset k options)))
