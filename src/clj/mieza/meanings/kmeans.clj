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
            [clojure.edn :as edn]
            [clojure.java.io :as io]
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
            [uncomplicate.commons.core :as uc]
            [uncomplicate.neanderthal.core :as ne]
            [uncomplicate.neanderthal.native :as ne-native]
            [taoensso.timbre :as log]
            [clj-fast.clojure.core :refer [get nth assoc get-in merge assoc-in update-in select-keys destructure let fn loop defn defn-]])
  (:import [mieza.meanings.records.cluster_result ClusterResult]
           [mieza.meanings.records.clustering_state KMeansState]
           [java.io RandomAccessFile]
           [java.nio ByteBuffer ByteOrder]
           [java.nio.channels FileChannel]
           [java.nio.file CopyOption Files StandardCopyOption]
           [org.apache.arrow.flatbuf Message MessageHeader RecordBatch]))





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
   :fused-assign true
   :fused-reduce false})


(defn- pad-8
  ^long [^long n]
  (let [padding (rem n 8)]
    (if (zero? padding)
      n
      (+ n (- 8 padding)))))

(defn- read-fully?
  [^FileChannel channel ^ByteBuffer buffer]
  (loop [read-any? false]
    (if (.hasRemaining buffer)
      (let [n (.read channel buffer)]
        (cond
          (pos? n) (recur true)
          (and (neg? n) read-any?) (throw (java.io.EOFException. "Unexpected EOF in Arrow metadata"))
          (neg? n) false
          :else (recur read-any?)))
      true)))

(defn- read-int-le
  [^FileChannel channel]
  (let [buffer (doto (ByteBuffer/allocate 4)
                 (.order ByteOrder/LITTLE_ENDIAN))]
    (when (read-fully? channel buffer)
      (.flip buffer)
      (.getInt buffer))))

(defn- read-arrow-message-size
  [^FileChannel channel]
  (when-let [size (read-int-le channel)]
    (if (= -1 size)
      (read-int-le channel)
      size)))

(defn- arrow-file?
  [^bytes prefix]
  (and (= 65 (aget prefix 0))
       (= 82 (aget prefix 1))
       (= 82 (aget prefix 2))
       (= 79 (aget prefix 3))
       (= 87 (aget prefix 4))
       (= 49 (aget prefix 5))
       (zero? (aget prefix 6))
       (zero? (aget prefix 7))))

(defn- arrow-metadata-row-count
  "Counts Arrow record-batch rows from IPC metadata while seeking over bodies."
  [filepath]
  (try
    (with-open [raf (RandomAccessFile. filepath "r")]
      (let [channel (.getChannel raf)
            prefix (ByteBuffer/allocate 8)]
        (when (read-fully? channel prefix)
          (if (arrow-file? (.array prefix))
            (.position channel 8)
            (.position channel 0))
          (loop [rows (long 0)]
            (if-let [message-size (read-arrow-message-size channel)]
              (if (zero? message-size)
                rows
                (let [metadata-size (pad-8 message-size)
                      metadata (doto (ByteBuffer/allocate (int metadata-size))
                                 (.order ByteOrder/LITTLE_ENDIAN))
                      _ (read-fully? channel metadata)
                      _ (.flip metadata)
                      message-buffer (doto (.duplicate metadata)
                                       (.limit (int message-size)))
                      message (Message/getRootAsMessage message-buffer)
                      body-length (long (.bodyLength message))
                      rows (if (= MessageHeader/RecordBatch (.headerType message))
                             (+ rows
                                (long (.length ^RecordBatch
                                               (.header message (RecordBatch.)))))
                             rows)]
                  (.position channel (+ (.position channel) (pad-8 body-length)))
                  (recur rows)))
              rows)))))
    (catch Exception _
      nil)))

(defn- scan-size
  [filepath]
  (let [stats (dsr/aggregate {"n" (dsr/row-count)} (persist/read-dataset-seq filepath))]
    (first (get stats "n"))))

(defn estimate-size
  "Estimates the number of rows in the dataset at filepath."
  [filepath]
  (or (when (#{:arrow :arrows} (persist/filename->format filepath))
        (arrow-metadata-row-count filepath))
      (scan-size filepath)))


(defn column-names
  [filepath]
  (vec (remove #{"assignments" "q(x)" :assignments} (ds/column-names (first (persist/read-dataset-seq filepath))))))


(defn initialize-k-means-state
  "Sets initial configuration options for the k means calculation."
  [points-file k options]
  (let [{:keys [format init distance-key m fused-assign fused-reduce on-progress
                lloyd-checkpoint-path lloyd-checkpoint-every]} (merge default-options options)
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
                     :fused-reduce fused-reduce
                     :distance-fn (distances/get-distance-fn distance-key))]
    (cond-> state
      on-progress (assoc :on-progress on-progress)
      lloyd-checkpoint-path (assoc :lloyd-checkpoint-path lloyd-checkpoint-path)
      lloyd-checkpoint-every (assoc :lloyd-checkpoint-every lloyd-checkpoint-every))))


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


(def ^:private byte-array-class (Class/forName "[B"))
(def ^:private short-array-class (Class/forName "[S"))
(def ^:private int-array-class (Class/forName "[I"))

(defn- assignment-at
  "Gets the assignment index from a polymorphic array (byte/short/int).
   Handles unsigned conversion for byte and short types."
  ^long [arr ^long i]
  (cond
    (instance? byte-array-class arr)
    (Byte/toUnsignedInt (aget ^bytes arr (int i)))

    (instance? short-array-class arr)
    (Short/toUnsignedInt (aget ^shorts arr (int i)))

    :else
    (aget ^ints arr (int i))))


(defn- accumulate-byte-assignments!
  [^floats points-arr ^bytes assignments-arr ^doubles centroid-sums ^ints centroid-counts
   n dims]
  (let [n (int n)
        dims (int dims)]
    (clojure.core/loop [i (int 0)]
      (when (< i n)
        (let [c (int (Byte/toUnsignedInt (aget assignments-arr i)))
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


(defn- accumulate-short-assignments!
  [^floats points-arr ^shorts assignments-arr ^doubles centroid-sums ^ints centroid-counts
   n dims]
  (let [n (int n)
        dims (int dims)]
    (clojure.core/loop [i (int 0)]
      (when (< i n)
        (let [c (int (Short/toUnsignedInt (aget assignments-arr i)))
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


(defn- accumulate-int-assignments!
  [^floats points-arr ^ints assignments-arr ^doubles centroid-sums ^ints centroid-counts
   n dims]
  (let [n (int n)
        dims (int dims)]
    (clojure.core/loop [i (int 0)]
      (when (< i n)
        (let [c (int (aget assignments-arr i))
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


(defn- accumulate-chunk!
  "Given a chunk's raw point data (float array, row-major [n, dims]) and
   assignment indices (byte/short/int array), accumulate into centroid-sums
   (double array for precision) and centroid-counts arrays."
  [^floats points-arr assignments-arr ^doubles centroid-sums ^ints centroid-counts
   n dims]
  (cond
    (instance? byte-array-class assignments-arr)
    (accumulate-byte-assignments! points-arr assignments-arr centroid-sums centroid-counts n dims)

    (instance? short-array-class assignments-arr)
    (accumulate-short-assignments! points-arr assignments-arr centroid-sums centroid-counts n dims)

    (instance? int-array-class assignments-arr)
    (accumulate-int-assignments! points-arr assignments-arr centroid-sums centroid-counts n dims)

    :else
    (throw (IllegalArgumentException.
            (str "Unsupported assignments array type: " (class assignments-arr))))))


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


(defn- point-manhattan-distance
  [^floats points-arr ^floats centroid-arr p-offset c-offset dims]
  (loop [d (long 0)
         dist (double 0.0)]
    (if (>= d dims)
      dist
      (let [diff (- (double (aget points-arr (int (+ p-offset d))))
                    (double (aget centroid-arr (int (+ c-offset d)))))]
        (recur (inc d) (+ dist (Math/abs diff)))))))


(defn- point-chebyshev-distance
  [^floats points-arr ^floats centroid-arr p-offset c-offset dims]
  (loop [d (long 0)
         dist (double 0.0)]
    (if (>= d dims)
      dist
      (let [diff (- (double (aget points-arr (int (+ p-offset d))))
                    (double (aget centroid-arr (int (+ c-offset d)))))]
        (recur (inc d) (Math/max dist (Math/abs diff)))))))


(defn- point-emd-distance
  [^floats points-arr ^floats centroid-arr p-offset c-offset dims]
  (loop [d (long 0)
         last-distance (double 0.0)
         total-distance (double 0.0)]
    (if (>= d dims)
      total-distance
      (let [current-distance (- (+ (double (aget points-arr (int (+ p-offset d))))
                                  last-distance)
                                (double (aget centroid-arr (int (+ c-offset d)))))]
        (recur (inc d)
               current-distance
               (+ total-distance (Math/abs current-distance)))))))


(defn- point-distance-via-fn
  [distance-fn ^floats points-arr ^floats centroid-arr p-offset c-offset dims]
  (let [point (double-array dims)
        centroid (double-array dims)]
    (dotimes [d dims]
      (aset point d (double (aget points-arr (int (+ p-offset d)))))
      (aset centroid d (double (aget centroid-arr (int (+ c-offset d))))))
    (double (distance-fn point centroid))))


(defn- point-distance
  [distance-key distance-fn ^floats points-arr ^floats centroid-arr
   p-offset c-offset dims]
  (case distance-key
    :euclidean-sq (point-squared-distance points-arr centroid-arr p-offset c-offset dims)
    :euclidean (Math/sqrt (point-squared-distance points-arr centroid-arr p-offset c-offset dims))
    :manhattan (point-manhattan-distance points-arr centroid-arr p-offset c-offset dims)
    :chebyshev (point-chebyshev-distance points-arr centroid-arr p-offset c-offset dims)
    :emd (point-emd-distance points-arr centroid-arr p-offset c-offset dims)
    (point-distance-via-fn distance-fn points-arr centroid-arr p-offset c-offset dims)))


(defn- chunk-inertia-byte-assignments
  [distance-key distance-fn ^floats points-arr ^bytes assignments-arr
   ^floats centroid-arr n dims]
  (loop [i (long 0)
         chunk-cost (double 0.0)]
    (if (>= i n)
      chunk-cost
      (let [c (int (Byte/toUnsignedInt (aget assignments-arr (int i))))
            p-off (* i dims)
            c-off (* (long c) dims)
            point-dist (point-distance distance-key distance-fn points-arr centroid-arr p-off c-off dims)]
        (recur (inc i) (+ chunk-cost point-dist))))))


(defn- chunk-inertia-short-assignments
  [distance-key distance-fn ^floats points-arr ^shorts assignments-arr
   ^floats centroid-arr n dims]
  (loop [i (long 0)
         chunk-cost (double 0.0)]
    (if (>= i n)
      chunk-cost
      (let [c (int (Short/toUnsignedInt (aget assignments-arr (int i))))
            p-off (* i dims)
            c-off (* (long c) dims)
            point-dist (point-distance distance-key distance-fn points-arr centroid-arr p-off c-off dims)]
        (recur (inc i) (+ chunk-cost point-dist))))))


(defn- chunk-inertia-int-assignments
  [distance-key distance-fn ^floats points-arr ^ints assignments-arr
   ^floats centroid-arr n dims]
  (loop [i (long 0)
         chunk-cost (double 0.0)]
    (if (>= i n)
      chunk-cost
      (let [c (int (aget assignments-arr (int i)))
            p-off (* i dims)
            c-off (* (long c) dims)
            point-dist (point-distance distance-key distance-fn points-arr centroid-arr p-off c-off dims)]
        (recur (inc i) (+ chunk-cost point-dist))))))


(defn- chunk-inertia
  [distance-key distance-fn ^floats points-arr assignments-arr
   ^floats centroid-arr n dims]
  (cond
    (instance? byte-array-class assignments-arr)
    (chunk-inertia-byte-assignments distance-key distance-fn points-arr assignments-arr centroid-arr n dims)

    (instance? short-array-class assignments-arr)
    (chunk-inertia-short-assignments distance-key distance-fn points-arr assignments-arr centroid-arr n dims)

    (instance? int-array-class assignments-arr)
    (chunk-inertia-int-assignments distance-key distance-fn points-arr assignments-arr centroid-arr n dims)

    :else
    (throw (IllegalArgumentException.
            (str "Unsupported assignments array type: " (class assignments-arr))))))


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


(defn- lloyd-fast-iteration
  "Runs one Lloyd iteration: GPU fused assign + CPU accumulation.
   Streams chunks from disk via mmap, releasing each Neanderthal matrix
   immediately after use to prevent native memory accumulation.
   Returns [new-centroid-arr inertia]."
  [^KMeansState conf ^floats centroid-arr k dims distance-key]
  (let [k (long k)
        dims (long dims)
        centroid-matrix (ne-native/fge k dims centroid-arr {:layout :row})
        _ (distances/write-centroids-buffer! distances/gpu-context centroid-matrix)
        ctx @distances/gpu-context
        distance-fn (distances/get-distance-fn distance-key)
        sums (double-array (* k dims))
        counts (int-array k)
        inertia-acc (atom 0.0)]
    (doseq [ds (persist/read-dataset-seq conf :points)]
      (let [matrix (distances/dataset->matrix conf ds)
            n (long (ne/mrows matrix))
            assignments-arr (distances/gpu-fused-assign ctx matrix)
            points-arr (distances/matrix->float-array matrix)
            _ (uc/release matrix)
            dims-i (long dims)]
        (accumulate-chunk! points-arr assignments-arr sums counts n dims-i)
        (swap! inertia-acc + (chunk-inertia distance-key distance-fn points-arr assignments-arr
                                            centroid-arr n dims-i))))
    (uc/release centroid-matrix)
    (distances/release-centroids-buffer! distances/gpu-context)
    (let [^floats new-arr (compute-centroids-from-sums sums counts k dims)]
      ;; Fill empty clusters from old centroids
      (dotimes [c k]
        (when (Float/isNaN (aget new-arr (* c dims)))
          (System/arraycopy centroid-arr (* c dims) new-arr (* c dims) dims)))
      [new-arr @inertia-acc])))


(def ^:private lloyd-checkpoint-schema-version 1)


(defn- float-array->vector
  [^floats arr]
  (mapv float arr))


(defn- checkpoint-centroids->float-array
  ^floats [centroids]
  (let [arr (float-array (count centroids))]
    (doseq [[i value] (map-indexed vector centroids)]
      (aset arr (int i) (float value)))
    arr))


(defn- same-long?
  [expected actual]
  (and (number? actual)
       (= (long expected) (long actual))))


(defn- checkpoint-due?
  [^long iteration ^long checkpoint-every]
  (== (long 0) (Long/remainderUnsigned iteration checkpoint-every)))


(defn- write-lloyd-checkpoint!
  "Atomically writes a Lloyd checkpoint as EDN. Centroids are serialized under
   `:centroids` as a vector of float values in row-major [k, dims] order so the
   read path can round-trip them without extra dependencies."
  [path {:keys [iteration centroids inertia k dims distance-key dataset-path]}]
  (let [target (.toPath (io/file path))
        tmp (.resolveSibling target (str (.getFileName target) ".tmp"))
        parent (.getParent target)
        checkpoint {:schema-version lloyd-checkpoint-schema-version
                    :iteration (long iteration)
                    :centroids (float-array->vector centroids)
                    :inertia (double inertia)
                    :k (long k)
                    :dims (long dims)
                    :distance-key distance-key
                    :dataset-path (str dataset-path)
                    :written-at (java.util.Date.)}]
    (when parent
      (Files/createDirectories parent (make-array java.nio.file.attribute.FileAttribute 0)))
    (spit (.toFile tmp) (pr-str checkpoint))
    (Files/move tmp target
                (into-array CopyOption [StandardCopyOption/ATOMIC_MOVE
                                        StandardCopyOption/REPLACE_EXISTING]))
    nil))


(defn- read-lloyd-checkpoint
  [path]
  (let [checkpoint-file (io/file path)]
    (when (.exists checkpoint-file)
      (try
        (let [checkpoint (edn/read-string (slurp checkpoint-file))]
          (when (map? checkpoint)
            checkpoint))
        (catch Exception _
          nil)))))


(defn- lloyd-checkpoint-mismatches
  [checkpoint ^KMeansState conf ^long k ^long dims]
  (let [centroids (:centroids checkpoint)
        expected-centroid-count (* k dims)
        centroid-shape-valid? (and (sequential? centroids)
                                   (= expected-centroid-count (count centroids)))
        centroid-values-valid? (and centroid-shape-valid?
                                    (every? number? centroids))]
    (cond-> []
      (not= lloyd-checkpoint-schema-version (:schema-version checkpoint))
      (conj (str ":schema-version expected " lloyd-checkpoint-schema-version
                 ", found " (:schema-version checkpoint)))

      (not (same-long? k (:k checkpoint)))
      (conj (str ":k expected " k ", found " (:k checkpoint)))

      (not (same-long? dims (:dims checkpoint)))
      (conj (str ":dims expected " dims ", found " (:dims checkpoint)))

      (not= (:distance-key conf) (:distance-key checkpoint))
      (conj (str ":distance-key expected " (:distance-key conf)
                 ", found " (:distance-key checkpoint)))

      (and (contains? checkpoint :dataset-path)
           (not= (str (:points conf)) (:dataset-path checkpoint)))
      (conj (str ":dataset-path expected " (str (:points conf))
                 ", found " (:dataset-path checkpoint)))

      (not (number? (:iteration checkpoint)))
      (conj (str ":iteration expected number, found " (:iteration checkpoint)))

      (not (number? (:inertia checkpoint)))
      (conj (str ":inertia expected number, found " (:inertia checkpoint)))

      (not centroid-shape-valid?)
      (conj (str ":centroids expected " expected-centroid-count
                 " values, found " (if (sequential? centroids)
                                     (count centroids)
                                     (type centroids))))

      (and centroid-shape-valid? (not centroid-values-valid?))
      (conj ":centroids contains non-numeric values"))))


(defn- resume-lloyd-checkpoint
  [path ^KMeansState conf ^long k ^long dims]
  (when-let [checkpoint (read-lloyd-checkpoint path)]
    (let [mismatches (lloyd-checkpoint-mismatches checkpoint conf k dims)]
      (if (seq mismatches)
        (do
          (log/warn "Ignoring Lloyd checkpoint" path
                    "because" (clojure.string/join "; " mismatches))
          nil)
        (let [checkpoint-iteration (long (:iteration checkpoint))
              start-iteration (inc checkpoint-iteration)
              prev-inertia (double (:inertia checkpoint))]
          (log/info "Resuming Lloyd checkpoint" path
                    "from completed iteration" checkpoint-iteration)
          {:centroids (checkpoint-centroids->float-array (:centroids checkpoint))
           :iteration start-iteration
           :prev-inertia prev-inertia})))))


(defn- merge-reduced-chunk!
  [^doubles centroid-sums ^ints centroid-counts reduced-chunk]
  (let [^doubles chunk-sums (:sums reduced-chunk)
        ^ints chunk-counts (:counts reduced-chunk)
        sum-count (alength centroid-sums)
        cluster-count (alength centroid-counts)]
    (dotimes [i sum-count]
      (aset centroid-sums i (+ (aget centroid-sums i)
                               (aget chunk-sums i))))
    (dotimes [c cluster-count]
      (aset centroid-counts c (int (+ (aget centroid-counts c)
                                      (aget chunk-counts c)))))
    (double (:inertia reduced-chunk))))


(defn- lloyd-fast-reduced-iteration
  "Runs one Lloyd iteration: GPU fused Euclidean-sq assign + block partial
   reduction, followed by CPU double reduction over the small block partials.
   Returns [new-centroid-arr inertia]."
  [^KMeansState conf ^floats centroid-arr ^long k ^long dims]
  (let [centroid-matrix (ne-native/fge k dims centroid-arr {:layout :row})
        sums (double-array (* k dims))
        counts (int-array k)]
    (distances/write-centroids-buffer! distances/gpu-context centroid-matrix)
    (let [inertia
          (try
            (reduce
             (fn [acc ds]
               (let [matrix (distances/dataset->matrix conf ds)
                     reduced-chunk (distances/gpu-fused-assign-and-reduce @distances/gpu-context matrix)]
                 ;; Correctness note: for :euclidean-sq, the kernel computes the
                 ;; same argmin assignment as the old fused path, then returns
                 ;; per-cluster sums/counts. Reducing those partials in double
                 ;; precision gives the same centroid means as accumulate-chunk!.
                 (+ (double acc)
                    (merge-reduced-chunk! sums counts reduced-chunk))))
             0.0
             (persist/read-dataset-seq conf :points))
            (finally
              (distances/release-centroids-buffer! distances/gpu-context)))]
      (let [^floats new-arr (compute-centroids-from-sums sums counts k dims)]
        (dotimes [c k]
          (when (Float/isNaN (aget new-arr (* c dims)))
            (System/arraycopy centroid-arr (* c dims) new-arr (* c dims) dims)))
        [new-arr inertia]))))


(defn lloyd-fast
  "Fast Lloyd iteration using fused GPU assignment + Java array accumulation.
   Bypasses the dataset abstraction layer for centroid updates entirely."
  ^ClusterResult [^KMeansState conf initial-centroids]
  (let [col-names (:col-names conf)
        dims (long (count col-names))
        k (long (:k conf))
        max-iterations (long (get conf :iterations 100))
        checkpoint-path (:lloyd-checkpoint-path conf)
        checkpoint-every (long (Math/max (long 1) (long (get conf :lloyd-checkpoint-every 1))))
        resume-state (when checkpoint-path
                       (resume-lloyd-checkpoint checkpoint-path conf k dims))
        progress-bar (pr/progress-bar max-iterations)]
    (println "Performing fast lloyd iteration...")
    (let [initial-arr (if resume-state
                        (:centroids resume-state)
                        (centroid-ds->float-array initial-centroids col-names))
          start-iteration (long (if resume-state
                                  (:iteration resume-state)
                                  0))
          start-prev-inertia (double (if resume-state
                                       (:prev-inertia resume-state)
                                       Double/MAX_VALUE))
          write-checkpoint (fn [^long iteration ^floats centroid-arr ^double inertia final?]
                             (when (and checkpoint-path
                                        (>= iteration 0)
                                        (or final?
                                            (checkpoint-due? iteration checkpoint-every)))
                               (write-lloyd-checkpoint!
                                checkpoint-path
                                {:iteration iteration
                                 :centroids centroid-arr
                                 :inertia inertia
                                 :k k
                                 :dims dims
                                 :distance-key (:distance-key conf)
                                 :dataset-path (:points conf)})))
          [final-arr final-inertia _]
          (loop [^floats centroid-arr initial-arr
                 iteration start-iteration
                 prev-inertia start-prev-inertia
                 last-completed-iteration (dec start-iteration)]
            (pr/print (pr/tick progress-bar iteration))
            (if (>= iteration max-iterations)
              (do (pr/print (pr/done (pr/tick progress-bar iteration)))
                  (write-checkpoint last-completed-iteration centroid-arr prev-inertia true)
                  [centroid-arr prev-inertia last-completed-iteration])
              (let [iteration-result (lloyd-fast-iteration conf centroid-arr k dims (:distance-key conf))
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
                      (write-checkpoint iteration new-arr new-inertia true)
                      [new-arr new-inertia iteration])
                  (do
                    (write-checkpoint iteration new-arr new-inertia false)
                    (recur new-arr (inc iteration) new-inertia iteration))))))
          final-centroids (float-array->centroid-ds final-arr k col-names)]
      (map->ClusterResult
       {:centroids final-centroids
        :cost final-inertia
        :configuration (.configuration conf)}))))


(defn lloyd-fast-reduced
  "Fast Lloyd iteration using GPU fused Euclidean-sq assignment plus per-block
   partial reduction. Only :euclidean-sq is supported by this kernel path."
  ^ClusterResult [^KMeansState conf initial-centroids]
  (when-not (distances/reduce-accelerated? conf)
    (throw (ex-info ":fused-reduce is currently supported only for :euclidean-sq."
                    {:distance-key (:distance-key conf)})))
  (let [col-names (:col-names conf)
        dims (long (count col-names))
        k (long (:k conf))
        max-iterations (long (get conf :iterations 100))
        progress-bar (pr/progress-bar max-iterations)]
    (println "Performing fast reduced lloyd iteration...")
    (let [initial-arr (centroid-ds->float-array initial-centroids col-names)
          [final-arr final-inertia]
          (loop [^floats centroid-arr initial-arr
                 iteration (long 0)
                 prev-inertia (double Double/MAX_VALUE)]
            (pr/print (pr/tick progress-bar iteration))
            (if (>= iteration max-iterations)
              (do (pr/print (pr/done (pr/tick progress-bar iteration)))
                  [centroid-arr prev-inertia])
              (let [iteration-result (lloyd-fast-reduced-iteration conf centroid-arr k dims)
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
    (when (and (:fused-reduce conf)
               (not (distances/reduce-accelerated? conf)))
      (throw (ex-info ":fused-reduce is currently supported only for :euclidean-sq."
                      {:distance-key (:distance-key conf)})))
    (distances/with-gpu-context conf
      (cond
        (:fused-reduce conf)
        (lloyd-fast-reduced conf centroids)

        (:fused-assign conf)
        (lloyd-fast conf centroids)

        :else
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
    | `:fused-reduce` | When true with `:euclidean-sq`, use fused assignment plus GPU block partial reduction |

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
