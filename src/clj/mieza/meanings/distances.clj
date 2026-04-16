(ns mieza.meanings.distances
  "Multimethod for distance calculations.
   
   The `get-distance-fn` multimethod dispatches based on identity to 
   determine what distance function to use when calculating distances. 
   By default the supported distance function keys are:
   
   - :euclidean
   - :manhattan
   - :chebyshev
   - :correlation
   - :canberra
   - :emd
   - :euclidean-sq
   - :discrete
   - :cosine
   - :angular
   - :jensen-shannon

   The default distance function is :emd which may not be appropriate for all
   use cases since it doesn't minimize the variational distance like euclidean
   would. If you don't know why :emd is the default you should probably switch 
   to using :euclidean.
   
   For some distance functions GPU support can be enabled by setting the :use-gpu
   flag to True. GPU distance support is available for:

   - :emd
   - :euclidean
   - :manhattan
   - :chebyshev
   - :euclidean-sq
   "
  (:refer-clojure
   :exclude
   [get nth assoc get-in merge assoc-in update-in select-keys destructure let fn loop defn defn-])
  (:require
   [clj-fast.clojure.core :refer [assoc defn fn let]]
   [clojure.core :as c]
   [clojure.core.async :refer [chan]]
   [clojure.java.io :as io]
   [fastmath.distance :as fmdistances]
   [ham-fisted.lazy-noncaching :as hfln]
   [mieza.meanings.persistence :as p]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.neanderthal :refer [dataset->dense]]
   [uncomplicate.clojurecl
             [core :as cl :refer :all]
             [info :refer :all]]
   [uncomplicate.clojurecl.core :refer :all]
   [uncomplicate.clojurecl.info :refer :all]
   [uncomplicate.commons
             [core :as clojurecl :refer [with-release]]
             [utils :refer [direct-buffer]]]
   [uncomplicate.neanderthal.block :refer [buffer]]
   [uncomplicate.neanderthal.core :refer [entry imin mrows ncols rows]]
   [uncomplicate.neanderthal.native :refer [fge fv]]) 
  (:import
   [org.bytedeco.javacpp FloatPointer]))

(set! *warn-on-reflection* true)

;; I don't want to lock people out of using their preferred distance 
;; functions. So I'm going to implement distance as a multimethod that 
;; way people can choose to provide their own. 
;; 
;; However, this isn't like with the initialization method. We're going to 
;; be calling this as part of our inner loop. In an inner-loop it makes 
;; little to no sense to continually get the distance we want to use. 
;; 
;; So the thing that is a protocol isn't the distance function itself, but 
;; rather the method by which we get the distance function. 
(defmulti get-distance-fn identity)

;; I'm going to start by supporting all the distance functions that fastmath 
;; supports. I'll worry about GPU support and how that might make this more 
;; complicated when I get to that stage. Until then I want to do the simplest 
;; thing which will work.
(defmethod get-distance-fn :euclidean [_] fmdistances/euclidean)
(defmethod get-distance-fn :manhattan [_] fmdistances/manhattan)
(defmethod get-distance-fn :chebyshev [_] fmdistances/chebyshev)
(defmethod get-distance-fn :correlation [_] fmdistances/correlation)
(defmethod get-distance-fn :canberra [_] fmdistances/canberra)
(defmethod get-distance-fn :emd [_] fmdistances/earth-movers)
(defmethod get-distance-fn :euclidean-sq [_] fmdistances/euclidean-sq)
(defmethod get-distance-fn :discrete [_] fmdistances/discrete)
(defmethod get-distance-fn :cosine [_] fmdistances/cosine)
(defmethod get-distance-fn :angular [_] fmdistances/angular)
(defmethod get-distance-fn :jensen-shannon [_] fmdistances/jensen-shannon)

;; To enable generative testing we are keeping track of the keys we make 
;; available. This isn't intended to stop someone from using a different 
;; key. 
(def distance-keys
  [:euclidean
   :manhattan
   :chebyshev
   :correlation
   :canberra
   :emd
   :euclidean-sq
   :discrete
   :cosine
   :angular
   :jensen-shannon])


(defn dataset->matrix
  ([conf ds]
   (let [col-names (:col-names conf)
         ds (if (= (vec col-names) (vec (ds/column-names ds)))
              ds
              (ds/select-columns ds col-names))]
     (dataset->dense ds :row :float32)))
  ([ds] 
   (dataset->dense ds :row :float32)))


(defn matrix->float-array
  "Extracts the raw float data from a Neanderthal matrix as a Java float array.
   Layout is row-major: [row0-col0, row0-col1, ..., row1-col0, ...]."
  ^floats [matrix]
  (let [ptr (buffer matrix)
        arr (float-array (.capacity ^org.bytedeco.javacpp.FloatPointer ptr))]
    (.get ^org.bytedeco.javacpp.FloatPointer ptr arr)
    arr))

(c/defn- float-pointer-byte-buffer
  "Returns a direct ByteBuffer view over a JavaCPP FloatPointer without copying."
  ^java.nio.ByteBuffer [^FloatPointer ptr]
  (.asByteBuffer ptr))


;; GPU based distance functions need to work a little differently.  For CPU 
;; functions we get the function and then we use the distance function during 
;; an inner-loop.  For GPU based distance functions we don't call the distance 
;; function as part of the inner loop, but as part of the outer loop.  The 
;; inner loop is taken care of on the GPUs.
;; 
;; In order to orchestrate the GPU distance calculations we will need a GPU 
;; program and a kernel.  Telling whether GPU support is then as simple as checking 
;; for the presence of the relevant key.
(def gpu-accelerated
  {:emd          {:program "emd_multi.c"          :kernel "wasserstein_distances"
                  :fused-program "fused_emd_assign.c"          :fused-kernel "fused_emd_assign"}
   :euclidean    {:program "euclidean_multi.c"    :kernel "euclidean_distances"
                  :fused-program "fused_euclidean_assign.c"    :fused-kernel "fused_euclidean_assign"}
   :manhattan    {:program "manhattan_multi.c"    :kernel "manhattan_distances"
                  :fused-program "fused_manhattan_assign.c"    :fused-kernel "fused_manhattan_assign"}
   :chebyshev    {:program "chebyshev_multi.c"    :kernel "chebyshev_distances"
                  :fused-program "fused_chebyshev_assign.c"    :fused-kernel "fused_chebyshev_assign"}
   :euclidean-sq {:program "euclidean_sq_multi.c" :kernel "euclidean_sq_distances"
                  :fused-program "fused_euclidean_sq_assign.c" :fused-kernel "fused_euclidean_sq_assign"}})

;; We can tell whether we can use an outerloop GPU program instead of innerloop 
;; function calls by calling a predicate function.  This will allow for seemless 
;; acceleration upgrade when acceleration is available while falling back to the 
;; innerloop function calls when it is not.
(defn is-gpu-accelerated? 
  "Returns true when GPU acceleration is available."
  [conf] 
  (and
   (:use-gpu conf)
   (contains? gpu-accelerated (:distance-key conf))))


;; WHen working on the GPU we have to transfer data and back forth.  We want to do 
;; that using the smallest sizes possible because data transfer between the GPU and 
;; CPU represents a large fraction of the compute time.
(c/defn size->bytes
  "Returns the size in bytes needed to identify all centroid indices."
  [size]
  (let [size (long size)]
    (cond
      (< size 256)
      1
      (< size 65536)
      2
      (< size 4294967296)
      4)))


(c/defn bytes->type
  "Returns the type OpenCL uses to represent a given number of bytes."
  [bytes]
  (let [bytes (long bytes)]
    (cond
      (= bytes 1)
      "uchar"

      (= bytes 2)
      "ushort"

      (= bytes 4)
      "uint")))

(c/defn- work-item-batches
  [n global-size]
  (let [n (long n)
        global-size (long global-size)
        batch-count (quot n global-size)]
    (if (zero? (rem n global-size))
      batch-count
      (inc batch-count))))

(c/defn- round-up-to-multiple
  [n size]
  (let [n (long n)
        size (long size)
        remainder (rem n size)]
    (if (zero? remainder)
      n
      (+ n (- size remainder)))))

;; To the use the GPU we need to setup a context through which we will interact with the GPU.
;; There can be potentially many GPUs and we want to be able to leverage all of them to gain the 
;; maximum possible speed.
(def gpu-context  (atom nil))
(def gpu-contexts (atom nil))

(defn setup-device
  "Sets up an OpenCL device."
  ([k distance-configuration length dev]
   (let [ctx (context [dev])
         channel (chan)
         cqueue (command-queue ctx dev)
         otype-str (when k (-> k size->bytes bytes->type))
         program-source (-> distance-configuration :program io/resource slurp)
         program (program-with-source ctx [program-source])
         prog (try
                (build-program! program (str "-DSIZE=" length) channel)
                (catch Exception _
                  (println (build-log program dev))))
         min-program-source (slurp (io/resource "min_index.c"))
         min-program (program-with-source ctx [min-program-source])
         min-prog (when otype-str
                    (try
                      (build-program!
                       min-program
                       (str "-DOTYPE=" otype-str)
                       channel)
                      (catch Exception _
                        (println (build-log min-program dev)))))
         sum-program-source (slurp (io/resource "centroids.c"))
         sum-program (program-with-source ctx [sum-program-source])
         sum-prog (try
                    (build-program! sum-program (str "-DSIZE=" length) channel)
                    (catch Exception _
                      (println (build-log sum-program dev))))
         ;; Fused distance+argmin kernel — compile alongside the two-pass kernels
         fused-program-source (when otype-str
                                (some-> distance-configuration :fused-program io/resource slurp))
         fused-program (when fused-program-source
                         (program-with-source ctx [fused-program-source]))
         fused-prog (when fused-program
                      (try
                        (build-program! fused-program
                                        (str "-DSIZE=" length " -DOTYPE=" otype-str)
                                        channel)
                        (catch Exception e
                          (println "Failed to compile fused kernel:" (.getMessage e))
                          (when fused-program (println (build-log fused-program dev)))
                          nil)))
         gpu-context-map {:chan channel
                          :dev dev
                          :kernel  (-> distance-configuration :kernel)
                          :ctx ctx
                          :cqueue cqueue
                          :program program
                          :prog prog
                          :min-program min-program
                          :min-prog min-prog
                          :min-kernel "minimum_index"
                          :sum-program sum-program
                          :sum-prog sum-prog
                          :sum-kernel "sum_by_group"
                          :fused-program fused-program
                          :fused-prog fused-prog
                          :fused-kernel (-> distance-configuration :fused-kernel)}]
     (reset! gpu-context gpu-context-map)
     gpu-context-map)))


(defn get-device-context
  [configuration matrix]
  (let [device  (->
                 (platforms)
                 (first)
                 (devices)
                 (first))
        distance-configuration (-> configuration
                                   :distance-key
                                   gpu-accelerated)
        k (-> configuration :k)]
    (setup-device k distance-configuration (ncols matrix) device)))


;; During Lloyd iteration it is common to re-use the same centroids while processing 
;; the same distance calculation.  In order to avoid having to write and rewrite the 
;; same centroid buffer once per sequence we split the writing of the centroid buffer 
;; from the writing of the dataset.
(defn write-centroids-buffer!
  [gpu-context centroids]
  (let [cqueue (:cqueue @gpu-context)
        ctx (:ctx @gpu-context)
        k (mrows centroids)
        cols (ncols centroids)
        cl-centroids (cl-buffer ctx (* k cols Float/BYTES) :read-only) 
        ^FloatPointer centroids-ptr (buffer centroids)
        centroids-buffer (float-pointer-byte-buffer centroids-ptr)]

    (enq-write! cqueue cl-centroids centroids-buffer)
    (swap! gpu-context assoc :cl-centroids cl-centroids)
    (swap! gpu-context assoc :k k)))

;; After processing an entire dataset sequence we need to clear the buffer so that 
;; we don't hold onto more resources than necessary.
(defn release-centroids-buffer!
  [gpu-context]
  (clojurecl/release (:cl-centroids @gpu-context))
  (swap! gpu-context assoc :cl-centroids nil)
  (swap! gpu-context assoc :k nil))

(defn centroid-feature-columns
  [centroids-ds]
  (remove #{:assignments "assignments"} (ds/column-names centroids-ds)))


(defmacro with-centroids
  [centroids-ds & forms]
  `(let [centroids# ~centroids-ds
         centroid-features# (ds/select-columns centroids# (centroid-feature-columns centroids#))]
     (write-centroids-buffer!
      gpu-context
      (dataset->matrix centroid-features#))
     (try
       ~@forms
       (finally
         (release-centroids-buffer! gpu-context)))))


(defn teardown-device
  "Tearsdown an OpenCL device."
  ([] (teardown-device @gpu-context))
  ([device-context]
   (doseq [k [:dev :ctx :cqueue :program :prog :min-program :min-prog :sum-program :sum-prog]]
     (clojurecl/release (k device-context)))
   ;; Release fused kernel resources if they were compiled
   (when-let [fp (:fused-program device-context)] (clojurecl/release fp))
   (when-let [fp (:fused-prog device-context)] (clojurecl/release fp))))


(defmacro with-gpu-context
  [conf & forms]
  `(do
     (get-device-context ~conf
                          (dataset->matrix
                           ~conf 
                           (if (not (nil? (:centroids ~conf)))
                             (:centroids ~conf)
                             (first (p/read-dataset-seq ~conf :points)))))
     (try
       ~@forms
       (finally
         (teardown-device)))))

;; When doing k means clustering there are three types of distances calculations 
;; that typically need to be done.  These are:
;; 
;; 1. Distance Calculation - Single Point
;; 
;; This typically happens during initialization methods that use the distance 
;; metric as a heuristic for approximating good initial clusters.
;; 
;; 2. Multi-Distance Calculation - Multipoint Minimum or Maximum
;; 
;; This type of distance calculation is like the previous distance calculation 
;; except that multiple distances are calculated and the minimum or maximum 
;; distance is returned rather than returning just one distance.  Again, this 
;; shows up during more sophisticated initialization schemes.
;;
;; 3. Multi-Distance Minimum Index
;;
;; This type of distance calculation is like the previous multi-point distance 
;; calculation but instead of returning the actual distance itself the returned 
;; value is the index of the distance.  This is the distance calculation method 
;; used when assigning clusterings during typical k means iterations.
(defn gpu-distance
  "Evaluates many distances in parallel."
  ([device-context matrix]
   (let [num-clusters (long (:k device-context))
         n (long (mrows matrix))
         cols (long (ncols matrix))
         num-distances (* n num-clusters)
         global-size (long 1024)
         num-per (work-item-batches n global-size)
         global-work-size [global-size]
         work-size (work-size global-work-size)
         ^java.nio.ByteBuffer host-msg (direct-buffer (* num-distances Float/BYTES))
         ^FloatPointer matrix-ptr (buffer matrix)
         matrix-array (float-array (.capacity matrix-ptr))
         _ (.get matrix-ptr matrix-array)
         cqueue (:cqueue device-context)
         cl-centroids (:cl-centroids device-context)]
     (with-release [cl-result (cl-buffer (:ctx device-context) (* num-distances Float/BYTES) :write-only)
                    cl-matrix (cl-buffer (:ctx device-context) (* n cols Float/BYTES) :read-only)
                    cl-kernel (kernel (:prog device-context) (:kernel device-context))]
       (set-args! cl-kernel cl-result cl-matrix cl-centroids (int-array [num-per]) (int-array [n]) (int-array [num-clusters]))
       (enq-write! cqueue cl-matrix matrix-array)
       (enq-kernel! cqueue cl-kernel work-size)
       (enq-read! cqueue cl-result host-msg)
       (finish! cqueue)
       (let [^java.nio.FloatBuffer data (.asFloatBuffer host-msg)
             res (float-array num-distances)]
         (dotimes [i num-distances]
           (aset ^floats res i (float (.get data))))
         res))))

  ([device-context matrix centroids]
   (let [num-clusters (long (mrows centroids))
         n (long (mrows matrix))
         cols (long (ncols matrix))
         centroid-cols (long (ncols centroids))
         num-distances (* n num-clusters)
         global-size (long 1024)
         num-per (work-item-batches n global-size)
         global-work-size [global-size]
         work-size (work-size global-work-size)
         ^java.nio.ByteBuffer host-msg (direct-buffer (* num-distances Float/BYTES))
         ^FloatPointer matrix-ptr (buffer matrix)
         ^FloatPointer centroids-ptr (buffer centroids)
         matrix-array (float-array (.capacity matrix-ptr))
         centroids-array (float-array (.capacity centroids-ptr))
         _ (.get matrix-ptr matrix-array)
         _ (.get centroids-ptr centroids-array)
         cqueue (:cqueue device-context)]
     (with-release [cl-result (cl-buffer (:ctx device-context) (* num-distances Float/BYTES) :write-only)
                    cl-matrix (cl-buffer (:ctx device-context) (* n cols Float/BYTES) :read-only)
                    cl-centroids (cl-buffer (:ctx device-context) (* num-clusters centroid-cols Float/BYTES) :read-only)
                    cl-kernel (kernel (:prog device-context) (:kernel device-context))]
       (set-args! cl-kernel cl-result cl-matrix cl-centroids (int-array [num-per]) (int-array [n]) (int-array [num-clusters]))
       (enq-write! cqueue cl-matrix matrix-array)
       (enq-write! cqueue cl-centroids centroids-array)
       (enq-kernel! cqueue cl-kernel work-size)
       (enq-read! cqueue cl-result host-msg)
       (finish! cqueue)
       (let [^java.nio.FloatBuffer data (.asFloatBuffer host-msg)
             res (float-array num-distances)]
         (dotimes [i num-distances]
           (aset ^floats res i (float (.get data))))
         res)))))


(defn create-min-index-buffer
  "The number of clusters determines the type of the assignments."
  [context ^long cluster-count ^long row-count]
  (cl-buffer context (* row-count (long (size->bytes cluster-count))) :write-only))


(defn create-min-index-result-array
  [^long cluster-count ^long num-rows]
  (let [array-type (case (int (size->bytes cluster-count))
                     1
                     byte-array
                     2
                     short-array
                     4
                     int-array)]
    (array-type num-rows)))


(defn gpu-distance-min-index
  "Evaluates many distances in parallel."
  ([device-context matrix]
   (let [num-clusters (long (:k device-context))
         n (long (mrows matrix))
         cols (long (ncols matrix))
         num-distances (* n num-clusters)
         global-size (long 1024)
         num-per (work-item-batches n global-size)
         global-work-size [global-size]
         work-size (work-size global-work-size)
         ^FloatPointer matrix-ptr (buffer matrix)
         matrix-array (float-array (.capacity matrix-ptr))
         _ (.get matrix-ptr matrix-array)
         min-indices (create-min-index-result-array num-clusters n)
         cqueue (:cqueue device-context)
         cl-centroids (:cl-centroids device-context)]

     (with-release [cl-result (cl-buffer (:ctx device-context) (* num-distances Float/BYTES) :read-write)
                    cl-matrix (cl-buffer (:ctx device-context) (* n cols Float/BYTES) :read-only)]
       (with-release [cl-kernel (kernel (:prog device-context) (:kernel device-context))]
         (set-args! cl-kernel cl-result cl-matrix cl-centroids
                    (int-array [num-per]) (int-array [n]) (int-array [num-clusters]))
         (enq-write! cqueue cl-matrix matrix-array)
         (enq-kernel! cqueue cl-kernel work-size)
         (finish! cqueue))

       (with-release [cl-min-indexes (create-min-index-buffer (:ctx device-context) num-clusters n)
                      cl-min-kernel (kernel (:min-prog device-context) (:min-kernel device-context))]
         (set-args! cl-min-kernel cl-result cl-min-indexes
                    (int-array [num-per]) (int-array [n]) (int-array [num-clusters]))
         (enq-kernel! cqueue cl-min-kernel work-size)
         (enq-read! cqueue cl-min-indexes min-indices)
         (finish! cqueue)
         min-indices))))

  ([device-context matrix assignments points]
   (let [num-clusters (long (:k device-context))
         n (long (mrows matrix))
         num-distances (* n num-clusters)
         global-size (long 1024)
         num-per (work-item-batches n global-size)
         global-work-size [global-size]
         work-size (work-size global-work-size)
         ^FloatPointer matrix-ptr (buffer matrix)
         matrix-array (float-array (.capacity matrix-ptr))
         _ (.get matrix-ptr matrix-array)
         cqueue (:cqueue device-context)
         min-indices (create-min-index-result-array num-clusters n)
         cl-centroids (:cl-centroids device-context)]

     (with-release [cl-result (cl-buffer (:ctx device-context) (* num-distances Float/BYTES) :read-write)]
       (with-release [cl-kernel (kernel (:prog device-context) (:kernel device-context))]
         (set-args! cl-kernel cl-result points cl-centroids
                    (int-array [num-per]) (int-array [n]) (int-array [num-clusters]))
         (enq-kernel! cqueue cl-kernel work-size)
         (finish! cqueue))

       (with-release [cl-min-kernel (kernel (:min-prog device-context) (:min-kernel device-context))]
         (set-args! cl-min-kernel cl-result assignments
                    (int-array [num-per]) (int-array [n]) (int-array [num-clusters]))
         (enq-kernel! cqueue cl-min-kernel work-size)
         (enq-read! cqueue assignments min-indices)
         (finish! cqueue)
         min-indices)))))



;; Fused distance+argmin kernel — computes assignments in a single pass
;; without materializing the N*K distance matrix in global memory.
;; This is the key optimization from the Flash-KMeans paper (arXiv:2603.09229):
;; each work-item streams through all centroids while maintaining a running
;; minimum in registers, writing only the final assignment index.
(defn gpu-fused-assign
  "Computes distance + argmin in a single kernel, returning assignment indices.
   Eliminates the intermediate N*K distance matrix entirely."
  ([device-context matrix]
   (gpu-fused-assign device-context matrix (or (:fused-local-size device-context) 128)))
  ([device-context matrix local-size]
   (let [num-clusters (long (:k device-context))
         n (long (mrows matrix))
         cols (long (ncols matrix))
         local-size (long local-size)
         _ (when-not (pos? local-size)
             (throw (IllegalArgumentException. "local-size must be positive")))
         global-size (long (round-up-to-multiple n local-size))
         global-work-size [global-size]
         local-work-size [local-size]
         work-size (work-size global-work-size local-work-size)
         ^FloatPointer matrix-ptr (buffer matrix)
         matrix-buffer (float-pointer-byte-buffer matrix-ptr)
         min-indices (create-min-index-result-array num-clusters n)
         cqueue (:cqueue device-context)
         cl-centroids (:cl-centroids device-context)]
     (with-release [cl-matrix (cl-buffer (:ctx device-context) (* n cols Float/BYTES) :read-only)
                    cl-assignments (create-min-index-buffer (:ctx device-context) num-clusters n)
                    cl-kernel (kernel (:fused-prog device-context) (:fused-kernel device-context))]
       (set-args! cl-kernel cl-assignments cl-matrix cl-centroids
                  (int-array [1]) (int-array [n]) (int-array [num-clusters]))
       (enq-write! cqueue cl-matrix matrix-buffer)
       (enq-kernel! cqueue cl-kernel work-size)
       (enq-read! cqueue cl-assignments min-indices)
       (finish! cqueue)
       min-indices))))


(defn fused-minimum-index
  "Returns nearest-centroid indices using the fused distance+argmin kernel."
  [conf ds]
  (let [ds-matrix (dataset->matrix conf ds)]
    (gpu-fused-assign @gpu-context ds-matrix)))


;; The GPU is so much faster than the CPU that we should be preferring
;; the way it approaches things to the way that the CPU might break down
;; the problem.  So our CPU outerloop returns arrays of floats just like
;; the GPU does.
(defn distances
  "Returns a vector of distance of the centroids from the point."
  [centroids distance-fn point]
  (map (partial distance-fn point) centroids))


(defn cpu-distance
  [configuration dataset centroids]
  (let [distance-fn (get-distance-fn (:distance-key configuration))
        centroids (ds/rowvecs centroids)
        distances-fn (partial distances centroids distance-fn)
        points (ds/rowvecs dataset)]
    (into-array Float/TYPE (mapcat distances-fn points))))


(defn minimum-distance
  [conf ds centroid-ds]
  (let [ds-matrix (dataset->matrix conf ds) 
        row-count (ds/row-count ds)
        centroid-matrix (dataset->matrix conf centroid-ds)
        col-count (ds/row-count centroid-ds) 
        distances (fge row-count col-count (gpu-distance @gpu-context ds-matrix centroid-matrix) {:layout :row})]
    (fv (hfln/map (fn [m] (entry m (imin m))) (rows distances)))))


(defn minimum-index
  [conf ds]
  (let [ds-matrix (dataset->matrix conf ds)]
    (gpu-distance-min-index @gpu-context ds-matrix)))
