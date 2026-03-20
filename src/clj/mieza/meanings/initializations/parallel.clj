(ns mieza.meanings.initializations.parallel
  (:require
   [taoensso.timbre :as log]
   [mieza.meanings.persistence :as persist]
   [clojure.spec.alpha :as s]
   [mieza.meanings.initializations.utils :refer
    [uniform-sample weighted-sample add-d2-weights d2-weight-col]])
  (:use
   [mieza.meanings.initializations.core]))

(def t-config :mieza.meanings.specs/configuration)
(def t-dataset :mieza.meanings.specs/dataset)

(defn- weighted-sample-vecs
  "Weighted sample that returns row vectors, extracting only data columns."
  [config ds-seq centers n]
  (let [col-names (:col-names config)
        rows (weighted-sample (add-d2-weights config ds-seq centers)
                              d2-weight-col
                              n)]
    (mapv (fn [row] (mapv #(get row %) col-names)) rows)))

(s/fdef k-means-parallel :args (s/cat :config t-config) :ret t-dataset)
(defn k-means-parallel
  [config]
  (log/info "Performing k means parallel initialization")
  (let [ds-seq (persist/read-dataset-seq config :points)
        k (:k config)
        oversample-factor (* 2 k)
        iterations 5
        rows->maps (partial persist/ds-seq->rows->maps ds-seq)
        k-means (:k-means config)]
    (loop [i 0 centers (uniform-sample ds-seq 1)]
      (if (= i iterations)
        (do
          (log/info "Finished oversampling. Reducing to k centroids")
          (:centroids (k-means (rows->maps centers) k :init :k-means-++ :distance-fn (:distance-key config))))
        (recur (inc i) (concat centers
                               (weighted-sample-vecs config ds-seq centers oversample-factor)))))))


(defmethod initialize-centroids
  :k-means-parallel
  [config]
  (k-means-parallel config))
