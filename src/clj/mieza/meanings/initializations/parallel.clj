(ns mieza.meanings.initializations.parallel
  (:require
   [taoensso.timbre :as log]
   [mieza.meanings.persistence :as persist]
   [clojure.spec.alpha :as s]
   [tech.v3.dataset :as ds]
   [mieza.meanings.initializations.utils :refer
    [uniform-sample weighted-sample add-d2-weights d2-weight-col]]
   [mieza.meanings.initializations.core :refer [initialize-centroids]]))

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
        k (long (:k config))
        oversample-factor (* 2 k)
        iterations 5
        k-means (:k-means config)]
    (loop [i 0 centers (uniform-sample ds-seq 1)]
      (if (= i iterations)
        (do
          (log/info "Finished oversampling. Reducing to k centroids")
          (let [center-ds (ds/->dataset (map (partial zipmap (:col-names config)) centers))
                tmp-file (str (java.util.UUID/randomUUID) ".arrow")]
            (persist/write-dataset tmp-file center-ds)
            (:centroids (k-means tmp-file k :init :k-means-++ :distance-key (:distance-key config)))))
        (recur (inc i) (concat centers
                               (weighted-sample-vecs config ds-seq centers oversample-factor)))))))


(defmethod initialize-centroids
  :k-means-parallel
  [config]
  (k-means-parallel config))
