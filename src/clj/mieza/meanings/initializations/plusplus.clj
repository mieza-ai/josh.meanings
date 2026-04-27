(ns mieza.meanings.initializations.plusplus
  (:require
   [taoensso.timbre :as log]
   [mieza.meanings.persistence :as persist]
   [clojure.spec.alpha :as s]
   [mieza.meanings.initializations.utils :refer
    [centroids->dataset uniform-sample weighted-sample add-d2-weights d2-weight-col]]
   [mieza.meanings.initializations.core :refer [initialize-centroids]]))

(def t-config :mieza.meanings.specs/configuration)
(def t-dataset :mieza.meanings.specs/dataset)

(defn- weighted-sample-vecs
  "Weighted sample that returns row vectors (matching uniform-sample output),
   extracting only the data columns specified in config."
  [config ds-seq centers]
  (let [col-names (:col-names config)
        rows (weighted-sample (add-d2-weights config ds-seq centers)
                              d2-weight-col
                              1)]
    (mapv (fn [row] (mapv #(get row %) col-names)) rows)))

(s/fdef k-means-++ :args (s/cat :config t-config) :ret t-dataset)
(defn- k-means-++
  [config]
  (log/info "Performing k means++ initialization")
  (centroids->dataset
   config
   (loop [centers (uniform-sample (persist/read-dataset-seq config :points) 1)]
     (if (= (:k config) (count centers))
       centers
       (recur (concat centers
                      (weighted-sample-vecs config
                                           (persist/read-dataset-seq config :points)
                                           centers)))))))

(defmethod initialize-centroids
  :k-means-++
  [k-means-state]
  (k-means-++ k-means-state))
