(ns mieza.meanings.initializations.plusplus
  (:require
   [taoensso.timbre :as log :refer [info]]
   [mieza.meanings.persistence :as persist]
   [clojure.spec.alpha :as s]
   [mieza.meanings.initializations.utils :refer
    [centroids->dataset uniform-sample weighted-sample shortest-distance-squared-*]])
  (:use
   [mieza.meanings.initializations.core]))

(def t-config :mieza.meanings.specs/configuration)
(def t-dataset :mieza.meanings.specs/dataset)

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
                      (weighted-sample (persist/read-dataset-seq config :points)
                                       (shortest-distance-squared-* config centers)
                                       1)))))))

(defmethod initialize-centroids
  :k-means-++
  [k-means-state]
  (k-means-++ k-means-state))