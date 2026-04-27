(ns mieza.meanings.initializations.naive
  "A random initialization strategy for k means which lacks theoretical 
   guarantees on solution quality for any individual run, but which will 
   complete in O(n + k*d) time and only takes O(k*d) space."
  (:require
   [taoensso.timbre :as log]
   [mieza.meanings.persistence :as persist]
   [mieza.meanings.initializations.utils :refer [centroids->dataset uniform-sample]]
   [clojure.spec.alpha :as s]
   [clojure.test :refer [is]]
   [mieza.meanings.initializations.core :refer [initialize-centroids]]))


;; Specs used in this file
(def t-config :mieza.meanings.specs/configuration)
(def t-dataset :mieza.meanings.specs/dataset)

(s/fdef naive-initialization :args (s/cat :config t-config) :ret t-dataset)
(defn- naive-initialization [config]
  {:pre [(is (s/valid? t-config config))] :post [(is (s/valid? t-dataset %))]}
  (log/info "Performing classical (naive) k means initialization")
  (centroids->dataset
   config
   (uniform-sample (persist/read-dataset-seq config :points) (:k config))))


(defmethod initialize-centroids
  :naive
  [k-means-state]
  (naive-initialization k-means-state))