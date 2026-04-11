(ns mieza.meanings.model-persistence-test
  "Tests for ClusterResult save/load round-trip and classification."
  (:require [clojure.test :refer [deftest testing is]]
            [clojure.java.io :as io]
            [clojure.data.csv :as csv]
            [mieza.meanings.kmeans :refer [k-means]]
            [mieza.meanings.protocols.savable :refer [save-model]]
            [mieza.meanings.protocols.classifier :refer [assignments]]
            [mieza.meanings.records.cluster-result :refer [load-model]]
            [tech.v3.dataset :as ds]))

(defn write-csv! [filename rows]
  (with-open [w (io/writer filename)]
    (csv/write-csv w rows)))

(defn cleanup! [& patterns]
  (doseq [f (file-seq (io/file "."))
          :when (and (.isFile f)
                     (some #(.contains (.getName f) %) patterns))]
    (io/delete-file f true)))

(defmacro with-cleanup [patterns & body]
  `(try
     (apply cleanup! ~patterns)
     ~@body
     (finally
       (apply cleanup! ~patterns))))

(def test-data
  [["x" "y"]
   [0 0] [1 1] [2 0]
   [100 100] [101 101] [100 99]
   [200 0] [201 1] [200 2]])

(defn make-model []
  (write-csv! "test.model.csv" test-data)
  (k-means "test.model.csv" 3
           :distance-key :euclidean
           :init :afk-mc
           :m 50))

;; --- Save and Load ---

(deftest test-save-load-round-trip
  (with-cleanup ["test.model"]
    (let [model (make-model)
          save-path "test.model.result.edn"]
      (save-model model save-path)
      (testing "Model file is created"
        (is (.exists (io/file save-path))))
      (let [loaded (load-model save-path)]
        (testing "Loaded model has same cost"
          (is (== (:cost model) (:cost loaded))))
        (testing "Loaded model has same number of centroids"
          (is (= (ds/row-count (:centroids model))
                 (ds/row-count (:centroids loaded)))))
        (testing "Configuration is preserved"
          (is (= (:k (:configuration model))
                 (:k (:configuration loaded))))
          (is (= (:distance-key (:configuration model))
                 (:distance-key (:configuration loaded))))
          (is (= (:init (:configuration model))
                 (:init (:configuration loaded)))))))))

;; --- ClusterResult fields ---

(deftest test-cluster-result-fields
  (with-cleanup ["test.fields"]
    (write-csv! "test.fields.csv" test-data)
    (let [model (k-means "test.fields.csv" 3
                         :distance-key :euclidean
                         :init :afk-mc
                         :m 50)]
      (testing ":centroids is a dataset"
        (is (ds/dataset? (:centroids model))))
      (testing ":cost is a non-negative number"
        (is (number? (:cost model)))
        (is (>= (:cost model) 0.0)))
      (testing ":configuration contains required keys"
        (let [conf (:configuration model)]
          (is (contains? conf :k))
          (is (contains? conf :m))
          (is (contains? conf :distance-key))
          (is (contains? conf :init))
          (is (contains? conf :col-names))
          (is (= 3 (:k conf))))))))

;; --- Multiple saves don't corrupt ---

(deftest test-save-twice-loads-correctly
  (with-cleanup ["test.twice"]
    (write-csv! "test.twice.csv" test-data)
    (let [model (k-means "test.twice.csv" 3
                         :distance-key :euclidean
                         :init :afk-mc
                         :m 50)]
      (save-model model "test.twice.v1.edn")
      (save-model model "test.twice.v2.edn")
      (let [v1 (load-model "test.twice.v1.edn")
            v2 (load-model "test.twice.v2.edn")]
        (is (== (:cost v1) (:cost v2)))
        (is (= (ds/row-count (:centroids v1))
               (ds/row-count (:centroids v2))))))))
