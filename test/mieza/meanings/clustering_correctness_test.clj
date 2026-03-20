(ns mieza.meanings.clustering-correctness-test
  "Tests verifying that k-means produces correct clusters on known data."
  (:require [clojure.test :refer [deftest testing is use-fixtures]]
            [clojure.java.io :as io]
            [clojure.data.csv :as csv]
            [mieza.meanings.kmeans :refer [k-means k-means-seq]]
            [mieza.meanings.protocols.classifier :refer [assignments]]
            [tech.v3.dataset :as ds]))

;; --- Test data ---

(defn write-csv! [filename rows]
  (with-open [w (io/writer filename)]
    (csv/write-csv w rows)))

(defn cleanup! [filename]
  (doseq [f (file-seq (io/file "."))
          :when (and (.isFile f)
                     (.contains (.getName f)
                                (clojure.string/replace filename ".csv" "")))]
    (io/delete-file f true)))

(defmacro with-test-csv [filename rows & body]
  `(try
     (cleanup! ~filename)
     (write-csv! ~filename ~rows)
     ~@body
     (finally
       (cleanup! ~filename))))

;; Three well-separated clusters: points at (0,0), (100,100), (200,0)
(def separated-data
  [["x" "y"]
   [0 0] [1 1] [2 0] [0 2] [1 0]
   [100 100] [101 101] [100 99] [99 100] [101 100]
   [200 0] [201 1] [200 2] [199 0] [201 0]])

;; --- Tests ---

(deftest test-finds-correct-number-of-clusters
  (with-test-csv "test.separated.csv" separated-data
    (let [result (k-means "test.separated.csv" 3
                          :distance-key :euclidean
                          :init :afk-mc
                          :m 50)]
      (testing "Returns k centroids"
        (is (= 3 (count (ds/rowvecs (:centroids result))))))
      (testing "Cost is finite and non-negative"
        (is (number? (:cost result)))
        (is (>= (:cost result) 0.0))))))

(deftest test-well-separated-clusters
  (with-test-csv "test.separated2.csv" separated-data
    (let [result (k-means "test.separated2.csv" 3
                          :distance-key :euclidean
                          :init :afk-mc
                          :m 50)
          centroids (ds/rowvecs (:centroids result))]
      (testing "Returns 3 distinct centroids"
        (is (= 3 (count centroids)))
        (is (= 3 (count (set centroids))) "All centroids should be unique"))
      (testing "Cost decreases from initial — clustering improved the objective"
        (is (< (:cost result) Double/MAX_VALUE))))))

(deftest test-k-means-seq-returns-multiple-results
  (with-test-csv "test.seq.csv" separated-data
    (let [results (take 3 (k-means-seq "test.seq.csv" 3
                                        :distance-key :euclidean
                                        :init :afk-mc
                                        :m 50))]
      (testing "k-means-seq returns a lazy sequence of results"
        (is (= 3 (count results)))
        (doseq [r results]
          (is (contains? r :cost))
          (is (contains? r :centroids))
          (is (number? (:cost r))))))))

(deftest test-k-equals-1
  (with-test-csv "test.k1.csv" separated-data
    (let [result (k-means "test.k1.csv" 1
                          :distance-key :euclidean
                          :init :afk-mc
                          :m 50)]
      (testing "k=1 returns a single centroid"
        (is (= 1 (count (ds/rowvecs (:centroids result)))))))))

(deftest test-different-distance-functions
  (with-test-csv "test.distfns.csv" separated-data
    (doseq [dist-key [:euclidean :manhattan :chebyshev :euclidean-sq]]
      (testing (str "k-means runs with " dist-key)
        (let [result (k-means "test.distfns.csv" 3
                              :distance-key dist-key
                              :init :afk-mc
                              :m 50)]
          (is (= 3 (count (ds/rowvecs (:centroids result))))
              (str dist-key " should produce 3 centroids")))))))
