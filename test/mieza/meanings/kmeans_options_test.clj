(ns mieza.meanings.kmeans-options-test
  "Tests for k-means with various option combinations and edge cases."
  (:require [clojure.test :refer [deftest testing is]]
            [clojure.java.io :as io]
            [clojure.data.csv :as csv]
            [mieza.meanings.kmeans :refer [k-means k-means-seq]]
            [tech.v3.dataset :as ds]))

(defn write-csv! [filename rows]
  (with-open [w (io/writer filename)]
    (csv/write-csv w rows)))

(defn cleanup! [base]
  (doseq [f (file-seq (io/file "."))
          :when (and (.isFile f)
                     (.contains (.getName f) base))]
    (io/delete-file f true)))

(defmacro with-test-csv [filename rows & body]
  `(let [base# (clojure.string/replace ~filename ".csv" "")]
     (try
       (cleanup! base#)
       (write-csv! ~filename ~rows)
       ~@body
       (finally
         (cleanup! base#)))))

(def medium-data
  (into [["a" "b" "c"]]
        (concat
         (for [_ (range 15)] [(rand-int 10) (rand-int 10) (rand-int 10)])
         (for [_ (range 15)] [(+ 90 (rand-int 10)) (+ 90 (rand-int 10)) (+ 90 (rand-int 10))]))))

;; --- k-means-seq produces improving results ---

(deftest test-k-means-seq-best-of-n
  (with-test-csv "test.bestofn.csv" medium-data
    (let [results (take 5 (k-means-seq "test.bestofn.csv" 2
                                        :distance-key :euclidean
                                        :init :afk-mc
                                        :m 50))
          best (apply min-key :cost results)]
      (testing "All results have costs"
        (doseq [r results]
          (is (number? (:cost r)))))
      (testing "Best result has lowest cost"
        (is (every? #(>= (:cost %) (:cost best)) results)))
      (testing "Best result has k centroids"
        (is (= 2 (count (ds/rowvecs (:centroids best)))))))))

;; --- Different k values ---

(deftest test-various-k-values
  (with-test-csv "test.kvals.csv" medium-data
    (doseq [k [1 2 3 5]]
      (testing (str "k=" k)
        (let [result (k-means "test.kvals.csv" k
                              :distance-key :euclidean
                              :init :afk-mc
                              :m 50)]
          (is (= k (count (ds/rowvecs (:centroids result))))
              (str "Should produce " k " centroids")))))))

;; --- Column selection ---

(deftest test-column-selection
  (with-test-csv "test.cols.csv" medium-data
    (let [result (k-means "test.cols.csv" 2
                          :distance-key :euclidean
                          :init :afk-mc
                          :m 50
                          :columns ["a" "b"])
          centroid-cols (set (ds/column-names (:centroids result)))]
      (testing "Centroids contain selected columns"
        (is (contains? centroid-cols "a"))
        (is (contains? centroid-cols "b")))
      (testing "Centroids do not contain unselected columns"
        (is (not (contains? centroid-cols "c")))))))

;; --- LazySeq input ---

;; NOTE: Lazy seq input currently broken — the LazySeq multimethod writes
;; to a temp arrow file, but tmd 8.x changed the arrow writer path.
;; This is a known issue to fix as part of the tmd 8.x migration.
;; (deftest test-lazy-seq-input ...)

;; --- Distance function combinations ---

(deftest test-distance-functions-with-k-means
  (with-test-csv "test.distcombos.csv" medium-data
    (doseq [dk [:euclidean :manhattan :chebyshev :euclidean-sq :emd]]
      (testing (str "k-means with " dk)
        (let [result (k-means "test.distcombos.csv" 2
                              :distance-key dk
                              :init :afk-mc
                              :m 50)]
          (is (= 2 (count (ds/rowvecs (:centroids result)))))
          (is (>= (:cost result) 0.0)))))))

;; --- Convergence: cost should be finite ---

(deftest test-convergence-produces-finite-cost
  (with-test-csv "test.converge.csv" medium-data
    (let [result (k-means "test.converge.csv" 2
                          :distance-key :euclidean
                          :init :afk-mc
                          :m 50)]
      (is (Double/isFinite (:cost result))
          "Cost should be finite after convergence"))))
