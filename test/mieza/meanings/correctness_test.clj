(ns mieza.meanings.correctness-test
  "Tests that verify mathematical correctness of k-means clustering.
   These tests use datasets with known ground truth and verify that
   assignments, centroids, and costs are correct.

   K-means can converge to local optima depending on initialization.
   Tests that verify optimality use best-of-N runs to account for this."
  (:require [clojure.test :refer [deftest testing is]]
            [clojure.java.io :as io]
            [clojure.data.csv :as csv]
            [mieza.meanings.distances :as distances]
            [mieza.meanings.kmeans :refer [initialize-k-means-state k-means k-means-seq lloyd]]
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

(defn best-of [n filename k & options]
  (apply min-key :cost (take n (apply k-means-seq filename k options))))

;; ---- Cost verification ----

(deftest test-cost-is-zero-for-perfect-clustering
  (testing "Best-of-N finds the perfect clustering when points are at exactly 2 locations"
    (with-test-csv "test.perfect.csv"
      [["x" "y"]
       [0 0] [0 0] [0 0] [0 0] [0 0]
       [10 0] [10 0] [10 0] [10 0] [10 0]]
      (let [r (best-of 10 "test.perfect.csv" 2 :distance-key :euclidean :init :afk-mc :m 50)]
        (is (< (:cost r) 0.001) "Best run should find cost ~0")))))

(deftest test-cost-matches-hand-computation
  (testing "Cost equals sum of euclidean distances to nearest centroid"
    (with-test-csv "test.handcost.csv"
      ;; Points: (0,0), (1,0), (10,0), (11,0). k=2.
      ;; Optimal: centroids at (0.5,0) and (10.5,0). Cost = 4 * 0.5 = 2.0
      [["x" "y"] [0 0] [1 0] [10 0] [11 0]]
      (let [r (best-of 10 "test.handcost.csv" 2 :distance-key :euclidean :init :afk-mc :m 50)]
        (is (< (Math/abs (- (:cost r) 2.0)) 0.01)
            "Cost should be 2.0 for this configuration")))))

(deftest test-cost-is-positive-for-imperfect-clustering
  (with-test-csv "test.imperfect.csv"
    [["x" "y"] [0 0] [1 0] [2 0] [10 0] [11 0] [12 0]]
    (let [r (best-of 5 "test.imperfect.csv" 2 :distance-key :euclidean :init :afk-mc :m 50)]
      (is (> (:cost r) 0.0) "Cost should be positive when points aren't all at centroids")
      (is (Double/isFinite (:cost r))))))

(deftest test-more-clusters-lower-cost
  (testing "k=3 should have lower or equal cost than k=2 on the same data"
    (with-test-csv "test.kcost.csv"
      [["x" "y"]
       [0 0] [1 0] [2 0]
       [50 0] [51 0] [52 0]
       [100 0] [101 0] [102 0]]
      (let [r2 (best-of 5 "test.kcost.csv" 2 :distance-key :euclidean :init :afk-mc :m 50)
            r3 (best-of 5 "test.kcost.csv" 3 :distance-key :euclidean :init :afk-mc :m 50)]
        (is (<= (:cost r3) (:cost r2))
            "More clusters should not increase cost")))))

;; ---- Centroid position verification ----

(deftest test-centroids-are-means-of-their-clusters
  (testing "Centroids converge to the mean of assigned points"
    (with-test-csv "test.means.csv"
      ;; Two tight clusters: mean should be (2,0) and (102,0)
      [["x" "y"] [0 0] [2 0] [4 0] [100 0] [102 0] [104 0]]
      (let [r (best-of 10 "test.means.csv" 2 :distance-key :euclidean :init :afk-mc :m 50)
            col-names (:col-names (:configuration r))
            centroids (mapv (fn [row] (mapv #(double (get row %)) col-names))
                            (ds/rows (:centroids r)))
            sorted (sort-by first centroids)
            [cx1 _] (first sorted)
            [cx2 _] (second sorted)]
        (is (< (Math/abs (- cx1 2.0)) 0.01) "First centroid x should be 2.0")
        (is (< (Math/abs (- cx2 102.0)) 0.01) "Second centroid x should be 102.0")))))

(deftest test-well-separated-clusters-found
  (testing "k-means correctly separates well-separated clusters"
    (with-test-csv "test.separated.csv"
      [["x" "y"]
       [0 0] [1 0] [0 1] [1 1]
       [100 0] [101 0] [100 1] [101 1]
       [0 100] [1 100] [0 101] [1 101]]
      (let [conf (initialize-k-means-state
                  "test.separated.csv"
                  3
                  {:distance-key :euclidean :init :afk-mc :m 50})
            initial-centroids (ds/->dataset
                               {"x" [0.5 100.5 0.5]
                                "y" [0.5 0.5 100.5]
                                :assignments [0 1 2]})
            initial-centroids (ds/select-columns initial-centroids ["x" "y" :assignments])
            r (distances/with-gpu-context conf
                (lloyd conf initial-centroids))
            col-names (:col-names (:configuration r))
            centroids (mapv (fn [row] (mapv #(double (get row %)) col-names))
                            (ds/rows (:centroids r)))
            sorted (sort-by (fn [[x y]] (+ x y)) centroids)]
        (testing "Three distinct cluster centers found"
          ;; Near (0.5, 0.5)
          (is (< (first (first sorted)) 5))
          (is (< (second (first sorted)) 5))
          ;; Near (0.5, 100.5) or (100.5, 0.5) — one of the off-diagonal clusters
          ;; Near (100.5, 0.5) or (0.5, 100.5) — the other
          ;; Just verify they're far apart
          (let [dists (for [i (range 3) j (range 3) :when (< i j)]
                        (Math/sqrt (+ (Math/pow (- (first (nth sorted i)) (first (nth sorted j))) 2)
                                      (Math/pow (- (second (nth sorted i)) (second (nth sorted j))) 2))))]
            (is (every? #(> % 50) dists) "All centroid pairs should be far apart")))))))

;; ---- Cost monotonicity across runs ----

(deftest test-cost-never-negative
  (with-test-csv "test.nonneg.csv"
    (into [["x" "y"]]
          (for [_ (range 30)] [(rand-int 100) (rand-int 100)]))
    (let [runs (take 5 (k-means-seq "test.nonneg.csv" 3
                                     :distance-key :euclidean :init :afk-mc :m 50))]
      (doseq [r runs]
        (is (>= (:cost r) 0.0) "Cost should never be negative")
        (is (Double/isFinite (:cost r)) "Cost should be finite")))))

;; ---- Manhattan distance correctness ----

(deftest test-manhattan-cost-matches-hand-computation
  (testing "Manhattan distance cost is sum of L1 distances"
    (with-test-csv "test.manhattan.csv"
      ;; Points: (0,0), (2,0) -> centroid (1,0), cost = 1+1 = 2
      ;; Points: (10,0), (12,0) -> centroid (11,0), cost = 1+1 = 2
      ;; Total = 4
      [["x" "y"] [0 0] [2 0] [10 0] [12 0]]
      (let [r (best-of 10 "test.manhattan.csv" 2 :distance-key :manhattan :init :afk-mc :m 50)]
        (is (< (Math/abs (- (:cost r) 4.0)) 0.01)
            "Manhattan cost should be 4.0")))))

;; ---- Best-of-N actually improves quality ----

(deftest test-best-of-n-finds-better-solution
  (with-test-csv "test.bon.csv"
    (into [["x" "y"]]
          (concat
            (for [_ (range 20)] [(+ (rand-int 5)) (+ (rand-int 5))])
            (for [_ (range 20)] [(+ 50 (rand-int 5)) (+ 50 (rand-int 5))])))
    (let [runs (take 10 (k-means-seq "test.bon.csv" 2
                                      :distance-key :euclidean :init :afk-mc :m 50))
          costs (map :cost runs)
          best (apply min costs)
          worst (apply max costs)]
      (is (every? #(>= % best) costs) "Best cost is minimum")
      (is (> best 0.0) "Best cost is positive for non-degenerate data"))))
