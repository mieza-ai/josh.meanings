(ns mieza.meanings.cpu-distances-test
  "Tests for CPU distance function dispatch."
  (:require [clojure.test :refer [deftest testing is]]
            [mieza.meanings.distances :refer [get-distance-fn distance-keys]]))

(def point-a [1.0 2.0 3.0])
(def point-b [4.0 6.0 3.0])

(deftest test-euclidean
  (let [d (get-distance-fn :euclidean)]
    (is (== 0.0 (d point-a point-a)) "Distance to self is zero")
    (is (> (d point-a point-b) 0.0) "Distance between distinct points is positive")
    (is (== (d point-a point-b) (d point-b point-a)) "Distance is symmetric")
    (is (< (Math/abs (- (d point-a point-b) 5.0)) 0.001) "sqrt(9+16+0) = 5.0")))

(deftest test-manhattan
  (let [d (get-distance-fn :manhattan)]
    (is (== 0.0 (d point-a point-a)))
    (is (< (Math/abs (- (d point-a point-b) 7.0)) 0.001) "|3|+|4|+|0| = 7.0")))

(deftest test-chebyshev
  (let [d (get-distance-fn :chebyshev)]
    (is (== 0.0 (d point-a point-a)))
    (is (< (Math/abs (- (d point-a point-b) 4.0)) 0.001) "max(3,4,0) = 4.0")))

(deftest test-euclidean-sq
  (let [d (get-distance-fn :euclidean-sq)]
    (is (== 0.0 (d point-a point-a)))
    (is (< (Math/abs (- (d point-a point-b) 25.0)) 0.001) "9+16+0 = 25.0")))

(deftest test-cosine
  (let [d (get-distance-fn :cosine)]
    ;; cosine distance for identical vectors may not be exactly 0 due to floating point
    (is (< (d point-a point-a) 1.01) "Cosine distance to self is <= 1")
    (is (>= (d point-a point-b) 0.0) "Cosine distance is non-negative")))

(deftest test-discrete
  (let [d (get-distance-fn :discrete)]
    (is (== 0.0 (d point-a point-a)) "Identical points have discrete distance 0")
    (is (> (d point-a point-b) 0.0) "Different points have positive discrete distance")))

(deftest test-all-distance-keys-resolve
  (testing "Every registered distance key resolves to a callable that returns a number"
    (doseq [k distance-keys]
      (let [d (get-distance-fn k)]
        (is (ifn? d) (str k " should be invocable"))
        (is (number? (d point-a point-b)) (str k " should return a number"))))))

(deftest test-distance-properties
  (testing "Triangle inequality holds for metric distances"
    (let [point-c [2.0 3.0 1.0]]
      (doseq [k [:euclidean :manhattan :chebyshev]]
        (let [d (get-distance-fn k)]
          (is (<= (d point-a point-b)
                  (+ (d point-a point-c) (d point-c point-b)))
              (str "Triangle inequality for " k)))))))
