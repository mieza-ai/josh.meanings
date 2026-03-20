(ns mieza.meanings.init-utils-test
  "Tests for initialization utility functions."
  (:require [clojure.test :refer [deftest testing is]]
            [mieza.meanings.initializations.utils :as utils]
            [mieza.meanings.records.clustering-state :refer [->KMeansState]]
            [tech.v3.dataset :as ds]))

;; Helper to make a minimal KMeansState for testing
(defn make-state [overrides]
  (merge (->KMeansState
          3           ;; k
          "test.csv"  ;; points
          :arrow      ;; format
          :afk-mc     ;; init
          :euclidean  ;; distance-key
          nil         ;; m (chain length)
          nil         ;; k-means fn
          100         ;; size-estimate
          ["x" "y"]   ;; col-names
          false)      ;; use-gpu
         overrides))

;; --- vector->dataset ---

(deftest test-vector->dataset
  (let [data [[1.0 2.0] [3.0 4.0] [5.0 6.0]]
        ds (utils/vector->dataset data ["x" "y"])]
    (is (ds/dataset? ds))
    (is (= 3 (ds/row-count ds)))
    (is (= #{"x" "y"} (set (ds/column-names ds))))))

;; --- centroids->dataset ---

(deftest test-centroids->dataset
  (let [state (make-state {:k 2})
        points [[1.0 2.0] [3.0 4.0]]
        ds (utils/centroids->dataset state points)]
    (is (ds/dataset? ds))
    (is (= 2 (ds/row-count ds)))
    (is (= #{"x" "y"} (set (ds/column-names ds))))))

;; --- uniform-sample ---

(deftest test-uniform-sample
  (let [ds1 (ds/->dataset {"x" (range 100) "y" (range 100)})
        samples (utils/uniform-sample [ds1] 10)]
    (is (= 10 (count samples)) "Should sample exactly 10 points")
    (is (every? sequential? samples) "Each sample is a sequence")))

(deftest test-uniform-sample-less-than-available
  (let [ds1 (ds/->dataset {"x" (range 5) "y" (range 5)})
        samples (utils/uniform-sample [ds1] 3)]
    (is (= 3 (count samples)))))

;; --- chain length ---

(deftest test-should-update-chain-length
  (is (true? (utils/should-update-chain-length? (make-state {:m nil})))
      "Should update when m is nil")
  (is (false? (utils/should-update-chain-length? (make-state {:m 50})))
      "Should not update when m is already set"))

(deftest test-update-chain-length
  (let [state (make-state {:k 5 :m nil})
        updated (utils/update-chain-length state)]
    (is (number? (:m updated)) "Chain length should be a number")
    (is (pos? (:m updated)) "Chain length should be positive")
    (is (< (:m updated) (:size-estimate state))
        "Chain length should be less than dataset size")))

(deftest test-add-default-chain-length
  (let [state (make-state {:k 3 :m nil})
        with-m (utils/add-default-chain-length state)]
    (is (number? (:m with-m)))
    (is (pos? (:m with-m)))))

(deftest test-add-default-chain-length-preserves-existing
  (let [state (make-state {:k 3 :m 42})
        with-m (utils/add-default-chain-length state)]
    (is (= 42 (:m with-m))
        "Should not override existing chain length")))

;; --- weighted-sample ---

(deftest test-weighted-sample
  (let [ds1 (ds/->dataset {"x" (range 50)
                            "y" (range 50)
                            "weight" (repeat 50 1.0)})
        samples (utils/weighted-sample [ds1] "weight" 5)]
    (is (= 5 (count samples)) "Should return exactly k samples")
    (is (every? map? samples) "Each sample is a map")))

;; --- sample-one ---

(deftest test-sample-one
  (let [state (make-state {:k 3 :m 50 :size-estimate 10})
        _ (spit "test_sample_one.csv" "x,y\n1,2\n3,4\n5,6\n7,8\n9,10")
        state (assoc state :points "test_sample_one.csv" :col-names ["x" "y"])
        result (utils/sample-one state)]
    (is (ds/dataset? result) "Returns a dataset")
    (is (= 1 (ds/row-count result)) "Returns exactly one row")
    (clojure.java.io/delete-file "test_sample_one.csv" true)))
