(ns mieza.meanings.distances-test
  (:require
   [clojure.string]
   [clojure.test :refer [deftest is testing]]
   [mieza.meanings.distances :refer [size->bytes cpu-distance gpu-distance
                                     get-device-context teardown-device
                                     centroid-feature-columns with-gpu-context
                                     with-centroids minimum-index]]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.neanderthal :as dsn]))


(defn near-equal?
  [^floats arr1 ^floats arr2 tolerance]
  (if (= (count arr1) (count arr2))
    (loop [i 0]
      (if (= i (count arr1))
        true
        (if (<= (Math/abs (- (aget arr1 i) (aget arr2 i))) tolerance)
          (recur (inc i))
          false)))
    false))


(defn str-float-array
  [arr]
  (let [arr-str (->> arr (map #(format "%.2f" %)) (clojure.string/join " "))]
    (str "[" arr-str "]")))


(deftest test-gpu-distance
  (testing "GPU distance calculations."
    (doseq [distance-key [:emd :euclidean :manhattan :chebyshev :euclidean-sq]]
      (doseq [test-config [{:dataset-size 10 :centroids 2}
                           {:dataset-size 100 :centroids 2}
                           {:dataset-size 100 :centroids 10}
                           {:dataset-size 1000 :centroids 30}
                           {:dataset-size 100000 :centroids 30}
                           {:dataset-size 100000 :centroids 100}]]
        (let [configuration {:distance-key distance-key :col-names ["x" "y" "z"]}
              gen-dataset (fn [n] (ds/->dataset (repeatedly n (fn [] {"x" (rand) "y" (rand) "z" (rand)}))))
              ds->matrix (fn [ds] (dsn/dataset->dense ds :row :float32))
              dataset (gen-dataset (:dataset-size test-config))
              centroids-dataset (gen-dataset (:centroids test-config))
              matrix (ds->matrix dataset)
              centroids-matrix (ds->matrix centroids-dataset)
              gpu-context (get-device-context configuration centroids-matrix)]
          (try
            (let [gpu-dist (gpu-distance gpu-context matrix centroids-matrix)
                  cpu-dist (cpu-distance configuration dataset centroids-dataset)]
              (is (near-equal? gpu-dist cpu-dist 0.01)
                  (str "Dataset "
                       (str-float-array gpu-dist)
                       (str-float-array cpu-dist))))
            (finally
              (teardown-device gpu-context))))))))


(deftest test-size->bytes
  (testing
   (is (= 1 (size->bytes 250))))
  (testing
   (is (= 2 (size->bytes 500)))))


(deftest test-centroid-feature-columns
  (testing "assignment helper columns are not centroid features"
    (let [centroids (ds/->dataset (array-map "x" [0.0 10.0]
                                             "y" [0.0 10.0]
                                             :assignments [1000 1000]
                                             "assignments" [900 900]))]
      (is (= ["x" "y"] (vec (centroid-feature-columns centroids)))))))


(deftest test-gpu-centroid-buffer-ignores-assignment-columns
  (testing "with-centroids writes only feature columns into the GPU centroid buffer"
    (let [points (ds/->dataset (array-map "x" [0.0 10.0]
                                          "y" [0.0 10.0]))
          centroids (ds/->dataset (array-map "x" [0.0 10.0]
                                             "y" [0.0 10.0]
                                             :assignments [1000 1000]
                                             "assignments" [900 900]))
          configuration {:k 2
                         :col-names ["x" "y"]
                         :distance-key :euclidean
                         :centroids centroids
                         :use-gpu true}]
      (with-gpu-context configuration
        (with-centroids centroids
          (is (= [0 1] (vec (minimum-index configuration points)))))))))
