(ns mieza.meanings.fused-assign-test
  "Tests that the fused distance+argmin kernel produces identical assignments
   to the two-pass (distance matrix + min_index) approach for all supported
   GPU-accelerated distance metrics."
  (:require
   [clojure.test :refer [deftest is testing]]
   [mieza.meanings.distances :refer [get-device-context teardown-device
                                     write-centroids-buffer! release-centroids-buffer!
                                     gpu-distance-min-index gpu-fused-assign
                                     gpu-context]]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.neanderthal :as dsn]))


(defn arrays-equal?
  "Compares two primitive arrays element-by-element. Works with byte[], short[], and int[]."
  [arr1 arr2]
  (java.util.Arrays/equals arr1 arr2))


(defn gen-dataset
  "Generates a random dataset with n rows and the given column names."
  [n col-names]
  (ds/->dataset
   (repeatedly n (fn [] (zipmap col-names (repeatedly (count col-names) rand))))))


(defn ds->matrix
  [ds]
  (dsn/dataset->dense ds :row :float32))


(deftest test-fused-assign-matches-two-pass
  (testing "Fused assign produces identical assignments to two-pass approach"
    (doseq [distance-key [:emd :euclidean :manhattan :chebyshev :euclidean-sq]]
      (doseq [test-config [{:dataset-size 10   :centroids 2}
                           {:dataset-size 100  :centroids 2}
                           {:dataset-size 100  :centroids 10}
                           {:dataset-size 1000 :centroids 30}
                           {:dataset-size 10000 :centroids 30}
                           {:dataset-size 10000 :centroids 100}]]
        (let [col-names ["x" "y" "z"]
              k (:centroids test-config)
              configuration {:distance-key distance-key :col-names col-names :k k}
              dataset (gen-dataset (:dataset-size test-config) col-names)
              centroids-dataset (gen-dataset k col-names)
              matrix (ds->matrix dataset)
              centroids-matrix (ds->matrix centroids-dataset)]
          (get-device-context configuration centroids-matrix)
          (try
            (write-centroids-buffer! gpu-context centroids-matrix)
            (let [ctx @gpu-context
                  two-pass-result (gpu-distance-min-index ctx matrix)
                  fused-result    (gpu-fused-assign ctx matrix)]
              (is (arrays-equal? two-pass-result fused-result)
                  (str "Mismatch for " distance-key
                       " N=" (:dataset-size test-config)
                       " K=" (:centroids test-config))))
            (finally
              (release-centroids-buffer! gpu-context)
              (teardown-device))))))))


(deftest test-fused-assign-higher-dimensions
  (testing "Fused assign works with higher dimensional data"
    (doseq [distance-key [:euclidean-sq :euclidean :manhattan]]
      (let [col-names (mapv #(str "f" %) (range 50))
            k 20
            configuration {:distance-key distance-key :col-names col-names :k k}
            dataset (gen-dataset 5000 col-names)
            centroids-dataset (gen-dataset k col-names)
            matrix (ds->matrix dataset)
            centroids-matrix (ds->matrix centroids-dataset)]
        (get-device-context configuration centroids-matrix)
        (try
          (write-centroids-buffer! gpu-context centroids-matrix)
          (let [ctx @gpu-context
                two-pass-result (gpu-distance-min-index ctx matrix)
                fused-result    (gpu-fused-assign ctx matrix)]
            (is (arrays-equal? two-pass-result fused-result)
                (str "Mismatch for " distance-key " with 50 dimensions")))
          (finally
            (release-centroids-buffer! gpu-context)
            (teardown-device)))))))
