(ns mieza.benchmarks.comparative.bench-fused
  "Benchmark comparing fused vs two-pass assignment kernels.
   Run with: clojure -A:dev -M -m mieza.benchmarks.comparative.bench-fused"
  (:require [mieza.meanings.distances :as d]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.neanderthal :as dsn]
            [uncomplicate.neanderthal.core :refer [mrows ncols]]))


(defn gen-dataset [n dims]
  (let [col-names (mapv #(str "f" %) (range dims))]
    {:col-names col-names
     :dataset (ds/->dataset (repeatedly n (fn [] (zipmap col-names (repeatedly dims rand)))))}))


(defn ds->matrix [ds]
  (dsn/dataset->dense ds :row :float32))


(defn bench-assign-kernel
  "Benchmarks a single assignment kernel call, returns elapsed ms.
   Runs warmup iterations then measures timing over the given repeats."
  [assign-fn device-ctx matrix warmup repeats]
  ;; warmup
  (dotimes [_ warmup]
    (assign-fn device-ctx matrix))
  ;; timed
  (let [start (System/nanoTime)]
    (dotimes [_ repeats]
      (assign-fn device-ctx matrix))
    (/ (- (System/nanoTime) start) (* repeats 1e6))))


(defn bench-config
  "Benchmarks both approaches for a given N, K, D, and distance metric."
  [distance-key n k dims & {:keys [warmup repeats] :or {warmup 3 repeats 10}}]
  (let [{:keys [col-names dataset]} (gen-dataset n dims)
        centroids-data (gen-dataset k dims)
        configuration {:distance-key distance-key :col-names col-names}
        matrix (ds->matrix dataset)
        centroids-matrix (ds->matrix (:dataset centroids-data))
        device-ctx (d/get-device-context configuration centroids-matrix)]
    (try
      (d/write-centroids-buffer! d/gpu-context centroids-matrix)
      (let [two-pass-ms (bench-assign-kernel d/gpu-distance-min-index device-ctx matrix warmup repeats)
            fused-ms    (bench-assign-kernel d/gpu-fused-assign device-ctx matrix warmup repeats)
            speedup     (/ two-pass-ms fused-ms)]
        (println (format "  %-14s N=%-8d K=%-6d D=%-4d | two-pass: %8.2f ms  fused: %8.2f ms  speedup: %.2fx"
                         (name distance-key) n k dims two-pass-ms fused-ms speedup))
        (flush)
        {:distance-key distance-key
         :n n :k k :dims dims
         :two-pass-ms two-pass-ms
         :fused-ms fused-ms
         :speedup speedup})
      (finally
        (d/release-centroids-buffer! d/gpu-context)
        (d/teardown-device device-ctx)))))


(defn -main [& _args]
  (println "=== Fused vs Two-Pass Assignment Kernel Benchmark ===\n")
  (let [configs (for [distance-key [:euclidean-sq :euclidean :manhattan :chebyshev :emd]
                      [n k dims] [[1000    10  3]
                                  [10000   30  3]
                                  [10000  100  3]
                                  [100000  30  3]
                                  [100000 100  3]
                                  [100000 100 50]]]
                  [distance-key n k dims])
        results (doall (map (fn [[dk n k d]] (bench-config dk n k d)) configs))]
    (println "\n=== Summary ===")
    (println (format "Average speedup: %.2fx" (/ (reduce + (map :speedup results)) (count results))))
    (println (format "Max speedup:     %.2fx" (apply max (map :speedup results))))
    (println (format "Min speedup:     %.2fx" (apply min (map :speedup results))))))
