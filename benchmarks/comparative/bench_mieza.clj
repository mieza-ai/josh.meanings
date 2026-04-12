(ns mieza.benchmarks.comparative.bench-mieza
  "Benchmark mieza.meanings k-means on generated datasets.
   Run with: clojure -A:dev -M -m mieza.benchmarks.comparative.bench-mieza [dataset-names...]"
  (:require [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [mieza.meanings.kmeans :refer [k-means]]
            [tech.v3.dataset :as ds]))

(def data-dir "benchmarks/comparative/data")
(def results-dir "benchmarks/comparative/results")

(defn load-configs []
  (with-open [r (io/reader (str data-dir "/configs.csv"))]
    (let [rows (doall (csv/read-csv r))]
      (into {}
            (map (fn [[name n_points n_dims k]]
                   [name {:n-points (parse-long n_points)
                          :n-dims (parse-long n_dims)
                          :k (parse-long k)}])
                 (rest rows))))))

(defn bench-once [dataset-path k distance-key]
  (let [start (System/nanoTime)
        result (k-means dataset-path k
                        :distance-key distance-key
                        :init :afk-mc
                        :m 50)
        elapsed (/ (- (System/nanoTime) start) 1e9)]
    {:elapsed-s (double elapsed)
     :cost (double (:cost result))}))

(defn bench-dataset [name config]
  (let [path (str data-dir "/" name ".csv")
        k (:k config)]
    (println (format "\n=== %s (%,d pts, %dd, k=%d) ==="
                     name (:n-points config) (:n-dims config) k))
    (flush)
    (when (.exists (io/file path))
      (println "  mieza.meanings (euclidean, GPU)...")
      (flush)
      (let [{:keys [elapsed-s cost]} (bench-once path k :euclidean)]
        (println (format "  RESULT: %.3fs, cost=%.2f" elapsed-s cost))
        (flush)
        {:dataset name
         :method "mieza.meanings"
         :n-points (:n-points config)
         :n-dims (:n-dims config)
         :k k
         :elapsed-s elapsed-s
         :cost cost}))))

(defn write-results [results]
  (.mkdirs (io/file results-dir))
  (let [path (str results-dir "/mieza_results.csv")]
    (with-open [w (io/writer path)]
      (csv/write-csv w
        (cons ["dataset" "method" "n_points" "n_dims" "k" "elapsed_s" "cost"]
              (map (fn [r]
                     [(:dataset r) (:method r) (:n-points r) (:n-dims r)
                      (:k r) (format "%.3f" (:elapsed-s r)) (format "%.2f" (:cost r))])
                   results))))
    (println (format "\nResults written to %s" path))))

(defn -main [& args]
  (let [configs (load-configs)
        names (if (seq args)
                (filter configs args)
                (sort (keys configs)))
        results (->> names
                     (map #(bench-dataset % (configs %)))
                     (remove nil?)
                     vec)]
    (write-results results)
    (System/exit 0)))
