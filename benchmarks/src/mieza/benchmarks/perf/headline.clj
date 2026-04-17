(ns mieza.benchmarks.perf.headline
  "Headline end-to-end benchmark for before/after perf comparisons.

   Runs k-means-via-file on the 1M-point med_50d.arrow fixture with
   k=10, :distance-key :euclidean-sq. Reports total wall-clock time."
  (:require
   [mieza.meanings.kmeans :as km]))

(defn- force-gc! []
  (dotimes [_ 3] (System/gc))
  (Thread/sleep 200))

(defn- fmt-secs [nanos]
  (format "%.2f s" (/ (double nanos) 1e9)))

(defn -main [& [points-file k-str reduce-str distance-str]]
  (let [points (or points-file "benchmarks/comparative/data/med_50d.arrow")
        k      (Long/parseLong (or k-str "10"))
        reduce? (Boolean/parseBoolean (or reduce-str "false"))
        distance-key (keyword (or distance-str "euclidean-sq"))
        opts  {:distance-key distance-key
               :fused-assign true
               :fused-reduce reduce?
               :init :afk-mc
               :format :arrow}]
    (println "Benchmark: " points "k=" k ":fused-reduce" reduce?)
    ;; warmup: let JIT settle on a smaller k
    (force-gc!)
    (println "warmup run (discarded)...")
    (apply km/k-means-via-file points 3 (apply concat opts))
    ;; three timed runs
    (doseq [run-idx [1 2 3]]
      (force-gc!)
      (let [t0 (System/nanoTime)
            _  (apply km/k-means-via-file points k (apply concat opts))
            t1 (System/nanoTime)]
        (println "run" run-idx "wall-time:" (fmt-secs (- t1 t0))))))
  (shutdown-agents)
  (System/exit 0))
