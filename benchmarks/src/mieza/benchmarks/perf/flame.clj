(ns mieza.benchmarks.perf.flame
  "Profile end-to-end k-means on a realistic dataset to understand where
   time is spent."
  (:require
   [clj-async-profiler.core :as prof]
   [mieza.meanings.kmeans :as km]))

(defn -main [& [points-file k-str reduce-str distance-str]]
  (let [points (or points-file "benchmarks/comparative/data/large_50d.arrow")
        k      (Long/parseLong (or k-str "10"))
        reduce? (Boolean/parseBoolean (or reduce-str "false"))
        distance-key (keyword (or distance-str "euclidean-sq"))
        opts  {:distance-key distance-key
               :fused-assign true
               :fused-reduce reduce?
               :init :afk-mc
               :format :arrow}]
    (println "Warmup on k=3 ...")
    (apply km/k-means-via-file points 3 (apply concat opts))
    (println "Profiled run, event=:itimer ...")
    ;; :itimer works without perf_events kernel permissions;
    ;; interval is default 10ms.
    (let [result (prof/profile
                  {:event :itimer}
                  (apply km/k-means-via-file points k (apply concat opts)))]
      (println "Profile complete.")
      (println "Result path (flamegraph SVG):" (str result))
      (println "Directory contents:")
      (doseq [f (sort-by #(.lastModified ^java.io.File %) >
                         (.listFiles (java.io.File. "/tmp/clj-async-profiler/results")))]
        (println " "
                 (str (.lastModified ^java.io.File f))
                 (.getPath ^java.io.File f)))))
  (shutdown-agents)
  (System/exit 0))
