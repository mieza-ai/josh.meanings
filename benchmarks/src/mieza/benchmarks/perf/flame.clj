(ns mieza.benchmarks.perf.flame
  "Profile end-to-end k-means on a realistic dataset to understand where
   time is spent."
  (:require
   [clj-async-profiler.core :as prof]
   [mieza.meanings.kmeans :as km]))

(defn -main [& [points-file k-str]]
  (let [points (or points-file "benchmarks/comparative/data/large_50d.arrow")
        k      (Long/parseLong (or k-str "10"))]
    (println "Warmup on k=3 ...")
    (km/k-means-via-file points 3
                         :distance-key :euclidean-sq
                         :fused-assign true
                         :init :afk-mc
                         :format :arrow)
    (println "Profiled run, event=:itimer ...")
    ;; :itimer works without perf_events kernel permissions;
    ;; interval is default 10ms.
    (let [result (prof/profile
                  {:event :itimer}
                  (km/k-means-via-file points k
                                       :distance-key :euclidean-sq
                                       :fused-assign true
                                       :init :afk-mc
                                       :format :arrow))]
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
