(ns mieza.benchmarks.perf.emd-check
  "Verify the EMD :fused-reduce path produces the same centroids and inertia
   as the EMD :fused-assign path when both start from identical centroids."
  (:require
   [mieza.meanings.distances :as distances]
   [mieza.meanings.kmeans :as km]
   [tech.v3.dataset :as ds]))

(defn- naive-initial-centroids
  "Deterministic seed: pick the first k rows of the dataset."
  [ds-path k col-names]
  (let [ds (first (mieza.meanings.persistence/read-dataset-seq ds-path))
        head (ds/head ds k)]
    (ds/select-columns head col-names)))

(defn -main [& _]
  (let [points "benchmarks/comparative/data/med_50d.arrow"
        k 10
        iterations 5
        base-opts {:distance-key :emd
                   :fused-assign true
                   :init :afk-mc
                   :format :arrow
                   :iterations iterations}
        conf-a (km/initialize-k-means-state points k (assoc base-opts :fused-reduce false))
        conf-b (km/initialize-k-means-state points k (assoc base-opts :fused-reduce true))
        seed (naive-initial-centroids points k (:col-names conf-a))
        ra (distances/with-gpu-context conf-a (km/lloyd-fast conf-a seed))
        rb (distances/with-gpu-context conf-b (km/lloyd-fast-reduced conf-b seed))
        ca (vec (sort (for [row (ds/rowvecs (:centroids ra))] (vec row))))
        cb (vec (sort (for [row (ds/rowvecs (:centroids rb))] (vec row))))
        max-diff (double
                  (apply max 0.0
                         (for [[ra rb] (map vector ca cb)
                               [x y] (map vector ra rb)]
                           (Math/abs (- (double x) (double y))))))
        cost-a (:cost ra)
        cost-b (:cost rb)]
    (println "rows : assign" (count ca) "reduce" (count cb))
    (println "max-centroid-abs-diff :" max-diff)
    (println "cost assign :" cost-a)
    (println "cost reduce :" cost-b)
    (println "cost-rel-diff :" (when (and cost-a cost-b (pos? cost-a))
                                  (/ (Math/abs (- (double cost-a) (double cost-b)))
                                     (double cost-a))))
    (shutdown-agents)
    (System/exit (if (< max-diff 1.0) 0 1))))
