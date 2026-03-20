(ns mieza.meanings.initialization-methods-test
  "Tests for all initialization methods via the k-means entry point."
  (:require [clojure.test :refer [deftest testing is]]
            [clojure.java.io :as io]
            [clojure.data.csv :as csv]
            [mieza.meanings.kmeans :refer [k-means k-means-seq]]
            [tech.v3.dataset :as ds]))

(defn write-csv! [filename rows]
  (with-open [w (io/writer filename)]
    (csv/write-csv w rows)))

(defn cleanup! [base]
  (doseq [f (file-seq (io/file "."))
          :when (and (.isFile f)
                     (.contains (.getName f) base))]
    (io/delete-file f true)))

(defmacro with-test-csv [filename rows & body]
  `(let [base# (clojure.string/replace ~filename ".csv" "")]
     (try
       (cleanup! base#)
       (write-csv! ~filename ~rows)
       ~@body
       (finally
         (cleanup! base#)))))

;; Three clear clusters with 10 points each.
(def init-test-data
  (into [["a" "b"]]
        (concat
         (for [_ (range 10)] [(+ (rand-int 5)) (+ (rand-int 5))])
         (for [_ (range 10)] [(+ 98 (rand-int 5)) (+ 98 (rand-int 5))])
         (for [_ (range 10)] [(+ 198 (rand-int 5)) (+ 198 (rand-int 5))]))))

(def all-init-keys [:afk-mc :k-means-++ :k-mc-squared :k-means-parallel])

;; Every init method produces k centroids

(deftest test-all-inits-produce-k-centroids
  (doseq [init-key all-init-keys]
    (let [fname (str "test.init." (name init-key) ".csv")]
      (with-test-csv fname init-test-data
        (let [result (k-means fname 3
                              :distance-key :euclidean
                              :init init-key
                              :m 10)]
          (testing (str init-key " produces 3 centroids")
            (is (= 3 (count (ds/rowvecs (:centroids result))))))
          (testing (str init-key " has non-negative cost")
            (is (>= (:cost result) 0.0))))))))

;; Every init method produces distinct centroids

(deftest test-all-inits-produce-distinct-centroids
  (doseq [init-key all-init-keys]
    (let [fname (str "test.init.dist." (name init-key) ".csv")]
      (with-test-csv fname init-test-data
        (let [result (k-means fname 3
                              :distance-key :euclidean
                              :init init-key
                              :m 10)
              centroids (ds/rowvecs (:centroids result))]
          (testing (str init-key " produces distinct centroids")
            (is (= 3 (count (set centroids))))))))))

;; AFK-MC with varying k

(deftest test-afk-mc-with-different-k
  (doseq [k [1 2 3 5]]
    (with-test-csv (str "test.init.afk.k" k ".csv") init-test-data
      (let [result (k-means (str "test.init.afk.k" k ".csv") k
                            :distance-key :euclidean
                            :init :afk-mc
                            :m 50)]
        (testing (str ":afk-mc with k=" k)
          (is (= k (count (ds/rowvecs (:centroids result))))))))))

;; AFK-MC with different distance functions

(deftest test-afk-mc-with-different-distances
  (doseq [dk [:euclidean :manhattan :chebyshev :euclidean-sq :emd]]
    (with-test-csv (str "test.init.afk." (name dk) ".csv") init-test-data
      (let [result (k-means (str "test.init.afk." (name dk) ".csv") 3
                            :distance-key dk
                            :init :afk-mc
                            :m 50)]
        (testing (str ":afk-mc with " dk)
          (is (= 3 (count (ds/rowvecs (:centroids result)))))
          (is (>= (:cost result) 0.0)))))))

;; Multiple runs via k-means-seq

(deftest test-multiple-runs-via-seq
  (with-test-csv "test.init.seq.csv" init-test-data
    (let [results (take 3 (k-means-seq "test.init.seq.csv" 3
                                        :distance-key :euclidean
                                        :init :afk-mc
                                        :m 50))]
      (testing "Multiple runs all produce valid results"
        (doseq [r results]
          (is (= 3 (count (ds/rowvecs (:centroids r)))))
          (is (>= (:cost r) 0.0))))
      (testing "Best-of-N is a valid strategy"
        (let [best (apply min-key :cost results)]
          (is (every? #(>= (:cost %) (:cost best)) results)))))))

;; Configuration preserved through init

(deftest test-configuration-preserved
  (with-test-csv "test.init.conf.csv" init-test-data
    (let [result (k-means "test.init.conf.csv" 3
                          :distance-key :euclidean
                          :init :afk-mc
                          :m 50)
          conf (:configuration result)]
      (is (= 3 (:k conf)))
      (is (= :euclidean (:distance-key conf)))
      (is (= :afk-mc (:init conf)))
      (is (vector? (:col-names conf)))
      (is (= #{"a" "b"} (set (:col-names conf)))))))
