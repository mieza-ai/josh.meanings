(ns mieza.meanings.persistence-write-test
  "Tests for dataset persistence: write, read, round-trip, format conversion."
  (:require [clojure.test :refer [deftest testing is]]
            [clojure.java.io :as io]
            [mieza.meanings.persistence :as p]
            [tech.v3.dataset :as ds]))

(defn make-dataset []
  (ds/->dataset {"x" [1.0 2.0 3.0 4.0 5.0]
                 "y" [10.0 20.0 30.0 40.0 50.0]}))

(defn cleanup-files! [& patterns]
  (doseq [f (file-seq (io/file "."))
          :when (and (.isFile f)
                     (some #(.contains (.getName f) %) patterns))]
    (io/delete-file f true)))

(defmacro with-cleanup [patterns & body]
  `(try
     (cleanup-files! ~@patterns)
     ~@body
     (finally
       (cleanup-files! ~@patterns))))

;; --- extension / filename->format ---

(deftest test-extension
  (is (= "csv" (p/extension "data.csv")))
  (is (= "parquet" (p/extension "path/to/data.parquet")))
  (is (= "arrow" (p/extension "data.arrow")))
  (is (= "arrows" (p/extension "data.arrows"))))

(deftest test-filename->format
  (is (= :csv (p/filename->format "data.csv")))
  (is (= :parquet (p/filename->format "data.parquet")))
  (is (= :arrow (p/filename->format "data.arrow")))
  (is (= :arrows (p/filename->format "data.arrows"))))

;; --- change-extension ---

(deftest test-change-extension
  (is (= "data.arrow" (p/change-extension "data.csv" :arrow)))
  (is (= "data.parquet" (p/change-extension "data.csv" :parquet)))
  (is (= "data.arrows" (p/change-extension "data.csv" :arrows)))
  (is (= "data.csv" (p/change-extension "data.arrow" :csv))))

;; --- file? ---

(deftest test-file-exists
  (with-cleanup ["persist_test_exists"]
    (spit "persist_test_exists.txt" "hello")
    (is (true? (p/file? "persist_test_exists.txt")))
    (is (false? (p/file? "nonexistent_file_xyz.txt")))))

;; --- write + read round-trips per format ---

(deftest test-csv-round-trip
  (with-cleanup ["persist_rt_csv"]
    (let [ds-in (make-dataset)
          fname "persist_rt_csv.csv"]
      (p/write-dataset fname ds-in)
      (is (p/file? fname))
      (let [ds-out (first (p/read-dataset-seq fname))]
        (is (= (ds/row-count ds-in) (ds/row-count ds-out)))
        (is (= (set (ds/column-names ds-in)) (set (ds/column-names ds-out))))))))

(deftest test-arrow-round-trip
  (with-cleanup ["persist_rt_arrow"]
    (let [ds-in (make-dataset)
          fname "persist_rt_arrow.arrow"]
      (p/write-dataset fname ds-in)
      (is (p/file? fname))
      (let [ds-out (first (p/read-dataset-seq fname))]
        (is (= (ds/row-count ds-in) (ds/row-count ds-out)))
        (is (= (set (ds/column-names ds-in)) (set (ds/column-names ds-out))))
        (testing "Values preserved"
          (is (= (vec (ds-in "x")) (vec (ds-out "x")))))))))

(deftest test-parquet-round-trip
  (with-cleanup ["persist_rt_parquet"]
    (let [ds-in (make-dataset)
          fname "persist_rt_parquet.parquet"]
      (p/write-dataset fname ds-in)
      (is (p/file? fname))
      (let [ds-out (first (p/read-dataset-seq fname))]
        (is (= (ds/row-count ds-in) (ds/row-count ds-out)))
        (is (= (set (ds/column-names ds-in)) (set (ds/column-names ds-out))))))))

;; --- write-datasets (sequence write) ---

(deftest test-write-dataset-seq
  (with-cleanup ["persist_seq_arrows"]
    (let [ds1 (ds/->dataset {"x" [1.0 2.0] "y" [3.0 4.0]})
          ds2 (ds/->dataset {"x" [5.0 6.0] "y" [7.0 8.0]})
          fname "persist_seq_arrows.arrows"]
      (p/write-datasets fname [ds1 ds2])
      (is (p/file? fname))
      (let [ds-seq (p/read-dataset-seq fname)]
        (is (= 4 (reduce + (map ds/row-count ds-seq))))))))

;; --- convert-file ---

(deftest test-convert-file
  (with-cleanup ["persist_conv"]
    (let [ds-in (make-dataset)
          csv-fname "persist_conv.csv"
          _ (p/write-dataset csv-fname ds-in)
          arrow-fname (p/convert-file csv-fname :arrow)]
      (is (= "persist_conv.arrow" arrow-fname))
      (is (p/file? arrow-fname))
      (let [ds-out (first (p/read-dataset-seq arrow-fname))]
        (is (= (ds/row-count ds-in) (ds/row-count ds-out)))))))

(deftest test-convert-file-same-format-noop
  (with-cleanup ["persist_noop"]
    (let [ds-in (make-dataset)
          fname "persist_noop.csv"]
      (p/write-dataset fname ds-in)
      (is (= fname (p/convert-file fname :csv))
          "Converting to same format returns same filename"))))

;; --- select-columns-seq ---

(deftest test-select-columns-seq
  (let [ds (ds/->dataset {"x" [1.0 2.0] "y" [3.0 4.0] "z" [5.0 6.0]})
        filtered (first (p/select-columns-seq [ds] ["x" "z"]))]
    (is (= #{"x" "z"} (set (ds/column-names filtered))))
    (is (not (contains? (set (ds/column-names filtered)) "y")))))

;; --- dataset-seq->column-names ---

(deftest test-dataset-seq-column-names
  (let [ds (ds/->dataset {"a" [1] "b" [2] "c" [3]})
        names (p/dataset-seq->column-names [ds])]
    (is (= #{"a" "b" "c"} (set names)))))
