(ns mieza.meanings.initializations.afk-test 
  (:require [clojure.test :refer [deftest is testing]]
            [mieza.meanings.initializations.afk :refer [afk-centroids-checkpoint-file
                                                        qx-file]]))

(deftest test-qx-file
  (testing "That the path returned is in the same directory as the original."
    (is (= (qx-file {:points "/media/joshua/a/database/user_files/river.arrow"})
           "/media/joshua/a/database/user_files/river.qx.arrow"))))

(deftest test-afk-checkpoint-file
  (testing "checkpoint paths are scoped to the source points file"
    (is (= (afk-centroids-checkpoint-file
            {:points "/media/joshua/a/database/user_files/river.arrow"})
           "/media/joshua/a/database/user_files/river.afk-centroids.arrow"))
    (is (not= (afk-centroids-checkpoint-file
               {:points "/media/joshua/a/database/user_files/river.arrow"})
              (afk-centroids-checkpoint-file
               {:points "/media/joshua/a/database/user_files/turn.arrow"})))))
