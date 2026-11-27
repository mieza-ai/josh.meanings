(ns mieza.meanings.initializations.afk-test 
  (:require [clojure.test :refer [deftest is testing]]
            [mieza.meanings.initializations.afk :refer [afk-centroids-checkpoint-file
                                                        chains-per-pass
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


(deftest chains-per-pass-is-bounded
  (testing "the configured count, capped by the centroids still to pick"
    (is (= 25 (chains-per-pass {:m 100 :size-estimate 1000000} 499)))
    (is (= 10 (chains-per-pass {:m 100 :size-estimate 1000000} 10)))
    (is (= 3 (chains-per-pass {:m 100 :size-estimate 1000000 :afk-chains-per-pass 3} 499))))
  (testing "a pass never draws more than a quarter of the points"
    (is (= 2 (chains-per-pass {:m 100 :size-estimate 800} 499))))
  (testing "small data keeps one chain per pass"
    (is (= 1 (chains-per-pass {:m 10 :size-estimate 30} 2)))))
