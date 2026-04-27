(ns mieza.meanings.advise
  "OS page-cache advice helpers."
  (:require
   [clojure.java.io :as io]
   [taoensso.timbre :as log])
  (:import
   [com.sun.jna Function NativeLibrary Platform]))

(def ^:private o-rdonly 0)
(def ^:private posix-fadv-dontneed 4)

(defn- libc-function
  ^Function [name]
  (.getFunction (NativeLibrary/getInstance Platform/C_LIBRARY_NAME) name))

(defn- open-read-only
  ^long [filepath]
  (.invokeInt (libc-function "open")
              (object-array [filepath (Integer/valueOf (int o-rdonly))])))

(defn- close-fd!
  [fd]
  (.invokeInt (libc-function "close")
              (object-array [(Integer/valueOf (int fd))])))

(defn- posix-fadvise-dontneed!
  [fd]
  (.invokeInt (libc-function "posix_fadvise")
              (object-array [(Integer/valueOf (int fd))
                             (Long/valueOf 0)
                             (Long/valueOf 0)
                             (Integer/valueOf (int posix-fadv-dontneed))])))

(defn dontneed!
  "Signals the OS that the contents of `filepath` are no longer needed and
   its page-cache pages may be reclaimed. Returns true on success, false on
   error (e.g. file not found, unsupported platform). Non-destructive:
   subsequent reads will re-fault from disk."
  [filepath]
  (if-not (Platform/isLinux)
    false
    (let [file (io/file filepath)]
      (if-not (.exists file)
        (do
          (log/warn "Cannot drop page cache for missing file" (str file))
          false)
        (try
          (let [fd (open-read-only (.getPath file))]
            (if (neg? fd)
              false
              (try
                (zero? (posix-fadvise-dontneed! fd))
                (finally
                  (close-fd! fd)))))
          (catch Throwable _
            false))))))
