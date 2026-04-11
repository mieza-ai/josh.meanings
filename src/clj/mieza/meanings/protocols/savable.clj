(ns mieza.meanings.protocols.savable)

(defprotocol Savable
  "Serialize a trained result (typically `mieza.meanings.records.cluster-result/ClusterResult`)
  to disk for later `mieza.meanings.records.cluster-result/load-model`."
  (save-model [this filename]
    "Writes `this` to `filename` as EDN. Centroids are stored as row maps; reload with `load-model`."))
