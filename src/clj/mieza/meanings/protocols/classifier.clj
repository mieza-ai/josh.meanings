(ns mieza.meanings.protocols.classifier)

(defprotocol Classifier
  "Assign cluster labels to rows of one or more datasets, using the clustering
  configuration and centroids carried by `this` — typically a
  `mieza.meanings.records.cluster-result/ClusterResult` after fitting, or a
  `mieza.meanings.records.clustering-state/KMeansState` while iterating.

  `datasets` must be a lazy (or finite) sequence of `tech.v3.dataset` datasets compatible
  with the column layout used when fitting."
  (assignments [this datasets]
    "Yields a sequence of datasets, each with an `:assignments` column (nearest centroid index per row)."))