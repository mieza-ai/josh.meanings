# mieza.meanings

[![Clojars Project](https://img.shields.io/clojars/v/ai.mieza/mieza.meanings.svg)](https://clojars.org/ai.mieza/mieza.meanings)
[![cljdoc](https://cljdoc.org/badge/ai.mieza/mieza.meanings)](https://cljdoc.org/d/ai.mieza/mieza.meanings)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

GPU-accelerated K-Means clustering for Clojure. Built to handle "medium data" workloads —
datasets too large to fit in memory, but not so large that the computation cannot be
persisted to disk.

Unlike most other K-means implementations, we employ several techniques which lend
themselves toward making this implementation quite a bit faster than alternatives:

1. We leverage memory mapping of the datasets.
2. We do our distance calculations on the GPU via OpenCL.
3. We implement initialization schemes from more recent research.

> [!NOTE]
> GPU acceleration is available for several distance functions including EMD,
> Euclidean, Manhattan, Chebyshev and Euclidean squared.

## Installation

If you use the Clojure CLI, add the library to your `deps.edn`:

```clojure
ai.mieza/mieza.meanings {:mvn/version "3.0.14"}
```

## Getting Started

```clojure
(require '[mieza.meanings.kmeans :refer [k-means k-means-seq]]
         '[mieza.meanings.protocols.savable :refer [save-model]]
         '[mieza.meanings.protocols.classifier :refer [assignments]]
         '[mieza.meanings.records.cluster-result :refer [load-model]])

;; Dataset: file path (CSV / Parquet / Arrow, etc.) or a lazy seq of datasets
(def dataset "your_dataset.csv")

(def k 10)

;; Single clustering run → ClusterResult
(def model (k-means dataset k))

;; Multiple runs — keep the lowest objective (:cost)
(def k-tries 10)
(def best (apply min-key :cost (take k-tries (k-means-seq dataset k))))

;; Persist and reload (EDN on disk)
(def path "cluster-model.edn")
(save-model best path)
(def reloaded (load-model path))

;; Centroids and cost are record fields
(:centroids reloaded)
(:cost reloaded)

;; Batch-assign new dataset chunks to clusters (lazy seq of datasets)
;; (assignments reloaded your-dataset-seq)
```

### Note on CHANGELOG / older examples

Some release notes refer to `.classify`, `.load-centroids`, or `.load-assignments` on
`ClusterResult`. The current public surface uses the `ClusterResult` record fields
and the `assignments` protocol for new data; see `llms-full.txt` or `doc/intro.md` for detail.

## Testing

Run the project's unit tests with:

```
lein test
```

Tests exercising the GPU code paths require a GPU with OpenCL support.

## History

This project was originally created as `josh.meanings` by Joshua Cole.
It is now maintained by [mieza.ai](https://mieza.ai).

## License

Distributed under the terms of the [MIT License](LICENSE).
