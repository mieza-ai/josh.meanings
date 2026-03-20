# mieza.meanings

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

[![Clojars Project](https://img.shields.io/clojars/v/ai.mieza/mieza.meanings.svg)](https://clojars.org/ai.mieza/mieza.meanings)

If you use the Clojure CLI, add the library to your `deps.edn`:

```clojure
ai.mieza/mieza.meanings {:mvn/version "3.0.14"}
```

## Getting Started

```clojure
(require '[mieza.meanings.kmeans :refer [k-means k-means-seq]]
         '[mieza.meanings.protocols.savable :refer [save-model]]
         '[mieza.meanings.protocols.classifier :refer [classify load-centroids load-assignments]])


;; Get a dataset.  You can pass in your dataset under a variety of formats.
;; See the docs for more details on supported formats.
(def dataset "your_dataset.csv")

;; Choose the number of clusters you want
(def k 10)


;; To get a single cluster model
(def model (k-means dataset k))

;; Alternatively you can run k means multiple times.  This is recommended because
;; some k means initializations don't give guarantees on the quality of a solution
;; and so you can get better results by running k means multiple times and taking
;; the best result.
(def model (apply min-key :cost (take k-tries (k-means-seq cluster-dataset-name k))))

;; Once you have a model you can save it.
(def model-path (.save-model model))

;; Later you can load that model
(def model (load-model model-path))

;; To load the assignments just
(.load-assignments model)

;; To classify a new entry
(.classify model [1 2 3])

;; To view the centroids
(.load-centroids model)
```

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
