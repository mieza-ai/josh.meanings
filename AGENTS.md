# AGENTS.md — mieza.meanings

Instructions for AI coding agents and humans automating changes to this repository.

## Project overview

mieza.meanings is a **GPU-accelerated k-means** library for Clojure, aimed at **medium data**: datasets too large to hold in memory at once, but manageable on disk with memory-mapped I/O and chunked processing.

- **GPU via OpenCL** (not CUDA), integrated with `uncomplicate/clojurecl`.
- Differentiators: memory-mapped Arrow/Parquet/CSV pipelines, multiple distance functions with optional GPU paths, modern initialization methods (e.g. AFK-MC2-style `:afk-mc`).
- Namespaces live under `mieza.meanings.*` (not `josh.meanings`).

## Build and run

- **Primary**: Leiningen — `lein test`, `lein repl` (init ns: `mieza.meanings.kmeans`), `lein codox` for local HTML API docs.
- **Also supported**: Clojure CLI — `deps.edn` mirrors dependencies; use your usual `clojure -M:dev` / `-X` patterns as you wire aliases locally.
- **JVM**: The `:dev` profile in `project.clj` sets incubator modules, native access, and stack size — keep similar flags when running tests or profiling outside Lein.

## Layout

| Path | Purpose |
|------|---------|
| `src/clj/` | Clojure sources (`mieza.meanings`, `mieza.file`) |
| `src/kernels/` | OpenCL C kernels (`.c`) |
| `test/` | `clojure.test` + `test.check` |
| `benchmarks/` | Benchmark harnesses |
| `theory/` | Reference PDFs (treat as binary) |
| `resources/` | e.g. `logback.xml` |
| `doc/` | Human docs; `intro.md` API-oriented guide |

## Testing

- Run **`lein test`** before proposing changes.
- GPU-heavy tests need an OpenCL-capable device; many tests degrade or skip appropriately — still run the full suite when possible.
- Generative tests cover distances, inits, persistence, and clustering correctness.

## Style and lint

- **2 spaces** for Clojure; kernels use **4 spaces** (see `.editorconfig`).
- **clj-kondo** config in `.clj-kondo/` — do not silence real diagnostics without cause.

## Architecture map

| Area | Namespace / location |
|------|----------------------|
| Entrypoints | `mieza.meanings.kmeans` — `k-means`, `k-means-seq`, Lloyd iteration |
| Distances | `mieza.meanings.distances` — multimethod `get-distance-fn`, GPU helpers |
| Initialization | `mieza.meanings.initializations.*` — multimethod `initialize-centroids` on `:init` |
| I/O | `mieza.meanings.persistence` — formats (`:arrow`, `:arrows`, `:parquet`, `:csv`) |
| Result type | `mieza.meanings.records.cluster-result` — `ClusterResult`, `load-model` |
| Protocols | `mieza.meanings.protocols.savable` / `classifier` |

## Ecosystem

Built on **tech.ml.dataset**, **dtype-next**, **fastmath**, **neanderthal**, **clojurecl**. The dependency tree pulls Hadoop/Arrow/Jackson — version bumps can cause subtle runtime issues; align transitive versions carefully.

## Boundaries — do not

- Change OpenCL kernels without reasoning about buffer sizes, work-groups, and barriers; validate on real hardware.
- Bump `project.clj` / `deps.edn` coordinates casually — run **full tests** and confirm both toolchains if you touch dependencies.
- Break public clustering entrypoints or on-disk model shape without a **CHANGELOG** entry and migration notes.

## Boundaries — do

- Preserve **backward compatibility** for documented public flows (`k-means`, `k-means-seq`, `save-model` / `load-model`, `ClusterResult` fields).
- Update **`CHANGELOG.md`** for user-visible fixes or API changes.
- Match existing naming, require style, and multimethod patterns when extending distances or initializations.

## Git

- Small, focused commits; clear subject lines.
- `CHANGELOG.md` follows **Keep a Changelog** style when you edit it.
