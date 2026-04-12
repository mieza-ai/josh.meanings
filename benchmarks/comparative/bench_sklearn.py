#!/usr/bin/env python3
"""Benchmark scikit-learn KMeans on generated datasets."""
import csv
import json
import os
import sys
import time
import traceback

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_configs():
    configs = {}
    with open(os.path.join(DATA_DIR, "configs.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            configs[row["name"]] = {
                "n_points": int(row["n_points"]),
                "n_dims": int(row["n_dims"]),
                "k": int(row["k"]),
            }
    return configs


def load_dataset(name):
    filepath = os.path.join(DATA_DIR, f"{name}.npy")
    return np.load(filepath)


def bench_kmeans(data, k, n_init=1, max_iter=100):
    """Benchmark standard KMeans. Returns (elapsed_seconds, inertia)."""
    km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=42)
    t0 = time.perf_counter()
    km.fit(data)
    elapsed = time.perf_counter() - t0
    return elapsed, float(km.inertia_)


def bench_minibatch_kmeans(data, k, n_init=1, max_iter=100, batch_size=10000):
    """Benchmark MiniBatchKMeans. Returns (elapsed_seconds, inertia)."""
    km = MiniBatchKMeans(
        n_clusters=k, n_init=n_init, max_iter=max_iter,
        batch_size=batch_size, random_state=42
    )
    t0 = time.perf_counter()
    km.fit(data)
    elapsed = time.perf_counter() - t0
    return elapsed, float(km.inertia_)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    configs = load_configs()

    names = list(configs.keys())
    if len(sys.argv) > 1:
        names = [n for n in sys.argv[1:] if n in configs]

    results = []

    for name in names:
        cfg = configs[name]
        k = cfg["k"]
        print(f"\n=== {name} ({cfg['n_points']:,} pts, {cfg['n_dims']}d, k={k}) ===")

        try:
            data = load_dataset(name)
            print(f"  Loaded: {data.shape}")
        except Exception as e:
            print(f"  SKIP (load failed): {e}")
            continue

        # Standard KMeans
        try:
            print(f"  sklearn.KMeans...", end="", flush=True)
            elapsed, inertia = bench_kmeans(data, k)
            print(f" {elapsed:.2f}s, inertia={inertia:.0f}")
            results.append({
                "dataset": name,
                "method": "sklearn.KMeans",
                "n_points": cfg["n_points"],
                "n_dims": cfg["n_dims"],
                "k": k,
                "elapsed_s": round(elapsed, 3),
                "inertia": round(inertia, 2),
            })
        except Exception as e:
            print(f" FAILED: {e}")
            traceback.print_exc()

        # MiniBatchKMeans
        try:
            print(f"  sklearn.MiniBatchKMeans...", end="", flush=True)
            elapsed, inertia = bench_minibatch_kmeans(data, k)
            print(f" {elapsed:.2f}s, inertia={inertia:.0f}")
            results.append({
                "dataset": name,
                "method": "sklearn.MiniBatchKMeans",
                "n_points": cfg["n_points"],
                "n_dims": cfg["n_dims"],
                "k": k,
                "elapsed_s": round(elapsed, 3),
                "inertia": round(inertia, 2),
            })
        except Exception as e:
            print(f" FAILED: {e}")
            traceback.print_exc()

    # Write results
    results_path = os.path.join(RESULTS_DIR, "sklearn_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    main()
