#!/usr/bin/env python3
"""Generate synthetic datasets for k-means benchmarking.

Creates well-separated Gaussian clusters. Writes:
- .npy files for scikit-learn (numpy native format)
- .arrow files for mieza.meanings (via pyarrow IPC)

For very large datasets, generates in chunks to avoid OOM during generation.
"""
import numpy as np
import os
import sys
import time

import pyarrow as pa
import pyarrow.ipc as ipc

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")

CONFIGS = [
    # (name, n_points, n_dims, k)
    # Small — for correctness / warmup only
    ("small_2d", 10_000, 2, 5),
    # Medium — where sklearn is fast
    ("med_50d", 1_000_000, 50, 10),
    # Large — where it gets interesting
    ("large_50d", 10_000_000, 50, 10),
    # XL — sklearn should struggle here (~20GB raw)
    ("xl_50d", 100_000_000, 50, 10),
    # XXL — sklearn should OOM here (~100GB raw)
    ("xxl_50d", 500_000_000, 50, 10),
]

CHUNK_SIZE = 1_000_000  # Generate 1M rows at a time to control memory


def generate_chunk(n_points, n_dims, n_clusters, centers, rng):
    points_per_cluster = n_points // n_clusters
    remainder = n_points % n_clusters
    data = []
    for i in range(n_clusters):
        count = points_per_cluster + (1 if i < remainder else 0)
        cluster_data = rng.normal(loc=centers[i], scale=1.0, size=(count, n_dims))
        data.append(cluster_data)
    return np.vstack(data).astype(np.float32)


def generate_centers(n_dims, n_clusters, seed=42):
    rng = np.random.RandomState(seed)
    return rng.uniform(-1000, 1000, size=(n_clusters, n_dims))


def write_npy(n_points, n_dims, n_clusters, centers, filepath):
    """Write full dataset as .npy. Will OOM for very large datasets — that's the point."""
    rng = np.random.RandomState(42)
    data = generate_chunk(n_points, n_dims, n_clusters, centers, rng)
    np.save(filepath, data)


def write_arrow_chunked(n_points, n_dims, n_clusters, centers, filepath):
    """Write dataset as Arrow IPC in chunks. Handles any size."""
    names = [f"d{i}" for i in range(n_dims)]
    schema = pa.schema([(name, pa.float32()) for name in names])

    remaining = n_points
    seed = 42
    with pa.OSFile(filepath, "wb") as f:
        writer = ipc.new_file(f, schema)
        while remaining > 0:
            chunk_size = min(CHUNK_SIZE, remaining)
            rng = np.random.RandomState(seed)
            data = generate_chunk(chunk_size, n_dims, n_clusters, centers, rng)
            arrays = [pa.array(data[:, i]) for i in range(n_dims)]
            batch = pa.record_batch(arrays, schema=schema)
            writer.write_batch(batch)
            remaining -= chunk_size
            seed += 1
        writer.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    configs = CONFIGS
    if len(sys.argv) > 1:
        names = set(sys.argv[1:])
        configs = [c for c in configs if c[0] in names]

    for name, n_points, n_dims, k in configs:
        npy_path = os.path.join(OUTPUT_DIR, f"{name}.npy")
        arrow_path = os.path.join(OUTPUT_DIR, f"{name}.arrow")
        raw_gb = n_points * n_dims * 4 / 1e9

        if os.path.exists(arrow_path):
            print(f"  {name}: arrow exists, skipping")
            if not os.path.exists(npy_path) and raw_gb < 50:
                print(f"    (regenerating npy)")
            else:
                continue

        centers = generate_centers(n_dims, k)

        print(f"  {name}: {n_points:,} x {n_dims}d, k={k}, ~{raw_gb:.1f}GB", flush=True)

        # Arrow — always generate (chunked, constant memory)
        if not os.path.exists(arrow_path):
            t0 = time.time()
            write_arrow_chunked(n_points, n_dims, k, centers, arrow_path)
            arrow_mb = os.path.getsize(arrow_path) / 1e6
            print(f"    arrow: {time.time()-t0:.1f}s, {arrow_mb:.0f}MB", flush=True)

        # Numpy — only for datasets that fit comfortably in memory
        if not os.path.exists(npy_path):
            if raw_gb < 50:
                t0 = time.time()
                write_npy(n_points, n_dims, k, centers, npy_path)
                npy_mb = os.path.getsize(npy_path) / 1e6
                print(f"    npy: {time.time()-t0:.1f}s, {npy_mb:.0f}MB", flush=True)
            else:
                print(f"    npy: SKIPPED (>{raw_gb:.0f}GB, would OOM sklearn anyway)")

    config_path = os.path.join(OUTPUT_DIR, "configs.csv")
    with open(config_path, "w") as f:
        f.write("name,n_points,n_dims,k\n")
        for name, n_points, n_dims, k in configs:
            f.write(f"{name},{n_points},{n_dims},{k}\n")

    print("Done.")


if __name__ == "__main__":
    main()
