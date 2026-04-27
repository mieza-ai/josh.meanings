#ifndef SIZE
#define SIZE 3
#endif

#ifndef MAX_CLUSTERS
#define MAX_CLUSTERS 2
#endif

#ifndef LOCAL_SIZE
#define LOCAL_SIZE 256
#endif

__kernel void fused_emd_assign_reduce(__global const float *points,
                                      __global const float *centroids,
                                      __global float *partial_sums,
                                      __global int *partial_counts,
                                      __global float *partial_inertia,
                                      uint total,
                                      int numClusters) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group = get_group_id(0);
    int local_size = get_local_size(0);

    __local int local_assignments[LOCAL_SIZE];
    __local float local_distances[LOCAL_SIZE];
    __local float local_sums[MAX_CLUSTERS * SIZE];
    __local int local_counts[MAX_CLUSTERS];

    for (int i = lid; i < numClusters * SIZE; i += local_size) {
        local_sums[i] = 0.0f;
    }

    for (int c = lid; c < numClusters; c += local_size) {
        local_counts[c] = 0;
    }

    local_assignments[lid] = -1;
    local_distances[lid] = 0.0f;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid < total) {
        int point_offset = gid * SIZE;
        float best_dist = MAXFLOAT;
        int best_cluster = 0;

        for (int c = 0; c < numClusters; c++) {
            int centroid_offset = c * SIZE;
            // EMD: cumulative prefix sum of (point - centroid), then sum of |.|
            float running = 0.0f;
            float dist = 0.0f;
            for (int d = 0; d < SIZE; d++) {
                running += points[point_offset + d] - centroids[centroid_offset + d];
                dist += fabs(running);
            }

            if (dist < best_dist) {
                best_dist = dist;
                best_cluster = c;
            }
        }

        local_assignments[lid] = best_cluster;
        local_distances[lid] = best_dist;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = lid; i < numClusters * SIZE; i += local_size) {
        int c = i / SIZE;
        int d = i - (c * SIZE);
        float sum = 0.0f;
        int count = 0;

        for (int j = 0; j < local_size; j++) {
            int point_index = group * local_size + j;
            if (point_index < total && local_assignments[j] == c) {
                sum += points[point_index * SIZE + d];
                if (d == 0) {
                    count++;
                }
            }
        }

        local_sums[i] = sum;
        if (d == 0) {
            local_counts[c] = count;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_distances[lid] += local_distances[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int block_sum_offset = group * numClusters * SIZE;
    int block_count_offset = group * numClusters;

    for (int i = lid; i < numClusters * SIZE; i += local_size) {
        partial_sums[block_sum_offset + i] = local_sums[i];
    }

    for (int c = lid; c < numClusters; c += local_size) {
        partial_counts[block_count_offset + c] = local_counts[c];
    }

    if (lid == 0) {
        partial_inertia[group] = local_distances[0];
    }
}
