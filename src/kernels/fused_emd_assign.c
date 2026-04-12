#ifndef SIZE
#define SIZE 3
#endif

__kernel void fused_emd_assign(__global OTYPE *assignments,
                                __global const float *points,
                                __global const float *centroids,
                                uint num_per, uint total, int numClusters) {
    int block = get_global_id(0);
    int start = block * num_per;
    int end = min(start + num_per, total);

    for (int idx = start; idx < end; idx++) {
        float point[SIZE];
        for (int i = 0; i < SIZE; i++) {
            point[i] = points[idx * SIZE + i];
        }

        float bestDist = MAXFLOAT;
        OTYPE bestIdx = 0;

        for (int c = 0; c < numClusters; c++) {
            float dist[SIZE];
            for (int i = 0; i < SIZE; i++) {
                dist[i] = point[i] - centroids[c * SIZE + i];
            }

            // Cumulative prefix sum (Wasserstein/EMD)
            for (int i = 1; i < SIZE; i++) {
                dist[i] += dist[i - 1];
            }

            float sum = 0.0f;
            for (int i = 0; i < SIZE; i++) {
                sum += fabs(dist[i]);
            }

            if (sum < bestDist) {
                bestDist = sum;
                bestIdx = c;
            }
        }
        assignments[idx] = bestIdx;
    }
}
