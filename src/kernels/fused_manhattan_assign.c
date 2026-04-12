#ifndef SIZE
#define SIZE 3
#endif

__kernel void fused_manhattan_assign(__global OTYPE *assignments,
                                     __global const float *points,
                                     __global const float *centroids,
                                     uint num_per, uint total, int numClusters) {
    int block = get_global_id(0);
    int start = block * num_per;
    int end = min(start + num_per, total);

    for (int idx = start; idx < end; idx++) {
        float bestDist = MAXFLOAT;
        OTYPE bestIdx = 0;

        for (int c = 0; c < numClusters; c++) {
            float sum = 0.0f;
            for (int i = 0; i < SIZE; i++) {
                float diff = points[idx * SIZE + i] - centroids[c * SIZE + i];
                sum += fabs(diff);
            }
            if (sum < bestDist) {
                bestDist = sum;
                bestIdx = c;
            }
        }
        assignments[idx] = bestIdx;
    }
}
