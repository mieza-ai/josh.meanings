#ifndef SIZE
#define SIZE 3
#endif

__kernel void fused_chebyshev_assign(__global OTYPE *assignments,
                                     __global const float *points,
                                     __global const float *centroids,
                                     uint num_per, uint total, int numClusters) {
    uint idx = get_global_id(0);
    if (idx >= total) {
        return;
    }

    float bestDist = MAXFLOAT;
    OTYPE bestIdx = 0;

    for (int c = 0; c < numClusters; c++) {
        float maxv = 0.0f;
        for (int i = 0; i < SIZE; i++) {
            float diff = fabs(points[idx * SIZE + i] - centroids[c * SIZE + i]);
            if (diff > maxv) {
                maxv = diff;
            }
        }
        if (maxv < bestDist) {
            bestDist = maxv;
            bestIdx = c;
        }
    }
    assignments[idx] = bestIdx;
}
