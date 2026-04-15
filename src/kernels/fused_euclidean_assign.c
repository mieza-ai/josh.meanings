#ifndef SIZE
#define SIZE 3
#endif

// For argmin, sqrt is monotonic so we can compare squared distances directly.
// This gives identical assignments to the two-pass euclidean kernel but avoids
// the sqrt cost per comparison.
__kernel void fused_euclidean_assign(__global OTYPE *assignments,
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
        float sum = 0.0f;
        for (int i = 0; i < SIZE; i++) {
            float diff = points[idx * SIZE + i] - centroids[c * SIZE + i];
            sum += diff * diff;
        }
        // sqrt is monotonic - skip it for argmin comparison
        if (sum < bestDist) {
            bestDist = sum;
            bestIdx = c;
        }
    }
    assignments[idx] = bestIdx;
}
