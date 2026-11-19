/* Histogram. Input streams, bins always hit. Expected: stream / none. */
#include "common.h"

#define N (1u << 20) /* 1M uint32 = 4 MB */

KERNEL void k_hist(const uint32_t *data, size_t n, uint32_t *bins) {
    for (size_t i = 0; i < n; i++)
        bins[data[i] & 0xff]++;
}

int main(void) {
    uint32_t *data = xalloc(N * sizeof *data);
    for (size_t i = 0; i < N; i++)
        data[i] = (uint32_t)rng();
    uint32_t bins[256] = {0};
    k_hist(data, N, bins);
    printf("%u\n", bins[0] + bins[255]);
    return 0;
}
