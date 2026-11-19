/* Descending scan. Expected: stride/stream; next-line goes the wrong way. */
#include "common.h"

#define N (1u << 20) /* 1M int32 = 4 MB */

KERNEL int64_t k_rsum(const int32_t *a, size_t n) {
    int64_t s = 0;
    for (size_t i = n; i-- > 0;)
        s += a[i];
    return s;
}

int main(void) {
    int32_t *a = xalloc(N * sizeof *a);
    for (size_t i = 0; i < N; i++)
        a[i] = (int32_t)(rng() & 0xff);
    printf("%lld\n", (long long)k_rsum(a, N));
    return 0;
}
