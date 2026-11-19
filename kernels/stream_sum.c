/* Sequential sum over 8 MB. Expected: stream / next-line. */
#include "common.h"

#define N (1u << 20) /* 1M int64 = 8 MB */

KERNEL int64_t k_sum(const int64_t *a, size_t n) {
    int64_t s = 0;
    for (size_t i = 0; i < n; i++)
        s += a[i];
    return s;
}

int main(void) {
    int64_t *a = xalloc(N * sizeof *a);
    for (size_t i = 0; i < N; i++)
        a[i] = (int64_t)(rng() & 0xff);
    printf("%lld\n", (long long)k_sum(a, N));
    return 0;
}
