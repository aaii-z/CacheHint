/* One field per 256 B struct: 4-line stride. Expected: stride. */
#include "common.h"

typedef struct {
    int64_t key;
    char pad[248];
} rec_t; /* 256 bytes */

#define N (1u << 15) /* 32K records = 8 MB */
#define REPS 8

KERNEL int64_t k_sum_keys(const rec_t *r, size_t n) {
    int64_t s = 0;
    for (size_t i = 0; i < n; i++)
        s += r[i].key;
    return s;
}

int main(void) {
    rec_t *r = xalloc(N * sizeof *r);
    for (size_t i = 0; i < N; i++)
        r[i].key = (int64_t)(rng() & 0xff);
    int64_t s = 0;
    for (int k = 0; k < REPS; k++)
        s += k_sum_keys(r, N);
    printf("%lld\n", (long long)s);
    return 0;
}
