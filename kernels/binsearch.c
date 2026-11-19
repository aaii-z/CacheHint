/* Binary search on 8 MB. Expected: none. */
#include "common.h"

#define N (1u << 21) /* 2M int32 = 8 MB */
#define NQ 60000u

KERNEL size_t k_search(const int32_t *a, size_t n, const int32_t *q, size_t nq) {
    size_t hits = 0;
    for (size_t i = 0; i < nq; i++) {
        size_t lo = 0, hi = n;
        int32_t key = q[i];
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (a[mid] < key)
                lo = mid + 1;
            else
                hi = mid;
        }
        hits += (lo < n && a[lo] == key);
    }
    return hits;
}

int main(void) {
    int32_t *a = xalloc(N * sizeof *a);
    for (size_t i = 0; i < N; i++)
        a[i] = (int32_t)(i * 3);
    int32_t *q = xalloc(NQ * sizeof *q);
    for (size_t i = 0; i < NQ; i++)
        q[i] = (int32_t)(rng() % (3ull * N));
    printf("%zu\n", k_search(a, N, q, NQ));
    return 0;
}
