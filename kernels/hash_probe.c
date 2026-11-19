/* Open-addressing hash lookups. Expected: queries stream, probes random. */
#include "common.h"

#define TBITS 20
#define TSIZE (1u << TBITS) /* 1M slots x 8 B = 8 MB */
#define NKEYS (600000u)
#define NQ (400000u)
#define EMPTY 0ull

static inline uint64_t mix(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdull;
    k ^= k >> 33;
    return k;
}

static void insert(uint64_t *t, uint64_t key) {
    uint64_t h = mix(key) & (TSIZE - 1);
    while (t[h] != EMPTY && t[h] != key)
        h = (h + 1) & (TSIZE - 1);
    t[h] = key;
}

KERNEL size_t k_lookup(const uint64_t *t, const uint64_t *q, size_t nq) {
    size_t found = 0;
    for (size_t i = 0; i < nq; i++) {
        uint64_t key = q[i];
        uint64_t h = mix(key) & (TSIZE - 1);
        while (t[h] != EMPTY) {
            if (t[h] == key) {
                found++;
                break;
            }
            h = (h + 1) & (TSIZE - 1);
        }
    }
    return found;
}

int main(void) {
    uint64_t *t = xalloc(TSIZE * sizeof *t);
    memset(t, 0, TSIZE * sizeof *t);
    uint64_t *keys = xalloc(NKEYS * sizeof *keys);
    for (size_t i = 0; i < NKEYS; i++) {
        keys[i] = rng() | 1;
        insert(t, keys[i]);
    }
    uint64_t *q = xalloc(NQ * sizeof *q);
    for (size_t i = 0; i < NQ; i++)
        q[i] = (i & 1) ? keys[rng() % NKEYS] : (rng() | 1); /* half hits */
    printf("%zu\n", k_lookup(t, q, NQ));
    return 0;
}
