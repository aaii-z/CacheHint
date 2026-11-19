/* Shared helpers. Only k_* functions (KERNEL) are traced and labelled. */
#ifndef CACHEHINT_COMMON_H
#define CACHEHINT_COMMON_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KERNEL __attribute__((noipa)) /* no inlining or IPA (e.g. hoisting pure calls) */

static uint64_t rng_state = 88172645463325252ull;

/* Deterministic xorshift64. */
static inline uint64_t rng(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return rng_state = x;
}

static void *xalloc(size_t n) {
    void *p = aligned_alloc(64, (n + 63) / 64 * 64);
    if (!p) {
        perror("aligned_alloc");
        exit(1);
    }
    return p;
}

static void shuffle_u32(uint32_t *a, size_t n) {
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = rng() % (i + 1);
        uint32_t t = a[i];
        a[i] = a[j];
        a[j] = t;
    }
}

#endif
