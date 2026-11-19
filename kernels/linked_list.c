/* Linked-list walk. Same binary, two inputs: `seq` (nodes in memory order)
 * and `shuf` (random order). Same PCs, different best prefetcher. */
#include "common.h"

typedef struct node {
    struct node *next;
    int64_t val;
    char pad[48];
} node_t; /* 64 bytes = one cache line */

#define N (1u << 17) /* 128K nodes = 8 MB */
#define REPS 4

KERNEL int64_t k_walk(const node_t *head) {
    int64_t s = 0;
    for (const node_t *p = head; p; p = p->next)
        s += p->val;
    return s;
}

int main(int argc, char **argv) {
    int shuffled = argc > 1 && strcmp(argv[1], "shuf") == 0;
    node_t *pool = xalloc(N * sizeof *pool);
    uint32_t *order = xalloc(N * sizeof *order);
    for (uint32_t i = 0; i < N; i++)
        order[i] = i;
    if (shuffled)
        shuffle_u32(order, N);
    for (uint32_t i = 0; i < N; i++) {
        node_t *n = &pool[order[i]];
        n->val = (int64_t)(rng() & 0xff);
        n->next = (i + 1 < N) ? &pool[order[i + 1]] : NULL;
    }
    int64_t s = 0;
    for (int k = 0; k < REPS; k++)
        s += k_walk(&pool[order[0]]);
    printf("%lld\n", (long long)s);
    return 0;
}
