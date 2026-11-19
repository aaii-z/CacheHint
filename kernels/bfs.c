/* BFS on a random CSR graph. adj streams, dist[u] random. Expected: mixed. */
#include "common.h"

#define V (1u << 17)
#define DEG 8

KERNEL size_t k_bfs(const uint32_t *row_ptr, const uint32_t *adj, int32_t *dist,
                    uint32_t *queue, uint32_t src) {
    size_t head = 0, tail = 0;
    dist[src] = 0;
    queue[tail++] = src;
    while (head < tail) {
        uint32_t v = queue[head++];
        for (uint32_t e = row_ptr[v]; e < row_ptr[v + 1]; e++) {
            uint32_t u = adj[e];
            if (dist[u] < 0) {
                dist[u] = dist[v] + 1;
                queue[tail++] = u;
            }
        }
    }
    return tail;
}

int main(void) {
    size_t ne = (size_t)V * DEG;
    uint32_t *row_ptr = xalloc((V + 1) * sizeof *row_ptr);
    uint32_t *adj = xalloc(ne * sizeof *adj);
    int32_t *dist = xalloc(V * sizeof *dist);
    uint32_t *queue = xalloc(V * sizeof *queue);
    for (size_t v = 0; v <= V; v++)
        row_ptr[v] = (uint32_t)(v * DEG);
    for (size_t e = 0; e < ne; e++)
        adj[e] = (uint32_t)(rng() % V);
    for (size_t v = 0; v < V; v++)
        dist[v] = -1;
    printf("%zu\n", k_bfs(row_ptr, adj, dist, queue, 0));
    return 0;
}
