/* 96 B structs, several fields per element. Expected: mixed per field. */
#include "common.h"

typedef struct {
    double x, y, z;
    double vx, vy, vz;
    double m;
    double pad[5];
} particle_t; /* 96 bytes */

#define N (1u << 16) /* 64K particles = 6 MB */
#define REPS 2

KERNEL void k_step(particle_t *p, size_t n, double dt) {
    for (size_t i = 0; i < n; i++) {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

int main(void) {
    particle_t *p = xalloc(N * sizeof *p);
    for (size_t i = 0; i < N; i++) {
        p[i].x = p[i].y = p[i].z = 0.0;
        p[i].vx = (double)(rng() & 0xff);
        p[i].vy = (double)(rng() & 0xff);
        p[i].vz = (double)(rng() & 0xff);
        p[i].m = 1.0;
    }
    for (int k = 0; k < REPS; k++)
        k_step(p, N, 0.01);
    printf("%f\n", p[N / 2].x + p[N - 1].z);
    return 0;
}
