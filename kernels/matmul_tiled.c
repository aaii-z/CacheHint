/* Tiled matmul. Tiles fit in L1. Expected: high reuse, little to gain. */
#include "common.h"

#define N 96
#define T 16

KERNEL void k_matmul_tiled(const double *A, const double *B, double *C, int n) {
    for (int ii = 0; ii < n; ii += T)
        for (int kk = 0; kk < n; kk += T)
            for (int jj = 0; jj < n; jj += T)
                for (int i = ii; i < ii + T; i++)
                    for (int k = kk; k < kk + T; k++) {
                        double a = A[i * n + k];
                        for (int j = jj; j < jj + T; j++)
                            C[i * n + j] += a * B[k * n + j];
                    }
}

int main(void) {
    double *A = xalloc(N * N * sizeof *A);
    double *B = xalloc(N * N * sizeof *B);
    double *C = xalloc(N * N * sizeof *C);
    for (int i = 0; i < N * N; i++) {
        A[i] = (double)(rng() & 0xf);
        B[i] = (double)(rng() & 0xf);
        C[i] = 0.0;
    }
    k_matmul_tiled(A, B, C, N);
    printf("%f\n", C[N + 1] + C[N * N - 1]);
    return 0;
}
