/* Naive i-j-k matmul. A by rows, B by columns. Expected: stride for B. */
#include "common.h"

#define N 96 /* 96x96 doubles = 72 KB per matrix, > 32 KB L1 */

KERNEL void k_matmul(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++)
                s += A[i * n + k] * B[k * n + j];
            C[i * n + j] = s;
        }
}

int main(void) {
    double *A = xalloc(N * N * sizeof *A);
    double *B = xalloc(N * N * sizeof *B);
    double *C = xalloc(N * N * sizeof *C);
    for (int i = 0; i < N * N; i++) {
        A[i] = (double)(rng() & 0xf);
        B[i] = (double)(rng() & 0xf);
    }
    k_matmul(A, B, C, N);
    printf("%f\n", C[N + 1] + C[N * N - 1]);
    return 0;
}
