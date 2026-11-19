/* CSR SpMV. col/val stream, x[col[j]] is random. Expected: mixed. */
#include "common.h"

#define ROWS (1u << 16)
#define NNZ_PER_ROW 8
#define XLEN (1u << 20) /* 8 MB vector */

KERNEL void k_spmv(const uint32_t *row_ptr, const uint32_t *col, const double *val,
                   const double *x, double *y, size_t rows) {
    for (size_t r = 0; r < rows; r++) {
        double s = 0.0;
        for (uint32_t j = row_ptr[r]; j < row_ptr[r + 1]; j++)
            s += val[j] * x[col[j]];
        y[r] = s;
    }
}

int main(void) {
    size_t nnz = (size_t)ROWS * NNZ_PER_ROW;
    uint32_t *row_ptr = xalloc((ROWS + 1) * sizeof *row_ptr);
    uint32_t *col = xalloc(nnz * sizeof *col);
    double *val = xalloc(nnz * sizeof *val);
    double *x = xalloc(XLEN * sizeof *x);
    double *y = xalloc(ROWS * sizeof *y);
    for (size_t r = 0; r <= ROWS; r++)
        row_ptr[r] = (uint32_t)(r * NNZ_PER_ROW);
    for (size_t j = 0; j < nnz; j++) {
        col[j] = (uint32_t)(rng() % XLEN);
        val[j] = (double)(rng() & 0xf);
    }
    for (size_t i = 0; i < XLEN; i++)
        x[i] = 1.0;
    k_spmv(row_ptr, col, val, x, y, ROWS);
    printf("%f\n", y[0] + y[ROWS - 1]);
    return 0;
}
