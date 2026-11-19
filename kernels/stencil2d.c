/* 5-point stencil, three rows streamed at once. Expected: stream. */
#include "common.h"

#define W 512
#define H 512 /* 512x512 doubles = 2 MB per grid, > 512 KB L2 */

KERNEL void k_stencil(const double *in, double *out, int w, int h) {
    for (int i = 1; i < h - 1; i++)
        for (int j = 1; j < w - 1; j++)
            out[i * w + j] = 0.2 * (in[i * w + j] + in[(i - 1) * w + j] +
                                    in[(i + 1) * w + j] + in[i * w + j - 1] +
                                    in[i * w + j + 1]);
}

int main(void) {
    double *in = xalloc(W * H * sizeof *in);
    double *out = xalloc(W * H * sizeof *out);
    for (int i = 0; i < W * H; i++) {
        in[i] = (double)(rng() & 0xff);
        out[i] = 0.0;
    }
    k_stencil(in, out, W, H);
    printf("%f\n", out[W + 1] + out[(H - 2) * W + W - 2]);
    return 0;
}
