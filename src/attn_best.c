/*
 * attn_naive.c -- naive single-head attention (given).
 *
 *     O = softmax(Q K^T / sqrt(d)) V        Q, K, V, O: N x d, row-major floats
 *
 * Each variant has its own file in src/ with this same driver; write your
 * code in its attention() function and keep the driver (main) unchanged.
 *
 * Usage:  attn_naive <input.bin> <output.bin>
 *   input.bin : int32 N, int32 d, then Q, K, V (N*d float32 each)
 *   output.bin: int32 N, int32 d, then O (N*d float32)
 * Prints the median time of 5 runs (after 1 warm-up run).
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------------- */
/* attention(): change this function in your variants.                        */
/* ------------------------------------------------------------------------- */
void attention(const float *restrict Q, const float *restrict K,
               const float *restrict V, float *restrict O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    float *S = malloc((size_t)N * (size_t)N * sizeof(float));
    const int BM = 64, BN = 64;
    for (int i0 = 0; i0 < N; i0 += BM) {
        int i1 = i0 + BM; if (i1 > N) i1 = N;
        for (int j0 = 0; j0 < N; j0 += BN) {
            int j1 = j0 + BN; if (j1 > N) j1 = N;
            for (int i = i0; i < i1; i++) {
                const float *q = Q + (size_t)i * d;
                int j = j0;
                for (; j + 4 <= j1; j += 4) {
                    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                    const float *k0 = K + (size_t)(j + 0) * d;
                    const float *k1 = K + (size_t)(j + 1) * d;
                    const float *k2 = K + (size_t)(j + 2) * d;
                    const float *k3 = K + (size_t)(j + 3) * d;
                    #pragma clang loop vectorize(enable) interleave_count(4)
                    for (int k = 0; k < d; k++) {
                        float qv = q[k];
                        a0 += qv * k0[k];
                        a1 += qv * k1[k];
                        a2 += qv * k2[k];
                        a3 += qv * k3[k];
                    }
                    S[(size_t)i * N + j + 0] = a0 * scale;
                    S[(size_t)i * N + j + 1] = a1 * scale;
                    S[(size_t)i * N + j + 2] = a2 * scale;
                    S[(size_t)i * N + j + 3] = a3 * scale;
                }
                for (; j < j1; j++) {
                    float a = 0.f;
                    const float *kj = K + (size_t)j * d;
                    for (int k = 0; k < d; k++) a += q[k] * kj[k];
                    S[(size_t)i * N + j] = a * scale;
                }
            }
        }
    }
    for (int i = 0; i < N; i++) {
        float *row = S + (size_t)i * N;
        float maxv = row[0];
        for (int j = 1; j < N; j++) if (row[j] > maxv) maxv = row[j];
        float sum = 0.f;
        for (int j = 0; j < N; j++) { row[j] = expf(row[j] - maxv); sum += row[j]; }
        float inv = 1.f / sum;
        for (int j = 0; j < N; j++) row[j] *= inv;
    }
    for (int i = 0; i < N; i++)
        for (int k = 0; k < d; k++) O[(size_t)i * d + k] = 0.f;
    for (int j0 = 0; j0 < N; j0 += BN) {
        int j1 = j0 + BN; if (j1 > N) j1 = N;
        for (int i = 0; i < N; i++) {
            float *o = O + (size_t)i * d;
            for (int j = j0; j < j1; j++) {
                float s = S[(size_t)i * N + j];
                const float *vj = V + (size_t)j * d;
                #pragma clang loop vectorize(enable)
                for (int k = 0; k < d; k++) o[k] += s * vj[k];
            }
        }
    }
    free(S);
}


/* Driver: do not change below this line.                                     */
/* ------------------------------------------------------------------------- */
#define REPS 5

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }
    FILE *in = fopen(argv[1], "rb");
    if (!in) {
        perror(argv[1]);
        return 1;
    }
    int32_t hdr[2];
    if (fread(hdr, sizeof hdr, 1, in) != 1) {
        fprintf(stderr, "%s: bad header\n", argv[1]);
        return 1;
    }
    int N = hdr[0], d = hdr[1];
    size_t n = (size_t)N * d;
    float *Q = malloc(n * sizeof(float)), *K = malloc(n * sizeof(float));
    float *V = malloc(n * sizeof(float)), *O = malloc(n * sizeof(float));
    if (!Q || !K || !V || !O || fread(Q, sizeof(float), n, in) != n ||
        fread(K, sizeof(float), n, in) != n || fread(V, sizeof(float), n, in) != n) {
        fprintf(stderr, "%s: could not read Q, K, V\n", argv[1]);
        return 1;
    }
    fclose(in);

    attention(Q, K, V, O, N, d); /* warm-up, not timed */
    double t[REPS];
    for (int r = 0; r < REPS; r++) {
        double t0 = now();
        attention(Q, K, V, O, N, d);
        t[r] = now() - t0;
    }
    qsort(t, REPS, sizeof t[0], cmp_double);
    double median = t[REPS / 2];

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const char *name = strrchr(argv[0], '/') ? strrchr(argv[0], '/') + 1 : argv[0];
    printf("%-18s N=%-5d d=%-4d threads=%-3d time_ms=%10.3f  GFLOP/s=%8.2f\n", name, N, d,
           threads, median * 1e3, 4.0 * N * (double)N * d / median * 1e-9);

    FILE *out = fopen(argv[2], "wb");
    if (!out || fwrite(hdr, sizeof hdr, 1, out) != 1 || fwrite(O, sizeof(float), n, out) != n) {
        perror(argv[2]);
        return 1;
    }
    fclose(out);
    free(Q);
    free(K);
    free(V);
    free(O);
    return 0;
}
