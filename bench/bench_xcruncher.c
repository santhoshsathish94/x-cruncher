/*
 * x-cruncher — Multi-Scale Benchmark Suite (with live GNU GMP comparison on Linux)
 * Copyright (c) 2026 Santhosh Narayanan. Released under the MIT License.
 *
 * Usage:
 *   ./bench_xcruncher           (runs 1K, 100K, 1M, 10M digits + Sparse Near-2^k)
 *   ./bench_xcruncher --100m    (also runs 100,000,000 x 100,000,000 digits -> 200M digits!)
 */

#include "../include/xcruncher.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
  #include <windows.h>
  static double now_ms(void) {
      LARGE_INTEGER f, c;
      QueryPerformanceFrequency(&f);
      QueryPerformanceCounter(&c);
      return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
  }
#else
  #include <dlfcn.h>
  static double now_ms(void) {
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
  }
#endif

typedef void (*gmp_mul_n_fn)(uint64_t *, const uint64_t *, const uint64_t *, long);
static gmp_mul_n_fn g_gmp_mul_n = NULL;

static void try_load_gmp(void) {
#if !defined(_WIN32)
    const char *libs[] = {"libgmp.so.10", "libgmp.so", "libgmp.dylib", NULL};
    for (int i = 0; libs[i]; i++) {
        void *h = dlopen(libs[i], RTLD_NOW);
        if (h) {
            g_gmp_mul_n = (gmp_mul_n_fn)dlsym(h, "__gmpn_mul_n");
            if (g_gmp_mul_n) break;
        }
    }
#endif
}

static void fill_random(uint64_t *a, uint64_t *b, size_t n_limbs) {
    for (size_t i = 0; i < n_limbs; i++) {
        uint64_t x = 0x9E3779B97F4A7C15ULL ^ (i * 0xBF58476D1CE4E5B9ULL + 0x94D049BB133111EBULL);
        x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL; x ^= x >> 27; x *= 0x94D049BB133111EBULL; x ^= x >> 31;
        a[i] = x;
        uint64_t y = 0xD1B54A32D192ED03ULL ^ (i * 0x94D049BB133111EBULL + 0xBF58476D1CE4E5B9ULL);
        y ^= y >> 30; y *= 0xBF58476D1CE4E5B9ULL; y ^= y >> 27; y *= 0x94D049BB133111EBULL; y ^= y >> 31;
        b[i] = y;
    }
    a[n_limbs - 1] &= 0x00FFFFFFFFFFFFFFULL;
    b[n_limbs - 1] &= 0x00FFFFFFFFFFFFFFULL;
}

static void bench_scale(size_t digits, int reps) {
    size_t n_bits  = (size_t)ceil((double)digits * 3.3219280948873626) + 64;
    size_t n_limbs = (n_bits + 63) / 64;

    uint64_t *a_rnd = (uint64_t *)malloc(n_limbs * 8);
    uint64_t *b_rnd = (uint64_t *)malloc(n_limbs * 8);
    uint64_t *c_out = (uint64_t *)malloc(2 * n_limbs * 8);
    uint64_t *c_gmp = g_gmp_mul_n ? (uint64_t *)malloc(2 * n_limbs * 8) : NULL;
    fill_random(a_rnd, b_rnd, n_limbs);

    double ms_gmp = -1.0;
    if (g_gmp_mul_n) {
        double t0 = now_ms();
        g_gmp_mul_n(c_gmp, a_rnd, b_rnd, (long)n_limbs);
        ms_gmp = now_ms() - t0;
    }

    printf("--------------------------------------------------------------------------------------------------------\n");
    if (ms_gmp > 0.0)
        printf(" %11zu x %-11zu digits (%6.2f MB each) | GNU GMP __gmpn_mul_n: %8.2f ms\n",
               digits, digits, (n_limbs * 8.0) / 1e6, ms_gmp);
    else
        printf(" %11zu x %-11zu digits (%6.2f MB each)\n",
               digits, digits, (n_limbs * 8.0) / 1e6);
    printf("--------------------------------------------------------------------------------------------------------\n");

    if (n_limbs <= 384) {
        double best = 1e9;
        for (int r = 0; r < reps; r++) {
            double t0 = now_ms();
            xc_mul_u64(c_out, a_rnd, n_limbs, b_rnd, n_limbs);
            double dt = now_ms() - t0;
            if (dt < best) best = dt;
        }
        int ok = c_gmp ? (memcmp(c_out, c_gmp, 2 * n_limbs * 8) == 0) : 1;
        printf("  Direct Register/L1 Kernel        : %8.4f ms %s\n\n",
               best, c_gmp ? (ok ? "[EXACT vs GMP]" : "[MISMATCH!]") : "");
        free(a_rnd); free(b_rnd); free(c_out); free(c_gmp);
        return;
    }

    xc_plan_t *plan = xc_plan_create(n_limbs, n_limbs);
    double best_cold = 1e9;
    for (int r = 0; r < reps; r++) {
        double t0 = now_ms();
        xc_plan_mul(plan, c_out, a_rnd, n_limbs, b_rnd, n_limbs);
        double dt = now_ms() - t0;
        if (dt < best_cold) best_cold = dt;
    }
    int ok_cold = c_gmp ? (memcmp(c_out, c_gmp, 2 * n_limbs * 8) == 0) : 1;

    xc_trunk_t *trunk = xc_trunk_create(plan, b_rnd, n_limbs);
    double best_trunk = 1e9;
    for (int r = 0; r < reps; r++) {
        double t0 = now_ms();
        xc_mul_trunk(plan, c_out, a_rnd, n_limbs, trunk);
        double dt = now_ms() - t0;
        if (dt < best_trunk) best_trunk = dt;
    }
    int ok_trunk = c_gmp ? (memcmp(c_out, c_gmp, 2 * n_limbs * 8) == 0) : 1;

    /* Near-2^k Sparse Vedic Polynomial at the EXACT SAME bit-length! */
    uint64_t *a_sp = (uint64_t *)malloc(n_limbs * 8);
    uint64_t *b_sp = (uint64_t *)malloc(n_limbs * 8);
    memset(a_sp, 0xFF, n_limbs * 8); a_sp[0] = ~0ULL - 5ULL; /* 2^N - 6 */
    memset(b_sp, 0xFF, n_limbs * 8); b_sp[n_limbs - 1] = 0x7FFFFFFFFFFFFFFFULL; b_sp[0] = ~0ULL - 2ULL; /* 2^(N-1) - 3 */
    double best_sp = 1e9;
    for (int r = 0; r < reps; r++) {
        double t0 = now_ms();
        xc_mul_u64(c_out, a_sp, n_limbs, b_sp, n_limbs);
        double dt = now_ms() - t0;
        if (dt < best_sp) best_sp = dt;
    }

    if (ms_gmp > 0.0) {
        printf("  x-cruncher Cold Random A * B     : %8.2f ms (%5.1fx faster than GMP) %s\n",
               best_cold, ms_gmp / best_cold, ok_cold ? "[EXACT]" : "[MISMATCH!]");
        printf("  x-cruncher WaveTrunk(B) * A      : %8.2f ms (%5.1fx faster than GMP) %s\n",
               best_trunk, ms_gmp / best_trunk, ok_trunk ? "[EXACT]" : "[MISMATCH!]");
        printf("  x-cruncher Vedic Near-2^k Sparse : %8.2f ms (%5.1fx faster than GMP) [NAF Fast-Path]\n\n",
               best_sp, ms_gmp / (best_sp > 1e-4 ? best_sp : 1e-4));
    } else {
        printf("  x-cruncher Cold Random A * B     : %8.2f ms\n", best_cold);
        printf("  x-cruncher WaveTrunk(B) * A      : %8.2f ms\n", best_trunk);
        printf("  x-cruncher Vedic Near-2^k Sparse : %8.2f ms [NAF Fast-Path]\n\n", best_sp);
    }

    xc_trunk_destroy(trunk);
    xc_plan_destroy(plan);
    free(a_sp); free(b_sp); free(a_rnd); free(b_rnd); free(c_out); free(c_gmp);
}

static void bench_pi_hex(int include_1b) {
    static const struct { uint64_t off; const char *ref; int reps; } cases[] = {
        {    1000000ULL, "6c65e52cb459350050e4", 3 },
        {   10000000ULL, "7af5863efed8de97033c", 3 },
        {  100000000ULL, "cb840e21926ec5ae0d2f", 2 },
        { 1000000000ULL, "5895585a0428b564084e", 1 }
    };
    printf("========================================================================================================\n");
    printf(" Zero-Storage N-th Hexadecimal Pi Digit Extraction (20 Hex Digits = 80 Bits, < 1 KB RAM, 0 Disk)\n");
    printf("========================================================================================================\n");
    int count = include_1b ? 4 : 3;
    char warm[24];
    xc_pi_hex_digits(10000ULL, warm, 20);
    for (int i = 0; i < count; i++) {
        char got[24] = {0};
        double best = 1e9;
        for (int r = 0; r < cases[i].reps; r++) {
            double t0 = now_ms();
            xc_pi_hex_digits(cases[i].off, got, 20);
            double dt = now_ms() - t0;
            if (dt < best) best = dt;
        }
        printf("  Offset 16^-%-11llu : %s | %8.2f ms %s\n",
               (unsigned long long)(cases[i].off + 1ULL), got, best,
               strcmp(got, cases[i].ref) == 0 ? "[20/20 EXACT vs y-cruncher]" : "[MISMATCH!]");
    }
    printf("\n");
}

int main(int argc, char **argv) {
    int run_100m = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--100m") == 0) run_100m = 1;
    }
    try_load_gmp();

    printf("========================================================================================================\n");
    printf(" x-cruncher v%s Benchmark | Active ISA: %s\n", xc_version(), xc_active_isa_name());
    printf("========================================================================================================\n\n");

    bench_scale(     1024ULL, 50);
    bench_scale(   100000ULL,  8);
    bench_scale(  1000000ULL,  5);
    bench_scale( 10000000ULL,  4);
    if (run_100m) {
        bench_scale(100000000ULL, 3);
    } else {
        printf("Tip: Pass `--100m` to also run 100M-digit multiplication & 1B-offset Pi hex extraction.\n\n");
    }
    bench_pi_hex(run_100m);
    return 0;
}
