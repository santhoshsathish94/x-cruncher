/*
 * x-cruncher — Comprehensive Unit & Exactness Test Suite
 * Copyright (c) 2026 Santhosh Narayanan. Released under the MIT License.
 */

#include "../include/xcruncher.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
    else      { g_fail++; printf("  [FAIL] %s (line %d)\n", msg, __LINE__); } \
} while (0)

static uint64_t xorshift64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return (*s = x);
}

/* Independent reference O(N*M) multiplier for ground-truth verification */
static void ref_mul(uint64_t *out, const uint64_t *a, size_t na, const uint64_t *b, size_t nb) {
    memset(out, 0, (na + nb) * sizeof(uint64_t));
    for (size_t i = 0; i < na; i++) {
        uint64_t carry = 0;
        for (size_t j = 0; j < nb; j++) {
#if defined(__SIZEOF_INT128__)
            __uint128_t p = (__uint128_t)a[i] * (__uint128_t)b[j] + out[i + j] + carry;
            out[i + j] = (uint64_t)p;
            carry = (uint64_t)(p >> 64);
#else
            uint64_t a0 = a[i] & 0xFFFFFFFFULL, a1 = a[i] >> 32;
            uint64_t b0 = b[j] & 0xFFFFFFFFULL, b1 = b[j] >> 32;
            uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
            uint64_t mid = (p00 >> 32) + (p01 & 0xFFFFFFFFULL) + (p10 & 0xFFFFFFFFULL);
            uint64_t lo = (p00 & 0xFFFFFFFFULL) | (mid << 32);
            uint64_t hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
            uint64_t s1 = out[i + j] + lo; uint64_t c1 = (s1 < lo);
            uint64_t s2 = s1 + carry;      uint64_t c2 = (s2 < s1);
            out[i + j] = s2; carry = hi + c1 + c2;
#endif
        }
        out[i + nb] += carry;
    }
}

int main(void) {
    printf("========================================================================\n");
    printf(" x-cruncher v%s Test Suite | Active ISA: %s\n", xc_version(), xc_active_isa_name());
    printf("========================================================================\n");

    uint64_t rng = 0xCAFEBABE98765432ULL;

    /* 1. Edge cases: zero & max 64-bit limb */
    {
        uint64_t a[2] = {0, 0}, b[2] = {12345ULL, 67890ULL}, out[4] = {9, 9, 9, 9};
        xc_mul_u64(out, a, 2, b, 2);
        CHECK(out[0]==0 && out[1]==0 && out[2]==0 && out[3]==0 && xc_last_path_used()==XC_PATH_ZERO,
              "Zero operand fast-path clears output cleanly");

        uint64_t m1 = ~0ULL, m2 = ~0ULL, mo[2] = {0}, mr[2] = {0};
        xc_mul_u64(mo, &m1, 1, &m2, 1);
        ref_mul(mr, &m1, 1, &m2, 1);
        CHECK(memcmp(mo, mr, 16) == 0, "Max uint64 (2^64 - 1)^2 carry exactness");
    }

    /* 2. Schoolbook & Karatsuba tiers (balanced & unbalanced) */
    {
        const size_t sizes[][2] = {{7, 7}, {32, 32}, {48, 48}, {96, 45}, {128, 128}, {250, 190}, {380, 380}};
        int all_ok = 1;
        for (size_t s = 0; s < sizeof(sizes)/sizeof(sizes[0]); s++) {
            size_t na = sizes[s][0], nb = sizes[s][1];
            uint64_t *a = (uint64_t *)malloc(na * 8), *b = (uint64_t *)malloc(nb * 8);
            uint64_t *got = (uint64_t *)malloc((na + nb) * 8), *exp = (uint64_t *)malloc((na + nb) * 8);
            for (size_t i = 0; i < na; i++) a[i] = xorshift64(&rng);
            for (size_t i = 0; i < nb; i++) b[i] = xorshift64(&rng);
            xc_mul_u64(got, a, na, b, nb);
            ref_mul(exp, a, na, b, nb);
            if (memcmp(got, exp, (na + nb) * 8) != 0) all_ok = 0;
            free(a); free(b); free(got); free(exp);
        }
        CHECK(all_ok, "Schoolbook & Karatsuba balanced/unbalanced exactness (7 .. 380 limbs)");
    }

    /* 3. Vedic Sparse Signed Power-of-2 (NAF) Fast-Path: (2^(64*N) - 6) * (2^(64*N - 1) - 3) */
    {
        const size_t N = 1024; /* 65,536-bit near-power-of-2 numbers! */
        uint64_t *a = (uint64_t *)malloc(N * 8), *b = (uint64_t *)malloc(N * 8);
        uint64_t *got = (uint64_t *)malloc(2 * N * 8), *exp = (uint64_t *)malloc(2 * N * 8);
        for (size_t i = 0; i < N; i++) { a[i] = ~0ULL; b[i] = ~0ULL; }
        a[0] = ~0ULL - 5ULL;            /* 2^(64*1024) - 6 */
        b[N - 1] = 0x7FFFFFFFFFFFFFFFULL;
        b[0] = ~0ULL - 2ULL;            /* 2^(64*1024 - 1) - 3 */

        xc_mul_u64(got, a, N, b, N);
        xc_path_t path_used = xc_last_path_used();
        ref_mul(exp, a, N, b, N);
        CHECK(path_used == XC_PATH_VEDIC_SPARSE && memcmp(got, exp, 2 * N * 8) == 0,
              "Vedic Sparse Signed-2^k (NAF) fast-path exact on (2^65536 - 6)*(2^65535 - 3)");
        free(a); free(b); free(got); free(exp);
    }

    /* 4. Crandall-Fagin 16-bit Vedic Wave Engine (Cold, Unbalanced, & All-Ones Carry Stress) */
    {
        const size_t sizes[][2] = {{512, 512}, {1500, 1500}, {3200, 900}, {6000, 6000}};
        int wave_ok = 1;
        for (size_t s = 0; s < sizeof(sizes)/sizeof(sizes[0]); s++) {
            size_t na = sizes[s][0], nb = sizes[s][1];
            uint64_t *a = (uint64_t *)malloc(na * 8), *b = (uint64_t *)malloc(nb * 8);
            uint64_t *got = (uint64_t *)malloc((na + nb) * 8), *exp = (uint64_t *)malloc((na + nb) * 8);
            for (size_t i = 0; i < na; i++) a[i] = xorshift64(&rng);
            for (size_t i = 0; i < nb; i++) b[i] = xorshift64(&rng);
            xc_mul_u64(got, a, na, b, nb);
            ref_mul(exp, a, na, b, nb);
            if (xc_last_path_used() != XC_PATH_WAVE_COLD || memcmp(got, exp, (na + nb) * 8) != 0) wave_ok = 0;
            free(a); free(b); free(got); free(exp);
        }
        CHECK(wave_ok, "Crandall-Fagin 16-bit Vedic Wave Engine exact (512 .. 6,000 limbs, ~115K digits)");
    }

    /* 5. Precomputed WaveTrunk(B) + Repeated-Block Hash Wave Collapse + Low-RAM Karatsuba */
    {
        const size_t N = 4096; /* ~79,000 digits */
        uint64_t *a_rnd = (uint64_t *)malloc(N * 8);
        uint64_t *a_rep = (uint64_t *)malloc(N * 8);
        uint64_t *b_rnd = (uint64_t *)malloc(N * 8);
        uint64_t *exp1  = (uint64_t *)malloc(2 * N * 8);
        uint64_t *exp2  = (uint64_t *)malloc(2 * N * 8);
        uint64_t *got   = (uint64_t *)malloc(2 * N * 8);

        for (size_t i = 0; i < N; i++) { a_rnd[i] = xorshift64(&rng); b_rnd[i] = xorshift64(&rng); }
        /* Tile 4 distinct 16-limb (64 uint16_t) slices across a_rep */
        for (size_t j = 0; j < N / 16; j++) {
            memcpy(a_rep + j * 16, a_rnd + (j & 3ULL) * 16, 16 * 8);
            a_rep[j * 16 + 15] &= 0x7FFFFFFFFFFFFFFFULL;
        }

        ref_mul(exp1, a_rnd, N, b_rnd, N);
        ref_mul(exp2, a_rep, N, b_rnd, N);

        xc_plan_t  *plan  = xc_plan_create(N, N);
        xc_trunk_t *trunk = xc_trunk_create(plan, b_rnd, N);

        xc_mul_trunk(plan, got, a_rnd, N, trunk);
        CHECK(memcmp(got, exp1, 2 * N * 8) == 0, "Precomputed Static WaveTrunk(B) * Random A exact");

        xc_mul_trunk(plan, got, a_rep, N, trunk);
        CHECK(memcmp(got, exp2, 2 * N * 8) == 0, "Precomputed WaveTrunk(B) + Block-Hash Wave Collapse exact");

        xc_mul_u64_lowmem(got, a_rnd, N, b_rnd, N);
        CHECK(xc_last_path_used() == XC_PATH_LOWMEM_KARA && memcmp(got, exp1, 2 * N * 8) == 0,
              "Low-RAM Raw-Bit Karatsuba Top-Split (50% wave RAM) exact");

        xc_trunk_destroy(trunk);
        xc_plan_destroy(plan);
        free(a_rnd); free(a_rep); free(b_rnd); free(exp1); free(exp2); free(got);
    }

    /* 6. Adversarial Worst-Case Carry-Free Vedic Saturation Test (all limbs = +2^(B-1) - 1, zero cancellation!)
          Tests both B = 16 (at 8,000 words) and adaptive B = 14 (at 60,000 words ~ 1.15M digits)! */
    {
        const size_t n1 = 8000;  /* B = 16 tier */
        uint64_t *a1 = (uint64_t *)malloc(n1 * 8), *g1 = (uint64_t *)malloc(2 * n1 * 8), *e1 = (uint64_t *)malloc(2 * n1 * 8);
        for (size_t i = 0; i < n1; i++) a1[i] = 0x7FFF7FFF7FFF7FFFULL;
        xc_mul_u64(g1, a1, n1, a1, n1);
        ref_mul(e1, a1, n1, a1, n1);
        int ok1 = (xc_last_limb_bits() == 16 && memcmp(g1, e1, 2 * n1 * 8) == 0);
        free(a1); free(g1); free(e1);

        const size_t n2 = 60000; /* ~1.15M digits -> triggers adaptive B = 14 worst-case safe tier! */
        uint64_t *a2 = (uint64_t *)calloc(n2, 8), *g2 = (uint64_t *)malloc(2 * n2 * 8), *e2 = (uint64_t *)malloc(2 * n2 * 8);
        /* Pack +8191 (2^13 - 1, max positive 14-bit Vedic limb!) into every single 14-bit slot */
        for (size_t m = 0; (m + 1) * 14 <= n2 * 64; m++) {
            size_t bit = m * 14, w = bit >> 6, sh = bit & 63;
            uint64_t v = 8191ULL;
            a2[w] |= (v << sh);
            if (sh > 50 && w + 1 < n2) a2[w + 1] |= (v >> (64 - sh));
        }
        xc_mul_u64(g2, a2, n2, a2, n2);
        int b_used = xc_last_limb_bits();
        /* Verify against Low-RAM Karatsuba (which uses smaller B=15/16 leaves) & spot-check ref_mul on 60K words */
        ref_mul(e2, a2, n2, a2, n2);
        int ok2 = (b_used <= 14 && memcmp(g2, e2, 2 * n2 * 8) == 0);
        free(a2); free(g2); free(e2);

        CHECK(ok1 && ok2, "Adversarial worst-case non-canceling Vedic limb saturation (1.15M digits, B=14) exact");
    }

    /* 7. Decimal string multiplication */
    {
        char *p = xc_mul_decimal("999999999999999999999999999999", "888888888888888888888888888888");
        CHECK(p != NULL && strcmp(p, "888888888888888888888888888887111111111111111111111111111112") == 0,
              "Decimal string Base-10^9 conversion & multiplication exact");
        xc_free(p);
    }

    /* 8. Zero-Storage N-th Hexadecimal Digit Extractor for Pi (verified vs y-cruncher v0.8.7) */
    {
        char h0[24] = {0}, h1m[24] = {0};
        int r0 = xc_pi_hex_digits(0ULL,       h0,  20);
        int r1 = xc_pi_hex_digits(1000000ULL, h1m, 20);
        CHECK(r0 == XC_OK && strcmp(h0,  "243f6a8885a308d31319") == 0 &&
              r1 == XC_OK && strcmp(h1m, "6c65e52cb459350050e4") == 0,
              "Zero-storage Pi hex digit extraction (20/20 hex digits at offset 0 & 1,000,000) exact");
    }

    printf("========================================================================\n");
    printf(" RESULT: %d passed, %d failed\n", g_pass, g_fail);
    printf("========================================================================\n");
    return (g_fail == 0) ? 0 : 1;
}
