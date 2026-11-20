/*
 * x-cruncher — High-Performance Arbitrary-Precision Multiplication Engine
 * Copyright (c) 2026 Santhosh Narayanan. Released under the MIT License.
 */

#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "../include/xcruncher.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
  #include <malloc.h>
#else
  #include <sys/mman.h>
  #if defined(__linux__)
    #include <sched.h>
    #include <unistd.h>
  #endif
#endif

#ifdef _OPENMP
  #include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
  #include <immintrin.h>
#endif

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

#define R_TOP 640

/* Radix-5 Winograd / Rader constants */
#define C5_1  0.30901699437494742410229341718281905886
#define S5_1 -0.95105651629515357211643933337938214340
#define C5_2 -0.80901699437494742410229341718281905886
#define S5_2 -0.58778525229247312916870595463907276859

/* Diagnostic state */
static xc_path_t g_last_path = XC_PATH_ZERO;
static int       g_last_uniq = 0;

/* =========================================================================================
 * CROSS-PLATFORM ALIGNED / HUGEPAGE ALLOCATOR & THREAD TOPOLOGY
 * ========================================================================================= */
static void *xc_aligned_alloc_buf(size_t bytes, int huge) {
    size_t align = huge ? (2ULL << 20) : 64ULL;
    size_t sz = (bytes + align - 1) & ~(align - 1);
    if (sz == 0) sz = align;
#if defined(_WIN32)
    (void)huge;
    return _aligned_malloc(sz, 64);
#else
    void *p = NULL;
    if (posix_memalign(&p, align, sz) != 0 || !p) return NULL;
  #if defined(__linux__) && defined(MADV_HUGEPAGE)
    if (huge) madvise(p, sz, MADV_HUGEPAGE);
  #endif
    return p;
#endif
}

static void xc_aligned_free_buf(void *p) {
    if (!p) return;
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

static int xc_wave_threads(size_t M) {
#ifdef _OPENMP
    if (M < 16384) return 1;
    static int best_nt = 0;
    if (best_nt == 0) {
        int max_t = omp_get_max_threads();
  #if defined(__linux__)
        int phys = max_t;
        FILE *f = fopen("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list", "r");
        if (f) {
            char buf[64] = {0};
            if (fgets(buf, sizeof(buf), f) && (strchr(buf, ',') || strchr(buf, '-'))) {
                phys = (max_t >= 4) ? (max_t / 2) : max_t;
            }
            fclose(f);
        }
        best_nt = (phys > 0) ? phys : 1;
        if (best_nt >= 8) {
            /* Spawn the full `best_nt` OpenMP pool FIRST, then pin each worker `tid` to physical core `tid` */
            #pragma omp parallel num_threads(best_nt)
            {
                int tid = omp_get_thread_num();
                cpu_set_t cs;
                CPU_ZERO(&cs);
                CPU_SET(tid, &cs);
                sched_setaffinity(0, sizeof(cs), &cs);
            }
        }
  #else
        best_nt = (max_t > 16) ? (max_t / 2) : (max_t > 0 ? max_t : 1);
  #endif
    }
    return best_nt;
#else
    (void)M;
    return 1;
#endif
}

/* =========================================================================================
 * RUNTIME ISA DETECTION (AVX-512 / AVX2+FMA / PORTABLE SCALAR)
 * ========================================================================================= */
typedef enum {
    XC_ISA_SCALAR = 0,
    XC_ISA_AVX2   = 1,
    XC_ISA_AVX512 = 2
} xc_isa_t;

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
  #define XC_HAS_X86_SIMD 1
  #define XC_TARGET_AVX512 __attribute__((target("avx512f,avx512dq,avx512bw,avx512vl,fma")))
  #define XC_TARGET_AVX2   __attribute__((target("avx2,fma")))
#else
  #define XC_HAS_X86_SIMD 0
  #define XC_TARGET_AVX512
  #define XC_TARGET_AVX2
#endif

static xc_isa_t xc_detect_isa(void) {
    static int cached = -1;
    if (cached >= 0) return (xc_isa_t)cached;
#if XC_HAS_X86_SIMD
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f") &&
        __builtin_cpu_supports("avx512dq") &&
        __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512vl") &&
        __builtin_cpu_supports("fma")) {
        cached = XC_ISA_AVX512;
        return XC_ISA_AVX512;
    }
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        cached = XC_ISA_AVX2;
        return XC_ISA_AVX2;
    }
#endif
    cached = XC_ISA_SCALAR;
    return XC_ISA_SCALAR;
}

XC_API const char *xc_version(void) { return XCRUNCHER_VERSION; }

XC_API const char *xc_active_isa_name(void) {
    xc_isa_t isa = xc_detect_isa();
    if (isa == XC_ISA_AVX512) return "AVX-512 (AoSoA8 ZMM + vmovntpd)";
    if (isa == XC_ISA_AVX2)   return "AVX2+FMA (AoSoA8 Dual-YMM)";
    return "Portable-Scalar (Auto-Vectorized)";
}

XC_API xc_path_t xc_last_path_used(void)    { return g_last_path; }
XC_API int       xc_last_unique_slices(void){ return g_last_uniq; }
XC_API void      xc_free(void *ptr)         { free(ptr); }

/* =========================================================================================
 * TIER 1: REGISTER-SCALE 64x64->128 SCHOOLBOOK & KARATSUBA
 * ========================================================================================= */
static inline void mul64x64(uint64_t a, uint64_t b, uint64_t *lo, uint64_t *hi) {
#if defined(__SIZEOF_INT128__)
    __uint128_t p = (__uint128_t)a * (__uint128_t)b;
    *lo = (uint64_t)p;
    *hi = (uint64_t)(p >> 64);
#elif defined(_MSC_VER) && defined(_M_X64)
    *lo = _umul128(a, b, hi);
#else
    uint64_t a0 = a & 0xFFFFFFFFULL, a1 = a >> 32;
    uint64_t b0 = b & 0xFFFFFFFFULL, b1 = b >> 32;
    uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    uint64_t mid = (p00 >> 32) + (p01 & 0xFFFFFFFFULL) + (p10 & 0xFFFFFFFFULL);
    *lo = (p00 & 0xFFFFFFFFULL) | (mid << 32);
    *hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
#endif
}

static void schoolbook_mul_u64(uint64_t *out, const uint64_t *a, size_t na,
                               const uint64_t *b, size_t nb) {
    memset(out, 0, (na + nb) * sizeof(uint64_t));
    for (size_t i = 0; i < na; i++) {
        uint64_t ai = a[i];
        if (!ai) continue;
        uint64_t carry = 0;
        for (size_t j = 0; j < nb; j++) {
            uint64_t lo, hi;
            mul64x64(ai, b[j], &lo, &hi);
            uint64_t s1 = out[i + j] + lo;
            uint64_t c1 = (s1 < lo);
            uint64_t s2 = s1 + carry;
            uint64_t c2 = (s2 < s1);
            out[i + j] = s2;
            carry = hi + c1 + c2;
        }
        out[i + nb] += carry;
    }
}

static uint64_t limbs_add(uint64_t *dst, const uint64_t *a, size_t na, const uint64_t *b, size_t nb) {
    size_t i = 0, min_n = na < nb ? na : nb;
    uint64_t c = 0;
    for (; i < min_n; i++) {
        uint64_t s1 = a[i] + b[i];
        uint64_t c1 = (s1 < a[i]);
        uint64_t s2 = s1 + c;
        c = c1 + (s2 < s1);
        dst[i] = s2;
    }
    for (; i < na; i++) { uint64_t s = a[i] + c; c = (s < a[i]); dst[i] = s; }
    for (; i < nb; i++) { uint64_t s = b[i] + c; c = (s < b[i]); dst[i] = s; }
    return c;
}

static void limbs_sub_inplace(uint64_t *dst, size_t nd, const uint64_t *sub, size_t ns) {
    uint64_t b = 0;
    size_t i = 0;
    for (; i < ns; i++) {
        uint64_t d1 = dst[i] - sub[i];
        uint64_t b1 = (d1 > dst[i]);
        uint64_t d2 = d1 - b;
        b = b1 + (d2 > d1);
        dst[i] = d2;
    }
    for (; b && i < nd; i++) {
        uint64_t d = dst[i] - b;
        b = (d > dst[i]);
        dst[i] = d;
    }
}

static void karatsuba_rec(uint64_t *out, const uint64_t *a, size_t na,
                          const uint64_t *b, size_t nb, uint64_t *scratch) {
    while (na > 0 && a[na - 1] == 0) na--;
    while (nb > 0 && b[nb - 1] == 0) nb--;
    if (na == 0 || nb == 0) return;
    if (na < nb) { const uint64_t *t = a; a = b; b = t; size_t tn = na; na = nb; nb = tn; }
    if (nb <= 36) {
        schoolbook_mul_u64(out, a, na, b, nb);
        return;
    }
    size_t half = na / 2;
    if (nb <= half) {
        /* Unbalanced split: A = A1 * 2^(64*half) + A0 */
        uint64_t *t0 = scratch;
        uint64_t *next_s = scratch + (na + nb + 4);
        memset(out, 0, (na + nb) * sizeof(uint64_t));
        memset(t0, 0, (half + nb) * sizeof(uint64_t));
        karatsuba_rec(t0, a, half, b, nb, next_s);
        memcpy(out, t0, (half + nb) * sizeof(uint64_t));

        size_t a1_n = na - half;
        memset(t0, 0, (a1_n + nb) * sizeof(uint64_t));
        karatsuba_rec(t0, a + half, a1_n, b, nb, next_s);
        uint64_t c = limbs_add(out + half, out + half, nb, t0, a1_n + nb);
        if (c && (half + a1_n + nb) < (na + nb)) out[half + a1_n + nb] += c;
        return;
    }

    size_t a1_n = na - half, b1_n = nb - half;
    size_t s_len = (a1_n > half ? a1_n : half) + 1;
    size_t t_len = (b1_n > half ? b1_n : half) + 1;
    uint64_t *sa = scratch;
    uint64_t *sb = sa + s_len;
    uint64_t *p2 = sb + t_len;
    uint64_t *next_s = p2 + (s_len + t_len + 2);

    memset(out, 0, (na + nb) * sizeof(uint64_t));
    /* p0 = A0 * B0 in out[0 .. 2*half - 1] */
    karatsuba_rec(out, a, half, b, half, next_s);
    /* p1 = A1 * B1 in out[2*half .. na + nb - 1] */
    karatsuba_rec(out + 2 * half, a + half, a1_n, b + half, b1_n, next_s);

    /* sa = A0 + A1, sb = B0 + B1 */
    sa[s_len - 1] = limbs_add(sa, a, half, a + half, a1_n);
    size_t sa_n = (sa[s_len - 1] != 0) ? s_len : (s_len - 1);
    sb[t_len - 1] = limbs_add(sb, b, half, b + half, b1_n);
    size_t sb_n = (sb[t_len - 1] != 0) ? t_len : (t_len - 1);

    size_t p2_n = sa_n + sb_n;
    memset(p2, 0, p2_n * sizeof(uint64_t));
    karatsuba_rec(p2, sa, sa_n, sb, sb_n, next_s);
    limbs_sub_inplace(p2, p2_n, out, 2 * half);
    limbs_sub_inplace(p2, p2_n, out + 2 * half, a1_n + b1_n);

    size_t rem = (na + nb) - half;
    if (p2_n > rem) p2_n = rem;
    uint64_t c = limbs_add(out + half, out + half, rem, p2, p2_n);
    (void)c;
}

/* =========================================================================================
 * TIER 2: VEDIC SIGNED POWER-OF-2 (CANONICAL NAF) SPARSE POLYNOMIAL SOLVER
 *   Collapses arbitrary-length runs of 1s (`111...1` -> +2^hi - 2^lo) and skips runs of 0s
 *   in O(1) per 64-bit word. Aborts in ~50 nanoseconds if input is dense random bits.
 * ========================================================================================= */
#define MAX_SPARSE_TERMS 256

typedef struct {
    size_t bit_exp;
    int    sign; /* +1 or -1 */
} xc_naf_term_t;

static int extract_sparse_naf(const uint64_t *a, size_t na, xc_naf_term_t *terms, int max_terms) {
    int count = 0;
    uint64_t carry = 0;
    size_t checked_words = 0;

    for (size_t w = 0; w < na; w++) {
        uint64_t word = a[w];
        /* Fast 64-bit skip: all-zeros with carry=0 OR all-ones with carry=1! */
        if (word == 0ULL && carry == 0ULL) continue;
        if (word == ~0ULL && carry == 1ULL) continue;

        checked_words++;
        /* Early-exit in ~50 ns on dense random numbers (> 48 terms in first 8 active words) */
        if (checked_words == 8 && count > 48) return -1;

        int b = 0;
        while (b < 64) {
            uint64_t v = ((word >> b) & 3ULL) + carry;
            if ((v & 1ULL) == 0ULL) {
                carry = v >> 1;
                b++;
            } else {
                if (count >= max_terms) return -1;
                int s = (v == 3ULL) ? -1 : +1;
                terms[count].bit_exp = w * 64ULL + (size_t)b;
                terms[count].sign    = s;
                count++;
                /* Advance carry across current + next bit (NAF guarantees next bit is 0!) */
                uint64_t two_bits = (b < 63) ? ((word >> b) & 3ULL)
                                             : (((word >> 63) & 1ULL) | (((w + 1 < na ? a[w + 1] : 0ULL) & 1ULL) << 1));
                uint64_t sum2 = two_bits + carry;
                carry = (s < 0) ? 1ULL : (sum2 >> 2);
                b += 2;
                if (b == 65) {
                    /* Bit 64 (bit 0 of next word) was consumed as zero; handle via carry into bit 1?
                       Simpler & exact: process 1 bit at a time inside active word (<= 64 iters)! */
                }
            }
        }
    }
    if (carry) {
        if (count >= max_terms) return -1;
        terms[count].bit_exp = na * 64ULL;
        terms[count].sign    = +1;
        count++;
    }
    return count;
}

/* Exact 1-bit-step NAF inside active words (skips 64-bit 0x00..00 and 0xFF..FF runs in 1 cycle!) */
static int extract_sparse_naf_exact(const uint64_t *a, size_t na, xc_naf_term_t *terms, int max_terms) {
    int count = 0;
    uint64_t c = 0;
    size_t active_words = 0;

    for (size_t w = 0; w < na; w++) {
        uint64_t x = a[w];
        if (x == 0ULL && c == 0ULL) continue;
        if (x == ~0ULL && c == 1ULL) continue;

        active_words++;
        if (active_words == 8 && count > 48) return -1;

        uint64_t next_lo = (w + 1 < na) ? (a[w + 1] & 1ULL) : 0ULL;
        /* Reitwiesner exact formula: xh = x + c; c_next = (x + c) >> 64;
           Actually bit-by-bit over 64 bits with lookahead bit i+1 is 8 lines and 100% exact! */
        for (int b = 0; b < 64; b++) {
            uint64_t bi  = (x >> b) & 1ULL;
            uint64_t bi1 = (b < 63) ? ((x >> (b + 1)) & 1ULL) : next_lo;
            uint64_t sum = bi + c; /* in {0, 1, 2} */
            if ((sum & 1ULL) == 0ULL) {
                c = sum >> 1;
            } else {
                /* sum == 1: look at bi1 + (sum >> 1) = bi1. If bi1 == 1, emit -1 and set c = 1! */
                if (count >= max_terms) return -1;
                if (bi1 == 1ULL) {
                    terms[count].bit_exp = w * 64ULL + (size_t)b;
                    terms[count].sign    = -1;
                    c = 1ULL;
                } else {
                    terms[count].bit_exp = w * 64ULL + (size_t)b;
                    terms[count].sign    = +1;
                    c = 0ULL;
                }
                count++;
            }
        }
    }
    if (c) {
        if (count >= max_terms) return -1;
        terms[count].bit_exp = na * 64ULL;
        terms[count].sign    = +1;
        count++;
    }
    return count;
}

typedef struct {
    size_t w_idx;
    int    bit_off;
    int    sign;
} xc_prod_event_t;

static int cmp_prod_event(const void *p1, const void *p2) {
    const xc_prod_event_t *a = (const xc_prod_event_t *)p1;
    const xc_prod_event_t *b = (const xc_prod_event_t *)p2;
    if (a->w_idx != b->w_idx) return (a->w_idx < b->w_idx) ? -1 : 1;
    return a->bit_off - b->bit_off;
}

/* One-pass O(M log M + N_words_memset) exact sparse polynomial solver! */
static int try_vedic_sparse_mul(uint64_t *out, const uint64_t *a, size_t na,
                                const uint64_t *b, size_t nb) {
    (void)extract_sparse_naf;
    xc_naf_term_t ta[MAX_SPARSE_TERMS], tb[MAX_SPARSE_TERMS];
    int ka = extract_sparse_naf_exact(a, na, ta, MAX_SPARSE_TERMS);
    if (ka < 0) return 0;
    int kb = extract_sparse_naf_exact(b, nb, tb, MAX_SPARSE_TERMS);
    if (kb < 0 || (long)ka * (long)kb > 16384L) return 0;

    size_t out_n = na + nb;
    int m_tot = ka * kb;
    xc_prod_event_t *ev = (xc_prod_event_t *)malloc((size_t)(m_tot + 1) * sizeof(xc_prod_event_t));
    if (!ev) return 0;

    int m = 0;
    for (int i = 0; i < ka; i++) {
        for (int j = 0; j < kb; j++) {
            size_t e = ta[i].bit_exp + tb[j].bit_exp;
            if (e >> 6 < out_n) {
                ev[m].w_idx   = e >> 6;
                ev[m].bit_off = (int)(e & 63ULL);
                ev[m].sign    = ta[i].sign * tb[j].sign;
                m++;
            }
        }
    }
    qsort(ev, (size_t)m, sizeof(xc_prod_event_t), cmp_prod_event);

    /* Single forward sweep across [0 .. out_n - 1] with portable 64-bit signed carry:
       Untouched word runs between events are filled in ONE memset(0x00 or 0xFF)! */
    size_t cur_w = 0;
    int64_t carry = 0;
    int idx = 0;
    while (idx < m) {
        size_t w = ev[idx].w_idx;
        if (w > cur_w) {
            if (carry == 0) {
                memset(out + cur_w, 0x00, (w - cur_w) * sizeof(uint64_t));
                cur_w = w;
            } else if (carry == -1) {
                memset(out + cur_w, 0xFF, (w - cur_w) * sizeof(uint64_t));
                cur_w = w;
            } else {
                while (cur_w < w && carry != 0 && carry != -1) {
                    out[cur_w++] = (uint64_t)carry;
                    carry = (carry < 0) ? -1 : 0;
                }
                if (cur_w < w) {
                    memset(out + cur_w, (carry == -1) ? 0xFF : 0x00, (w - cur_w) * sizeof(uint64_t));
                    cur_w = w;
                }
            }
        }
        uint64_t lo = (uint64_t)carry;
        int64_t  hi = (carry < 0) ? -1 : 0;
        while (idx < m && ev[idx].w_idx == w) {
            uint64_t bit = 1ULL << ev[idx].bit_off;
            if (ev[idx].sign > 0) {
                uint64_t s = lo + bit;
                if (s < lo) hi++;
                lo = s;
            } else {
                uint64_t d = lo - bit;
                if (d > lo) hi--;
                lo = d;
            }
            idx++;
        }
        out[w] = lo;
        carry  = hi;
        cur_w  = w + 1;
    }
    while (cur_w < out_n && carry != 0) {
        out[cur_w++] = (uint64_t)carry;
        carry = (carry < 0) ? -1 : 0;
    }
    if (cur_w < out_n) {
        memset(out + cur_w, 0x00, (out_n - cur_w) * sizeof(uint64_t));
    }
    free(ev);
    g_last_path = XC_PATH_VEDIC_SPARSE;
    return 1;
}

/* =========================================================================================
 * TIER 3: CRANDALL-FAGIN RIGHT-ANGLE (Y^M + i = 0) 16-BIT VEDIC VINCULUM WAVE ENGINE
 *   Unified AoSoA8 [re[0..7], im[0..7]] memory layout across AVX-512, AVX2+FMA, and Scalar!
 * ========================================================================================= */
struct xc_plan_s {
    int      log_c;
    size_t   C_SUB;
    size_t   M;
    xc_isa_t isa;
    int      bitrev64[64];

    double gam640_re[640], gam640_im[640];
    double g2560_re[2560], g2560_im[2560];
    double w640_re[320],   w640_im[320];
    double w320_re[5][64], w320_im[5][64];
    double w64_re[64],     w64_im[64];

    double  fine_re[4096], fine_im[4096];
    double *coarse_re, *coarse_im;
    double *sub_re,    *sub_im;

    double  *Az;
    double  *Bz;
    int64_t *tile_carry;
};

struct xc_trunk_s {
    int     log_c;
    size_t  C_SUB;
    size_t  M;
    size_t  nb;
    int     is_zero;
    double *Bz_trunk;
};

static int choose_log_c(size_t na, size_t nb) {
    size_t max_n = (na > nb) ? na : nb;
    size_t req_m = max_n * 4ULL + 8ULL; /* uint16_t count + Vedic sign carry */
    int log_c = 4; /* min C_SUB = 16 => M = 10,240 */
    while (((size_t)R_TOP << log_c) < req_m) log_c++;
    return log_c;
}

static int plan_init_internal(xc_plan_t *E, int log_c, int alloc_bz) {
    memset(E, 0, sizeof(*E));
    E->log_c = log_c;
    E->C_SUB = (size_t)1ULL << log_c;
    E->M     = (size_t)R_TOP * E->C_SUB;
    E->isa   = xc_detect_isa();

    for (int i = 0; i < 64; i++) {
        int r = 0;
        for (int b = 0; b < 6; b++) r = (r << 1) | ((i >> b) & 1);
        E->bitrev64[i] = r;
    }
    for (int j = 0; j < 640; j++) {
        double a = -M_PI * (double)j / 1280.0;
        E->gam640_re[j] = cos(a); E->gam640_im[j] = sin(a);
    }
    for (int m = 0; m < 2560; m++) {
        double a = -M_PI * (double)m / 1280.0;
        E->g2560_re[m] = cos(a); E->g2560_im[m] = sin(a);
    }
    for (int p = 0; p < 320; p++) {
        double a = -2.0 * M_PI * (double)p / 640.0;
        E->w640_re[p] = cos(a); E->w640_im[p] = sin(a);
    }
    for (int r1 = 0; r1 < 5; r1++) {
        for (int j0 = 0; j0 < 64; j0++) {
            double a = -2.0 * M_PI * (double)(j0 * r1) / 320.0;
            E->w320_re[r1][j0] = cos(a); E->w320_im[r1][j0] = sin(a);
        }
    }
    for (int k = 0; k < 32; k++) {
        double a = -2.0 * M_PI * (double)k / 64.0;
        E->w64_re[k] = cos(a); E->w64_im[k] = sin(a);
    }
    for (int f = 0; f < 4096; f++) {
        double a = -M_PI * (double)f / (double)(2 * E->M);
        E->fine_re[f] = cos(a); E->fine_im[f] = sin(a);
    }

    size_t n_coarse = (4 * E->M + 4095) / 4096;
    E->coarse_re = (double *)xc_aligned_alloc_buf(n_coarse * sizeof(double), 0);
    E->coarse_im = (double *)xc_aligned_alloc_buf(n_coarse * sizeof(double), 0);
    size_t half_sub = E->C_SUB / 2;
    E->sub_re    = (double *)xc_aligned_alloc_buf(half_sub * sizeof(double), 0);
    E->sub_im    = (double *)xc_aligned_alloc_buf(half_sub * sizeof(double), 0);
    int use_huge = (2 * E->M * sizeof(double) >= (2ULL << 20));
    E->Az        = (double *)xc_aligned_alloc_buf(2 * E->M * sizeof(double), use_huge);
    E->Bz        = alloc_bz ? (double *)xc_aligned_alloc_buf(2 * E->M * sizeof(double), use_huge) : NULL;
    E->tile_carry= (int64_t *)xc_aligned_alloc_buf(2 * R_TOP * (E->C_SUB / 8) * sizeof(int64_t), 0);

    if (!E->coarse_re || !E->coarse_im || !E->sub_re || !E->sub_im ||
        !E->Az || (alloc_bz && !E->Bz) || !E->tile_carry) {
        return XC_ERR_NOMEM;
    }

    for (size_t c = 0; c < n_coarse; c++) {
        double a = -M_PI * (double)(c * 4096ULL) / (double)(2 * E->M);
        E->coarse_re[c] = cos(a); E->coarse_im[c] = sin(a);
    }
    for (size_t k = 0; k < half_sub; k++) {
        double a = -2.0 * M_PI * (double)k / (double)E->C_SUB;
        E->sub_re[k] = cos(a); E->sub_im[k] = sin(a);
    }

    int nt = xc_wave_threads(E->M);
    int n_pages = (int)((2 * E->M + 511) / 512);
    int pg;
    #pragma omp parallel for num_threads(nt) schedule(static)
    for (pg = 0; pg < n_pages; pg++) {
        size_t i = (size_t)pg * 512;
        E->Az[i] = 0.0;
        if (E->Bz) E->Bz[i] = 0.0;
    }
    return XC_OK;
}

static void plan_free_internal(xc_plan_t *E) {
    if (!E) return;
    xc_aligned_free_buf(E->coarse_re);
    xc_aligned_free_buf(E->coarse_im);
    xc_aligned_free_buf(E->sub_re);
    xc_aligned_free_buf(E->sub_im);
    xc_aligned_free_buf(E->Az);
    xc_aligned_free_buf(E->Bz);
    xc_aligned_free_buf(E->tile_carry);
}

/* -----------------------------------------------------------------------------------------
 * PORTABLE SCALAR / AUTO-VECTORIZED KERNELS (Fallback for ARM64 / non-AVX2 x86)
 *   Uses exact same AoSoA8 [re[8], im[8]] layout so all 3 ISAs are 100% interoperable!
 * ----------------------------------------------------------------------------------------- */
typedef struct { double v[8]; } vec8d;

static inline void sc_load_vedic8(const uint16_t *A_u16, size_t n_u16, size_t u_pos, vec8d *out) {
    for (int m = 0; m < 8; m++) {
        size_t idx = u_pos + (size_t)m;
        int32_t u  = (idx < n_u16) ? (int32_t)A_u16[idx] : 0;
        int32_t s  = u >> 15;
        int32_t sp = (idx > 0 && idx - 1 < n_u16) ? (int32_t)(A_u16[idx - 1] >> 15) : 0;
        out->v[m]  = (double)(u - (s << 16) + sp);
    }
}

static inline void sc_lut_fourstep(const xc_plan_t *E, size_t b0, int r, vec8d *tr, vec8d *ti) {
    const size_t fourM = 4 * E->M;
    const size_t stride = (size_t)(4 * r + 1);
    size_t idx = (b0 * stride) % fourM;
    for (int m = 0; m < 8; m++) {
        size_t f = idx & 4095ULL, c = idx >> 12;
        double cr = E->coarse_re[c], ci = E->coarse_im[c];
        double fr = E->fine_re[f],   fi = E->fine_im[f];
        tr->v[m] = cr * fr - ci * fi;
        ti->v[m] = cr * fi + ci * fr;
        idx += stride;
        if (idx >= fourM) idx -= fourM;
    }
}

static void sc_dft64_fwd(vec8d *re, vec8d *im, const xc_plan_t *E) {
    for (int len = 32, step = 1; len >= 1; len >>= 1, step <<= 1) {
        for (int base = 0; base < 64; base += 2 * len) {
            for (int j = 0; j < len; j++) {
                int i0 = base + j, i1 = i0 + len;
                double wr = E->w64_re[j * step], wi = E->w64_im[j * step];
                for (int m = 0; m < 8; m++) {
                    double ur = re[i0].v[m], ui = im[i0].v[m];
                    double vr = re[i1].v[m], vi = im[i1].v[m];
                    double dr = ur - vr, di = ui - vi;
                    re[i0].v[m] = ur + vr; im[i0].v[m] = ui + vi;
                    if (j == 0) { re[i1].v[m] = dr; im[i1].v[m] = di; }
                    else { re[i1].v[m] = dr * wr - di * wi; im[i1].v[m] = dr * wi + di * wr; }
                }
            }
        }
    }
}

static void sc_dft64_inv(vec8d *re, vec8d *im, const xc_plan_t *E) {
    for (int len = 1, step = 32; len <= 32; len <<= 1, step >>= 1) {
        for (int base = 0; base < 64; base += 2 * len) {
            for (int j = 0; j < len; j++) {
                int i0 = base + j, i1 = i0 + len;
                double wr = E->w64_re[j * step], wi = E->w64_im[j * step];
                for (int m = 0; m < 8; m++) {
                    double ur = re[i0].v[m], ui = im[i0].v[m];
                    double vr = re[i1].v[m], vi = im[i1].v[m];
                    if (j > 0) {
                        double tvr = vr * wr + vi * wi;
                        vi = vi * wr - vr * wi;
                        vr = tvr;
                    }
                    re[i0].v[m] = ur + vr; im[i0].v[m] = ui + vi;
                    re[i1].v[m] = ur - vr; im[i1].v[m] = ui - vi;
                }
            }
        }
    }
}

static void sc_dft640_fwd_real(const vec8d *in_re, int n_nz, vec8d *out_re, vec8d *out_im, const xc_plan_t *E) {
    vec8d h_re[2][320], h_im[2][320];
    for (int p = 0; p < 320; p++) {
        double g0r = E->gam640_re[p], g0i = E->gam640_im[p];
        double g1r = E->gam640_re[p + 320], g1i = E->gam640_im[p + 320];
        double wr = E->w640_re[p], wi = E->w640_im[p];
        for (int m = 0; m < 8; m++) {
            double x0r = (p < n_nz) ? in_re[p].v[m] * g0r : 0.0;
            double x0i = (p < n_nz) ? in_re[p].v[m] * g0i : 0.0;
            double x1r = (p + 320 < n_nz) ? in_re[p + 320].v[m] * g1r : 0.0;
            double x1i = (p + 320 < n_nz) ? in_re[p + 320].v[m] * g1i : 0.0;
            h_re[0][p].v[m] = x0r + x1r; h_im[0][p].v[m] = x0i + x1i;
            double dr = x0r - x1r, di = x0i - x1i;
            h_re[1][p].v[m] = dr * wr - di * wi;
            h_im[1][p].v[m] = dr * wi + di * wr;
        }
    }
    for (int r2 = 0; r2 < 2; r2++) {
        vec8d b_re[5][64], b_im[5][64];
        for (int j0 = 0; j0 < 64; j0++) {
            for (int m = 0; m < 8; m++) {
                double x0r = h_re[r2][j0].v[m],     x0i = h_im[r2][j0].v[m];
                double x1r = h_re[r2][j0+64].v[m],  x1i = h_im[r2][j0+64].v[m];
                double x2r = h_re[r2][j0+128].v[m], x2i = h_im[r2][j0+128].v[m];
                double x3r = h_re[r2][j0+192].v[m], x3i = h_im[r2][j0+192].v[m];
                double x4r = h_re[r2][j0+256].v[m], x4i = h_im[r2][j0+256].v[m];
                double s1r = x1r + x4r, s1i = x1i + x4i, d1r = x1r - x4r, d1i = x1i - x4i;
                double s2r = x2r + x3r, s2i = x2i + x3i, d2r = x2r - x3r, d2i = x2i - x3i;
                b_re[0][j0].v[m] = x0r + s1r + s2r; b_im[0][j0].v[m] = x0i + s1i + s2i;
                double m1r = x0r + C5_1*s1r + C5_2*s2r, m1i = x0i + C5_1*s1i + C5_2*s2i;
                double m2r = x0r + C5_2*s1r + C5_1*s2r, m2i = x0i + C5_2*s1i + C5_1*s2i;
                double q1r = S5_1*d1r + S5_2*d2r, q1i = S5_1*d1i + S5_2*d2i;
                double q2r = S5_2*d1r - S5_1*d2r, q2i = S5_2*d1i - S5_1*d2i;
                double y[5][2] = {{0,0}, {m1r-q1i, m1i+q1r}, {m2r-q2i, m2i+q2r}, {m2r+q2i, m2i-q2r}, {m1r+q1i, m1i-q1r}};
                for (int r1 = 1; r1 < 5; r1++) {
                    double wr = E->w320_re[r1][j0], wi = E->w320_im[r1][j0];
                    b_re[r1][j0].v[m] = y[r1][0]*wr - y[r1][1]*wi;
                    b_im[r1][j0].v[m] = y[r1][0]*wi + y[r1][1]*wr;
                }
            }
        }
        for (int r1 = 0; r1 < 5; r1++) {
            sc_dft64_fwd(b_re[r1], b_im[r1], E);
            for (int r0 = 0; r0 < 64; r0++) {
                int br = E->bitrev64[r0], r = r2 + 2 * r1 + 10 * r0;
                out_re[r] = b_re[r1][br]; out_im[r] = b_im[r1][br];
            }
        }
    }
}

static void sc_dft640_inv(const vec8d *in_re, const vec8d *in_im, vec8d *out_re, vec8d *out_im, const xc_plan_t *E) {
    vec8d h_re[2][320], h_im[2][320];
    const double vs1 = -S5_1, vs2 = -S5_2;
    for (int r2 = 0; r2 < 2; r2++) {
        vec8d b_re[5][64], b_im[5][64];
        for (int r1 = 0; r1 < 5; r1++) {
            for (int r0 = 0; r0 < 64; r0++) {
                int br = E->bitrev64[r0], r = r2 + 2 * r1 + 10 * r0;
                b_re[r1][br] = in_re[r]; b_im[r1][br] = in_im[r];
            }
            sc_dft64_inv(b_re[r1], b_im[r1], E);
        }
        for (int j0 = 0; j0 < 64; j0++) {
            for (int m = 0; m < 8; m++) {
                double xr[5], xi[5];
                xr[0] = b_re[0][j0].v[m]; xi[0] = b_im[0][j0].v[m];
                for (int r1 = 1; r1 < 5; r1++) {
                    double wr = E->w320_re[r1][j0], wi = E->w320_im[r1][j0];
                    double vr = b_re[r1][j0].v[m],  vi = b_im[r1][j0].v[m];
                    xr[r1] = vr*wr + vi*wi; xi[r1] = vi*wr - vr*wi;
                }
                double s1r = xr[1]+xr[4], s1i = xi[1]+xi[4], d1r = xr[1]-xr[4], d1i = xi[1]-xi[4];
                double s2r = xr[2]+xr[3], s2i = xi[2]+xi[3], d2r = xr[2]-xr[3], d2i = xi[2]-xi[3];
                h_re[r2][j0].v[m] = xr[0] + s1r + s2r; h_im[r2][j0].v[m] = xi[0] + s1i + s2i;
                double m1r = xr[0] + C5_1*s1r + C5_2*s2r, m1i = xi[0] + C5_1*s1i + C5_2*s2i;
                double m2r = xr[0] + C5_2*s1r + C5_1*s2r, m2i = xi[0] + C5_2*s1i + C5_1*s2i;
                double q1r = vs1*d1r + vs2*d2r, q1i = vs1*d1i + vs2*d2i;
                double q2r = vs2*d1r - vs1*d2r, q2i = vs2*d1i - vs1*d2i;
                h_re[r2][j0+64].v[m]  = m1r - q1i; h_im[r2][j0+64].v[m]  = m1i + q1r;
                h_re[r2][j0+256].v[m] = m1r + q1i; h_im[r2][j0+256].v[m] = m1i - q1r;
                h_re[r2][j0+128].v[m] = m2r - q2i; h_im[r2][j0+128].v[m] = m2i + q2r;
                h_re[r2][j0+192].v[m] = m2r + q2i; h_im[r2][j0+192].v[m] = m2i - q2r;
            }
        }
    }
    for (int p = 0; p < 320; p++) {
        double wr = E->w640_re[p], wi = E->w640_im[p];
        double g0r = E->gam640_re[p], g0i = E->gam640_im[p];
        double g1r = E->gam640_re[p+320], g1i = E->gam640_im[p+320];
        for (int m = 0; m < 8; m++) {
            double h1r = h_re[1][p].v[m], h1i = h_im[1][p].v[m];
            double t1r = h1r*wr + h1i*wi, t1i = h1i*wr - h1r*wi;
            double a0r = h_re[0][p].v[m] + t1r, a0i = h_im[0][p].v[m] + t1i;
            double a1r = h_re[0][p].v[m] - t1r, a1i = h_im[0][p].v[m] - t1i;
            out_re[p].v[m]     = a0r*g0r + a0i*g0i; out_im[p].v[m]     = a0i*g0r - a0r*g0i;
            out_re[p+320].v[m] = a1r*g1r + a1i*g1i; out_im[p+320].v[m] = a1i*g1r - a1r*g1i;
        }
    }
}

static void sc_sub_dif(double *row, size_t C, const double *sub_re, const double *sub_im) {
    for (size_t half = C >> 1, step = 1; half >= 8; half >>= 1, step <<= 1) {
        for (size_t base = 0; base < C; base += 2 * half) {
            for (size_t j = 0; j < half; j += 8) {
                double *p0 = row + (base + j) * 2, *p1 = row + (base + j + half) * 2;
                for (int m = 0; m < 8; m++) {
                    double ur = p0[m], ui = p0[8 + m], vr = p1[m], vi = p1[8 + m];
                    double dr = ur - vr, di = ui - vi;
                    p0[m] = ur + vr; p0[8 + m] = ui + vi;
                    size_t tk = (j + (size_t)m) * step;
                    double wr = sub_re[tk], wi = sub_im[tk];
                    p1[m] = dr*wr - di*wi; p1[8 + m] = dr*wi + di*wr;
                }
            }
        }
    }
    const size_t s4 = C >> 3, s2 = C >> 2;
    for (size_t base = 0; base < C; base += 8) {
        double *p = row + base * 2;
        double r[8], im[8];
        memcpy(r, p, 64); memcpy(im, p + 8, 64);
        for (int j = 0; j < 4; j++) {
            double ur = r[j], ui = im[j], vr = r[j+4], vi = im[j+4];
            r[j] = ur + vr; im[j] = ui + vi;
            double dr = ur - vr, di = ui - vi, wr = sub_re[(size_t)j*s4], wi = sub_im[(size_t)j*s4];
            r[j+4] = dr*wr - di*wi; im[j+4] = dr*wi + di*wr;
        }
        for (int b = 0; b < 8; b += 4) {
            for (int j = 0; j < 2; j++) {
                double ur = r[b+j], ui = im[b+j], vr = r[b+j+2], vi = im[b+j+2];
                r[b+j] = ur + vr; im[b+j] = ui + vi;
                double dr = ur - vr, di = ui - vi, wr = sub_re[(size_t)j*s2], wi = sub_im[(size_t)j*s2];
                r[b+j+2] = dr*wr - di*wi; im[b+j+2] = dr*wi + di*wr;
            }
        }
        for (int b = 0; b < 8; b += 2) {
            double ur = r[b], ui = im[b], vr = r[b+1], vi = im[b+1];
            r[b] = ur + vr; im[b] = ui + vi; r[b+1] = ur - vr; im[b+1] = ui - vi;
        }
        memcpy(p, r, 64); memcpy(p + 8, im, 64);
    }
}

static void sc_sub_idit(double *row, size_t C, const double *sub_re, const double *sub_im) {
    const size_t s4 = C >> 3, s2 = C >> 2;
    for (size_t base = 0; base < C; base += 8) {
        double *p = row + base * 2;
        double r[8], im[8];
        memcpy(r, p, 64); memcpy(im, p + 8, 64);
        for (int b = 0; b < 8; b += 2) {
            double ur = r[b], ui = im[b], vr = r[b+1], vi = im[b+1];
            r[b] = ur + vr; im[b] = ui + vi; r[b+1] = ur - vr; im[b+1] = ui - vi;
        }
        for (int b = 0; b < 8; b += 4) {
            for (int j = 0; j < 2; j++) {
                double ur = r[b+j], ui = im[b+j], vr = r[b+j+2], vi = im[b+j+2];
                double wr = sub_re[(size_t)j*s2], wi = sub_im[(size_t)j*s2];
                double tvr = vr*wr + vi*wi, tvi = vi*wr - vr*wi;
                r[b+j] = ur + tvr; im[b+j] = ui + tvi; r[b+j+2] = ur - tvr; im[b+j+2] = ui - tvi;
            }
        }
        for (int j = 0; j < 4; j++) {
            double ur = r[j], ui = im[j], vr = r[j+4], vi = im[j+4];
            double wr = sub_re[(size_t)j*s4], wi = sub_im[(size_t)j*s4];
            double tvr = vr*wr + vi*wi, tvi = vi*wr - vr*wi;
            r[j] = ur + tvr; im[j] = ui + tvi; r[j+4] = ur - tvr; im[j+4] = ui - tvi;
        }
        memcpy(p, r, 64); memcpy(p + 8, im, 64);
    }
    for (size_t half = 8, step = (C >> 4); half < C; half <<= 1, step >>= 1) {
        for (size_t base = 0; base < C; base += 2 * half) {
            for (size_t j = 0; j < half; j += 8) {
                double *p0 = row + (base + j) * 2, *p1 = row + (base + j + half) * 2;
                for (int m = 0; m < 8; m++) {
                    double ur = p0[m], ui = p0[8 + m], vr = p1[m], vi = p1[8 + m];
                    size_t tk = (j + (size_t)m) * step;
                    double wr = sub_re[tk], wi = sub_im[tk];
                    double tvr = vr*wr + vi*wi, tvi = vi*wr - vr*wi;
                    p0[m] = ur + tvr; p0[8 + m] = ui + tvi;
                    p1[m] = ur - tvr; p1[8 + m] = ui - tvi;
                }
            }
        }
    }
}

static void pass1_single_scalar(const xc_plan_t *E, const uint16_t *A_u16, size_t n_u16, double *DstZ) {
    const size_t C = E->C_SUB, n_vec = C / 8;
    const int n_nz = (int)((n_u16 + 1 + C - 1) / C);
    int nt = xc_wave_threads(E->M);

    #pragma omp parallel num_threads(nt)
    {
        vec8d in_re[640], out_re[640], out_im[640];
        int v, n_vec_i = (int)n_vec;
        #pragma omp for schedule(static)
        for (v = 0; v < n_vec_i; v++) {
            size_t b0 = (size_t)v * 8;
            for (int j = 0; j < n_nz; j++) sc_load_vedic8(A_u16, n_u16, (size_t)j * C + b0, &in_re[j]);
            sc_dft640_fwd_real(in_re, n_nz, out_re, out_im, E);
            for (int r = 0; r < 640; r++) {
                vec8d tr, ti;
                sc_lut_fourstep(E, b0, r, &tr, &ti);
                double *dst = DstZ + ((size_t)r * C + b0) * 2;
                for (int m = 0; m < 8; m++) {
                    double ar = out_re[r].v[m], ai = out_im[r].v[m];
                    dst[m]     = ar * tr.v[m] - ai * ti.v[m];
                    dst[8 + m] = ar * ti.v[m] + ai * tr.v[m];
                }
            }
        }
    }
}

static void pass2_convolve_scalar(const xc_plan_t *E, const double *Bz_src, int b_is_trunk) {
    const size_t C = E->C_SUB;
    const double scale = b_is_trunk ? 1.0 : (1.0 / (double)E->M);
    int nt = xc_wave_threads(E->M);
    int r;

    #pragma omp parallel for num_threads(nt) schedule(static)
    for (r = 0; r < 640; r++) {
        double *rowA = E->Az + (size_t)r * C * 2;
        double *rowB = (double *)(uintptr_t)(Bz_src + (size_t)r * C * 2);
        sc_sub_dif(rowA, C, E->sub_re, E->sub_im);
        if (!b_is_trunk) sc_sub_dif(rowB, C, E->sub_re, E->sub_im);
        for (size_t j = 0; j < C; j += 8) {
            double *pa = rowA + 2 * j;
            const double *pb = Bz_src + ((size_t)r * C + j) * 2;
            for (int m = 0; m < 8; m++) {
                double ar = pa[m], ai = pa[8+m], br = pb[m], bi = pb[8+m];
                pa[m]   = (ar*br - ai*bi) * scale;
                pa[8+m] = (ar*bi + ai*br) * scale;
            }
        }
        sc_sub_idit(rowA, C, E->sub_re, E->sub_im);
    }
}

static void ripple_tile_carries(xc_plan_t *E, uint16_t *Out_u16, size_t out_n_u16) {
    const size_t total_tiles = (out_n_u16 + 7) / 8;
    int nt = xc_wave_threads(E->M);
    if (nt > 64) nt = 64;
    int64_t thread_carry_out[64] = {0};
    size_t  chunk_end_pos[64]    = {0};

    #pragma omp parallel num_threads(nt)
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        size_t t_beg = (total_tiles * (size_t)tid) / (size_t)nt;
        size_t t_end = (total_tiles * (size_t)(tid + 1)) / (size_t)nt;
        chunk_end_pos[tid] = t_end * 8;

        int64_t run_c = 0;
        for (size_t t = t_beg; t < t_end; t++) {
            int64_t c_in = run_c;
            run_c = E->tile_carry[t];
            if (c_in != 0) {
                size_t pos = t * 8, lim = (t + 1) * 8;
                if (lim > out_n_u16) lim = out_n_u16;
                for (size_t p = pos; p < lim && c_in != 0; p++) {
                    int64_t v = (int64_t)Out_u16[p] + c_in;
                    Out_u16[p] = (uint16_t)(v & 0xFFFF);
                    c_in = v >> 16;
                }
                run_c += c_in;
            }
        }
        thread_carry_out[tid] = run_c;
    }

    for (int tid = 0; tid < nt; tid++) {
        int64_t c = thread_carry_out[tid];
        size_t  p = chunk_end_pos[tid];
        while (c != 0 && p < out_n_u16) {
            int64_t v = (int64_t)Out_u16[p] + c;
            Out_u16[p] = (uint16_t)(v & 0xFFFF);
            c = v >> 16;
            p++;
        }
    }
}

static void pass3_unpack_scalar(xc_plan_t *E, uint16_t *Out_u16, size_t out_n_u16) {
    const size_t C = E->C_SUB, M = E->M, n_vec = C / 8, half_tiles = (size_t)R_TOP * n_vec;
    int nt = xc_wave_threads(E->M);

    #pragma omp parallel num_threads(nt)
    {
        vec8d in_re[640], in_im[640], out_re[640], out_im[640];
        int v, n_vec_i = (int)n_vec;
        #pragma omp for schedule(static)
        for (v = 0; v < n_vec_i; v++) {
            size_t b0 = (size_t)v * 8;
            for (int r = 0; r < 640; r++) {
                const double *src = E->Az + ((size_t)r * C + b0) * 2;
                vec8d tr, ti;
                sc_lut_fourstep(E, b0, r, &tr, &ti);
                for (int m = 0; m < 8; m++) {
                    double yr = src[m], yi = src[8 + m];
                    in_re[r].v[m] = yr * tr.v[m] + yi * ti.v[m];
                    in_im[r].v[m] = yi * tr.v[m] - yr * ti.v[m];
                }
            }
            sc_dft640_inv(in_re, in_im, out_re, out_im, E);

            for (int j = 0; j < 640; j++) {
                size_t t_lo = (size_t)j * n_vec + (size_t)v, pos_lo = t_lo * 8;
                if (pos_lo < out_n_u16) {
                    int64_t acc = 0;
                    for (int m = 0; m < 8; m++) {
                        acc += llround(out_re[j].v[m]);
                        if (pos_lo + (size_t)m < out_n_u16) Out_u16[pos_lo + m] = (uint16_t)(acc & 0xFFFF);
                        acc >>= 16;
                    }
                    E->tile_carry[t_lo] = acc;
                } else E->tile_carry[t_lo] = 0;

                size_t t_hi = half_tiles + t_lo, pos_hi = M + pos_lo;
                if (pos_hi < out_n_u16) {
                    int64_t acc = 0;
                    for (int m = 0; m < 8; m++) {
                        acc += llround(-out_im[j].v[m]);
                        if (pos_hi + (size_t)m < out_n_u16) Out_u16[pos_hi + m] = (uint16_t)(acc & 0xFFFF);
                        acc >>= 16;
                    }
                    E->tile_carry[t_hi] = acc;
                } else E->tile_carry[t_hi] = 0;
            }
        }
    }
    ripple_tile_carries(E, Out_u16, out_n_u16);
}

#if XC_HAS_X86_SIMD
/* =========================================================================================
 * AVX2+FMA KERNELS (AoSoA8 Dual-YMM — Runs on every modern Intel/AMD laptop & desktop!)
 * ========================================================================================= */
#define CMUL256(ar, ai, br, bi, or_, oi_) do { \
    __m256d _tr = _mm256_fmsub_pd((ar), (br), _mm256_mul_pd((ai), (bi))); \
    (oi_) = _mm256_fmadd_pd((ar), (bi), _mm256_mul_pd((ai), (br))); \
    (or_) = _tr; \
} while (0)

#define CMUL256_CONJ(ar, ai, br, bi, or_, oi_) do { \
    __m256d _tr = _mm256_fmadd_pd((ar), (br), _mm256_mul_pd((ai), (bi))); \
    (oi_) = _mm256_fnmadd_pd((ar), (bi), _mm256_mul_pd((ai), (br))); \
    (or_) = _tr; \
} while (0)

XC_TARGET_AVX2 static void avx2_sub_dif(double *row, size_t C, const double *sub_re, const double *sub_im) {
    for (size_t half = C >> 1, step = 1; half >= 8; half >>= 1, step <<= 1) {
        for (size_t base = 0; base < C; base += 2 * half) {
            for (size_t j = 0; j < half; j += 8) {
                double *p0 = row + (base + j) * 2, *p1 = row + (base + j + half) * 2;
                for (int h = 0; h < 2; h++) {
                    size_t off = (size_t)h * 4;
                    __m256d ur = _mm256_load_pd(p0 + off),     ui = _mm256_load_pd(p0 + 8 + off);
                    __m256d vr = _mm256_load_pd(p1 + off),     vi = _mm256_load_pd(p1 + 8 + off);
                    __m256d dr = _mm256_sub_pd(ur, vr),        di = _mm256_sub_pd(ui, vi);
                    _mm256_store_pd(p0 + off,     _mm256_add_pd(ur, vr));
                    _mm256_store_pd(p0 + 8 + off, _mm256_add_pd(ui, vi));
                    size_t tk0 = (j + off) * step;
                    __m256d wr = _mm256_setr_pd(sub_re[tk0], sub_re[tk0+step], sub_re[tk0+2*step], sub_re[tk0+3*step]);
                    __m256d wi = _mm256_setr_pd(sub_im[tk0], sub_im[tk0+step], sub_im[tk0+2*step], sub_im[tk0+3*step]);
                    __m256d or_, oi_;
                    CMUL256(dr, di, wr, wi, or_, oi_);
                    _mm256_store_pd(p1 + off, or_); _mm256_store_pd(p1 + 8 + off, oi_);
                }
            }
        }
    }
    const size_t s4 = C >> 3, s2 = C >> 2;
    const __m256d w4r = _mm256_setr_pd(1.0, sub_re[s4], sub_re[2*s4], sub_re[3*s4]);
    const __m256d w4i = _mm256_setr_pd(0.0, sub_im[s4], sub_im[2*s4], sub_im[3*s4]);
    const __m256d w2r = _mm256_setr_pd(1.0, sub_re[s2], 1.0, sub_re[s2]);
    const __m256d w2i = _mm256_setr_pd(0.0, sub_im[s2], 0.0, sub_im[s2]);

    for (size_t base = 0; base < C; base += 8) {
        double *p = row + base * 2;
        __m256d r0 = _mm256_load_pd(p),     r1 = _mm256_load_pd(p + 4);
        __m256d i0 = _mm256_load_pd(p + 8), i1 = _mm256_load_pd(p + 12);
        /* half = 4: r0 +/- r1 */
        __m256d dr = _mm256_sub_pd(r0, r1), di = _mm256_sub_pd(i0, i1);
        r0 = _mm256_add_pd(r0, r1);         i0 = _mm256_add_pd(i0, i1);
        CMUL256(dr, di, w4r, w4i, r1, i1);

        /* half = 2 & 1 on each YMM */
        __m256d yr[2] = {r0, r1}, yi[2] = {i0, i1};
        for (int h = 0; h < 2; h++) {
            __m256d rr = yr[h], ii = yi[h];
            __m256d lo_r = _mm256_permute2f128_pd(rr, rr, 0x00), hi_r = _mm256_permute2f128_pd(rr, rr, 0x11);
            __m256d lo_i = _mm256_permute2f128_pd(ii, ii, 0x00), hi_i = _mm256_permute2f128_pd(ii, ii, 0x11);
            __m256d sgn2 = _mm256_setr_pd(1.0, 1.0, -1.0, -1.0);
            rr = _mm256_fmadd_pd(sgn2, hi_r, lo_r); ii = _mm256_fmadd_pd(sgn2, hi_i, lo_i);
            __m256d tr, ti; CMUL256(rr, ii, w2r, w2i, tr, ti);
            rr = _mm256_blend_pd(rr, tr, 0xC); ii = _mm256_blend_pd(ii, ti, 0xC);

            __m256d ev_r = _mm256_movedup_pd(rr), od_r = _mm256_unpackhi_pd(rr, rr);
            __m256d ev_i = _mm256_movedup_pd(ii), od_i = _mm256_unpackhi_pd(ii, ii);
            __m256d sgn1 = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
            yr[h] = _mm256_fmadd_pd(sgn1, od_r, ev_r);
            yi[h] = _mm256_fmadd_pd(sgn1, od_i, ev_i);
        }
        _mm256_store_pd(p, yr[0]);     _mm256_store_pd(p + 4, yr[1]);
        _mm256_store_pd(p + 8, yi[0]); _mm256_store_pd(p + 12, yi[1]);
    }
}

XC_TARGET_AVX2 static void avx2_sub_idit(double *row, size_t C, const double *sub_re, const double *sub_im) {
    const size_t s4 = C >> 3, s2 = C >> 2;
    const __m256d w4r = _mm256_setr_pd(1.0, sub_re[s4], sub_re[2*s4], sub_re[3*s4]);
    const __m256d w4i = _mm256_setr_pd(0.0, sub_im[s4], sub_im[2*s4], sub_im[3*s4]);
    const __m256d w2r = _mm256_setr_pd(1.0, sub_re[s2], 1.0, sub_re[s2]);
    const __m256d w2i = _mm256_setr_pd(0.0, sub_im[s2], 0.0, sub_im[s2]);

    for (size_t base = 0; base < C; base += 8) {
        double *p = row + base * 2;
        __m256d yr[2] = {_mm256_load_pd(p), _mm256_load_pd(p + 4)};
        __m256d yi[2] = {_mm256_load_pd(p + 8), _mm256_load_pd(p + 12)};
        for (int h = 0; h < 2; h++) {
            __m256d rr = yr[h], ii = yi[h];
            __m256d ev_r = _mm256_movedup_pd(rr), od_r = _mm256_unpackhi_pd(rr, rr);
            __m256d ev_i = _mm256_movedup_pd(ii), od_i = _mm256_unpackhi_pd(ii, ii);
            __m256d sgn1 = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
            rr = _mm256_fmadd_pd(sgn1, od_r, ev_r); ii = _mm256_fmadd_pd(sgn1, od_i, ev_i);

            __m256d tr, ti; CMUL256_CONJ(rr, ii, w2r, w2i, tr, ti);
            rr = _mm256_blend_pd(rr, tr, 0xC); ii = _mm256_blend_pd(ii, ti, 0xC);
            __m256d lo_r = _mm256_permute2f128_pd(rr, rr, 0x00), hi_r = _mm256_permute2f128_pd(rr, rr, 0x11);
            __m256d lo_i = _mm256_permute2f128_pd(ii, ii, 0x00), hi_i = _mm256_permute2f128_pd(ii, ii, 0x11);
            __m256d sgn2 = _mm256_setr_pd(1.0, 1.0, -1.0, -1.0);
            yr[h] = _mm256_fmadd_pd(sgn2, hi_r, lo_r);
            yi[h] = _mm256_fmadd_pd(sgn2, hi_i, lo_i);
        }
        __m256d vr, vi;
        CMUL256_CONJ(yr[1], yi[1], w4r, w4i, vr, vi);
        _mm256_store_pd(p,      _mm256_add_pd(yr[0], vr)); _mm256_store_pd(p + 4,  _mm256_sub_pd(yr[0], vr));
        _mm256_store_pd(p + 8,  _mm256_add_pd(yi[0], vi)); _mm256_store_pd(p + 12, _mm256_sub_pd(yi[0], vi));
    }
    for (size_t half = 8, step = (C >> 4); half < C; half <<= 1, step >>= 1) {
        for (size_t base = 0; base < C; base += 2 * half) {
            for (size_t j = 0; j < half; j += 8) {
                double *p0 = row + (base + j) * 2, *p1 = row + (base + j + half) * 2;
                for (int h = 0; h < 2; h++) {
                    size_t off = (size_t)h * 4;
                    __m256d ur = _mm256_load_pd(p0 + off), ui = _mm256_load_pd(p0 + 8 + off);
                    __m256d vr = _mm256_load_pd(p1 + off), vi = _mm256_load_pd(p1 + 8 + off);
                    size_t tk0 = (j + off) * step;
                    __m256d wr = _mm256_setr_pd(sub_re[tk0], sub_re[tk0+step], sub_re[tk0+2*step], sub_re[tk0+3*step]);
                    __m256d wi = _mm256_setr_pd(sub_im[tk0], sub_im[tk0+step], sub_im[tk0+2*step], sub_im[tk0+3*step]);
                    CMUL256_CONJ(vr, vi, wr, wi, vr, vi);
                    _mm256_store_pd(p0 + off,     _mm256_add_pd(ur, vr));
                    _mm256_store_pd(p0 + 8 + off, _mm256_add_pd(ui, vi));
                    _mm256_store_pd(p1 + off,     _mm256_sub_pd(ur, vr));
                    _mm256_store_pd(p1 + 8 + off, _mm256_sub_pd(ui, vi));
                }
            }
        }
    }
}

XC_TARGET_AVX2 static void pass2_convolve_avx2(const xc_plan_t *E, const double *Bz_src, int b_is_trunk) {
    const size_t C = E->C_SUB;
    const __m256d scale = _mm256_set1_pd(b_is_trunk ? 1.0 : (1.0 / (double)E->M));
    int nt = xc_wave_threads(E->M);
    int r;

    #pragma omp parallel for num_threads(nt) schedule(static)
    for (r = 0; r < 640; r++) {
        double *rowA = E->Az + (size_t)r * C * 2;
        double *rowB = (double *)(uintptr_t)(Bz_src + (size_t)r * C * 2);
        avx2_sub_dif(rowA, C, E->sub_re, E->sub_im);
        if (!b_is_trunk) avx2_sub_dif(rowB, C, E->sub_re, E->sub_im);
        for (size_t j = 0; j < C; j += 8) {
            for (int h = 0; h < 2; h++) {
                size_t off = 2 * j + (size_t)h * 4;
                __m256d ar = _mm256_load_pd(rowA + off),     ai = _mm256_load_pd(rowA + off + 8);
                __m256d br = _mm256_load_pd(rowB + off),     bi = _mm256_load_pd(rowB + off + 8);
                __m256d pr, pi;
                CMUL256(ar, ai, br, bi, pr, pi);
                if (!b_is_trunk) { pr = _mm256_mul_pd(pr, scale); pi = _mm256_mul_pd(pi, scale); }
                _mm256_store_pd(rowA + off, pr); _mm256_store_pd(rowA + off + 8, pi);
            }
        }
        avx2_sub_idit(rowA, C, E->sub_re, E->sub_im);
    }
}

/* =========================================================================================
 * AVX-512 KERNELS (AoSoA8 ZMM + Non-Temporal vmovntpd + Hardware Rounding)
 * ========================================================================================= */
#define CMUL512(ar, ai, br, bi, or_, oi_) do { \
    __m512d _tr = _mm512_fmsub_pd((ar), (br), _mm512_mul_pd((ai), (bi))); \
    (oi_) = _mm512_fmadd_pd((ar), (bi), _mm512_mul_pd((ai), (br))); \
    (or_) = _tr; \
} while (0)

#define CMUL512_CONJ(ar, ai, br, bi, or_, oi_) do { \
    __m512d _tr = _mm512_fmadd_pd((ar), (br), _mm512_mul_pd((ai), (bi))); \
    (oi_) = _mm512_fnmadd_pd((ar), (bi), _mm512_mul_pd((ai), (br))); \
    (or_) = _tr; \
} while (0)

XC_TARGET_AVX512 static inline void dft64_fwd_8x(__m512d *re, __m512d *im, const xc_plan_t *E) {
    for (int len = 32, step = 1; len >= 1; len >>= 1, step <<= 1) {
        for (int base = 0; base < 64; base += 2 * len) {
            for (int j = 0; j < len; j++) {
                int i0 = base + j, i1 = i0 + len;
                __m512d u_r = re[i0], u_i = im[i0], v_r = re[i1], v_i = im[i1];
                __m512d d_r = _mm512_sub_pd(u_r, v_r), d_i = _mm512_sub_pd(u_i, v_i);
                re[i0] = _mm512_add_pd(u_r, v_r); im[i0] = _mm512_add_pd(u_i, v_i);
                if (j == 0) { re[i1] = d_r; im[i1] = d_i; }
                else {
                    CMUL512(d_r, d_i, _mm512_set1_pd(E->w64_re[j*step]), _mm512_set1_pd(E->w64_im[j*step]), re[i1], im[i1]);
                }
            }
        }
    }
}

XC_TARGET_AVX512 static inline void dft64_inv_8x(__m512d *re, __m512d *im, const xc_plan_t *E) {
    for (int len = 1, step = 32; len <= 32; len <<= 1, step >>= 1) {
        for (int base = 0; base < 64; base += 2 * len) {
            for (int j = 0; j < len; j++) {
                int i0 = base + j, i1 = i0 + len;
                __m512d u_r = re[i0], u_i = im[i0], v_r = re[i1], v_i = im[i1];
                if (j > 0) {
                    CMUL512_CONJ(v_r, v_i, _mm512_set1_pd(E->w64_re[j*step]), _mm512_set1_pd(E->w64_im[j*step]), v_r, v_i);
                }
                re[i0] = _mm512_add_pd(u_r, v_r); im[i0] = _mm512_add_pd(u_i, v_i);
                re[i1] = _mm512_sub_pd(u_r, v_r); im[i1] = _mm512_sub_pd(u_i, v_i);
            }
        }
    }
}

XC_TARGET_AVX512 static void dft640_fwd_real_8x(const __m512d *in_re, int n_nz_slices,
                                                __m512d *out_re, __m512d *out_im, const xc_plan_t *E) {
    __m512d h_re[2][320], h_im[2][320];
    __m512d z0 = _mm512_setzero_pd();

    for (int p = 0; p < 320; p++) {
        __m512d x0r = z0, x0i = z0, x1r = z0, x1i = z0;
        if (p < n_nz_slices) {
            x0r = _mm512_mul_pd(in_re[p], _mm512_set1_pd(E->gam640_re[p]));
            x0i = _mm512_mul_pd(in_re[p], _mm512_set1_pd(E->gam640_im[p]));
        }
        if (p + 320 < n_nz_slices) {
            x1r = _mm512_mul_pd(in_re[p + 320], _mm512_set1_pd(E->gam640_re[p + 320]));
            x1i = _mm512_mul_pd(in_re[p + 320], _mm512_set1_pd(E->gam640_im[p + 320]));
        }
        h_re[0][p] = _mm512_add_pd(x0r, x1r);
        h_im[0][p] = _mm512_add_pd(x0i, x1i);
        __m512d dr = _mm512_sub_pd(x0r, x1r), di = _mm512_sub_pd(x0i, x1i);
        CMUL512(dr, di, _mm512_set1_pd(E->w640_re[p]), _mm512_set1_pd(E->w640_im[p]), h_re[1][p], h_im[1][p]);
    }

    const __m512d vc1 = _mm512_set1_pd(C5_1), vs1 = _mm512_set1_pd(S5_1);
    const __m512d vc2 = _mm512_set1_pd(C5_2), vs2 = _mm512_set1_pd(S5_2);

    for (int r2 = 0; r2 < 2; r2++) {
        __m512d b_re[5][64], b_im[5][64];
        for (int j0 = 0; j0 < 64; j0++) {
            __m512d x0r = h_re[r2][j0],       x0i = h_im[r2][j0];
            __m512d x1r = h_re[r2][j0 + 64],  x1i = h_im[r2][j0 + 64];
            __m512d x2r = h_re[r2][j0 + 128], x2i = h_im[r2][j0 + 128];
            __m512d x3r = h_re[r2][j0 + 192], x3i = h_im[r2][j0 + 192];
            __m512d x4r = h_re[r2][j0 + 256], x4i = h_im[r2][j0 + 256];

            __m512d s1r = _mm512_add_pd(x1r, x4r), s1i = _mm512_add_pd(x1i, x4i);
            __m512d d1r = _mm512_sub_pd(x1r, x4r), d1i = _mm512_sub_pd(x1i, x4i);
            __m512d s2r = _mm512_add_pd(x2r, x3r), s2i = _mm512_add_pd(x2i, x3i);
            __m512d d2r = _mm512_sub_pd(x2r, x3r), d2i = _mm512_sub_pd(x2i, x3i);

            b_re[0][j0] = _mm512_add_pd(x0r, _mm512_add_pd(s1r, s2r));
            b_im[0][j0] = _mm512_add_pd(x0i, _mm512_add_pd(s1i, s2i));

            __m512d m1r = _mm512_fmadd_pd(vc1, s1r, _mm512_fmadd_pd(vc2, s2r, x0r));
            __m512d m1i = _mm512_fmadd_pd(vc1, s1i, _mm512_fmadd_pd(vc2, s2i, x0i));
            __m512d m2r = _mm512_fmadd_pd(vc2, s1r, _mm512_fmadd_pd(vc1, s2r, x0r));
            __m512d m2i = _mm512_fmadd_pd(vc2, s1i, _mm512_fmadd_pd(vc1, s2i, x0i));

            __m512d q1r = _mm512_fmadd_pd(vs1, d1r, _mm512_mul_pd(vs2, d2r));
            __m512d q1i = _mm512_fmadd_pd(vs1, d1i, _mm512_mul_pd(vs2, d2i));
            __m512d q2r = _mm512_fmsub_pd(vs2, d1r, _mm512_mul_pd(vs1, d2r));
            __m512d q2i = _mm512_fmsub_pd(vs2, d1i, _mm512_mul_pd(vs1, d2i));

            __m512d y1r = _mm512_sub_pd(m1r, q1i), y1i = _mm512_add_pd(m1i, q1r);
            __m512d y4r = _mm512_add_pd(m1r, q1i), y4i = _mm512_sub_pd(m1i, q1r);
            __m512d y2r = _mm512_sub_pd(m2r, q2i), y2i = _mm512_add_pd(m2i, q2r);
            __m512d y3r = _mm512_add_pd(m2r, q2i), y3i = _mm512_sub_pd(m2i, q2r);

            CMUL512(y1r, y1i, _mm512_set1_pd(E->w320_re[1][j0]), _mm512_set1_pd(E->w320_im[1][j0]), b_re[1][j0], b_im[1][j0]);
            CMUL512(y2r, y2i, _mm512_set1_pd(E->w320_re[2][j0]), _mm512_set1_pd(E->w320_im[2][j0]), b_re[2][j0], b_im[2][j0]);
            CMUL512(y3r, y3i, _mm512_set1_pd(E->w320_re[3][j0]), _mm512_set1_pd(E->w320_im[3][j0]), b_re[3][j0], b_im[3][j0]);
            CMUL512(y4r, y4i, _mm512_set1_pd(E->w320_re[4][j0]), _mm512_set1_pd(E->w320_im[4][j0]), b_re[4][j0], b_im[4][j0]);
        }
        for (int r1 = 0; r1 < 5; r1++) {
            dft64_fwd_8x(b_re[r1], b_im[r1], E);
            for (int r0 = 0; r0 < 64; r0++) {
                int br = E->bitrev64[r0], r = r2 + 2 * r1 + 10 * r0;
                out_re[r] = b_re[r1][br]; out_im[r] = b_im[r1][br];
            }
        }
    }
}

XC_TARGET_AVX512 static void dft640_inv_8x(const __m512d *in_re, const __m512d *in_im,
                                           __m512d *out_re, __m512d *out_im, const xc_plan_t *E) {
    const __m512d vc1 = _mm512_set1_pd(C5_1), vs1 = _mm512_set1_pd(-S5_1);
    const __m512d vc2 = _mm512_set1_pd(C5_2), vs2 = _mm512_set1_pd(-S5_2);

    __m512d h_re[2][320], h_im[2][320];
    for (int r2 = 0; r2 < 2; r2++) {
        __m512d b_re[5][64], b_im[5][64];
        for (int r1 = 0; r1 < 5; r1++) {
            for (int r0 = 0; r0 < 64; r0++) {
                int br = E->bitrev64[r0], r = r2 + 2 * r1 + 10 * r0;
                b_re[r1][br] = in_re[r]; b_im[r1][br] = in_im[r];
            }
            dft64_inv_8x(b_re[r1], b_im[r1], E);
        }
        for (int j0 = 0; j0 < 64; j0++) {
            __m512d x0r = b_re[0][j0], x0i = b_im[0][j0];
            __m512d x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i;
            CMUL512_CONJ(b_re[1][j0], b_im[1][j0], _mm512_set1_pd(E->w320_re[1][j0]), _mm512_set1_pd(E->w320_im[1][j0]), x1r, x1i);
            CMUL512_CONJ(b_re[2][j0], b_im[2][j0], _mm512_set1_pd(E->w320_re[2][j0]), _mm512_set1_pd(E->w320_im[2][j0]), x2r, x2i);
            CMUL512_CONJ(b_re[3][j0], b_im[3][j0], _mm512_set1_pd(E->w320_re[3][j0]), _mm512_set1_pd(E->w320_im[3][j0]), x3r, x3i);
            CMUL512_CONJ(b_re[4][j0], b_im[4][j0], _mm512_set1_pd(E->w320_re[4][j0]), _mm512_set1_pd(E->w320_im[4][j0]), x4r, x4i);

            __m512d s1r = _mm512_add_pd(x1r, x4r), s1i = _mm512_add_pd(x1i, x4i);
            __m512d d1r = _mm512_sub_pd(x1r, x4r), d1i = _mm512_sub_pd(x1i, x4i);
            __m512d s2r = _mm512_add_pd(x2r, x3r), s2i = _mm512_add_pd(x2i, x3i);
            __m512d d2r = _mm512_sub_pd(x2r, x3r), d2i = _mm512_sub_pd(x2i, x3i);

            h_re[r2][j0] = _mm512_add_pd(x0r, _mm512_add_pd(s1r, s2r));
            h_im[r2][j0] = _mm512_add_pd(x0i, _mm512_add_pd(s1i, s2i));

            __m512d m1r = _mm512_fmadd_pd(vc1, s1r, _mm512_fmadd_pd(vc2, s2r, x0r));
            __m512d m1i = _mm512_fmadd_pd(vc1, s1i, _mm512_fmadd_pd(vc2, s2i, x0i));
            __m512d m2r = _mm512_fmadd_pd(vc2, s1r, _mm512_fmadd_pd(vc1, s2r, x0r));
            __m512d m2i = _mm512_fmadd_pd(vc2, s1i, _mm512_fmadd_pd(vc1, s2i, x0i));

            __m512d q1r = _mm512_fmadd_pd(vs1, d1r, _mm512_mul_pd(vs2, d2r));
            __m512d q1i = _mm512_fmadd_pd(vs1, d1i, _mm512_mul_pd(vs2, d2i));
            __m512d q2r = _mm512_fmsub_pd(vs2, d1r, _mm512_mul_pd(vs1, d2r));
            __m512d q2i = _mm512_fmsub_pd(vs2, d1i, _mm512_mul_pd(vs1, d2i));

            h_re[r2][j0 + 64]  = _mm512_sub_pd(m1r, q1i); h_im[r2][j0 + 64]  = _mm512_add_pd(m1i, q1r);
            h_re[r2][j0 + 256] = _mm512_add_pd(m1r, q1i); h_im[r2][j0 + 256] = _mm512_sub_pd(m1i, q1r);
            h_re[r2][j0 + 128] = _mm512_sub_pd(m2r, q2i); h_im[r2][j0 + 128] = _mm512_add_pd(m2i, q2r);
            h_re[r2][j0 + 192] = _mm512_add_pd(m2r, q2i); h_im[r2][j0 + 192] = _mm512_sub_pd(m2i, q2r);
        }
    }

    for (int p = 0; p < 320; p++) {
        __m512d t1r, t1i;
        CMUL512_CONJ(h_re[1][p], h_im[1][p], _mm512_set1_pd(E->w640_re[p]), _mm512_set1_pd(E->w640_im[p]), t1r, t1i);
        __m512d a0r = _mm512_add_pd(h_re[0][p], t1r), a0i = _mm512_add_pd(h_im[0][p], t1i);
        CMUL512_CONJ(a0r, a0i, _mm512_set1_pd(E->gam640_re[p]), _mm512_set1_pd(E->gam640_im[p]), out_re[p], out_im[p]);
        __m512d a1r = _mm512_sub_pd(h_re[0][p], t1r), a1i = _mm512_sub_pd(h_im[0][p], t1i);
        CMUL512_CONJ(a1r, a1i, _mm512_set1_pd(E->gam640_re[p + 320]), _mm512_set1_pd(E->gam640_im[p + 320]),
                     out_re[p + 320], out_im[p + 320]);
    }
}

XC_TARGET_AVX512 static inline __m512d load_vedic8_real_512(const uint16_t *A_u16, size_t n_u16, size_t u_pos) {
    if (__builtin_expect(u_pos > 0 && u_pos + 8 <= n_u16, 1)) {
        __m128i cur = _mm_loadu_si128((const __m128i *)(A_u16 + u_pos));
        __m128i prv = _mm_loadu_si128((const __m128i *)(A_u16 + u_pos - 1));
        __m256i u32 = _mm256_cvtepu16_epi32(cur);
        __m256i s_cur = _mm256_srli_epi32(u32, 15);
        __m256i s_prv = _mm256_srli_epi32(_mm256_cvtepu16_epi32(prv), 15);
        __m256i d32 = _mm256_add_epi32(_mm256_sub_epi32(u32, _mm256_slli_epi32(s_cur, 16)), s_prv);
        return _mm512_cvtepi32_pd(d32);
    } else {
        double dr[8];
        for (int m = 0; m < 8; m++) {
            size_t idx = u_pos + (size_t)m;
            int32_t u  = (idx < n_u16) ? (int32_t)A_u16[idx] : 0;
            int32_t s  = u >> 15;
            int32_t sp = (idx > 0 && idx - 1 < n_u16) ? (int32_t)(A_u16[idx - 1] >> 15) : 0;
            dr[m] = (double)(u - (s << 16) + sp);
        }
        return _mm512_loadu_pd(dr);
    }
}

XC_TARGET_AVX512 static inline void lut_fourstep_8x_512(const xc_plan_t *E, size_t b0, int r, __m512d *tr_out, __m512d *ti_out) {
    const size_t fourM = 4 * E->M;
    const size_t stride = (size_t)(4 * r + 1);
    size_t idx = (b0 * stride) % fourM;
    double tr[8], ti[8];
    for (int m = 0; m < 8; m++) {
        size_t f = idx & 4095ULL, c = idx >> 12;
        double cr = E->coarse_re[c], ci = E->coarse_im[c];
        double fr = E->fine_re[f],   fi = E->fine_im[f];
        tr[m] = cr * fr - ci * fi;
        ti[m] = cr * fi + ci * fr;
        idx += stride;
        if (idx >= fourM) idx -= fourM;
    }
    *tr_out = _mm512_loadu_pd(tr);
    *ti_out = _mm512_loadu_pd(ti);
}

XC_TARGET_AVX512 static void pass1_single_avx512_worker(const xc_plan_t *E, const uint16_t *A_u16, size_t n_u16,
                                                        double *DstZ, size_t v_beg, size_t v_end) {
    const size_t C = E->C_SUB;
    const int n_nz = (int)((n_u16 + 1 + C - 1) / C);
    __m512d in_re[640], out_re[640], out_im[640];
    for (size_t v = v_beg; v < v_end; v++) {
        size_t b0 = v * 8;
        for (int j = 0; j < n_nz; j++) in_re[j] = load_vedic8_real_512(A_u16, n_u16, (size_t)j * C + b0);
        dft640_fwd_real_8x(in_re, n_nz, out_re, out_im, E);
        for (int r = 0; r < 640; r++) {
            __m512d tr, ti, yr, yi;
            lut_fourstep_8x_512(E, b0, r, &tr, &ti);
            CMUL512(out_re[r], out_im[r], tr, ti, yr, yi);
            double *dst = DstZ + ((size_t)r * C + b0) * 2;
            _mm512_stream_pd(dst, yr);
            _mm512_stream_pd(dst + 8, yi);
        }
    }
    _mm_sfence();
}

static void pass1_single_avx512(const xc_plan_t *E, const uint16_t *A_u16, size_t n_u16, double *DstZ) {
    const size_t n_vec = E->C_SUB / 8;
    int nt = xc_wave_threads(E->M);
    #pragma omp parallel num_threads(nt)
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num(), nthr = omp_get_num_threads();
#else
        int tid = 0, nthr = 1;
#endif
        size_t v_beg = (n_vec * (size_t)tid) / (size_t)nthr;
        size_t v_end = (n_vec * (size_t)(tid + 1)) / (size_t)nthr;
        pass1_single_avx512_worker(E, A_u16, n_u16, DstZ, v_beg, v_end);
    }
}

XC_TARGET_AVX512 static void pass1_dual_avx512_worker(const xc_plan_t *E, const uint16_t *A_u16, size_t na_u16,
                                                      const uint16_t *B_u16, size_t nb_u16, size_t v_beg, size_t v_end) {
    const size_t C = E->C_SUB;
    const int n_nza = (int)((na_u16 + 1 + C - 1) / C);
    const int n_nzb = (int)((nb_u16 + 1 + C - 1) / C);
    __m512d in_re[640], outA_re[640], outA_im[640], outB_re[640], outB_im[640];
    for (size_t v = v_beg; v < v_end; v++) {
        size_t b0 = v * 8;
        for (int j = 0; j < n_nza; j++) in_re[j] = load_vedic8_real_512(A_u16, na_u16, (size_t)j * C + b0);
        dft640_fwd_real_8x(in_re, n_nza, outA_re, outA_im, E);

        for (int j = 0; j < n_nzb; j++) in_re[j] = load_vedic8_real_512(B_u16, nb_u16, (size_t)j * C + b0);
        dft640_fwd_real_8x(in_re, n_nzb, outB_re, outB_im, E);

        for (int r = 0; r < 640; r++) {
            __m512d tr, ti, yar, yai, ybr, ybi;
            lut_fourstep_8x_512(E, b0, r, &tr, &ti);
            CMUL512(outA_re[r], outA_im[r], tr, ti, yar, yai);
            CMUL512(outB_re[r], outB_im[r], tr, ti, ybr, ybi);
            size_t off = ((size_t)r * C + b0) * 2;
            _mm512_stream_pd(E->Az + off, yar); _mm512_stream_pd(E->Az + off + 8, yai);
            _mm512_stream_pd(E->Bz + off, ybr); _mm512_stream_pd(E->Bz + off + 8, ybi);
        }
    }
    _mm_sfence();
}

static void pass1_dual_avx512(const xc_plan_t *E, const uint16_t *A_u16, size_t na_u16,
                              const uint16_t *B_u16, size_t nb_u16) {
    const size_t n_vec = E->C_SUB / 8;
    int nt = xc_wave_threads(E->M);
    #pragma omp parallel num_threads(nt)
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num(), nthr = omp_get_num_threads();
#else
        int tid = 0, nthr = 1;
#endif
        size_t v_beg = (n_vec * (size_t)tid) / (size_t)nthr;
        size_t v_end = (n_vec * (size_t)(tid + 1)) / (size_t)nthr;
        pass1_dual_avx512_worker(E, A_u16, na_u16, B_u16, nb_u16, v_beg, v_end);
    }
}

XC_TARGET_AVX512 static void sub_dif_avx512(double *row, size_t C, const double *sub_re, const double *sub_im) {
    for (size_t half = C >> 1, step = 1; half >= 8; half >>= 1, step <<= 1) {
        for (size_t base = 0; base < C; base += 2 * half) {
            for (size_t j = 0; j < half; j += 8) {
                double *p0 = row + (base + j) * 2, *p1 = row + (base + j + half) * 2;
                __m512d u_r = _mm512_load_pd(p0), u_i = _mm512_load_pd(p0 + 8);
                __m512d v_r = _mm512_load_pd(p1), v_i = _mm512_load_pd(p1 + 8);
                __m512d d_r = _mm512_sub_pd(u_r, v_r), d_i = _mm512_sub_pd(u_i, v_i);
                _mm512_store_pd(p0, _mm512_add_pd(u_r, v_r));
                _mm512_store_pd(p0 + 8, _mm512_add_pd(u_i, v_i));

                double wr_arr[8], wi_arr[8];
                for (int m = 0; m < 8; m++) { size_t tk = (j + (size_t)m) * step; wr_arr[m] = sub_re[tk]; wi_arr[m] = sub_im[tk]; }
                __m512d o_r, o_i;
                CMUL512(d_r, d_i, _mm512_loadu_pd(wr_arr), _mm512_loadu_pd(wi_arr), o_r, o_i);
                _mm512_store_pd(p1, o_r); _mm512_store_pd(p1 + 8, o_i);
            }
        }
    }
    const size_t s4 = C >> 3, s2 = C >> 2;
    const __m512d w4r = _mm512_setr_pd(1.0, sub_re[s4], sub_re[2*s4], sub_re[3*s4], 1.0, sub_re[s4], sub_re[2*s4], sub_re[3*s4]);
    const __m512d w4i = _mm512_setr_pd(0.0, sub_im[s4], sub_im[2*s4], sub_im[3*s4], 0.0, sub_im[s4], sub_im[2*s4], sub_im[3*s4]);
    const __m512d w2r = _mm512_setr_pd(1.0, sub_re[s2], 1.0, sub_re[s2], 1.0, sub_re[s2], 1.0, sub_re[s2]);
    const __m512d w2i = _mm512_setr_pd(0.0, sub_im[s2], 0.0, sub_im[s2], 0.0, sub_im[s2], 0.0, sub_im[s2]);

    const __m512i lo4 = _mm512_setr_epi64(0,1,2,3, 0,1,2,3), hi4 = _mm512_setr_epi64(4,5,6,7, 4,5,6,7);
    const __m512i lo2 = _mm512_setr_epi64(0,1,0,1, 4,5,4,5), hi2 = _mm512_setr_epi64(2,3,2,3, 6,7,6,7);
    const __m512i lo1 = _mm512_setr_epi64(0,0,2,2, 4,4,6,6), hi1 = _mm512_setr_epi64(1,1,3,3, 5,5,7,7);
    const __m512d sgn4 = _mm512_setr_pd(1,1,1,1, -1,-1,-1,-1);
    const __m512d sgn2 = _mm512_setr_pd(1,1,-1,-1, 1,1,-1,-1);
    const __m512d sgn1 = _mm512_setr_pd(1,-1,1,-1, 1,-1,1,-1);

    for (size_t base = 0; base < C; base += 8) {
        double *p = row + base * 2;
        __m512d r = _mm512_load_pd(p), i = _mm512_load_pd(p + 8), ur, ui, vr, vi, tr, ti;
        ur = _mm512_permutexvar_pd(lo4, r); ui = _mm512_permutexvar_pd(lo4, i);
        vr = _mm512_permutexvar_pd(hi4, r); vi = _mm512_permutexvar_pd(hi4, i);
        r = _mm512_fmadd_pd(sgn4, vr, ur); i = _mm512_fmadd_pd(sgn4, vi, ui);
        CMUL512(r, i, w4r, w4i, tr, ti);
        r = _mm512_mask_blend_pd(0xF0, r, tr); i = _mm512_mask_blend_pd(0xF0, i, ti);

        ur = _mm512_permutexvar_pd(lo2, r); ui = _mm512_permutexvar_pd(lo2, i);
        vr = _mm512_permutexvar_pd(hi2, r); vi = _mm512_permutexvar_pd(hi2, i);
        r = _mm512_fmadd_pd(sgn2, vr, ur); i = _mm512_fmadd_pd(sgn2, vi, ui);
        CMUL512(r, i, w2r, w2i, tr, ti);
        r = _mm512_mask_blend_pd(0xCC, r, tr); i = _mm512_mask_blend_pd(0xCC, i, ti);

        ur = _mm512_permutexvar_pd(lo1, r); ui = _mm512_permutexvar_pd(lo1, i);
        vr = _mm512_permutexvar_pd(hi1, r); vi = _mm512_permutexvar_pd(hi1, i);
        r = _mm512_fmadd_pd(sgn1, vr, ur); i = _mm512_fmadd_pd(sgn1, vi, ui);
        _mm512_store_pd(p, r); _mm512_store_pd(p + 8, i);
    }
}

XC_TARGET_AVX512 static void sub_idit_avx512(double *row, size_t C, const double *sub_re, const double *sub_im) {
    const size_t s4 = C >> 3, s2 = C >> 2;
    const __m512d w4r = _mm512_setr_pd(1.0, sub_re[s4], sub_re[2*s4], sub_re[3*s4], 1.0, sub_re[s4], sub_re[2*s4], sub_re[3*s4]);
    const __m512d w4i = _mm512_setr_pd(0.0, sub_im[s4], sub_im[2*s4], sub_im[3*s4], 0.0, sub_im[s4], sub_im[2*s4], sub_im[3*s4]);
    const __m512d w2r = _mm512_setr_pd(1.0, sub_re[s2], 1.0, sub_re[s2], 1.0, sub_re[s2], 1.0, sub_re[s2]);
    const __m512d w2i = _mm512_setr_pd(0.0, sub_im[s2], 0.0, sub_im[s2], 0.0, sub_im[s2], 0.0, sub_im[s2]);

    const __m512i lo4 = _mm512_setr_epi64(0,1,2,3, 0,1,2,3), hi4 = _mm512_setr_epi64(4,5,6,7, 4,5,6,7);
    const __m512i lo2 = _mm512_setr_epi64(0,1,0,1, 4,5,4,5), hi2 = _mm512_setr_epi64(2,3,2,3, 6,7,6,7);
    const __m512i lo1 = _mm512_setr_epi64(0,0,2,2, 4,4,6,6), hi1 = _mm512_setr_epi64(1,1,3,3, 5,5,7,7);
    const __m512d sgn4 = _mm512_setr_pd(1,1,1,1, -1,-1,-1,-1);
    const __m512d sgn2 = _mm512_setr_pd(1,1,-1,-1, 1,1,-1,-1);
    const __m512d sgn1 = _mm512_setr_pd(1,-1,1,-1, 1,-1,1,-1);

    for (size_t base = 0; base < C; base += 8) {
        double *p = row + base * 2;
        __m512d r = _mm512_load_pd(p), i = _mm512_load_pd(p + 8), ur, ui, vr, vi, tr, ti;
        ur = _mm512_permutexvar_pd(lo1, r); ui = _mm512_permutexvar_pd(lo1, i);
        vr = _mm512_permutexvar_pd(hi1, r); vi = _mm512_permutexvar_pd(hi1, i);
        r = _mm512_fmadd_pd(sgn1, vr, ur); i = _mm512_fmadd_pd(sgn1, vi, ui);

        CMUL512_CONJ(r, i, w2r, w2i, tr, ti);
        r = _mm512_mask_blend_pd(0xCC, r, tr); i = _mm512_mask_blend_pd(0xCC, i, ti);
        ur = _mm512_permutexvar_pd(lo2, r); ui = _mm512_permutexvar_pd(lo2, i);
        vr = _mm512_permutexvar_pd(hi2, r); vi = _mm512_permutexvar_pd(hi2, i);
        r = _mm512_fmadd_pd(sgn2, vr, ur); i = _mm512_fmadd_pd(sgn2, vi, ui);

        CMUL512_CONJ(r, i, w4r, w4i, tr, ti);
        r = _mm512_mask_blend_pd(0xF0, r, tr); i = _mm512_mask_blend_pd(0xF0, i, ti);
        ur = _mm512_permutexvar_pd(lo4, r); ui = _mm512_permutexvar_pd(lo4, i);
        vr = _mm512_permutexvar_pd(hi4, r); vi = _mm512_permutexvar_pd(hi4, i);
        r = _mm512_fmadd_pd(sgn4, vr, ur); i = _mm512_fmadd_pd(sgn4, vi, ui);
        _mm512_store_pd(p, r); _mm512_store_pd(p + 8, i);
    }

    for (size_t half = 8, step = (C >> 4); half < C; half <<= 1, step >>= 1) {
        for (size_t base = 0; base < C; base += 2 * half) {
            for (size_t j = 0; j < half; j += 8) {
                double *p0 = row + (base + j) * 2, *p1 = row + (base + j + half) * 2;
                __m512d u_r = _mm512_load_pd(p0), u_i = _mm512_load_pd(p0 + 8);
                __m512d v_r = _mm512_load_pd(p1), v_i = _mm512_load_pd(p1 + 8);
                double wr_arr[8], wi_arr[8];
                for (int m = 0; m < 8; m++) { size_t tk = (j + (size_t)m) * step; wr_arr[m] = sub_re[tk]; wi_arr[m] = sub_im[tk]; }
                CMUL512_CONJ(v_r, v_i, _mm512_loadu_pd(wr_arr), _mm512_loadu_pd(wi_arr), v_r, v_i);
                _mm512_store_pd(p0, _mm512_add_pd(u_r, v_r)); _mm512_store_pd(p0 + 8, _mm512_add_pd(u_i, v_i));
                _mm512_store_pd(p1, _mm512_sub_pd(u_r, v_r)); _mm512_store_pd(p1 + 8, _mm512_sub_pd(u_i, v_i));
            }
        }
    }
}

XC_TARGET_AVX512 static void pass2_convolve_avx512_row(const xc_plan_t *E, const double *Bz_src, int b_is_trunk, int r_beg, int r_end) {
    const size_t C = E->C_SUB;
    const __m512d scale = _mm512_set1_pd(b_is_trunk ? 1.0 : (1.0 / (double)E->M));
    for (int r = r_beg; r < r_end; r++) {
        double *rowA = E->Az + (size_t)r * C * 2;
        double *rowB = (double *)(uintptr_t)(Bz_src + (size_t)r * C * 2);
        sub_dif_avx512(rowA, C, E->sub_re, E->sub_im);
        if (!b_is_trunk) sub_dif_avx512(rowB, C, E->sub_re, E->sub_im);
        for (size_t j = 0; j < C; j += 8) {
            __m512d ar = _mm512_load_pd(rowA + 2*j), ai = _mm512_load_pd(rowA + 2*j + 8);
            __m512d br = _mm512_load_pd(rowB + 2*j), bi = _mm512_load_pd(rowB + 2*j + 8);
            __m512d pr, pi;
            CMUL512(ar, ai, br, bi, pr, pi);
            if (!b_is_trunk) { pr = _mm512_mul_pd(pr, scale); pi = _mm512_mul_pd(pi, scale); }
            _mm512_store_pd(rowA + 2*j, pr); _mm512_store_pd(rowA + 2*j + 8, pi);
        }
        sub_idit_avx512(rowA, C, E->sub_re, E->sub_im);
    }
}

static void pass2_convolve_avx512(const xc_plan_t *E, const double *Bz_src, int b_is_trunk) {
    int nt = xc_wave_threads(E->M);
    #pragma omp parallel num_threads(nt)
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num(), nthr = omp_get_num_threads();
#else
        int tid = 0, nthr = 1;
#endif
        int r_beg = (640 * tid) / nthr;
        int r_end = (640 * (tid + 1)) / nthr;
        pass2_convolve_avx512_row(E, Bz_src, b_is_trunk, r_beg, r_end);
    }
}

XC_TARGET_AVX512 static void pass3_unpack_avx512_worker(xc_plan_t *E, uint16_t *Out_u16, size_t out_n_u16,
                                                        size_t v_beg, size_t v_end) {
    const size_t C = E->C_SUB, M = E->M, n_vec = C / 8, half_tiles = (size_t)R_TOP * n_vec;
    __m512d in_re[640], in_im[640], out_re[640], out_im[640];
    const __m512d z0 = _mm512_setzero_pd();
    for (size_t v = v_beg; v < v_end; v++) {
        size_t b0 = v * 8;
        for (int r = 0; r < 640; r++) {
            const double *src = E->Az + ((size_t)r * C + b0) * 2;
            __m512d yr = _mm512_load_pd(src), yi = _mm512_load_pd(src + 8), tr, ti;
            lut_fourstep_8x_512(E, b0, r, &tr, &ti);
            CMUL512_CONJ(yr, yi, tr, ti, in_re[r], in_im[r]);
        }
        dft640_inv_8x(in_re, in_im, out_re, out_im, E);

        for (int j = 0; j < 640; j++) {
            size_t t_lo = (size_t)j * n_vec + v, pos_lo = t_lo * 8;
            if (pos_lo < out_n_u16) {
                __m512i ir = _mm512_cvt_roundpd_epi64(out_re[j], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                int64_t rr[8]; _mm512_storeu_si512((__m512i *)rr, ir);
                int64_t acc = 0;
                if (__builtin_expect(pos_lo + 8 <= out_n_u16, 1)) {
                    uint16_t loc[8];
                    for (int m = 0; m < 8; m++) { acc += rr[m]; loc[m] = (uint16_t)(acc & 0xFFFF); acc >>= 16; }
                    memcpy(Out_u16 + pos_lo, loc, 16);
                } else {
                    for (int m = 0; m < 8; m++) { acc += rr[m]; if (pos_lo + (size_t)m < out_n_u16) Out_u16[pos_lo + m] = (uint16_t)(acc & 0xFFFF); acc >>= 16; }
                }
                E->tile_carry[t_lo] = acc;
            } else E->tile_carry[t_lo] = 0;

            size_t t_hi = half_tiles + t_lo, pos_hi = M + pos_lo;
            if (pos_hi < out_n_u16) {
                __m512i ii = _mm512_cvt_roundpd_epi64(_mm512_sub_pd(z0, out_im[j]), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                int64_t ri[8]; _mm512_storeu_si512((__m512i *)ri, ii);
                int64_t acc = 0;
                if (__builtin_expect(pos_hi + 8 <= out_n_u16, 1)) {
                    uint16_t loc[8];
                    for (int m = 0; m < 8; m++) { acc += ri[m]; loc[m] = (uint16_t)(acc & 0xFFFF); acc >>= 16; }
                    memcpy(Out_u16 + pos_hi, loc, 16);
                } else {
                    for (int m = 0; m < 8; m++) { acc += ri[m]; if (pos_hi + (size_t)m < out_n_u16) Out_u16[pos_hi + m] = (uint16_t)(acc & 0xFFFF); acc >>= 16; }
                }
                E->tile_carry[t_hi] = acc;
            } else E->tile_carry[t_hi] = 0;
        }
    }
}

static void pass3_unpack_avx512(xc_plan_t *E, uint16_t *Out_u16, size_t out_n_u16) {
    const size_t n_vec = E->C_SUB / 8;
    int nt = xc_wave_threads(E->M);
    #pragma omp parallel num_threads(nt)
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num(), nthr = omp_get_num_threads();
#else
        int tid = 0, nthr = 1;
#endif
        size_t v_beg = (n_vec * (size_t)tid) / (size_t)nthr;
        size_t v_end = (n_vec * (size_t)(tid + 1)) / (size_t)nthr;
        pass3_unpack_avx512_worker(E, Out_u16, out_n_u16, v_beg, v_end);
    }
    ripple_tile_carries(E, Out_u16, out_n_u16);
}

XC_TARGET_AVX512 static void pass1_memo_avx512_worker(const xc_plan_t *E, const uint16_t *A_u16, size_t n_u16,
                                                      double *DstZ, int U, const int *rep_j,
                                                      const double (*W_re)[44], const double (*W_im)[44],
                                                      size_t v_beg, size_t v_end) {
    const size_t C = E->C_SUB;
    __m512d u_re[44];
    for (size_t v = v_beg; v < v_end; v++) {
        size_t b0 = v * 8;
        for (int u = 0; u < U; u++) u_re[u] = load_vedic8_real_512(A_u16, n_u16, (size_t)rep_j[u] * C + b0);
        for (int r = 0; r < 640; r++) {
            __m512d sr = _mm512_setzero_pd(), si = _mm512_setzero_pd();
            for (int u = 0; u < U; u++) {
                sr = _mm512_fmadd_pd(u_re[u], _mm512_set1_pd(W_re[r][u]), sr);
                si = _mm512_fmadd_pd(u_re[u], _mm512_set1_pd(W_im[r][u]), si);
            }
            __m512d tr, ti, yr, yi;
            lut_fourstep_8x_512(E, b0, r, &tr, &ti);
            CMUL512(sr, si, tr, ti, yr, yi);
            double *dst = DstZ + ((size_t)r * C + b0) * 2;
            _mm512_stream_pd(dst, yr); _mm512_stream_pd(dst + 8, yi);
        }
    }
    _mm_sfence();
}

static void pass1_memo_avx512(const xc_plan_t *E, const uint16_t *A_u16, size_t n_u16,
                              double *DstZ, int U, const int *rep_j, const int *slice_uid, int n_nz) {
    const size_t n_vec = E->C_SUB / 8;
    int nt = xc_wave_threads(E->M);
    double W_re[640][44], W_im[640][44];
    int r;

    #pragma omp parallel for num_threads(nt) schedule(static)
    for (r = 0; r < 640; r++) {
        for (int u = 0; u < U; u++) { W_re[r][u] = 0.0; W_im[r][u] = 0.0; }
        for (int j = 0; j < n_nz; j++) {
            int u = slice_uid[j], idx = (j * (4 * r + 1)) % 2560;
            W_re[r][u] += E->g2560_re[idx];
            W_im[r][u] += E->g2560_im[idx];
        }
    }

    #pragma omp parallel num_threads(nt)
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num(), nthr = omp_get_num_threads();
#else
        int tid = 0, nthr = 1;
#endif
        size_t v_beg = (n_vec * (size_t)tid) / (size_t)nthr;
        size_t v_end = (n_vec * (size_t)(tid + 1)) / (size_t)nthr;
        pass1_memo_avx512_worker(E, A_u16, n_u16, DstZ, U, rep_j,
                                 (const double (*)[44])W_re, (const double (*)[44])W_im, v_beg, v_end);
    }
}
#endif /* XC_HAS_X86_SIMD */

/* -----------------------------------------------------------------------------------------
 * UNIFIED DISPATCH WRAPPERS
 * ----------------------------------------------------------------------------------------- */
static void dispatch_pass1_single(const xc_plan_t *E, const uint16_t *A_u16, size_t n_u16, double *DstZ) {
#if XC_HAS_X86_SIMD
    if (E->isa == XC_ISA_AVX512) { pass1_single_avx512(E, A_u16, n_u16, DstZ); return; }
#endif
    pass1_single_scalar(E, A_u16, n_u16, DstZ);
}

static int dispatch_pass1_auto_memo(const xc_plan_t *E, const uint16_t *A_u16, size_t n_u16, double *DstZ) {
    const size_t C = E->C_SUB;
    const int n_nz = (int)((n_u16 + 1 + C - 1) / C);
    const int n_full = (int)(n_u16 / C);

    if (n_full >= 16) {
        int slice_uid[640], rep_j[640];
        uint64_t hashes[640];
        int nt = xc_wave_threads(E->M);
        int j_omp;

        #pragma omp parallel for num_threads(nt) schedule(static)
        for (j_omp = 0; j_omp < n_full; j_omp++) {
            const uint64_t *p64 = (const uint64_t *)(A_u16 + (size_t)j_omp * C);
            size_t n64 = C / 4;
            uint16_t prev_sign = (j_omp > 0) ? (A_u16[(size_t)j_omp * C - 1] >> 15) : 0;
            uint64_t h = 14695981039346656037ULL ^ prev_sign;
            for (size_t k = 0; k < n64; k++) { h ^= p64[k]; h *= 1099511628211ULL; }
            hashes[j_omp] = h;
        }

        int U = 0;
        for (int j = 0; j < n_full && U <= 40; j++) {
            uint16_t prev_j = (j > 0) ? (A_u16[(size_t)j * C - 1] >> 15) : 0;
            int match = -1;
            for (int u = 0; u < U; u++) {
                int rj = rep_j[u];
                uint16_t prev_r = (rj > 0) ? (A_u16[(size_t)rj * C - 1] >> 15) : 0;
                if (hashes[j] == hashes[rj] && prev_j == prev_r &&
                    memcmp(A_u16 + (size_t)j * C, A_u16 + (size_t)rj * C, C * 2) == 0) {
                    match = u; break;
                }
            }
            if (match >= 0) slice_uid[j] = match;
            else { rep_j[U] = j; slice_uid[j] = U; U++; }
        }
        for (int j = n_full; j < n_nz && U <= 42; j++) { rep_j[U] = j; slice_uid[j] = U; U++; }

        if (U <= 40 && U * 2 < n_nz) {
            g_last_uniq = U;
#if XC_HAS_X86_SIMD
            if (E->isa == XC_ISA_AVX512) {
                pass1_memo_avx512(E, A_u16, n_u16, DstZ, U, rep_j, slice_uid, n_nz);
                return 1;
            }
#endif
        }
    }
    g_last_uniq = n_nz;
    dispatch_pass1_single(E, A_u16, n_u16, DstZ);
    return 0;
}

static void dispatch_pass2(const xc_plan_t *E, const double *Bz_src, int b_is_trunk) {
#if XC_HAS_X86_SIMD
    if (E->isa == XC_ISA_AVX512) { pass2_convolve_avx512(E, Bz_src, b_is_trunk); return; }
    if (E->isa == XC_ISA_AVX2)   { pass2_convolve_avx2(E, Bz_src, b_is_trunk);   return; }
#endif
    pass2_convolve_scalar(E, Bz_src, b_is_trunk);
}

static void dispatch_pass3(xc_plan_t *E, uint16_t *Out_u16, size_t out_n_u16) {
#if XC_HAS_X86_SIMD
    if (E->isa == XC_ISA_AVX512) { pass3_unpack_avx512(E, Out_u16, out_n_u16); return; }
#endif
    pass3_unpack_scalar(E, Out_u16, out_n_u16);
}

/* =========================================================================================
 * PUBLIC PLAN, TRUNK, & MULTIPLICATION APIS
 * ========================================================================================= */
XC_API xc_plan_t *xc_plan_create(size_t max_na, size_t max_nb) {
    if (max_na == 0 || max_nb == 0) return NULL;
    int log_c = choose_log_c(max_na, max_nb);
    if (log_c > 16) return NULL; /* > ~250M input digits per limb array: use xc_mul_u64_lowmem */
    xc_plan_t *plan = (xc_plan_t *)malloc(sizeof(xc_plan_t));
    if (!plan) return NULL;
    if (plan_init_internal(plan, log_c, 1) != XC_OK) {
        plan_free_internal(plan);
        free(plan);
        return NULL;
    }
    return plan;
}

XC_API void xc_plan_destroy(xc_plan_t *plan) {
    if (!plan) return;
    plan_free_internal(plan);
    free(plan);
}

XC_API int xc_plan_mul(xc_plan_t *plan, uint64_t *out,
                       const uint64_t *a, size_t na,
                       const uint64_t *b, size_t nb) {
    if (!plan || !out || (!a && na) || (!b && nb)) return XC_ERR_ARG;
    size_t orig_out = na + nb;
    while (na > 0 && a[na - 1] == 0) na--;
    while (nb > 0 && b[nb - 1] == 0) nb--;
    if (na == 0 || nb == 0) {
        memset(out, 0, orig_out * sizeof(uint64_t));
        g_last_path = XC_PATH_ZERO;
        return XC_OK;
    }
    if (choose_log_c(na, nb) > plan->log_c) return XC_ERR_RANGE;

    memset(out, 0, orig_out * sizeof(uint64_t));
    size_t na_u16 = na * 4, nb_u16 = nb * 4, out_u16 = (na + nb) * 4;

#if XC_HAS_X86_SIMD
    if (plan->isa == XC_ISA_AVX512) {
        pass1_dual_avx512(plan, (const uint16_t *)a, na_u16, (const uint16_t *)b, nb_u16);
    } else
#endif
    {
        dispatch_pass1_single(plan, (const uint16_t *)a, na_u16, plan->Az);
        dispatch_pass1_single(plan, (const uint16_t *)b, nb_u16, plan->Bz);
    }
    dispatch_pass2(plan, plan->Bz, 0);
    dispatch_pass3(plan, (uint16_t *)out, out_u16);
    g_last_path = XC_PATH_WAVE_COLD;
    return XC_OK;
}

XC_API xc_trunk_t *xc_trunk_create(const xc_plan_t *plan, const uint64_t *b, size_t nb) {
    if (!plan || (!b && nb)) return NULL;
    size_t orig_nb = nb;
    while (nb > 0 && b[nb - 1] == 0) nb--;
    if (nb > 0 && choose_log_c(nb, nb) > plan->log_c) return NULL;

    xc_trunk_t *T = (xc_trunk_t *)calloc(1, sizeof(xc_trunk_t));
    if (!T) return NULL;
    T->log_c = plan->log_c;
    T->C_SUB = plan->C_SUB;
    T->M     = plan->M;
    T->nb    = orig_nb;
    if (nb == 0) { T->is_zero = 1; return T; }

    int use_huge = (2 * plan->M * sizeof(double) >= (2ULL << 20));
    T->Bz_trunk = (double *)xc_aligned_alloc_buf(2 * plan->M * sizeof(double), use_huge);
    if (!T->Bz_trunk) { free(T); return NULL; }

    dispatch_pass1_single(plan, (const uint16_t *)b, nb * 4, T->Bz_trunk);
    const size_t C = plan->C_SUB;
    const double invM = 1.0 / (double)plan->M;
    int nt = xc_wave_threads(plan->M);
    int r;

    #pragma omp parallel for num_threads(nt) schedule(static)
    for (r = 0; r < 640; r++) {
        double *rowB = T->Bz_trunk + (size_t)r * C * 2;
#if XC_HAS_X86_SIMD
        if (plan->isa == XC_ISA_AVX512) sub_dif_avx512(rowB, C, plan->sub_re, plan->sub_im);
        else if (plan->isa == XC_ISA_AVX2) avx2_sub_dif(rowB, C, plan->sub_re, plan->sub_im);
        else
#endif
            sc_sub_dif(rowB, C, plan->sub_re, plan->sub_im);

        for (size_t k = 0; k < 2 * C; k++) rowB[k] *= invM;
    }
    return T;
}

XC_API void xc_trunk_destroy(xc_trunk_t *trunk) {
    if (!trunk) return;
    xc_aligned_free_buf(trunk->Bz_trunk);
    free(trunk);
}

XC_API int xc_mul_trunk(xc_plan_t *plan, uint64_t *out,
                        const uint64_t *a, size_t na,
                        const xc_trunk_t *b_trunk) {
    if (!plan || !out || !b_trunk || (!a && na)) return XC_ERR_ARG;
    if (plan->log_c != b_trunk->log_c) return XC_ERR_ARG;
    size_t orig_out = na + b_trunk->nb;
    while (na > 0 && a[na - 1] == 0) na--;
    if (na == 0 || b_trunk->is_zero) {
        memset(out, 0, orig_out * sizeof(uint64_t));
        g_last_path = XC_PATH_ZERO;
        return XC_OK;
    }
    if (choose_log_c(na, b_trunk->nb) > plan->log_c) return XC_ERR_RANGE;

    memset(out, 0, orig_out * sizeof(uint64_t));
    int used_hash = dispatch_pass1_auto_memo(plan, (const uint16_t *)a, na * 4, plan->Az);
    dispatch_pass2(plan, b_trunk->Bz_trunk, 1);
    dispatch_pass3(plan, (uint16_t *)out, orig_out * 4);
    g_last_path = used_hash ? XC_PATH_WAVE_HASH : XC_PATH_WAVE_TRUNK;
    return XC_OK;
}

XC_API int xc_mul_u64(uint64_t *out, const uint64_t *a, size_t na,
                      const uint64_t *b, size_t nb) {
    if (!out || (!a && na) || (!b && nb)) return XC_ERR_ARG;
    size_t orig_out = na + nb;
    while (na > 0 && a[na - 1] == 0) na--;
    while (nb > 0 && b[nb - 1] == 0) nb--;
    if (na == 0 || nb == 0) {
        memset(out, 0, orig_out * sizeof(uint64_t));
        g_last_path = XC_PATH_ZERO;
        return XC_OK;
    }

    memset(out, 0, orig_out * sizeof(uint64_t));
    size_t min_n = (na < nb) ? na : nb;
    size_t max_n = (na > nb) ? na : nb;

    /* 1) Register-scale Schoolbook */
    if (min_n <= 40 || (na + nb) <= 64) {
        schoolbook_mul_u64(out, a, na, b, nb);
        g_last_path = XC_PATH_SCHOOLBOOK;
        return XC_OK;
    }

    /* 2) L1-cache Karatsuba */
    if (max_n <= 384) {
        size_t sc_words = 8 * (na + nb) + 256;
        uint64_t *scratch = (uint64_t *)malloc(sc_words * sizeof(uint64_t));
        if (!scratch) return XC_ERR_NOMEM;
        karatsuba_rec(out, a, na, b, nb, scratch);
        free(scratch);
        g_last_path = XC_PATH_KARATSUBA;
        return XC_OK;
    }

    /* 3) Vedic Signed Power-of-2 (NAF) Sparse Polynomial Fast-Path */
    if (try_vedic_sparse_mul(out, a, na, b, nb)) {
        return XC_OK;
    }

    /* 4) Ultra-huge fallback (> ~250M input digits): Raw-Bit Karatsuba Top-Split */
    int log_c = choose_log_c(na, nb);
    if (log_c > 16) {
        return xc_mul_u64_lowmem(out, a, na, b, nb);
    }

    /* 5) Crandall-Fagin 16-Bit Vedic Vinculum Wave Engine */
    xc_plan_t plan;
    if (plan_init_internal(&plan, log_c, 1) != XC_OK) {
        plan_free_internal(&plan);
        return XC_ERR_NOMEM;
    }
    int rc = xc_plan_mul(&plan, out, a, na, b, nb);
    plan_free_internal(&plan);
    return rc;
}

/* Low-RAM Raw-Bit Karatsuba Top-Split (cuts peak wave RAM by 50%!) */
XC_API int xc_mul_u64_lowmem(uint64_t *out, const uint64_t *a, size_t na,
                             const uint64_t *b, size_t nb) {
    if (!out || (!a && na) || (!b && nb)) return XC_ERR_ARG;
    size_t orig_out = na + nb;
    while (na > 0 && a[na - 1] == 0) na--;
    while (nb > 0 && b[nb - 1] == 0) nb--;
    if (na == 0 || nb == 0) {
        memset(out, 0, orig_out * sizeof(uint64_t));
        g_last_path = XC_PATH_ZERO;
        return XC_OK;
    }
    if ((na < nb ? na : nb) <= 384) return xc_mul_u64(out, a, na, b, nb);

    memset(out, 0, orig_out * sizeof(uint64_t));
    size_t half = ((na > nb ? na : nb) + 1) / 2;
    if (na <= half || nb <= half) return xc_mul_u64(out, a, na, b, nb);

    size_t a1_n = na - half, b1_n = nb - half;
    size_t sa_cap = (a1_n > half ? a1_n : half) + 1;
    size_t sb_cap = (b1_n > half ? b1_n : half) + 1;

    /* Reuse ONE single half-size wave plan across all 3 Karatsuba leaves! */
    xc_plan_t *half_plan = xc_plan_create(sa_cap, sb_cap);
    if (!half_plan) return XC_ERR_NOMEM;

    uint64_t *sa = (uint64_t *)malloc(sa_cap * sizeof(uint64_t));
    uint64_t *sb = (uint64_t *)malloc(sb_cap * sizeof(uint64_t));
    uint64_t *p2 = (uint64_t *)calloc(sa_cap + sb_cap, sizeof(uint64_t));
    if (!sa || !sb || !p2) {
        free(sa); free(sb); free(p2); xc_plan_destroy(half_plan);
        return XC_ERR_NOMEM;
    }

    int rc = xc_plan_mul(half_plan, out, a, half, b, half);
    if (rc == XC_OK) rc = xc_plan_mul(half_plan, out + 2 * half, a + half, a1_n, b + half, b1_n);

    if (rc == XC_OK) {
        sa[sa_cap - 1] = limbs_add(sa, a, half, a + half, a1_n);
        size_t sa_n = sa[sa_cap - 1] ? sa_cap : (sa_cap - 1);
        sb[sb_cap - 1] = limbs_add(sb, b, half, b + half, b1_n);
        size_t sb_n = sb[sb_cap - 1] ? sb_cap : (sb_cap - 1);

        rc = xc_plan_mul(half_plan, p2, sa, sa_n, sb, sb_n);
        if (rc == XC_OK) {
            size_t p2_n = sa_n + sb_n;
            limbs_sub_inplace(p2, p2_n, out, 2 * half);
            limbs_sub_inplace(p2, p2_n, out + 2 * half, a1_n + b1_n);
            size_t rem = (na + nb) - half;
            if (p2_n > rem) p2_n = rem;
            limbs_add(out + half, out + half, rem, p2, p2_n);
        }
    }

    free(sa); free(sb); free(p2);
    xc_plan_destroy(half_plan);
    g_last_path = XC_PATH_LOWMEM_KARA;
    return rc;
}

/* =========================================================================================
 * DECIMAL STRING CONVERSION HELPERS (Base 10^9 <-> uint64_t limbs)
 * ========================================================================================= */
XC_API int xc_from_decimal(const char *dec_str, uint64_t **out_limbs, size_t *out_nlimbs) {
    if (!dec_str || !out_limbs || !out_nlimbs) return XC_ERR_ARG;
    while (*dec_str == ' ' || *dec_str == '\t' || *dec_str == '+') dec_str++;
    while (*dec_str == '0' && dec_str[1] >= '0' && dec_str[1] <= '9') dec_str++;
    size_t len = 0;
    while (dec_str[len] >= '0' && dec_str[len] <= '9') len++;
    if (len == 0) return XC_ERR_ARG;

    size_t cap = (size_t)((double)len * 0.052) + 2;
    uint64_t *limbs = (uint64_t *)calloc(cap, sizeof(uint64_t));
    if (!limbs) return XC_ERR_NOMEM;
    size_t n = 1;

    size_t pos = 0;
    while (pos < len) {
        size_t chunk = (len - pos >= 9) ? 9 : (len - pos);
        uint64_t mul = 1, val = 0;
        for (size_t k = 0; k < chunk; k++) {
            mul *= 10ULL;
            val = val * 10ULL + (uint64_t)(dec_str[pos + k] - '0');
        }
        pos += chunk;

        uint64_t carry = val;
        for (size_t i = 0; i < n; i++) {
            uint64_t lo, hi;
            mul64x64(limbs[i], mul, &lo, &hi);
            uint64_t s = lo + carry;
            carry = hi + (s < lo);
            limbs[i] = s;
        }
        if (carry) limbs[n++] = carry;
    }
    while (n > 1 && limbs[n - 1] == 0) n--;
    *out_limbs  = limbs;
    *out_nlimbs = n;
    return XC_OK;
}

XC_API char *xc_to_decimal(const uint64_t *limbs, size_t nlimbs) {
    if (!limbs && nlimbs) return NULL;
    while (nlimbs > 0 && limbs[nlimbs - 1] == 0) nlimbs--;
    if (nlimbs == 0) {
        char *z = (char *)malloc(2);
        if (z) { z[0] = '0'; z[1] = '\0'; }
        return z;
    }

    uint64_t *tmp = (uint64_t *)malloc(nlimbs * sizeof(uint64_t));
    if (!tmp) return NULL;
    memcpy(tmp, limbs, nlimbs * sizeof(uint64_t));

    size_t max_chunks = nlimbs * 3 + 4;
    uint32_t *chunks = (uint32_t *)malloc(max_chunks * sizeof(uint32_t));
    if (!chunks) { free(tmp); return NULL; }
    size_t nc = 0;

    while (nlimbs > 0) {
        uint64_t rem = 0;
        for (size_t i = nlimbs; i-- > 0;) {
#if defined(__SIZEOF_INT128__)
            __uint128_t cur = ((__uint128_t)rem << 64) | tmp[i];
            tmp[i] = (uint64_t)(cur / 1000000000ULL);
            rem    = (uint64_t)(cur % 1000000000ULL);
#else
            /* Portable 32-bit half-word division by 10^9 */
            uint64_t hi = (rem << 32) | (tmp[i] >> 32);
            uint64_t q1 = hi / 1000000000ULL; rem = hi % 1000000000ULL;
            uint64_t lo = (rem << 32) | (tmp[i] & 0xFFFFFFFFULL);
            uint64_t q0 = lo / 1000000000ULL; rem = lo % 1000000000ULL;
            tmp[i] = (q1 << 32) | q0;
#endif
        }
        chunks[nc++] = (uint32_t)rem;
        while (nlimbs > 0 && tmp[nlimbs - 1] == 0) nlimbs--;
    }
    free(tmp);

    char *str = (char *)malloc(nc * 9 + 2);
    if (!str) { free(chunks); return NULL; }
    int written = snprintf(str, 16, "%u", chunks[nc - 1]);
    char *p = str + written;
    for (size_t i = nc - 1; i-- > 0;) {
        uint32_t v = chunks[i];
        for (int d = 8; d >= 0; d--) { p[d] = (char)('0' + (v % 10U)); v /= 10U; }
        p += 9;
    }
    *p = '\0';
    free(chunks);
    return str;
}

XC_API char *xc_mul_decimal(const char *dec_a, const char *dec_b) {
    uint64_t *la = NULL, *lb = NULL;
    size_t na = 0, nb = 0;
    if (xc_from_decimal(dec_a, &la, &na) != XC_OK) return NULL;
    if (xc_from_decimal(dec_b, &lb, &nb) != XC_OK) { free(la); return NULL; }
    uint64_t *lc = (uint64_t *)calloc(na + nb, sizeof(uint64_t));
    if (!lc) { free(la); free(lb); return NULL; }
    int rc = xc_mul_u64(lc, la, na, lb, nb);
    free(la); free(lb);
    if (rc != XC_OK) { free(lc); return NULL; }
    char *res = xc_to_decimal(lc, na + nb);
    free(lc);
    return res;
}
