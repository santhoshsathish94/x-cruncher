/* =========================================================================================
 * bench/repro_raw_limbs.c — Standalone Reproducible Raw-Limb Benchmark & GMP Verifier
 *
 * Allows independent evaluators to benchmark x-cruncher on:
 *   1) Identical external binary limb files (little-endian uint64_t[], e.g. exported Pi limbs)
 *   2) Deterministic synthetic 100M-digit (5,190,513 limbs) operands (--synthetic-100m)
 * with per-run timings, Best / Median / Worst summary, thread count, and optional live
 * bit-for-bit verification against GNU GMP (__gmpn_mul / __gmpn_mul_n).
 *
 * Build (Linux / macOS):
 *   cc -O3 -std=c11 -Wall -Wextra -fopenmp -Iinclude \
 *      bench/repro_raw_limbs.c src/xcruncher.c -o repro_raw_limbs -fopenmp -lm -ldl
 * ========================================================================================= */
#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
  #define _CRT_SECURE_NO_WARNINGS
#endif
#if !defined(_WIN32) && !defined(_FILE_OFFSET_BITS)
  #define _FILE_OFFSET_BITS 64
#endif
#if defined(__linux__) && !defined(_GNU_SOURCE)
  #define _GNU_SOURCE
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
  #define _DARWIN_C_SOURCE
#endif

#include "xcruncher.h"
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifdef _OPENMP
  #include <omp.h>
#endif

#if defined(_WIN32)
  #include <windows.h>
  static double now_ms(void) {
      LARGE_INTEGER f, c;
      QueryPerformanceFrequency(&f);
      QueryPerformanceCounter(&c);
      return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
  }
#else
  #include <time.h>
  #include <dlfcn.h>
  #include <sys/time.h>
  #include <sys/resource.h>
  static double now_ms(void) {
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1e-6;
  }
#endif

typedef uint64_t (*gmp_mul_fn)(uint64_t *, const uint64_t *, intptr_t, const uint64_t *, intptr_t);

static gmp_mul_fn load_gmp_mul(void) {
#if defined(_WIN32)
    const char *names[] = {
        "libgmp-10.dll", "gmp.dll",
        "C:\\msys64\\ucrt64\\bin\\libgmp-10.dll",
        "C:\\msys64\\mingw64\\bin\\libgmp-10.dll", NULL
    };
    for (int i = 0; names[i]; i++) {
        HMODULE h = LoadLibraryA(names[i]);
        if (h) {
            FARPROC p = GetProcAddress(h, "__gmpn_mul");
            if (p) {
                gmp_mul_fn fn = NULL;
                memcpy(&fn, &p, sizeof(fn));
                return fn;
            }
        }
    }
    return NULL;
#else
    const char *names[] = {
        "libgmp.so.10", "libgmp.so", "libgmp.dylib",
        "/opt/homebrew/lib/libgmp.dylib", "/usr/local/lib/libgmp.dylib", NULL
    };
    for (int i = 0; names[i]; i++) {
        void *h = dlopen(names[i], RTLD_LAZY);
        if (h) {
            void *p = dlsym(h, "__gmpn_mul");
            if (p) {
                gmp_mul_fn fn = NULL;
                memcpy(&fn, &p, sizeof(fn));
                return fn;
            }
        }
    }
    return NULL;
#endif
}

static uint64_t *alloc_u64(size_t n) {
    if (n == 0 || n > SIZE_MAX / sizeof(uint64_t)) return NULL;
    return (uint64_t *)malloc(n * sizeof(uint64_t));
}

static void free_u64(uint64_t *p) {
    free(p);
}

static int file_size_i64(FILE *f, int64_t *out_sz) {
#if defined(_WIN32)
    if (_fseeki64(f, 0, SEEK_END) != 0) return -1;
    int64_t sz = _ftelli64(f);
    if (sz < 0 || _fseeki64(f, 0, SEEK_SET) != 0) return -1;
    *out_sz = sz;
    return 0;
#else
    if (fseeko(f, 0, SEEK_END) != 0) return -1;
    off_t sz = ftello(f);
    if (sz < 0 || fseeko(f, 0, SEEK_SET) != 0) return -1;
    *out_sz = (int64_t)sz;
    return 0;
#endif
}

static uint64_t *read_limb_file(const char *path, size_t *out_n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    int64_t sz = 0;
    if (file_size_i64(f, &sz) != 0 || sz <= 0 || (sz % 8) != 0) {
        fprintf(stderr, "Error: %s size (%lld bytes) must be a positive multiple of 8.\n",
                path, (long long)sz);
        fclose(f);
        return NULL;
    }
    uint64_t n64 = (uint64_t)sz / 8ULL;
    if (n64 > (uint64_t)(SIZE_MAX / sizeof(uint64_t))) {
        fprintf(stderr, "Error: %s exceeds addressable size_t limb limit.\n", path);
        fclose(f);
        return NULL;
    }
    size_t n = (size_t)n64;
    uint64_t *buf = alloc_u64(n);
    if (!buf) {
        fprintf(stderr, "Error: out of memory allocating %zu limbs for %s\n", n, path);
        fclose(f);
        return NULL;
    }
    if (fread(buf, 8, n, f) != n) {
        fprintf(stderr, "Error: short read on %s\n", path);
        free_u64(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *out_n = n;
    return buf;
}

static int parse_positive_runs(const char *arg, int *out_runs) {
    if (!arg || arg[0] == '\0') return 0;
    errno = 0;
    char *endp = NULL;
    long v = strtol(arg, &endp, 10);
    if (errno != 0 || endp == arg || *endp != '\0' || v < 1L || v > 100000L) return 0;
    *out_runs = (int)v;
    return 1;
}

static int cmp_double(const void *p1, const void *p2) {
    double a = *(const double *)p1, b = *(const double *)p2;
    return (a > b) - (a < b);
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s <limbs_a.bin> [limbs_b.bin] [--runs N] [--verify-gmp] [-o out_limbs.bin]\n"
        "  %s --synthetic-100m [--distinct] [--runs N] [--verify-gmp]\n",
        prog, prog);
}

int main(int argc, char **argv) {
    const char *file_a = NULL, *file_b = NULL, *out_path = NULL;
    int synthetic_100m = 0, distinct = 0, verify_gmp = 0, runs = 7;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--synthetic-100m") == 0) synthetic_100m = 1;
        else if (strcmp(argv[i], "--distinct") == 0) distinct = 1;
        else if (strcmp(argv[i], "--verify-gmp") == 0) verify_gmp = 1;
        else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            if (!parse_positive_runs(argv[++i], &runs)) {
                fprintf(stderr, "Error: --runs expects an integer in 1..100000 (got '%s')\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) out_path = argv[++i];
        else if (argv[i][0] == '-') { usage(argv[0]); return 1; }
        else if (!file_a) file_a = argv[i];
        else if (!file_b) file_b = argv[i];
        else { usage(argv[0]); return 1; }
    }
    if (!synthetic_100m && !file_a) { usage(argv[0]); return 1; }

    int exit_code = 1;
    size_t na = 0, nb = 0, n_out = 0;
    uint64_t *a = NULL, *b = NULL, *out = NULL, *ref = NULL;
    double *times = NULL;

    if (synthetic_100m) {
        na = 5190513ULL; /* floor(10^100000000) scale = 5,190,513 uint64_t limbs */
        nb = na;
        a = alloc_u64(na);
        if (!a) goto cleanup;
        uint64_t s = 0x3141592653589793ULL;
        for (size_t i = 0; i < na; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; a[i] = s; }
        if (distinct) {
            b = alloc_u64(nb);
            if (!b) goto cleanup;
            for (size_t i = 0; i < nb; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; b[i] = s; }
        } else {
            b = a;
        }
    } else {
        a = read_limb_file(file_a, &na);
        if (!a) goto cleanup;
        if (file_b) {
            b = read_limb_file(file_b, &nb);
            if (!b) goto cleanup;
        } else {
            b = a;
            nb = na;
        }
    }

    if (na > SIZE_MAX - nb) {
        fprintf(stderr, "Error: na + nb overflows size_t.\n");
        goto cleanup;
    }
    n_out = na + nb;
    out = alloc_u64(n_out);
    times = (double *)malloc((size_t)runs * sizeof(double));
    if (!out || !times) {
        fprintf(stderr, "Error: out of memory allocating output/timing buffers.\n");
        goto cleanup;
    }

    int omp_max = 1;
#ifdef _OPENMP
    omp_max = omp_get_max_threads();
#endif
    printf("========================================================================\n");
    printf(" x-cruncher v%s Reproducibility Harness\n", xc_version());
    printf(" Active ISA      : %s\n", xc_active_isa_name());
    printf(" OpenMP Max Thr  : %d (OMP_NUM_THREADS=%s)\n",
           omp_max, getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "auto");
    printf(" Mode            : %s (%zu x %zu uint64_t limbs -> %zu output limbs)\n",
           (a == b) ? "Square (A * A)" : "Multiply (A * B)", na, nb, n_out);
    printf("========================================================================\n");

    for (int r = 0; r < runs; r++) {
        double t0 = now_ms();
        int rc = xc_mul_u64(out, a, na, b, nb);
        double dt = now_ms() - t0;
        if (rc != XC_OK) {
            fprintf(stderr, "xc_mul_u64 failed with code %d\n", rc);
            exit_code = 2;
            goto cleanup;
        }
        times[r] = dt;
        printf("  Run %2d : %9.3f ms\n", r + 1, dt);
    }

    qsort(times, (size_t)runs, sizeof(double), cmp_double);
    double median = (runs & 1) ? times[runs / 2] : 0.5 * (times[runs / 2 - 1] + times[runs / 2]);
    printf("------------------------------------------------------------------------\n");
    printf("  Best   : %9.3f ms\n", times[0]);
    printf("  Median : %9.3f ms\n", median);
    printf("  Worst  : %9.3f ms\n", times[runs - 1]);
#if !defined(_WIN32)
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
  #if defined(__APPLE__)
        printf("  PeakRSS: %9.1f MB\n", (double)ru.ru_maxrss / (1024.0 * 1024.0));
  #else
        printf("  PeakRSS: %9.1f MB\n", (double)ru.ru_maxrss / 1024.0);
  #endif
    }
#endif

    if (out_path) {
        FILE *fo = fopen(out_path, "wb");
        if (!fo || fwrite(out, 8, n_out, fo) != n_out) {
            perror(out_path);
            if (fo) fclose(fo);
            exit_code = 3;
            goto cleanup;
        }
        fclose(fo);
        printf("  Output : wrote %zu limbs to %s\n", n_out, out_path);
    }

    exit_code = 0;
    if (verify_gmp) {
        gmp_mul_fn gmp_mul = load_gmp_mul();
        if (!gmp_mul) {
            fprintf(stderr, "  GMP    : libgmp not found on dynamic library path (skipped)\n");
        } else {
            ref = alloc_u64(n_out);
            if (!ref) {
                fprintf(stderr, "Error: out of memory allocating GMP reference buffer.\n");
                exit_code = 1;
                goto cleanup;
            }
            double tg0 = now_ms();
            if (na >= nb) gmp_mul(ref, a, (intptr_t)na, b, (intptr_t)nb);
            else          gmp_mul(ref, b, (intptr_t)nb, a, (intptr_t)na);
            double tg = now_ms() - tg0;
            int exact = (memcmp(out, ref, n_out * sizeof(uint64_t)) == 0);
            printf("  GMP 1T : %9.3f ms | Exact bit-for-bit match: %s\n",
                   tg, exact ? "YES (100%)" : "NO (MISMATCH!)");
            if (!exact) exit_code = 4;
        }
    }
    printf("========================================================================\n");

cleanup:
    free_u64(ref);
    free(times);
    free_u64(out);
    if (b && b != a) free_u64(b);
    free_u64(a);
    return exit_code;
}
