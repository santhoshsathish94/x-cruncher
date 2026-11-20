# Contributing to x-cruncher

Thank you for your interest in inspecting, benchmarking, or contributing to `x-cruncher`! Because `x-cruncher` is an exact arithmetic engine, **deterministic correctness and reproducible measurements always take precedence over raw speed.**

---

## 1. Non-Negotiable Correctness Requirements

1. **100% Deterministic Worst-Case Exactness:**
   - Every multiplication path (`xc_mul_u64`, `xc_mul_u64_lowmem`, `xc_plan_mul`, `xc_mul_trunk`) must return the **exact** integer product for **every possible input bit-pattern** within its documented size bound — never only "with high probability" on random inputs.
   - Any floating-point (`IEEE-754 binary64`) transform change must include an explicit worst-case bound on $\max_k |c_k|$ under adversarial non-canceling limb saturation (e.g., all balanced limbs equal to $\pm 2^{B-1}$) keeping total rounding error strictly below $0.25 < 0.5$, plus an adversarial saturation test in [`tests/test_xcruncher.c`](tests/test_xcruncher.c).
   - Any modular NTT / CRT change must prove $\max_k |c_k| \le \lfloor (M - 1) / 2 \rfloor$ (signed) or $< M$ (unsigned) across the full supported limb count and test both positive and negative CRT saturation extremes.

2. **Mandatory Test & Sanitizer Gate:**
   Before opening a pull request, all of the following must pass with **zero warnings and zero leaks**:
   ```bash
   # 1. Release build + C & Python test suites:
   make clean && make test

   # 2. Strict C11 + AddressSanitizer + UBSan + LeakSanitizer:
   cc -O3 -g -std=c11 -Wall -Wextra -Werror -fsanitize=address,undefined \
      -fno-omit-frame-pointer -fopenmp -Iinclude \
      tests/test_xcruncher.c src/xcruncher.c -lm -o test_san
   ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
   UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 ./test_san
   rm -f test_san
   ```

---

## 2. Coding & Portability Conventions

- **Zero External Runtime Dependencies:** Keep [`src/xcruncher.c`](src/xcruncher.c) and [`include/xcruncher.h`](include/xcruncher.h) self-contained ISO C11 (using only standard C headers, OpenMP when `_OPENMP` is enabled, and optional OS virtual-memory hints). Keep [`python/xcruncher/__init__.py`](python/xcruncher/__init__.py) pure stdlib `ctypes`.
- **Single-Binary Runtime ISA Dispatch:** Never require `-march=native` to build `src/xcruncher.c`. Use function-level target attributes (`XC_TARGET_IFMA_NTT`, `XC_TARGET_IFMA52`, `XC_TARGET_AVX512`, `XC_TARGET_AVX2`) guarded by runtime CPU feature checks (`__builtin_cpu_supports`), and always keep the portable scalar fallback working.
- **No Local Stack Arrays Inside `XC_TARGET_*` Functions:** GCC 14 at `-O3 -fsanitize=address` can emit `64-byte`-aligned `vmovdqa64` stack loads/stores on ASan-instrumented stack frames. Keep SIMD temporaries in vector registers (`__m512i` / `__m256d`) or pre-allocated `64-byte`-aligned heap scratch buffers.
- **No `#pragma omp` Inside `XC_TARGET_*` Functions:** Clang outlines `#pragma omp` regions into helper functions that do not inherit `__attribute__((target(...)))`. Place OpenMP parallel loops in non-target wrapper functions that call `XC_TARGET_*` tile/row kernels.
- **Windows `LLP64` Safety:** On 64-bit Windows, `long` is 32-bit. Always use `size_t`, `uint64_t`, `int64_t`, or `intptr_t` for limb counts, byte offsets, and `GMP` `mp_size_t` FFI signatures.
- **Library-Friendly Threading:** Never call `sched_setaffinity` or mutate the host application's thread affinity mask inside library functions, and always honor `OMP_NUM_THREADS` when set by the caller.

---

## 3. Benchmark Requirements for Performance PRs

Every pull request claiming a performance improvement must include reproducible before/after numbers in the PR description:

1. **CPU model, core/thread count, and OS/kernel** (e.g., `AMD Ryzen 9 7950X3D`, `OMP_NUM_THREADS=4` and full-socket, `Linux 6.8`).
2. **Compiler version and exact flags** (e.g., `GCC 13.3.0`, `-O3 -Wall -Wextra -fopenmp`).
3. **Exact benchmark command** (`./bench_xcruncher --100m` and/or `./repro_raw_limbs --synthetic-100m --runs 7 --verify-gmp`).
4. **Best, Median, and Worst** timings across at least `5–7` runs, confirming `EXACT` / `memcmp == 0` against `GNU GMP` or `ref_mul`.

---

## 4. High-Impact Areas Open for Community Contributions

- **Dedicated `AVX2` 3-Prime `30-Bit` / `31-Bit` Integer NTT:** Closing the gap on pre-`IFMA52` x86-64 CPUs (`Haswell`–`Raptor Lake`, `Zen 1`–`Zen 3`) at `10M–100M` digits.
- **`ARM64 NEON` / `SVE2` Integer NTT:** Accelerating `1M–100M`-digit exact integer multiplication on Apple Silicon (`M1–M4`), AWS Graviton, and Android devices.
- **Sub-Quadratic Base Conversion ($O(M(N) \log N)$ Divide-and-Conquer):** Replacing the $O(N^2)$ `xc_from_decimal` / `xc_to_decimal` convenience helpers for multi-million-digit decimal I/O.
- **Newton–Raphson Big-Integer Division & Square Root:** Building on `xc_trunk_t` (`WaveTrunk` / `NTT Trunk`) for fast reciprocal iteration.
