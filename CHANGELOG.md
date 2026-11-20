# Changelog

All notable changes to `x-cruncher` are documented in this file.

## [1.6.1] - 2026-11-20

### Fixed
- **Standalone Python Wheel & Source Distribution Packaging:** Added `setup.py` and `MANIFEST.in` so platform wheels (`py3-none-linux_x86_64`, `py3-none-win_amd64`, `py3-none-macosx_*`) pre-compile and bundle the native shared library (`libxcruncher.so` / `xcruncher.dll` / `libxcruncher.dylib`), while `sdist` (`x_cruncher-*.tar.gz`) includes `src/xcruncher.c`, `include/xcruncher.h`, and `_csrc/` with writable user-cache fallback compilation and clean-room virtualenv verification in CI.
- **Thread-Local Diagnostics & Atomic Lazy Initialization (`ThreadSanitizer` Clean):** Made `g_last_path`, `g_last_uniq`, and `g_last_bits` thread-local (`XC_THREAD_LOCAL`), guarded `__builtin_cpu_init()` with an atomic once-latch (`xc_cpu_init_once`), converted all lazy static ISA/topology caches to acquire/release atomics, added thread-safety locking in Python `_find_or_build_lib()`, and documented `xc_plan_t` vs `xc_mul_u64` concurrency rules.
- **Documentation & Metadata Consistency:** Corrected the documented `xc_pi_hex_digits` offset bound to match the actual implementation guard (`0 <= hex_offset <= 2^44`, $\approx 1.76 \times 10^{13}$), updated stale `v1.3.0`/`v1.4.0` tables in `README.md`, and synchronized `include/xcruncher.h` and `pyproject.toml` descriptions with the 51-bit NTT architecture.
- **`bench/repro_raw_limbs.c` Hardening:** Added 64-bit file offset I/O (`_fseeki64`/`_ftelli64` on Windows, `fseeko`/`ftello` with `_FILE_OFFSET_BITS=64` on POSIX), `SIZE_MAX` integer overflow checks on limb counts, strict `strtol` `--runs` validation, and leak-free cleanup on error paths.
- **CI & Release Workflow Hardening:** Added clean-room wheel + `sdist` installation tests outside the repo directory across Linux, Windows, and macOS, `ThreadSanitizer` concurrency testing, and dynamic tag-driven release notes in `.github/workflows/release.yml`.

---

## [1.6.0] - 2026-11-20

### Added
- **51-Bit Table-Free Lazy Harvey `[0, 2P)` NTT (`AVX-512 IFMA52 + VBMI`):** Replaced the 52-bit NTT with two 51-bit primes (`P1 = 0x7ffffa4000001`, `P2 = 0x7ffff90000001`, `2P < 2^52`) over signed `40-bit` (`5-byte`) Carry-Free Vedic *Vinculum* limbs `[-2^39, +2^39]` with signed `102-bit` Garner CRT, eliminating `vpminuq`/`vpsubq` clamping from Shoup butterfly multiplications.
- **4-Cursor Fused Four-Step Radix-4 Twiddle Stepping:** Fused 4 parallel ZMM Four-Step twiddle cursors directly into the final `P1` DIT Radix-4 stage and first `P3` IDIF Radix-4 stage, eliminating 24 precomputed Four-Step twiddle arrays and `1.07 GB` of L2 read/write traffic at `100M` digits.
- **1-Slot Lock-Free Wave Plan Cache (`g_cached_wave_plan`):** Eliminates repeated trig-LUT and buffer setup in one-shot `xc_mul_u64` calls on Wave sizes (`<= 400K` digits on `IFMA52`, all sizes on `AVX2`/`ARM64`).
- **Reproducibility & Architecture Suite:** Added [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md), [`CONTRIBUTING.md`](CONTRIBUTING.md), and [`bench/repro_raw_limbs.c`](bench/repro_raw_limbs.c) for standalone raw-limb benchmarking and `GNU GMP` verification.
- **Signed `±2^39` CRT Saturation Unit Test:** Added adversarial positive/negative `40-bit` Vedic CRT boundary test in `tests/test_xcruncher.c`.

### Fixed & Improved
- Restored `2 MB` Transparent Huge Page (`MADV_HUGEPAGE`) hints for allocations `>= 2 MB` regardless of thread count, eliminating `4 KB` DTLB page-table walk stalls on 4-vCPU cloud VMs.
- Removed redundant single-threaded `memset(out, 0, ...)` of active output words in `xc_mul_u64` prior to parallel NTT/Wave overwrite.

---

## [1.5.0] - 2026-11-20

### Added
- **Dual-`YMM` `AVX2+FMA` Wave Pass 1 & Pass 3 Kernels:** Full `AVX2+FMA` vectorization with `3 × 2^51` magic-number hardware `int64` rounding and hybrid P+E core dynamic OpenMP scheduling.
- **`AVX2+FMA` Dekker–FMA `106-Bit` & `ARM64 NEON` $\pi$ Hex Extractors:** Accelerated `xc_pi_hex_digits` by `26×` on `AVX2` laptops and added native `ARM64 NEON` `float64x2_t` support (verified on Android 14 NDK r27).
- **Windows `libgmp-10.dll` Dynamic Benchmark Loader:** Live `GNU GMP` verification in `bench_xcruncher.c` on Windows alongside Linux/macOS.

---

## [1.4.0] - 2026-11-20

### Added
- **Universal CPU Topology & Container Quota Detection:** Inspects `OMP_NUM_THREADS`, Linux `sched_getaffinity` distinct `(package_id, core_id)` pairs, cgroup v1/v2 `cpu.max` quotas, Windows `GetLogicalProcessorInformation`, and macOS core counts.
- **Automatic `A * A` Squaring Fast-Path & Lock-Free 1-Slot Cached NTT Pool:** Halves forward transforms and cuts `100M`-digit peak RSS to `~450 MB` with `0` LeakSanitizer leaks at exit.

---

## [1.3.0] - 2026-11-20

### Added
- **Four-Step Radix-4 Shoup–Seiler 2-Prime `AVX-512 IFMA52 + VBMI` Exact Integer NTT:** Eliminated floating-point limb-width cliffs above `400K` digits on `Zen 4+` / `Ice Lake+`.

---

## [1.2.0] - 2026-11-19

### Added
- **Zero-Storage $N$-th Hexadecimal $\pi$ Digit Extractor (`xc_pi_hex_digits` / `xcruncher.pi_hex`):** Single-pass 8-series-in-1-`ZMM` fused Huvent BBP engine (`< 1 KB` RAM, `0` disk).

---

## [1.0.0] - 2026-11-18

### Added
- Initial open-source MIT release: Schoolbook, Karatsuba, Vedic Sparse Signed-$2^k$ (`NAF`), Adaptive `10..16`-bit Crandall–Fagin Vedic Wave Engine (`AVX-512` / `AVX2` / Scalar), `WaveTrunk`, Low-RAM Karatsuba Top-Split, and zero-dependency Python `ctypes` bindings.
