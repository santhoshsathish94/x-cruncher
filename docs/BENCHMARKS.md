# Reproducible Benchmarks (`x-cruncher v1.6.1`)

Reproducibility is a first-class requirement of `x-cruncher`. Every benchmark reported in this repository includes the exact hardware, OS/kernel, compiler flags, thread count, and shell commands needed for another developer to verify both **speed** and **bit-for-bit exactness**.

---

## 1. Quick Reproduction Commands

### A. Built-In Multi-Scale Benchmark (`1K` to `100M` Decimal Digits + Zero-Storage $\pi$)
```bash
git clone https://github.com/santhoshsathish94/x-cruncher.git
cd x-cruncher
make clean all
make test

# Quick suite (1K, 100K, 1M, 10M digits + 1M/10M Pi hex):
./bench_xcruncher

# Full suite including 100,000,000 x 100,000,000 digits (200M output digits) + 100M/1B Pi hex:
./bench_xcruncher --100m

# Single-threaded apples-to-apples comparison against single-threaded GNU GMP:
OMP_NUM_THREADS=1 ./bench_xcruncher --100m

# 4-thread pinned run (e.g., matching a 4-vCPU cloud VM):
OMP_NUM_THREADS=4 ./bench_xcruncher --100m
```

If `libgmp` (`libgmp.so.10` on Linux, `libgmp-10.dll` on Windows, `libgmp.dylib` on macOS) is installed on the host, `bench_xcruncher` dynamically loads `__gmpn_mul_n`, times live `GNU GMP` on the **exact same operand buffers**, and verifies `memcmp == 0` across every output limb.

### B. Identical-Operand Raw-Limb Harness (`bench/repro_raw_limbs.c`)
For external head-to-head validation on arbitrary binary limb files (such as exported `100M`-digit $\pi$ limbs converted to little-endian `uint64_t[]`):
```bash
cc -O3 -std=c11 -Wall -Wextra -fopenmp -Iinclude \
   bench/repro_raw_limbs.c src/xcruncher.c -o repro_raw_limbs -fopenmp -lm -ldl

# Synthetic 100M-digit (5,190,513 limbs) 7-run squaring benchmark + live GMP verification:
./repro_raw_limbs --synthetic-100m --runs 7 --verify-gmp

# External identical-operand file squaring (A * A) or multiplication (A * B):
./repro_raw_limbs pi_100m_limbs.bin --runs 7 --verify-gmp -o x_pi_sq.bin
./repro_raw_limbs limbs_a.bin limbs_b.bin --runs 7 --verify-gmp
```

---

## 2. Primary Reference System & Recorded Results (`v1.6.0`)

### Hardware & Software Environment
- **CPU:** `AMD Ryzen 9 7950X3D` (`16` physical Zen 4 cores / `32` threads, `AVX-512 IFMA52 + VBMI`, DDR5)
- **OS / Kernel:** Ubuntu 24.04 LTS, `Linux 6.8.0-147-generic x86_64`
- **Compiler & Flags:** `gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`, `-O3 -Wall -Wextra -fopenmp -Iinclude` (runtime CPU dispatch; no `-march=native` required)
- **Reference Library:** `GNU GMP 6.3.0` (`libgmp10:amd64 2:6.3.0+dfsg-2ubuntu6.1`, `__gmpn_mul_n`)

### Table 2A — 16 Physical Cores (Default Auto-Topology, `./bench_xcruncher --100m`)

| Input Size (Decimal Digits) | Limbs (`uint64_t`) | `GNU GMP 6.3.0` (`1T`) | `x-cruncher` Cold `A * B` | `x-cruncher` Square `A * A` | `x-cruncher` `Trunk(B) * A` | `x-cruncher` Sparse `NAF` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`1,024 × 1,024`** | `54` | `0.02 ms` | `0.0024 ms` | `0.0024 ms` | — | `< 0.001 ms` |
| **`100,000 × 100,000`** | `5,191` | `0.78 ms` | `0.18 ms` (`4.3×`) | `0.13 ms` (`6.1×`) | `0.13 ms` (`6.1×`) | `0.01 ms` (`94×`) |
| **`1,000,000 × 1,000,000`** | `51,906` | `9.85 ms` | `0.85 ms` (`11.6×`) | `0.59 ms` (`16.8×`) | `0.54 ms` (`18.1×`) | `0.08 ms` (`130×`) |
| **`10,000,000 × 10,000,000`** | `519,052` | `133.57 ms` | `5.02 ms` (`26.6×`) | `3.38 ms` (`39.5×`) | `3.85 ms` (`34.7×`) | `0.49 ms` (`274×`) |
| **`100,000,000 × 100,000,000`** | `5,190,513` | `1,701.74 ms` | **`59.83 ms` (`28.4×`)** | **`38.18 ms` (`44.6×`)** | **`44.32 ms` (`38.4×`)** | **`6.38 ms` (`267×`)** |

### Table 2B — Single-Thread Apples-to-Apples (`OMP_NUM_THREADS=1 ./bench_xcruncher --100m`)

Because `GNU GMP`'s `__gmpn_mul_n` is single-threaded, running `x-cruncher` with `OMP_NUM_THREADS=1` isolates pure algorithmic + `AVX-512 IFMA52` instruction-level speedup without multi-core parallelism:

| Input Size (Decimal Digits) | `GNU GMP 6.3.0` (`1T`) | `x-cruncher` Cold `A * B` (`1T`) | `x-cruncher` Square `A * A` (`1T`) | `x-cruncher` `Trunk(B) * A` (`1T`) |
| ---: | ---: | ---: | ---: | ---: |
| **`100,000 × 100,000`** | `0.75 ms` | `0.81 ms` (`0.9×`) | `0.46 ms` (`1.6×`) | `0.44 ms` (`1.7×`) |
| **`1,000,000 × 1,000,000`** | `9.65 ms` | `4.26 ms` (`2.3×`) | `2.95 ms` (`3.3×`) | `2.93 ms` (`3.3×`) |
| **`10,000,000 × 10,000,000`** | `124.40 ms` | `42.42 ms` (`2.9×`) | `28.92 ms` (`4.3×`) | `29.74 ms` (`4.2×`) |
| **`100,000,000 × 100,000,000`** | `1,705.92 ms` | **`410.47 ms` (`4.2×`)** | **`288.63 ms` (`5.9×`)** | **`299.45 ms` (`5.7×`)** |

### Table 2C — 4-Thread Pinned (`2C/4T` Zen 4 Slice: `taskset -c 0,16,1,17`, 7-Run One-Shot `100M` Square)

Command: `taskset -c 0,16,1,17 ./repro_raw_limbs --synthetic-100m --runs 7 --verify-gmp` (`5,190,513` limbs $\times$ `5,190,513` limbs $\to$ `10,381,026` exact output limbs):

| Run | `2C/4T` (`taskset -c 0,16,1,17`) | `16C/32T` (Full Socket) | Notes |
| :---: | ---: | ---: | :--- |
| **Run 1** | `210.577 ms` | `110.300 ms` | Includes initial OS `mmap` + `2 MB` THP page-fault allocation (`~268 MB` NTT + `83 MB` output) |
| **Run 2** | `141.568 ms` | `38.309 ms` | Cached workspace (`g_cached_ntt`) |
| **Run 3** | `140.084 ms` | `36.798 ms` | Cached workspace |
| **Run 4** | `140.380 ms` | `37.150 ms` | Cached workspace |
| **Run 5** | `138.359 ms` | `36.960 ms` | Cached workspace |
| **Run 6** | `138.683 ms` | `36.627 ms` | Cached workspace |
| **Run 7** | `141.312 ms` | `37.811 ms` | Cached workspace |
| **Best** | **`138.359 ms`** | **`36.627 ms`** | `EXACT_VS_GMP = 1` (`memcmp == 0`) |
| **Median** | **`140.380 ms`** | **`37.150 ms`** | `EXACT_VS_GMP = 1` (`memcmp == 0`) |
| **Worst** | `210.577 ms` | `110.300 ms` | Cold first-call allocation |

---

## 3. Secondary Hardware Validation (`AVX2+FMA` Laptop & `Android 14 / ARM64`)

| Platform | Compiler & OS | `1M × 1M` (`Trunk` / `GMP`) | `10M × 10M` (`Trunk` / `GMP`) | `100M × 100M` (`Trunk` / `Square` / `GMP`) | `100M` $\pi$ Hex |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **`Intel Core i7-1355U` (2P+8E, 15W)** | MinGW GCC 16 / Windows 11 (`AVX2+FMA`) | `4.52 ms` / `12.43 ms` | `36.65 ms` / `213.0 ms` (`5.8×`) | `1,139 ms` / `1,587 ms` / `5,010 ms` (`4.4×`) | `1,772 ms` (`20/20` Exact) |
| **Android 14 (`ARM64 NEON` / `NDK r27`)** | `aarch64-linux-android29-clang -O3 -fopenmp` | `2.53 ms` | `28.20 ms` | *(in-RAM mobile tier)* | `20/20` Exact |

---

## 4. Comparison Methodology & Important Caveats (`GNU GMP` & `y-cruncher`)

To interpret benchmark comparisons fairly and accurately:

1. **Single-Thread vs Multi-Thread (`GNU GMP`):**
   - `GNU GMP`'s low-level `mpn_mul` / `__gmpn_mul_n` runs on a **single CPU thread**, whereas `x-cruncher` uses OpenMP multi-threading by default above `16,384` transform elements. Always check both Table 2A (multi-core) and Table 2B (`OMP_NUM_THREADS=1`) depending on whether your workload is single-threaded or multi-threaded.
2. **Full Integer Product vs Truncated `BigFloat` Precision (`y-cruncher`):**
   - In `y-cruncher`'s custom formula `Multiply` mode configured at `Decimal Digits: 100,000,000` (`-dec:100m`), `y-cruncher` computes a **floating-point product truncated to `100,000,000` decimal digits** (and operates on its internal high-precision radix representation).
   - `x-cruncher` computes the **complete `200,000,000`-digit (`~664.4M`-bit) exact integer product** from `64-bit` binary limbs (`uint64_t[]`). While `100M`-digit $\pi \times \pi$ tests use the same mathematical scale, the internal binary representations and output precisions are not bit-for-bit identical workloads.
3. **Cold First Call vs Subsequent Calls (Demand Paging):**
   - The very first large `xc_mul_u64` call in a fresh OS process allocates `~268 MB` (squaring) or `~536 MB` (`A * B`) via `mmap` and faults pages into physical RAM (`Run 1`: `~210 ms` on 4 threads / `~110 ms` on 16 threads). Subsequent calls reuse the lock-free 1-slot cached workspace (`Run 2+`: `~140 ms` on 4 threads / `~37–38 ms` on 16 threads). Applications performing a single one-shot multiplication before process exit will observe the `Run 1` cold-page cost.
4. **Cloud VM & SMT Scheduling Variance (`Best` vs `Median`):**
   - Virtualized cloud instances (`4 vCPU` quotas, shared L3 caches, hypervisor steal time, or background `khugepaged` compaction) typically exhibit `5%–25%` higher run-to-run variance than bare-metal pinned cores.
   - Note the reporting distinction between the two benchmark tools: `bench_xcruncher` reports **best-of-repetitions** (designed to filter OS scheduling interrupts on sub-millisecond `1K–1M` kernels), whereas `repro_raw_limbs` reports **Best, Median, and Worst** across `N` runs (recommended whenever publishing `10M–100M` multi-core comparisons).
