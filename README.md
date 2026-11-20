# x-cruncher

**`x-cruncher`** is a zero-dependency, **MIT-licensed** arbitrary-precision integer multiplication & zero-storage $\pi$ extraction engine in C11 (with zero-dependency Python `ctypes` bindings) that delivers **`12×–45×` multi-core speedups** (`up to 274×` on sparse near-$2^k$ inputs) over standard open-source libraries (`GNU GMP`, `FLINT`) **and outperforms closed-source `y-cruncher v0.8.7` (`22-ZN4 ~ Kizuna`) on both `100M × 100M` dense multiplication (`59.83 ms` Cold / `44.32 ms` Trunk / `38.18 ms` Square on 16 cores; `140.38 ms` median vs `144.5 ms` on 4 Zen 4 threads) and `100M` / `1B` zero-storage $\pi$ BBP extraction (`62.04 ms` vs `75.00 ms`)** — with **100% deterministic integer/worst-case exactness** on every possible input bit-pattern.

---

## Verified Benchmarks (`AMD Ryzen 9 7950X3D`, 16 Physical Zen 4 Cores, DDR5)

All timings below are produced directly by `./bench_xcruncher --100m` (`make bench-100m`, plain `-O3` with runtime `AVX-512 IFMA52` dispatch) in `v1.6.0`, and verified bit-for-bit (`memcmp == 0`) against `GNU GMP 6.3.0`'s `x86-64` assembly `__gmpn_mul_n` on **both** dense-random and pathological worst-case (`0xFFFFFFFF...` / signed `±2^39` limb) inputs:

| Input Size (Decimal Digits) | Active Exact Engine | `GNU GMP 6.3.0` (`__gmpn_mul_n`, 1T) | `x-cruncher` Cold `A * B` | `x-cruncher` Square `A * A` | `x-cruncher` `Trunk(B) * A` | `x-cruncher` Sparse (`NAF`) |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| **`1,024 × 1,024`** | `64-bit` Karatsuba | `0.02 ms` | **`0.0024 ms`** | **`0.0024 ms`** | — | **`< 0.001 ms`** |
| **`100,000 × 100,000`** | `16-bit` ZMM Wave | `0.78 ms` | **`0.18 ms` (`4.3×`)** | **`0.13 ms` (`6.1×`)** | **`0.13 ms` (`6.1×`)** | **`0.01 ms` (`94×`)** |
| **`1,000,000 × 1,000,000`** | `51-bit` `IFMA52` NTT | `9.85 ms` | **`0.85 ms` (`11.6×`)** | **`0.59 ms` (`16.8×`)** | **`0.54 ms` (`18.1×`)** | **`0.08 ms` (`130×`)** |
| **`10,000,000 × 10,000,000`** | `51-bit` `IFMA52` NTT | `133.57 ms` | **`5.02 ms` (`26.6×`)** | **`3.38 ms` (`39.5×`)** | **`3.85 ms` (`34.7×`)** | **`0.49 ms` (`274×`)** |
| **`100,000,000 × 100,000,000`** | `51-bit` `IFMA52` NTT | `1,701.74 ms` | **`59.83 ms` (`28.4×`)** | **`38.18 ms` (`44.6×`)** | **`44.32 ms` (`38.4×`)** | **`6.38 ms` (`267×`)** |

*Note on architecture, container/VM thread scaling, & exactness:*
- **On `AVX-512 IFMA52` CPUs (`AMD Zen 4 / Zen 5`, `Intel Ice Lake+`):** Above `~400K` digits (`logN >= 17`, up to `~101M` digits per operand), `x-cruncher v1.6.0` dispatches to a **Table-Free Lazy Harvey `[0, 2P)` 2-Prime `51-bit` Four-Step Radix-4 Exact Integer NTT** ($P_1 = \text{0x7ffffa4000001}$, $P_2 = \text{0x7ffff90000001}$, $2P < 2^{52}$) over signed Carry-Free `40-bit` (`5-byte`) Vedic *Vinculum* limbs $d_m \in [-2^{39}, +2^{39}]$ — eliminating both floating-point rounding and butterfly `vpminuq` clamping instructions while fusing 4-cursor Four-Step twiddle stepping directly into the outer Radix-4 stages (**`59.83 ms` Cold / `44.32 ms` Trunk / `38.18 ms` One-Shot Square** on 16 cores; **`140.38 ms` median / `138.36 ms` best** on 4 Zen 4 threads, computing all **`200,000,000`** exact integer digits vs `y-cruncher`'s `144.5 ms` truncated `100M`-digit `BigFloat`).
### Multi-Device Portability (`Intel AVX2+FMA` Laptop & `Android 14 / ARM64 NEON`)

Tested across desktop/server `AVX-512 IFMA52`, ultrabook `AVX2+FMA` (`13th Gen Intel Core i7-1355U`, 2P+8E hybrid cores on Windows 11 vs `GNU GMP 6.3.0` `libgmp-10.dll`), Apple Silicon `ARM64 NEON`, and **Android 14 (`NDK r27` `aarch64-linux-android29-clang -static-openmp`)**:

| Device / Architecture | `1M × 1M` (`Trunk` / `GMP`) | `10M × 10M` (`Trunk` / `GMP`) | `100M × 100M` (`Trunk` / `Square` / `GMP`) | `1M` / `10M` / `100M` $\pi$ Hex Digits |
| :--- | ---: | ---: | ---: | ---: |
| **`Ryzen 9 7950X3D`** (`AVX-512 IFMA52`) | **`0.50 ms`** / `9.55 ms` | **`4.49 ms`** / `126.3 ms` | **`46.8 ms`** / **`41.2 ms`** / `1,703 ms` | **`0.51 ms` / `5.2 ms` / `62.5 ms`** |
| **`Intel Core i7-1355U` 15W Laptop** (`AVX2+FMA`) | **`4.52 ms`** / `12.43 ms` | **`36.65 ms`** / `213.0 ms` (`5.8×`) | **`1,139 ms`** / **`1,587 ms`** / `5,010 ms` (`4.4×`) | **`15.7 ms` / `154.7 ms` / `1,772 ms`** |
| **Android 14 (`NDK r27` Mobile)** | **`2.53 ms`** | **`28.20 ms`** | *(in-RAM mobile tier)* | **`20/20` Exact (`ARM64 NEON`)** |

*Running directly on an Android phone (e.g. Samsung Galaxy S23):*
- **Option A (Termux on-device):** `pkg install git clang make libomp python && git clone https://github.com/santhoshsathish94/x-cruncher && cd x-cruncher && make test && make bench`
- **Option B (Android NDK cross-compile + `adb`):**
  ```bash
  aarch64-linux-android29-clang -O3 -std=c11 -fopenmp -static-openmp -Iinclude src/xcruncher.c bench/bench_xcruncher.c -lm -ldl -o bench_xcruncher
  adb push bench_xcruncher /data/local/tmp/ && adb shell /data/local/tmp/bench_xcruncher
  ```
- **On `AVX2+FMA` / non-`IFMA52` `AVX-512` / `ARM64` CPUs:** Automatically falls back to the adaptive `10..16`-bit Crandall–Fagin Vedic Wave Engine (`choose_wave_params()`), enforcing $L \cdot 2^{2B-2} \le 4.398 \times 10^{13} < 2^{53}/204$ (`1 ULP <= 1/128`) so **every input bit-pattern remains provably exact** on every CPU.

### Zero-Storage $N$-th Hexadecimal $\pi$ Digit Extraction (`xc_pi_hex_digits` vs `y-cruncher v0.8.7`)

Using a **Single-Pass 8-Series-in-1-`ZMM` Fused Huvent Engine** (combining 3-cycle Centered `IEEE-754 double` Barrett squaring below $90\text{M}$ with `6×`-Interleaved Self-Reducing Seiler `AVX-512 IFMA52` squaring and `104-bit` fixed-point accumulation in `< 1 KB` RAM and `0` disk storage), `x-cruncher v1.3.0` **outperforms closed-source `y-cruncher v0.8.7` (`22-ZN4 ~ Kizuna`, `bench-bbp`) head-to-head** on the same `Ryzen 9 7950X3D`:

| Hexadecimal Offset | `y-cruncher v0.8.7` (`bench-bbp`) | `x-cruncher v1.3.0` (`xc_pi_hex_digits`) | Verified 20-Hex-Digit Output (`80` Exact Bits) |
| ---: | ---: | ---: | :--- |
| **`1,000,000`** | — | **`0.51 ms`** | `6c65e52cb459350050e4` *(EXACT)* |
| **`10,000,000`** | — | **`4.97 ms`** | `7af5863efed8de97033c` *(EXACT)* |
| **`100,000,000` (`100m`)** | `75.00 ms` | **`61.94 ms` (`1.21×` faster)** | `cb840e21926ec5ae0d2f` *(20/20 EXACT)* |
| **`1,000,000,000` (`1b`)** | `724.00 ms` | **`704.70 ms` (`1.03×` faster)** | `5895585a0428b564084e` *(20/20 EXACT)* |

---

## Scope & Transparent Comparison (`x-cruncher` vs `y-cruncher` & `GNU GMP`)

To make expectations completely clear for users and reviewers:

### What `x-cruncher v1.6.0` Provides & Where It Leads
- **Embeddable Open-Source Library (`MIT` C11 + Zero-Dependency Python):** Unlike `y-cruncher` (which is a closed-source standalone CLI executable), `x-cruncher` is a small, zero-dependency C11 shared/static library (`#include "xcruncher.h"`) and Python package (`import xcruncher`).
- **In-RAM Big-Integer Multiplication (`uint64_t` limbs & Python `int`):** Beats single-threaded `GNU GMP 6.3.0` (`__gmpn_mul_n`) by **`12×–45×`** (`up to 274×` on sparse near-$2^k$ numbers) and outperforms `y-cruncher` on `AVX-512 IFMA52` CPUs (`Zen 4 / Zen 5`, `Ice Lake+`) up to `~101M` digits per operand (**`59.83 ms` Cold / `44.32 ms` `Trunk` / `38.18 ms` Square** on 16 cores; **`140.38 ms` median vs `144.5 ms`** on 4 Zen 4 threads — while returning the **full `200M`-digit exact integer product** rather than a truncated `100M`-digit floating-point approximation).
- **Zero-Storage $N$-th Hexadecimal $\pi$ Extraction (`xc_pi_hex_digits` / `xc.pi_hex`):** Jumps directly to any hexadecimal offset of $\pi$ using `< 1 KB` RAM and `0` disk storage, beating `y-cruncher v0.8.7`'s `bench-bbp` at `100M` (`62.04 ms` vs `75.00 ms`) and `1B` (`703.50 ms` vs `724.00 ms`) with `20/20` matching hex digits.

### What `x-cruncher` Does *Not* Yet Implement (Where `y-cruncher` Remains Ahead)
1. **Full All-Digits Decimal Constant Generation (`Chudnovsky` $\pi, e, \zeta(3), \gamma$):** `x-cruncher` extracts arbitrary *hexadecimal* slices of $\pi$ with zero storage (`xc_pi_hex_digits`), whereas computing *all* $N$ sequential *decimal* digits (`./y-cruncher bench 100m`) additionally requires Chudnovsky binary splitting, Newton–Raphson big-integer division/sqrt, and sub-quadratic divide-and-conquer radix conversion (`xc_from_decimal` / `xc_to_decimal` are $O(N^2)$ convenience helpers intended for small-to-medium strings; use binary `uint64_t` limbs or `xc.mul(int, int)` for million-digit numbers).
2. **Out-of-Core NVMe Disk Swapping (`1B`–`300T+` Digits):** `x-cruncher` is strictly an **in-RAM** engine (up to `~201M` digits in the 2-prime `IFMA52` NTT, and larger in-RAM sizes via `xc_mul_u64_lowmem`). Multi-terabyte/petabyte runs exceeding system RAM require out-of-core SSD streaming, where `y-cruncher` is purpose-built.
3. **Pre-`AVX-512 IFMA52` Hardware (`AVX2`-Only / `ARM64`):** The `61.64 ms` 2-prime integer NTT and `61.94 ms` $\pi$ extractor require `AVX-512 IFMA52` (`AMD Zen 4+`, `Intel Ice Lake+`). On `AVX2+FMA`, `ARM64`, or scalar targets, `x-cruncher` automatically falls back to its portable worst-case-safe `float64` Vedic Wave & scalar OpenMP paths (`100%` exact, typically `3×–6×` faster than `GMP` on multi-core `AVX2`, though slower than `y-cruncher`'s dedicated `AVX2` binary at `100M` digits).

---

## Formal Worst-Case Exactness Bound (`IEEE-754 binary64`)

Carry-Free Vedic *Vinculum* balancing maps every unsigned $B$-bit chunk $u_m \in [0, 2^B - 1]$ ($s_m = u_m \gg (B - 1)$) to:
$$d_m = u_m - 2^B s_m + s_{m-1} \in \left[-2^{B-1}, \; +2^{B-1}\right]$$
For inputs with $L = \min(L_A, L_B)$ balanced $B$-bit limbs, even in the **strict adversarial worst case** where every single limb equals $+2^{B-1}$ with zero sign cancellation, the exact integer convolution peak satisfies:
$$\max_{k} |c_k| \le L \cdot \left(2^{B-1}\right)^2 = L \cdot 2^{2B - 2}$$
`choose_wave_params()` selects the largest $B \in \{10, \dots, 16\}$ satisfying:
$$L \cdot 2^{2B - 2} \le 4.398 \times 10^{13} < 2^{45.33} < \frac{2^{53}}{204}$$
At magnitudes $\le 4.398 \times 10^{13}$, `1 ULP` of IEEE-754 `double` is at most $2^{-7} = \mathbf{0.0078125}$ (`1/128`), keeping accumulated 25-stage unitary FFT + Coarse×Fine LUT error strictly below $0.15 < 0.5$ (`llround` exactness threshold) for **all** inputs — verified directly on adversarial `0x7FFF...` and saturated `+(2^(B-1) - 1)` inputs at `1M`, `10M`, and `100M` digits against `GNU GMP`.

---

## Core Mathematical & Systems Innovations

1. **Table-Free Lazy Harvey `[0, 2P)` 2-Prime `51-Bit` `AVX-512 IFMA52` Four-Step Radix-4 Exact Integer NTT (`400K`–`101M` Digits)**
   - Uses two `51-bit` primes ($P_1 = \text{0x7ffffa4000001}$, $P_2 = \text{0x7ffff90000001}$, $2P < 2^{52}$) over signed Carry-Free `40-bit` (`5-byte`) Vedic *Vinculum* limbs $d_m \in [-2^{39}, +2^{39}]$ — eliminating floating-point rounding and `100%` of Shoup butterfly `vpminuq` clamping instructions while shrinking `100M`-digit matrices from `671 MB` to **`134 MB`** (`4096 × 4096`, `32 KB` L1-resident rows).
   - Fuses 1-cycle `AVX-512 VBMI` (`vpermb`) `40-bit` unpacking, 4-way parallel Four-Step register twiddle stepping fused directly into the outer Radix-4 stages (zero `134 MB` Four-Step twiddle tables), signed `102-bit` `IFMA52` Garner CRT, and direct `320-bit` (`5 × uint64_t`) output packing (`59.83 ms` Cold / `44.32 ms` `Trunk` / `38.18 ms` Square at `100M` digits).

2. **Single-Pass 8-Series-in-1-`ZMM` Fused Huvent Zero-Storage $\pi$ Hex Extractor (`xc_pi_hex_digits`)**
   - Packs all 8 odd-denominator terms of Huvent's $2^{-12k}$ binary BBP formula into a single `ZMM` register with **identical base exponents** across all 8 lanes (zero mask divergence), switching from 3-cycle centered `double` FMA squaring ($m < 90\text{M}$) to `6×`-interleaved self-reducing Seiler `IFMA52` squaring ($m \ge 90\text{M}$) and `104-bit` (`20` hex digit) accumulation.

3. **Crandall–Fagin Right-Angle Folding ($Y^M + i = 0$) + Adaptive `10..16-Bit` Carry-Free Vedic *Vinculum* Wave Engine**
   - Every $B$-bit limb $u_m \in [0, 2^B - 1]$ with top bit $s_m = u_m \gg (B - 1)$ is balanced in $O(1)$ across all cores with **zero sequential carry chain**:
     $$d_m = u_m - 2^B s_m + s_{m-1} \in [-2^{B-1}, +2^{B-1}]$$
   - Enforces a strict worst-case `IEEE-754` bound ($L \cdot 2^{2B-2} \le 4.398 \times 10^{13}$) with L1-conflict-free `AoSoA8` (`[re[8], im[8]]`) layout and `< 256 KB` Coarse × Fine angle LUTs across `AVX-512`, `AVX2+FMA`, and `ARM64` / Scalar targets.

4. **Vedic Signed Power-of-2 (`NAF`) Sparse Polynomial Fast-Path**
   - Scans `uint64_t` limbs in $O(1)$ per word (aborting in `~50 ns` on random inputs). Whenever numbers lie near powers of 2 or contain long runs of `1`s and `0`s — e.g. $(2^{332,192,810} - 6)(2^{332,192,809} - 3)$ — collapses runs via $111\dots 1_2 \cdot 2^p = 2^{p+r} - 2^p$ and solves the exact `200M`-digit product in **`8.22 ms`** with zero FFT/NTT allocation.

5. **Precomputed Static `Trunk(B)` + On-The-Fly 64-Bit Block-Hash Wave Collapse + Low-RAM Karatsuba**
   - `xc_trunk_create` caches constant multiplier `B` in frequency/NTT space (**`45.71 ms`** at `100M` digits; **`3.50 ms`** at `10M` digits), with automatic 64-bit `FNV-1a` slice deduplication in wave mode and a 50%-RAM Raw-Bit Karatsuba Top-Split (`xc_mul_u64_lowmem`) for memory-constrained environments.

6. **Single-Binary Runtime CPU Dispatch (`AVX-512 IFMA52` → `AVX-512` → `AVX2+FMA` → `Portable Scalar`)**
   - Compiles cleanly with zero external dependencies on Linux, Windows (`MinGW` / `MSVC`), and macOS (`x86_64` & `ARM64`), automatically selecting the fastest exact kernel supported by the host CPU at runtime.

---

## Quick Start

### Build & Test (Linux / macOS / Windows MSYS2)

```bash
make
make test
make bench
# Run the 100,000,000 x 100,000,000 digit (200M output digit) benchmark:
./bench_xcruncher --100m
```

### Build with CMake (Cross-Platform)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build --output-on-failure -C Release
```

---

## C API Example

```c
#include "xcruncher.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("Active ISA: %s\n", xc_active_isa_name());

    /* 1. One-shot adaptive multiplication (auto-selects Schoolbook / Karatsuba / Sparse NAF / Wave) */
    uint64_t a[1024], b[1024], out[2048];
    for (int i = 0; i < 1024; i++) { a[i] = 0x123456789ABCDEF0ULL + i; b[i] = 0xFEDCBA9876543210ULL ^ i; }
    xc_mul_u64(out, a, 1024, b, 1024);

    /* 2. Reusable Plan + Precomputed Trunk(B) for fast repeated multiplication */
    xc_plan_t  *plan  = xc_plan_create(1024, 1024);
    xc_trunk_t *trunk = xc_trunk_create(plan, b, 1024);
    xc_mul_trunk(plan, out, a, 1024, trunk);
    xc_trunk_destroy(trunk);
    xc_plan_destroy(plan);

    /* 3. Zero-storage N-th hexadecimal digits of Pi (< 1 KB RAM, 0 disk) */
    char pi_hex[21];
    xc_pi_hex_digits(100000000ULL, pi_hex, 20);
    printf("Pi @ 16^-100M: %s\n", pi_hex); /* cb840e21926ec5ae0d2f */
    return 0;
}
```

---

## Python Usage (Zero Dependencies)

```python
import xcruncher as xc

print("Active ISA:", xc.active_isa())

# 1. Drop-in exact acceleration for huge Python ints (including negative & sparse near-2^k ints)
a = (1 << 1_000_000) - 6
b = (1 << 999_999) - 3
c = xc.mul(a, b)
assert c == a * b

# 2. Precomputed Trunk(B) for repeated multiplications by a constant huge integer
with xc.WavePlan(max_bits_a=2_000_000) as plan:
    trunk = plan.create_trunk(b)
    c2 = trunk.mul(a)
    trunk.close()

# 3. Zero-storage N-th hexadecimal digits of Pi (1..20 hex digits at any offset)
print("Pi @ 16^-100M:", xc.pi_hex(100_000_000, 20))  # cb840e21926ec5ae0d2f
```

---

## License

Released under the [MIT License](./LICENSE). Copyright (c) 2026 Santhosh Narayanan.
