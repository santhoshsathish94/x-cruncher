# x-cruncher v1.6.1 Architecture Reference

This document specifies the internal mathematical foundations, worst-case exactness proofs, SIMD kernels, memory layout, concurrency model, and fallback hierarchy of `x-cruncher v1.6.1` ([`src/xcruncher.c`](../src/xcruncher.c)).

---

## 1. Six-Tier Dispatch Hierarchy

Every call to `xc_mul_u64(out, a, na, b, nb)` strips trailing zero `uint64_t` limbs (clearing any stripped high output words) and dispatches deterministically through six exact tiers:

| Tier | Trigger Condition | Active Engine | Worst-Case Exactness Mechanism |
| :---: | :--- | :--- | :--- |
| **0** | `na == 0 \|\| nb == 0` | Zero Fast-Path | `memset(out, 0, ...)` |
| **1** | `min(na, nb) <= 40` or `na + nb <= 64` | Register Schoolbook (`64 × 64 -> 128-bit`) | Exact unsigned integer carry chain |
| **2** | `max(na, nb) <= 384` (`<= 24,576` bits) | L1-Resident Karatsuba | Exact unsigned integer sub-products |
| **3** | $\le 128$ signed NAF terms per operand ($k_A k_B \le 16384$) | Vedic Sparse Signed-$2^k$ (`NAF`) | Exact sparse event sweep (`O(k_A k_B log(k_A k_B))`) |
| **4** | `AVX-512 IFMA52 + VBMI` CPU, `17 <= logN <= 24`, `min_l40 <= 8,388,500` (`~400K`–`101M` digits/op) | **2-Prime `51-bit` Lazy Harvey Four-Step Radix-4 NTT** | Exact modular arithmetic + signed `102-bit` Garner CRT |
| **5** | `log_c > 17` (`> ~400M` digits) or `xc_mul_u64_lowmem` | Raw-Bit Karatsuba Top-Split (`50%` RAM) | Exact recursive decomposition into Tier 4 / Tier 6 halves |
| **6** | All other sizes & non-`IFMA52` CPUs (`AVX-512`, `AVX2+FMA`, `ARM64`, Scalar) | **Crandall–Fagin Adaptive `10..16`-Bit Vedic Wave Engine** | Provable worst-case `IEEE-754 binary64` bound ($|\text{err}| < 0.15 < 0.5$) |

When both operands share the same pointer (`a == b`) or identical limb contents (`xc_is_same_operand`), Tiers 4 and 6 automatically execute a **squaring fast-path** (`A * A`), transforming only a single operand and reducing peak workspace memory by `33%–40%`.

---

## 2. Tier 4: 51-Bit Table-Free Lazy Harvey `[0, 2P)` NTT (`AVX-512 IFMA52 + VBMI`)

### 2.1 Prime Selection & Elimination of Butterfly Clamping
On `AVX-512 IFMA52` processors (`AMD Zen 4 / Zen 5`, `Intel Ice Lake+`), the `vpmadd52luq` and `vpmadd52huq` instructions multiply two unsigned `52-bit` integers and accumulate the low or high `52` bits into a `64-bit` lane in a single instruction.

In a conventional `52-bit` NTT ($P \approx 2^{52}$), lazy values in $[0, 2P)$ exceed $2^{52}$, forcing every butterfly to clamp inputs back into $[0, 2^{52})$ with extra `vpminuq + vpsubq` instructions.

`x-cruncher v1.6.0` instead chooses two **`51-bit` NTT-friendly primes** satisfying $2P < 2^{52}$ and $2^{26} \mid (P - 1)$:

$$P_1 = \text{0x7ffffa4000001} = 2^{51} - 367 \cdot 2^{26} + 1 \quad (g_1 = 5, \; -P_1^{-1} \bmod 2^{52} = \text{0x800005c000001})$$
$$P_2 = \text{0x7ffff90000001} = 2^{51} - 448 \cdot 2^{26} + 1 \quad (g_2 = 3, \; -P_2^{-1} \bmod 2^{52} = \text{0x8000070000001})$$

Because $2P < 2^{52}$, any lazy residue $d \in [0, 2P)$ fits natively inside an unsigned `52-bit` `IFMA52` operand without pre-clamping. Furthermore, for any precomputed Shoup twiddle $w_m \in [0, P)$ with $w_p = (w_m \cdot (-P^{-1})) \bmod 2^{52}$, we have:

$$d \cdot w_m < 2 P^2 < 2^{52} P \implies \left\lfloor \frac{d \cdot w_m}{2^{52}} \right\rfloor < P$$

Therefore, `ntt_mul_shoup_2p`:
```c
__m512i hi = _mm512_madd52hi_epu64(vP, d, wm);   /* P + floor(d * wm / 2^52) in [P, 2P) */
__m512i q  = _mm512_madd52lo_epu64(z,  d, wp);   /* q in [0, 2^52)                      */
return _mm512_sub_epi64(hi, _mm512_madd52hi_epu64(z, q, vP)); /* strictly in (0, 2P)!   */
```
executes in **3 vector instructions** with **zero `vpminuq` and zero `vpsubq`**.

> **Dynamic × Dynamic Subtlety (`ntt_mul_mont_2p`):** When multiplying two *dynamic* values $a, b$ that can *both* lie in $[0, 2P)$ (pointwise row convolution/squaring and Four-Step register twiddle multiplication), $a \cdot b < 4 P^2 < 2 \cdot 2^{52} P$, so `hi - q_hi` lands in $(0, 3P)$. Those sites explicitly call `ntt_mul_mont_2p`, which appends `_mm512_min_epu64(t, _mm512_sub_epi64(t, v2P))` to restore $[0, 2P)$.

### 2.2 Carry-Free `40-Bit` Vedic *Vinculum* Balancing & Exactness Bound
Unsigned `40-bit` (`5-byte`) chunks $u_m \in [0, 2^{40} - 1]$ with top sign-borrow bit $s_m = u_m \gg 39 \in \{0, 1\}$ are mapped in `O(1)` without sequential carry propagation via `AVX-512 VBMI` (`vpermb`) byte gather:

$$d_m = u_m - 2^{40} s_m + s_{m-1} \in \left[-2^{39}, \; +2^{39}\right]$$

with one top carry limb $d_L = s_{L-1} \in \{0, 1\}$. Correctness follows from the telescoping sum:

$$\sum_{m=0}^{L} d_m \, 2^{40m} = \sum_{m=0}^{L-1} u_m \, 2^{40m}$$

Under any input bit-pattern with $L_{\min} = \min(L_A, L_B) \le 8,388,500$ limbs (`~101,007,600` decimal digits per operand), the worst-case signed convolution magnitude is strictly bounded by:

$$\max_k |c_k| \le 8,388,500 \times \left(2^{39}\right)^2 = 8,388,500 \times 2^{78} \approx 2^{100.99998} < \frac{P_1 P_2 - 1}{2} \approx 2^{100.999996}$$

Hence every exact signed convolution coefficient $c_k \in \left[-\lfloor P_1 P_2 / 2 \rfloor, +\lfloor P_1 P_2 / 2 \rfloor\right]$ is uniquely and deterministically recovered by signed `102-bit` Garner CRT (`ntt_col_idif_crt_pack_tile8`):

1. Reduce $r_1 \in [0, P_1)$ and $r_2 \in [0, P_2)$ from Montgomery domain.
2. Compute $k = (r_2 - r_1) \cdot P_1^{-1} \bmod P_2 \in [0, P_2)$.
3. Test `is_neg = (k > P2 / 2)` (since $|c| < P_1(P_2 - 1)/2$ implies $k = \lfloor P_2 / 2 \rfloor \implies c \ge 0$).
4. For negative lanes, form $|c| = (P_2 - k) P_1 - r_1$ in `104-bit` (`52 + 52`) `IFMA52` limb pairs and negate before accumulating `320-bit` (`8 × 40-bit` = `5 × uint64_t`) block carries.

Inputs exceeding `min_l40 > 8,388,500` automatically route to Tier 5 (`xc_mul_u64_lowmem` Karatsuba top-split), whose sub-problems fall cleanly back inside Tier 4.

### 2.3 Fused Four-Step Radix-4 Decomposition (`N = R × C`)
To avoid strided DRAM thrashing at $N = 2^{24}$ (`134 MB` per prime matrix), `x-cruncher` decomposes the length-$N$ NTT into $R \times C$ (`4096 × 4096`, padded stride `C + 8` to prevent cache-set aliasing):

1. **Pass 1 (`ntt_unpack_col_dit_dual_tile8`):** Unpacks `8`-column strips of `5-byte` limbs via `vpermb`, places them in bit-reversed row slots in L2-resident thread scratch (`tbufs1/2`), executes a Radix-4 **DIT** column transform (bit-reversed in $\to$ natural $k_1$ out), and **fuses the final `half = R / 4` Radix-4 stage directly with 4 independent ZMM twiddle cursors** ($c_0, c_1, c_2, c_3$ separated by $w_N^{c \cdot (R/4)}$) and non-temporal `vmovntdq` streaming stores into `mat1` and `mat2`. This deletes all `24` Four-Step twiddle arrays, cuts loop-carried `IFMA52` latency by `4×`, and saves `536 MB` of L2 read/write traffic.
2. **Pass 2 (`ntt_row_fused_convolve_C` / `ntt_row_fused_square_C`):** Each `32 KB` row (`4096 × 8` bytes) stays hot in L1D across forward Radix-4 DIF, pointwise `ntt_mul_mont_2p`, and inverse Radix-4 IDIT.
3. **Pass 3 (`ntt_col_idif_crt_pack_tile8` + `ntt_resolve_block_carries`):** Fuses the transpose read from `mat1/2` and 4-cursor inverse Four-Step twiddle multiply directly into the **first `half = R / 2` IDIF Radix-4 stage** (saving another `536 MB` of L2 traffic), completes the column IDIF to bit-reversed output, runs signed `102-bit` Garner CRT in ZMM registers, packs `8 × 40-bit` limbs into `5 × uint64_t` (`320` bits), and resolves inter-tile carries in parallel.

---

## 3. Tier 6: Portable Crandall–Fagin `10..16`-Bit Vedic Wave Engine

On architectures without `AVX-512 IFMA52` (`Intel AVX2+FMA` laptops/servers, older `AVX-512` CPUs, `ARM64 NEON`, and portable scalar builds), `x-cruncher` uses a right-angle negacyclic ($Y^M + i = 0$) complex FFT over balanced $B$-bit Vedic *Vinculum* limbs $d_m \in [-2^{B-1}, +2^{B-1}]$.

### Formal Worst-Case `IEEE-754 binary64` Exactness Guarantee
`choose_wave_params()` selects the largest $B \in \{10, \dots, 16\}$ satisfying the strict adversarial bound:

$$L_{\min} \cdot 2^{2B - 2} \le 4.398 \times 10^{13} < \frac{2^{53}}{204}$$

Even if **every** balanced limb equals $+2^{B-1}$ with zero sign cancellation, the exact integer convolution peak never exceeds $4.398 \times 10^{13}$, where `1 ULP` of `IEEE-754 double` is at most $2^{-7} = 1/128 = 0.0078125$. Accumulated multi-stage unitary FFT + Coarse×Fine angle LUT error remains $< 0.15 < 0.5$, guaranteeing exact `llround` integer recovery on all bit patterns:

- **`AVX-512` (`ZMM`):** `AoSoA8` (`[re[8], im[8]]`) layout with `vmovntpd` streaming writes and `_mm512_vrndscale_pd` rounding.
- **`AVX2+FMA` (`Dual-YMM`):** `AoSoA8` layout processed as two `YMM` halves with `3 × 2^51` (`6755399441055744.0`) hardware FMA magic-number `int64` rounding (`zero scalar libm calls`).
- **`ARM64` / Portable Scalar:** Auto-vectorizable C11 loops with OpenMP dynamic scheduling across hybrid P+E / big.LITTLE cores.

---

## 4. Zero-Storage $N$-th Hexadecimal $\pi$ Extractor (`xc_pi_hex_digits`)

`xc_pi_hex_digits(offset, out_hex, count)` extracts up to `20` exact hexadecimal digits (`80` bits) of $\pi$ starting at `16^-offset` (`0`-indexed after `3.`) for **`0 <= offset <= 2^44` (`17,592,186,044,416`, $\approx 1.76 \times 10^{13}$)** using `< 1 KB` of stack/register state and `0` bytes of heap or disk storage via Huvent's $2^{-12k}$ binary BBP-Bellard series (returning `XC_ERR_ARG` for `offset > 2^44`):

- **`AVX-512 IFMA52` (`0 <= offset <= 2^44`):** Packs all 8 odd-denominator terms into a single `ZMM` register with identical shift exponents across lanes; uses 3-cycle centered `double` Barrett squaring for moduli $m < 90\text{M}$ and `6×`-interleaved self-reducing Seiler `IFMA52` Montgomery squaring above $90\text{M}$ into `104-bit` fixed-point accumulators.
- **`AVX2+FMA` (`0 <= offset <= 2^42`, `128-bit` integer fallback up to `2^44`):** `2×`-interleaved **Dekker–FMA `106-bit`** exact modular exponentiation (`vfmadd213pd` + `vfmsub213pd` error-free product transformation) for moduli up to $2^{46}$, falling back automatically to exact `128-bit` integer modular exponentiation for `2^42 < offset <= 2^44`.
- **`ARM64 NEON` (`0 <= offset <= 2^42`, `128-bit` integer fallback up to `2^44`):** `float64x2_t` hardware FMA (`vfmaq_f64`) + `vrndnq_f64` + `vcvtnq_s64_f64` dual-term modular exponentiation below $90\text{M}$ with exact `64×64 -> 128-bit` integer fallback above $90\text{M}$.

---

## 5. Memory Management, Caching & Concurrency Contract

- **Thread-Safe One-Shot & Stateless APIs:** `xc_mul_u64`, `xc_mul_u64_lowmem`, `xc_pi_hex_digits`, `xc_from_decimal`, `xc_to_decimal`, `xc_mul_decimal`, `xc_active_isa_name`, and `xc_version` are **safe for concurrent calls across multiple application threads** (assuming callers pass distinct `out` buffers).
  - **Thread-Local Diagnostics:** `g_last_path`, `g_last_uniq`, and `g_last_bits` use `XC_THREAD_LOCAL` (`_Thread_local` / `__declspec(thread)`), so `xc_last_path_used()`, `xc_last_unique_slices()`, and `xc_last_limb_bits()` reflect the calling thread's most recent operation without cross-thread data races.
  - **Atomic One-Time ISA & Topology Initialization:** `__builtin_cpu_init()` is guarded by a 3-state acquire/release atomic once-latch (`xc_cpu_init_once`), and all cached ISA/topology integers (`xc_detect_isa`, `xc_has_ifma_ntt`, `xc_has_ifma52`, `xc_wave_threads`, `xc_compute_threads`) use acquire/release atomic loads and stores (`ThreadSanitizer` clean).
- **Reusable Plan & Trunk Ownership (`xc_plan_t` vs `xc_trunk_t`):**
  - A single `xc_plan_t` owns mutable scratch matrices (`Az`, `Bz`, `A1`, `A2`, `tbufs`) and **must not** be invoked concurrently from multiple threads at the same time; multi-threaded callers using plans should allocate one `xc_plan_t` per thread.
  - Once built via `xc_trunk_create`, a `const xc_trunk_t *` is immutable and **may** be shared read-only across concurrent threads that each supply their own compatible `xc_plan_t`.
- **Lock-Free 1-Slot Workspace Caches (`g_cached_ntt`, `g_cached_wave_plan`):** One-shot `xc_mul_u64` atomically acquires (`__atomic_exchange_n`) and recycles a single NTT or Wave workspace across sequential calls of the same `logN` / `(B, log_c)`. Concurrent callers that miss the empty slot transparently allocate an independent temporary workspace and either populate the cache or free on return. Both caches release memory in `__attribute__((destructor))` at library unload (`0` LeakSanitizer leaks).
- **Transparent Huge Pages (`2 MB` THP):** Allocations $\ge 2\text{ MB}$ on Linux issue `madvise(..., MADV_HUGEPAGE)` regardless of thread count, avoiding `68,000+` `4 KB` DTLB page-table walks during strided `268 MB` Four-Step matrix transposes.
- **Universal Topology & Quota Detection (`xc_wave_threads`):** Honors explicit `OMP_NUM_THREADS` first; otherwise inspects Linux `sched_getaffinity` distinct `(package_id, core_id)` pairs, cgroup v1/v2 `cpu.max` container/VM quotas, Windows `GetLogicalProcessorInformation`, and macOS core counts — never mutating caller thread affinity.
