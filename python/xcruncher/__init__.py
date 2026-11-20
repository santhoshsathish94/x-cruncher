"""
x-cruncher Python Bindings (zero third-party dependencies, pure ctypes).

Accelerates arbitrary-precision Python `int` multiplication & zero-storage Pi extraction via:
  - Register/L1 raw 64-bit schoolbook & Karatsuba
  - Vedic Signed Power-of-2 (NAF) sparse polynomial solver
  - 51-Bit Table-Free Lazy Harvey 2-Prime AVX-512 IFMA52 + VBMI Four-Step Radix-4 NTT
  - Crandall-Fagin Right-Angle Adaptive 10..16-bit Carry-Free Vedic Vinculum Wave Engine
  - Precomputed Static Trunks (`WaveTrunk`) & Zero-Storage Huvent Hexadecimal Pi Extractor
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

__version__ = "1.6.1"

_LIB: Optional[ctypes.CDLL] = None
_LIB_LOCK = threading.Lock()
_MAX_PI_HEX_OFFSET = 1 << 44


def _load_cdll(path: Path) -> ctypes.CDLL:
    if platform.system() == "Windows" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(path.parent))
        for entry in os.environ.get("PATH", "").split(os.pathsep):
            if entry and os.path.isdir(entry):
                try:
                    os.add_dll_directory(entry)
                except OSError:
                    pass
    return _bind_signatures(ctypes.CDLL(str(path)))


def _user_cache_dir() -> Path:
    if platform.system() == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir())
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    d = base / "xcruncher" / f"v{__version__}-{platform.machine()}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _find_or_build_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB

    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB

        sys_name = platform.system()
        if sys_name == "Windows":
            names = ["xcruncher.dll", "libxcruncher.dll"]
        elif sys_name == "Darwin":
            names = ["libxcruncher.dylib", "xcruncher.dylib"]
        else:
            names = ["libxcruncher.so", "xcruncher.so"]

        env_lib = os.environ.get("XCRUNCHER_LIB_PATH")
        if env_lib and Path(env_lib).is_file():
            _LIB = _load_cdll(Path(env_lib))
            return _LIB

        pkg_dir = Path(__file__).resolve().parent
        repo_root = pkg_dir.parent.parent
        search_dirs = [
            pkg_dir,
            repo_root,
            repo_root / "build",
            repo_root / "build" / "Release",
        ]
        try:
            search_dirs.append(_user_cache_dir())
        except OSError:
            pass

        for d in search_dirs:
            for name in names:
                cand = d / name
                if cand.is_file():
                    _LIB = _load_cdll(cand)
                    return _LIB

        # Fallback auto-build from bundled `xcruncher/_csrc/` or repository `src/`
        source_pairs = [
            (pkg_dir / "_csrc" / "xcruncher.c", pkg_dir / "_csrc"),
            (repo_root / "src" / "xcruncher.c", repo_root / "include"),
        ]
        for c_src, inc_dir in source_pairs:
            if not c_src.is_file():
                continue
            out_dir = pkg_dir if os.access(pkg_dir, os.W_OK) else _user_cache_dir()
            out_lib = out_dir / names[0]
            cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang") or shutil.which("cc")
            if cc:
                shared_flag = "-dynamiclib" if sys_name == "Darwin" else "-shared"
                cmd = [cc, "-O3", "-std=c11", shared_flag, str(c_src), "-I", str(inc_dir), "-o", str(out_lib)]
                if sys_name == "Windows":
                    cmd.extend(["-DXCRUNCHER_BUILD_DLL", "-static"])
                else:
                    cmd.insert(4, "-fPIC")
                    cmd.append("-lm")
                try:
                    subprocess.run(cmd + ["-fopenmp"], check=True, capture_output=True)
                except subprocess.CalledProcessError:
                    subprocess.run(cmd, check=True, capture_output=True)
                _LIB = _load_cdll(out_lib)
                return _LIB

        raise RuntimeError(
            "Could not locate or build xcruncher shared library. "
            "Install a platform wheel (`pip install x-cruncher`) or build with `make` / `cmake`."
        )


def _bind_signatures(lib: ctypes.CDLL) -> ctypes.CDLL:
    u64_p = ctypes.POINTER(ctypes.c_uint64)

    lib.xc_version.argtypes = []
    lib.xc_version.restype = ctypes.c_char_p

    lib.xc_active_isa_name.argtypes = []
    lib.xc_active_isa_name.restype = ctypes.c_char_p

    lib.xc_last_path_used.argtypes = []
    lib.xc_last_path_used.restype = ctypes.c_int

    lib.xc_mul_u64.argtypes = [u64_p, u64_p, ctypes.c_size_t, u64_p, ctypes.c_size_t]
    lib.xc_mul_u64.restype = ctypes.c_int

    lib.xc_mul_u64_lowmem.argtypes = [u64_p, u64_p, ctypes.c_size_t, u64_p, ctypes.c_size_t]
    lib.xc_mul_u64_lowmem.restype = ctypes.c_int

    lib.xc_plan_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
    lib.xc_plan_create.restype = ctypes.c_void_p

    lib.xc_plan_destroy.argtypes = [ctypes.c_void_p]
    lib.xc_plan_destroy.restype = None

    lib.xc_plan_mul.argtypes = [ctypes.c_void_p, u64_p, u64_p, ctypes.c_size_t, u64_p, ctypes.c_size_t]
    lib.xc_plan_mul.restype = ctypes.c_int

    lib.xc_trunk_create.argtypes = [ctypes.c_void_p, u64_p, ctypes.c_size_t]
    lib.xc_trunk_create.restype = ctypes.c_void_p

    lib.xc_trunk_destroy.argtypes = [ctypes.c_void_p]
    lib.xc_trunk_destroy.restype = None

    lib.xc_mul_trunk.argtypes = [ctypes.c_void_p, u64_p, u64_p, ctypes.c_size_t, ctypes.c_void_p]
    lib.xc_mul_trunk.restype = ctypes.c_int

    lib.xc_pi_hex_digits.argtypes = [ctypes.c_uint64, ctypes.c_char_p, ctypes.c_size_t]
    lib.xc_pi_hex_digits.restype = ctypes.c_int

    return lib


def _int_to_u64_buf(x: int) -> tuple[ctypes.Array, int]:
    ax = abs(x)
    n_limbs = max(1, (ax.bit_length() + 63) // 64)
    raw = ax.to_bytes(n_limbs * 8, "little")
    arr = (ctypes.c_uint64 * n_limbs).from_buffer_copy(raw)
    return arr, n_limbs


def active_isa() -> str:
    """Return the active hardware SIMD path selected at runtime."""
    return _find_or_build_lib().xc_active_isa_name().decode("utf-8")


def mul(a: int, b: int, *, lowmem: bool = False) -> int:
    """Multiply two arbitrary-size Python ints using x-cruncher."""
    if a == 0 or b == 0:
        return 0
    # Below ~4,096 bits, CPython's internal C loop avoids ctypes call overhead
    if a.bit_length() < 4096 and b.bit_length() < 4096 and not lowmem:
        return a * b

    lib = _find_or_build_lib()
    sign = -1 if (a < 0) ^ (b < 0) else 1
    ba, na = _int_to_u64_buf(a)
    bb, nb = _int_to_u64_buf(b)
    out = (ctypes.c_uint64 * (na + nb))()
    fn = lib.xc_mul_u64_lowmem if lowmem else lib.xc_mul_u64
    rc = fn(out, ba, na, bb, nb)
    if rc != 0:
        raise RuntimeError(f"xc_mul_u64 failed with code {rc}")
    res = int.from_bytes(bytes(out), "little")
    return -res if sign < 0 else res


class WavePlan:
    """Reusable wave workspace & precomputed WaveTrunk factory."""

    def __init__(self, max_bits_a: int, max_bits_b: Optional[int] = None) -> None:
        lib = _find_or_build_lib()
        na = max(1, (max_bits_a + 63) // 64)
        nb = max(1, ((max_bits_b or max_bits_a) + 63) // 64)
        self._lib = lib
        self._ptr = lib.xc_plan_create(na, nb)
        if not self._ptr:
            raise MemoryError("xc_plan_create failed")

    def close(self) -> None:
        if getattr(self, "_ptr", None):
            self._lib.xc_plan_destroy(self._ptr)
            self._ptr = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "WavePlan":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        sign = -1 if (a < 0) ^ (b < 0) else 1
        ba, na = _int_to_u64_buf(a)
        bb, nb = _int_to_u64_buf(b)
        out = (ctypes.c_uint64 * (na + nb))()
        rc = self._lib.xc_plan_mul(self._ptr, out, ba, na, bb, nb)
        if rc != 0:
            raise RuntimeError(f"xc_plan_mul failed with code {rc}")
        res = int.from_bytes(bytes(out), "little")
        return -res if sign < 0 else res

    def create_trunk(self, b: int) -> "WaveTrunk":
        return WaveTrunk(self, b)


class WaveTrunk:
    """Precomputed frequency-domain WaveTrunk(B) for fast repeated multiplication."""

    def __init__(self, plan: WavePlan, b: int) -> None:
        self._lib = plan._lib
        self._plan = plan
        self._sign = -1 if b < 0 else 1
        self._is_zero = b == 0
        bb, nb = _int_to_u64_buf(b)
        self._nb = nb
        self._ptr = self._lib.xc_trunk_create(plan._ptr, bb, nb)
        if not self._ptr:
            raise MemoryError("xc_trunk_create failed")

    def close(self) -> None:
        if getattr(self, "_ptr", None):
            self._lib.xc_trunk_destroy(self._ptr)
            self._ptr = None

    def __del__(self) -> None:
        self.close()

    def mul(self, a: int) -> int:
        if a == 0 or self._is_zero:
            return 0
        sign = self._sign * (-1 if a < 0 else 1)
        ba, na = _int_to_u64_buf(a)
        out = (ctypes.c_uint64 * (na + self._nb))()
        rc = self._lib.xc_mul_trunk(self._plan._ptr, out, ba, na, self._ptr)
        if rc != 0:
            raise RuntimeError(f"xc_mul_trunk failed with code {rc}")
        res = int.from_bytes(bytes(out), "little")
        return -res if sign < 0 else res


def pi_hex(hex_offset: int, digits: int = 16) -> str:
    """Extract `digits` (1..20) exact hexadecimal digits of Pi after 16^(-hex_offset) for 0 <= hex_offset <= 2^44."""
    if not (0 <= hex_offset <= _MAX_PI_HEX_OFFSET) or not (1 <= digits <= 20):
        raise ValueError(f"hex_offset must be in 0..{_MAX_PI_HEX_OFFSET} (2^44) and digits must be in 1..20")
    lib = _find_or_build_lib()
    buf = ctypes.create_string_buffer(digits + 1)
    rc = lib.xc_pi_hex_digits(ctypes.c_uint64(hex_offset), buf, ctypes.c_size_t(digits))
    if rc != 0:
        raise RuntimeError(f"xc_pi_hex_digits failed with code {rc}")
    return buf.value.decode("ascii")


__all__ = ["__version__", "active_isa", "mul", "pi_hex", "WavePlan", "WaveTrunk"]
