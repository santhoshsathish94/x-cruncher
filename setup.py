from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # pragma: no cover
    _bdist_wheel = None


def _shared_lib_name() -> str:
    sys_name = platform.system()
    if sys_name == "Windows":
        return "xcruncher.dll"
    if sys_name == "Darwin":
        return "libxcruncher.dylib"
    return "libxcruncher.so"


def _compile_native_lib(repo_root: Path, out_lib: Path) -> bool:
    c_src = repo_root / "src" / "xcruncher.c"
    inc_dir = repo_root / "include"
    if not c_src.is_file() or not inc_dir.is_dir():
        return False

    out_lib.parent.mkdir(parents=True, exist_ok=True)
    sys_name = platform.system()

    if sys_name == "Windows":
        gcc = shutil.which("gcc")
        if gcc:
            base = [
                gcc, "-O3", "-std=c11", "-shared",
                "-DXCRUNCHER_BUILD_DLL", "-static",
                "-I", str(inc_dir), str(c_src), "-o", str(out_lib),
            ]
            for extra in (["-fopenmp"], []):
                r = subprocess.run(base + extra, capture_output=True)
                if r.returncode == 0 and out_lib.is_file():
                    return True

        cmake = shutil.which("cmake")
        if cmake and (repo_root / "CMakeLists.txt").is_file():
            with tempfile.TemporaryDirectory() as tmp:
                r1 = subprocess.run(
                    [cmake, "-B", tmp, "-S", str(repo_root), "-DCMAKE_BUILD_TYPE=Release"],
                    capture_output=True,
                )
                r2 = subprocess.run(
                    [cmake, "--build", tmp, "--config", "Release", "--target", "xcruncher"],
                    capture_output=True,
                )
                if r1.returncode == 0 and r2.returncode == 0:
                    for dll in Path(tmp).rglob("*.dll"):
                        if "xcruncher" in dll.name.lower():
                            shutil.copy2(dll, out_lib)
                            return True
        return False

    cc = (
        os.environ.get("CC")
        or shutil.which("gcc")
        or shutil.which("clang")
        or shutil.which("cc")
    )
    if not cc:
        return False

    shared_flag = "-dynamiclib" if sys_name == "Darwin" else "-shared"
    base = [
        cc, "-O3", "-std=c11", shared_flag, "-fPIC",
        "-I", str(inc_dir), str(c_src), "-o", str(out_lib), "-lm",
    ]
    for omp_flags in (["-fopenmp"], []):
        r = subprocess.run(base + omp_flags, capture_output=True)
        if r.returncode == 0 and out_lib.is_file():
            return True
    return False


class BuildPyWithNative(_build_py):
    def run(self) -> None:
        super().run()
        repo_root = Path(__file__).resolve().parent
        pkg_out = Path(self.build_lib) / "xcruncher"
        pkg_out.mkdir(parents=True, exist_ok=True)

        # Always bundle C11 source + header inside `xcruncher/_csrc/` for inspection & fallback
        csrc_dir = pkg_out / "_csrc"
        csrc_dir.mkdir(parents=True, exist_ok=True)
        if (repo_root / "src" / "xcruncher.c").is_file():
            shutil.copy2(repo_root / "src" / "xcruncher.c", csrc_dir / "xcruncher.c")
        if (repo_root / "include" / "xcruncher.h").is_file():
            shutil.copy2(repo_root / "include" / "xcruncher.h", csrc_dir / "xcruncher.h")

        # Compile native shared library directly into the wheel/installed package
        out_lib = pkg_out / _shared_lib_name()
        if not _compile_native_lib(repo_root, out_lib):
            raise RuntimeError(
                "Failed to compile native xcruncher shared library during Python package build. "
                "Ensure a C11 compiler (gcc, clang, or MSVC + CMake) is installed."
            )


cmdclass: dict[str, type] = {"build_py": BuildPyWithNative}

if _bdist_wheel is not None:
    class BdistWheelCTypes(_bdist_wheel):
        def finalize_options(self) -> None:
            super().finalize_options()
            self.root_is_pure = False

        def get_tag(self) -> tuple[str, str, str]:
            _, _, plat = super().get_tag()
            # Pure C ABI loaded via ctypes: compatible across all Python 3 versions on this OS/arch
            return ("py3", "none", plat)

    cmdclass["bdist_wheel"] = BdistWheelCTypes


setup(cmdclass=cmdclass)
