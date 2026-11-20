"""End-to-end Python binding tests against CPython arbitrary-precision ground truth."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
import xcruncher as xc


def main() -> None:
    print(f"Testing Python xcruncher v{xc.__version__} | Active ISA: {xc.active_isa()}")
    rng = random.Random(20261120)

    # 1. Signs & small boundaries
    assert xc.mul(0, 123456789) == 0
    assert xc.mul(-9876543210123456789, 1234567890987654321) == -9876543210123456789 * 1234567890987654321

    # 2. Vedic Sparse NAF fast-path: (2^200000 - 6) * (2^199999 - 3)
    a_sp = (1 << 200_000) - 6
    b_sp = (1 << 199_999) - 3
    assert xc.mul(a_sp, b_sp) == a_sp * b_sp
    print("  [PASS] Python Vedic Sparse NAF (2^200000 - 6) * (2^199999 - 3)")

    # 3. Crandall-Fagin 16-bit Wave Engine + Low-RAM Karatsuba (~300,000 bits)
    a_big = rng.getrandbits(300_000)
    b_big = -rng.getrandbits(300_000)
    expected = a_big * b_big
    assert xc.mul(a_big, b_big) == expected
    assert xc.mul(a_big, b_big, lowmem=True) == expected
    print("  [PASS] Python Cold Wave & Low-RAM 50%-Memory Karatsuba (300,000 bits)")

    # 4. Precomputed WaveTrunk(B)
    with xc.WavePlan(320_000) as plan:
        trunk = plan.create_trunk(b_big)
        assert trunk.mul(a_big) == expected
        a2 = rng.getrandbits(280_000)
        assert trunk.mul(a2) == a2 * b_big
        trunk.close()
    print("  [PASS] Python Precomputed WaveTrunk(B) reuse")
    print("ALL PYTHON TESTS PASSED!")


if __name__ == "__main__":
    main()
