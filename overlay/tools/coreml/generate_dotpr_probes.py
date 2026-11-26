#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Generate exact float32 inputs for the isolated vDSP_dotpr probe."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

MAGIC = 0x44505431


def generated_bits(count: int, seed: int) -> list[int]:
    state = seed
    values = []
    for _ in range(count):
        state ^= state << 13 & 0xFFFFFFFF
        state ^= state >> 17
        state ^= state << 5 & 0xFFFFFFFF
        state &= 0xFFFFFFFF
        sign = state & 0x80000000
        exponent = 127 + ((state >> 24) % 17) - 8
        values.append(sign | exponent << 23 | state & 0x7FFFFF)
    return values


def probe_cases() -> list[tuple[str, list[int], list[int]]]:
    cases = [
        (
            "fma-two-term",
            [0x3C565E69, 0xC1480539],
            [0x408A288C, 0x3C5F70C6],
        ),
        (
            "double-rounding-boundary",
            [226492416, 1065353219],
            [1065353216, 1069547520],
        ),
        (
            "overflow-midpoint-boundary",
            [2373976064, 1526202368],
            [1065353216, 1677992200],
        ),
        (
            "vector-four",
            [0x4115B804, 0x3DBE3ACE, 0xBD94FC9D, 0x42EED71E],
            [0xBBA12958, 0x40C6ADFD, 0xC05058A1, 0x3EC6EFA3],
        ),
    ]
    for name, length, seed in (
        ("association-eight", 8, 0x8A2C91E3),
        ("vector-tail-seventeen", 17, 0x117A91E3),
        ("production-length-257", 257, 0x257A91E3),
    ):
        bits = generated_bits(length * 2, seed)
        cases.append((name, bits[:length], bits[length:]))
    cases.append(("signed-zero", [0x80000000], [0x3F800000]))
    return cases


def write_input(path: Path) -> None:
    cases = probe_cases()
    with path.open("wb") as stream:
        stream.write(struct.pack("<II", MAGIC, len(cases)))
        for name, lhs, rhs in cases:
            encoded = name.encode()
            stream.write(struct.pack("<I", len(encoded)))
            stream.write(encoded)
            stream.write(struct.pack("<I", len(lhs)))
            stream.write(struct.pack(f"<{len(lhs)}I", *lhs))
            stream.write(struct.pack(f"<{len(rhs)}I", *rhs))
    print(f"wrote {len(cases)} cases to {path}")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} OUT.bin", file=sys.stderr)
        return 2
    write_input(Path(argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
