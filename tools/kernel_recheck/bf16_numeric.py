#!/usr/bin/env python3
"""Device check of the literal `bfloat16_t` round trip on float bit patterns.

A float32 buffer of chosen bit patterns goes through `bfloat16_t v = x[i]; y[i] = v;`
on the GPU. Non-NaN inputs must equal round-to-nearest-even to the bf16 grid
(`defs.bf16`); NaN inputs must stay NaN. MLX's own CPU conversion
(`mlx/types/bf16.h`) returns 0x7FC0 for every NaN, so the preferred NaN bits are
0x7FC00000; any NaN passes, the bits are printed.

Usage (repo root): python3 -m tools.kernel_recheck.bf16_numeric
Exit 0 when every row passes, 1 otherwise.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import mlx_provenance  # noqa: E402

from . import defs

BITS = [
    ("nan_low_payload", 0x7F800001),
    ("nan_full_payload", 0x7FFFFFFF),
    ("nan_negative_full", 0xFFFFFFFF),
    ("nan_negative_low", 0xFF800001),
    ("nan_quiet", 0x7FC00000),
    ("inf_positive", 0x7F800000),
    ("inf_negative", 0xFF800000),
    ("zero_positive", 0x00000000),
    ("zero_negative", 0x80000000),
    ("tie_to_even_down", 0x3F808000),
    ("tie_to_even_up", 0x3F818000),
    ("just_below_tie", 0x3F807FFF),
    ("just_above_tie", 0x3F808001),
    ("max_finite_rounds_to_inf", 0x7F7FFFFF),
    ("subnormal_tiny", 0x00000001),
    ("subnormal_tie_even", 0x00008000),
    ("subnormal_tie_odd", 0x00018000),
    ("one", 0x3F800000),
    ("minus_pi", 0xC0490FDB),
]

SOURCE = """
    uint i = thread_position_in_grid.x;
    bfloat16_t v = x[i];
    y[i] = v;
"""


def main():
    import mlx.core as mx

    # verified=mismatch is expected for a libmlx rebuilt from source and swapped in;
    # the printed sha-256 prefixes identify the binary that produced the rows.
    print(mlx_provenance.provenance_line(mlx_provenance.installed_provenance()))
    bits = np.array([b for _, b in BITS], dtype=np.uint32)
    x = bits.view(np.float32)
    kernel = mx.fast.metal_kernel(
        name="bf16_numeric", input_names=["x"], output_names=["y"], source=SOURCE)
    (out,) = kernel(inputs=[mx.array(x)], template=[], grid=(len(BITS), 1, 1),
                    threadgroup=(len(BITS), 1, 1), output_shapes=[x.shape],
                    output_dtypes=[mx.float32], stream=mx.gpu)
    mx.eval(out)
    got = np.array(out).view(np.uint32)
    want = defs.bf16(x).view(np.uint32)
    failures = 0
    for (name, b), g, w in zip(BITS, got, want):
        is_nan_in = (b & 0x7FFFFFFF) > 0x7F800000
        if is_nan_in:
            ok = (int(g) & 0x7FFFFFFF) > 0x7F800000
            note = "preferred" if int(g) == 0x7FC00000 else "other-nan-bits"
        else:
            ok = int(g) == int(w)
            note = ""
        failures += 0 if ok else 1
        print(f"{'PASS' if ok else 'WRONG'} {name} in=0x{b:08x} got=0x{int(g):08x} "
              f"want={'any-nan' if is_nan_in else '0x%08x' % int(w)} {note}".rstrip())
    print(f"SUMMARY rows={len(BITS)} wrong={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
