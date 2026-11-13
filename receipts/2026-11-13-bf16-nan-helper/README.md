# bf16 conversion helper keeps NaN a NaN (2026-11-13)

Machine: 16-inch M1 Max (G13C), Linux 7.1.12, Honeykrisp (Mesa e7631595df),
CPU-only libmlx rebuilds; GPU runs through the idle-guard queue (class
validation).

Check: `python3 -m tools.kernel_recheck.bf16_numeric` runs 19 float bit
patterns through a literal `bfloat16_t v = x[i]; y[i] = v;` kernel. Non-NaN
rows must equal round-to-nearest-even to the bf16 grid (`defs.bf16`); NaN
rows must stay NaN. MLX's CPU conversion (`mlx/types/bf16.h`) returns 0x7FC0
for every NaN, so 0x7FC00000 is the preferred output; any NaN passes.

## Provenance

Each log starts with the `scripts/mlx_provenance.py` line. Both runs report
`verified=mismatch` on purpose: libmlx.so was rebuilt from source and swapped
into a copy of the wheel venv, so it cannot match the wheel RECORD. The
sha-256 prefixes on that line (core extension 615f0eb19f8f7d17; libmlx
e5ba3c26d8d993fb before, aa048347f850593d after) name the binaries that
produced the rows. `harness=unknown`: the probe ran from a plain copy of
`tools/kernel_recheck` and `scripts`, not a git checkout; the probe source is
the one in this PR.

## Before (`before-fix.log`)

libmlx built from main at 371b114d (literal-bf16 rounding from #420, no NaN
guard), libmlx sha-256 starting e5ba3c26: 15 PASS / 4 WRONG.

| input | bits | got |
|---|---|---|
| nan_low_payload | 0x7f800001 | 0x7f800000 (+inf) |
| nan_full_payload | 0x7fffffff | 0x80000000 (-0) |
| nan_negative_full | 0xffffffff | 0x00000000 (+0) |
| nan_negative_low | 0xff800001 | 0xff800000 (-inf) |

The bias add `bits + 0x7fff + lsb` carries into the exponent for NaN
payloads. My prediction named only the first two rows; the two negative NaN
rows fail by the same mechanism and I did not list them.

An earlier run on a wheel that predates #420 returned every input unchanged
(literal `bfloat16_t` was not rounded at all there, the helper was never
called); it is not a valid before-run for this bug and is not included.

## After (`after-fix.log`)

libmlx built from this branch rebased on 371b114d, sha-256 starting aa048347:
19 PASS / 0 WRONG, RC=0. All four NaN inputs return 0x7fc00000; ties,
below/above ties, max finite (-> inf), subnormals and -pi are exact.

## Not measured

- Metal's native `bfloat` conversion for NaN input (sign handling). The
  check accepts any NaN and prints the bits.
- Kernels with bf16 buffers: this run goes through the literal-local path
  only.
