# Small-row (2-4) word kernel measurement (jwm1, M1 G13G)

Date: 2026-11-15. Machine: jwm1, Apple M1 (G13G B1), Borealis Vulkan.
Wheel: mlx-omarchy 0.32.4.dev202611160110+diag.5e40ec (branch
w7q/small-row-qmv, narrowed gate m in 2..4). Shape: 4-bit group-64 affine
transposed QuantizedMatmul, K=2048, N=5120, bf16 x/scales/biases/out.

## Command

```
~/scratch/w7q-rows-venv/bin/python /tmp/spike-qmv-target.py
```

`tools/spike-qmv-target.py`: 5 warm + 20 timed
`mx.eval(quantized_matmul(...))` per m, median of wall ms divided by rows.
Targets are 1.2x better than the tile path (tile baselines from the 11-15
descending spike on the same machine).

## Result (2026-11-15 19:21 CST, free GPU)

| m | tile path | Rows8 | ratio |
|---|---:|---:|---:|
| 2 | 0.299 | 0.204 | 1.46x |
| 3 | not measured | 0.145 | no ratio claimed |
| 4 | 0.150 | 0.122 | 1.23x |

An earlier 2x target (0.204 vs <0.150 at m=2, 0.122 vs <0.075 at m=4) was
set from a weight-traffic model and MISSED. Claimed range 1.2-1.5x at m=2
and m=4; no 2x claim. m=3 raw number given with no tile baseline, so no
ratio is claimed there.

## Why not faster (analysis)

Weight traffic is only part of a submit (scales/biases per-row reads,
x loads, subgroup reduce, dispatch overhead cap the win — Amdahl). At
m >= 8 the scalar-FMA row chain loses to the coopmat tile's matrix units
(measured m=8: 1.3x regression; m=16: 2.4x regression), so the gate stays
m in 2..4 and wider rows keep the tile path.

## Verification (same tree)

New doctest pins per-row bit-identity (exact float equality) vs
single-row dispatches across m in {2,3,4} x K in {320,2048} x
N in {513,5120}, plus the kernel-id assertion (Rows8 fires) and the m=8
fallthrough gate check: 101,492/101,492 assertions pass on jwm1
(2026-11-15 re-run on the rebased tree). BATCH SPIR-V byte-identical to
main (shipped path untouched).
`profile/rows-wheel-profile.ndjson`: GPU profile of the wheel run —
QuantizedMatmul dispatches with e=548 (QmmVecQ4WordSubgroupRows8BF16)
at m=2/3/4.

## Scope limit

M1 (G13G) only. M1 Max (G13C) and M2 Max (G14C) re-run parked on fleet
access (jw16 off-network since 2026-11-07).

## Files

- `tools/spike-qmv-target.py`: the failing-first benchmark (narrowed).
- `profile/rows-wheel-profile.ndjson`: GPU profile backing the kernel-id
  claim.
- `SHA256SUMS`: checksums of this directory.
