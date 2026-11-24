# QMV small-T threshold audit (w76 task, 2026-11-24)

Sweep: rows 2/4/8/16 x 2B (2048x6144, 6144x2048, 2048x2048) and 4B
(2560x9728, 9728x2560, 2560x2560) shapes, q4 g64 transposed bf16.
Arms via live env flags, ABBA-interleaved, median of 5 reps x 2 blocks:
def (shipped), small (QMV_SMALL_T=8), tile (QMM_MMA_ROWS=1 + QMM_ROWS=0),
sm8 (SMALL_T=8 + SMALL_T_CAP=8). Bitwise vs single-row def reference.
Harness: scratch/wB1/qmv-sweep/qmv_sweep.py + qmv_sweep2.py on
jwm1 (G13G, honeykrisp e7631595df, mlx 0.32.4.dev202611230356+9b8185041).

## jwm1 (G13G) results — clean run (sweep-jwm1.json)

| cell | m=2 | m=4 | m=8 | m=16 |
|---|---|---|---|---|
| small vs def | -0.7..+15.2% | +15.4..+27.9% | ~0 (tile) | ~0 (tile) |
| tile vs def | -59..-131% | -20..-41% | ~0 | ~0 |

- m<=4: batch-4 short-pass GEMV beats the shipped row chain at every
  shape (m=4: +15-28%), each row bit-identical (id_small=True).
- m=8/16: all arms route to the coopmat tile; parity within noise.
  Routing already correct.
- tile arm at m<=4 is 1.2-2.3x slower: the mma path does NOT win at
  low row counts on G13G (contrast upstream Metal qmv_mma on M3 Ultra).

## Decode ids check (qmv_ids.py, qwen3-4b-4bit, spec T=2/4/8/16)
72/72 arm runs emitted ids identical to def. Routing flags change kernel
choice only; small arm is bit-identical per row by construction.

## sm8 arm (SMALL_T_CAP=8, batch-8 dispatch) — sweep2c clean run
m=8 wins at every shape vs def/tile: +3.7% (2b 2048x6144), +20.3%
(2b 6144x2048), +12.2% (2b 2048x2048), +3.8% (4b 2560x9728), +5.6%
(4b 9728x2560), +5.0% (4b 2560x2560); bit-identical (sm8_id=True).
m=16: falls back to tile (cap 8), parity -2.1..+0.5% (noise).

## Flip
G13G: MLX_OMARCHY_QMV_SMALL_T default 0 -> 8 (qmv_small_t_default_part
in device.h). Covers M=2..8 short-pass GEMV; M>=9 keeps the tile.
M=5..7 unswept (same single batch-8 dispatch M=8 wins with).
G14C (M2, w9C leg, prod venv 0.32.4.dev202611230356+9b818504): default
0 -> 8 too. Per-cell: m=8 sm8 +10.3..+56.0% at every shape; m<=4 batch-4
~+7% median (-4.2..+25.0, m=2 cells noisy); m=16 tile parity. Bitwise:
small identical at m<=4, sm8 identical through m=8, tile never. Medians
across the 24 cells: def 0.3860, small 0.3544 (-8%), sm8 0.3024 (-22%),
tile 0.5553 (+44% slower). M2 ids step skipped (gated HF repo, no
token); moot — engaged rows are bit-identical so decode ids cannot
diverge. G17C: stays 0.
Note: the 2026-09-29 negative receipt probed coopmat-family variants,
not this comparison; the small-row chain shipped 2026-11-15 and was
never re-raced against the batch GEMV on G13G.

## G13C (16-inch M1 Max, jw16) — wAW replicate + wB1 route probe
wAW ran qmv_sweep2.py verbatim on the prod wheel +bf5631d
(issuecomment-7119188736). Read with the device name in mind — jw16
reports "Apple M1 Max (G13C C0)", so its small_t default was already 8
in that wheel: every m<=8 cell compares the SAME route to itself, and
the observed -13..+22% spread is harness noise, not a route change.
The flip this PR adds stays scoped to G13G and G14C.

m=16 tall-K: wAW's table showed def 19-33% slower than the alt arms at
K=6144->2048 and K=9728->2560 and suggested a tile rule. wB1 probe of
the actual dispatches (MLX_OMARCHY_GPU_PROFILE, diag wheel 5ef3034,
same shapes): every arm emits the identical QmmTileF32 dispatch — the
m>8 path has a single route, so no threshold can change it. Rep-level
ABBA on wAW's exact venv (/usr/lib/omarchy-mlx/venv, +bf5631d), same
cell, 10 interleaved reps: def 2.058 / small 2.063 / tile 2.066 /
sm8 2.059 ms — parity within 0.4%. The reported gap is an ABBA
position/order artifact in the sweep harness, and the *_ident=False
flags at m=16 mark that route's nondeterministic accumulation, not a
reroute. No G13C m=16 rule added; not confirmed.
