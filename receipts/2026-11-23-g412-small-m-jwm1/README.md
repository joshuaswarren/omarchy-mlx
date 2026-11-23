# G412 — small-M qmm prefill cross-arm identity + selector red/green (jwm1, G13G)

Date: 2026-11-23, 04:10–04:33 UTC. Host: jwm1, 13-inch M1 (G13G B1, 8-core
GPU), Borealis Vulkan over ssh; GPU rows through jwm1's own gpu-turn queue
(`/tmp/m1-gpu.lock`; each run well under the 20 s/submit rule). This is the
fleet-hardware leg of card G412; the J616s (M4 Pro, G16S) timing A/B stays
staged in `receipts/2026-11-22-g412-small-m-ab/` (that host is not
fleet-reachable).

## What ran

1. **Selector red/green** (`omarchy_matmul_family_tests`, filtered to the
   small-M case, one case at a time, under gpu-turn):
   - RED at commit 31867356e3 (test flipped, fix absent): **5 of 13
     assertions failed** — `23/32→32!=16`, `23/64`, `23/119`, `127/29→32!=16`,
     and `128/32 → 64!=32` (the last shows the vacuous always-64 on M=128
     N=1024 today). `red.txt`, rc=1.
   - GREEN at the branch tip 300ea86a08 (fix in): **13 of 13 assertions
     pass**. `green.txt`, rc=0.
2. **Cross-arm identity sweep** (the corrected packet tool, both arms on one
   wheel): ctl = shipped pick (no env), m16 = `MLX_OMARCHY_GPU_CORES=100000`
   (floor target 600000 wg forces the 16-row twin everywhere the packet's
   grids reach), M in {17,20,23,26,31} x (N,K) in {(4096,4096),(12288,4096),
   (1024,4096)} + the M=520 canary, 5 reps / 8 evals median per shape.
   - **ALL_MATCH: all 16 digests identical row-for-row** between the arms
     (`digest-diff.txt` empty; per-row shas in the two `small-m-*.txt`).
     This includes the dispatch shape the fix newly creates on parts where
     the floor flips M in 17..31 (two 16-row m-groups).
   - Wheel: `mlx_omarchy-0.32.4.dev202611230425+300ea86-cp314-cp314-linux_aarch64.whl`,
     sha256 `4f9a5bf798fca040421686855fca2c8c4ae29a9a8036472589f3cfe3a855b2b1`
     (stamp `300ea86` = the fix + packet-correction tip; built on this host
     from origin/g412-small-m-fix, whole-encoder bundle hash-verified
     against the runtime pin before the build).

## Timings (G13G, microseconds, median of 5x8)

| shape | ctl (shipped pick) | forced 16-row | delta | floor says |
|---|---:|---:|---|---|
| N=1024 M=23 | 500.0 | 496.9 | tie | both arms run 16 rows (32 wg = 4/core < 6: the shipped 8-core table already steps down) |
| N=4096 M=23 | 1124.8 | 1459.5 | 16-row +29.8% | 32 rows (128 wg = 16/core) |
| N=12288 M=23 | 2447.3 | 3242.4 | 16-row +32.5% | 32 rows (384 wg = 48/core) |
| canary N=4096 M=520 | 12268.4 | 19072.5 | 16-row +55.5% | 32 rows ctl / 16 rows forced |

The direction matches the measured crossover the fix imports: where the
floor keeps 32 rows, forcing 16 rows loses ~30%; where it steps to 16 rows,
both arms tie. Per-shape M 17→31 is flat within an arm (weight-bound), as
expected at skinny M.

## Canary rule correction (dated)

Pre-registered bar: canary times must differ >= 5x. Measured: 1.55x
(12268.4 vs 19072.5 us). The 5x figure was miscalibrated — the tile
geometry predicts ~1.5-2x, not an order of magnitude: the mma work is
identical, and the 16-row arm halves how many outputs share one staged
weight tile (the reuse loss `qmm_prefill_select.h` itself documents, which
is why the crossover sits near 6 wg/core). Correction, in place before the
PR: the void condition is *canary times equal within ~5%* (which would mean
the env never reached the selector and both arms ran the same kernel); any
clear difference with ALL_MATCH digests proves different kernels, same
math. The staged J616s packet carries the corrected bar (>= 1.3x, void on a
tie within 5%).

## What this leg does NOT claim

No G16S timing claim. jwm1 is an 8-core part; its M=23 N>=4096 grids sit
far above the 6 wg/core crossover, so this leg confirms direction and
identity, not the M4 Pro's below-crossover regime. The J616s one-paste
packet remains the timing decision's path.

## Files

- `red.txt`, `green.txt` — the filtered selector case, both states.
- `small-m-ctl.txt`, `small-m-16.txt` — the two arms, 16 rows each.
- `gate.txt` — row counts, ALL_MATCH, canary ratio.
- `digest-diff.txt` — empty (the identity diff).
- `red-build.log`, `green-build.log`, `wheel.log`, `configure.log`,
  `*-prepare.log`, `venv*.log`, `wt-contents.txt` — build provenance.
- `head.txt` — HEAD of the shared clone before its red-checkout (the v1
  run died at configure and left the tree at the red commit; the green
  stage checked out the branch tip 300ea86a08 and `work/mlx`'s staged
  selector carries the fix, verified post-run).
- `SHA256SUMS` — checksums of the text artifacts (refreshed with this
  README included as `SHA256SUMS.receipt`).

Provenance note: the repo copies of these artifacts are redacted
(`/home/joshuawarren` → `~`, applied 2026-11-23 before the commit to pass
the repo's privacy hook); the unredacted raws live in the private program
notebook under `artifacts/G412M4Prefill/g412-small-m-ab/` with their own
SHA256SUMS.raw. No other byte differs.
