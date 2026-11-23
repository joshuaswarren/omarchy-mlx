# G412 — G16S small-M qmm prefill A/B packet (staged for J616s)

Card G412 (Main, 2026-11-22). The G401 audit's conclusion: the M4 Pro kit's
own workload (prompt_tokens 23) lands in the small-M arm of the G16S qmm
prefill route — below the 64-row gate (M >= 128) and above the M <= 16
16-row arm, so it runs the 32-row tile with 9 of 32 rows dead. Which lever
wins (a 16-row small-M arm vs a dispatch-count change) is a measurement
question; this packet measures it on the part, following the tile64
receipt's method (isolated quantized_matmul per shape, bit-identity gates,
interleaved e2e) at the skinny-M shapes.

The default now picked by the selector (after the G412 fix that teaches
`apple_gpu_cores()` the M4 Pro's 20-core count) is the occupancy floor's
pick: grids under 20*6 = 120 workgroups step to the 16-row twin, wider
grids keep 32 rows. This A/B is the on-part validation of that default.

Hardware: one Apple M4 Pro (G16S B1) J616s with the shipped omarchy driver.
No two-machine setup; everything runs on the one Mac with the mlx-omarchy
release venv.

## Arm mechanism (corrected 2026-11-23)

The staged 2026-11-22 packet forced its arms with
`MLX_OMARCHY_QMM_COOPMAT_ROWS=32` / `=16`. Both are no-ops for the tile
pick at M in 17..127: the env only gates the 64-row route at M >= 128
(`qmm_prefill_select.h`, kill-switch list), so both staged arms dispatched
the same 32-row kernel and an A/B run with it would have compared nothing.
The arms below force the real dispatch through the occupancy-floor selector
itself:

- ctl arm: no env — the shipped pick (on G16S after the fix, and on any
  part with a known core count).
- m16 arm: `MLX_OMARCHY_GPU_CORES=100000` — the floor target becomes
  600000 workgroups, so `coopmat_tile_rows()` steps every grid in this
  packet to the 16-row twin (the M16 route, unpadded B-tile stride on
  non-G14C parts, same as the ctl arm's FullN build at these shapes).

The canary row (M=520, N=4096) proves the two arms dispatched different
kernels: ctl keeps 32 rows there, the forced floor takes 16, and the times
must differ by >= 1.3x (the tile geometry predicts 1.5-2x: identical mma
work, halved weight-tile reuse). Canary times equal within ~5% = the env
did not reach the selector = void run. (Pre-registered bar corrected
2026-11-23: the original 5x was miscalibrated; see
receipts/2026-11-23-g412-small-m-jwm1/README.md, where the same canary
measured 1.55x with ALL_MATCH digests.)

## One-paste packet

```sh
REPO=~/src/mlx-omarchy   # any checkout of the G412 fix branch or later
KIT=$REPO/receipts/2026-11-22-g412-small-m-ab
PY=~/.local/share/mlx-omarchy/venv/bin/python
cd "$KIT/tools"
env -u MLX_OMARCHY_GPU_CORES MESA_SHADER_CACHE_DISABLE=true \
  $PY small_m_sweep.py | tee small-m-ctl.txt
MLX_OMARCHY_GPU_CORES=100000 MESA_SHADER_CACHE_DISABLE=true \
  $PY small_m_sweep.py | tee small-m-16.txt
# bit-identity gate: the sha column must match row for row (arm and us dropped)
diff <(awk '{print $2,$3,$4,$6}' small-m-ctl.txt) \
     <(awk '{print $2,$3,$4,$6}' small-m-16.txt) && echo ALL_MATCH
# canary check: the last row's us must differ by >= 5x between the files
paste small-m-ctl.txt small-m-16.txt | tail -1
# timing comparison for the default decision
paste small-m-ctl.txt small-m-16.txt
```

Pass conditions:

1. `ALL_MATCH` — every arm's 16 digests identical (bit-identity gate; the
   pick must never move outputs). The 16-row twin keeps the per-output
   ascending-k f32 chain; identity is doctest-pinned on the M1 Max and
   M2 Max parts, and this gate closes it on G16S hardware.
2. Canary row (M=520) times differ by >= 1.3x between the arms; a tie
   within ~5% voids the run (the env did not reach the selector).
3. The M in 17..31 `us` columns are the decision input: they compare the
   selector's default pick (32-row at N >= 3840, 16-row below) against the
   uniform 16-row twin on the same shapes. If the default's pick is slower
   than uniform-16 anywhere the default dispatches 32 rows, that shape's
   threshold moves; if uniform-16 loses everywhere, the floor's pick
   stands. Either way record the wheel sha, driver, and host line
   (`system_profiler SPHardwareDataType | grep -E "Chip|Model"`) at the
   top of the receipt file.

## e2e confirmation

Reuse the tile64 receipt's e2e method with a 23-token prompt
(`receipts/2026-11-15-m4-pro-g16s-qmm-bf16w/tools/run_prefill.py`,
`--targets 23 --max-tokens 32`, 5 interleaved runs per arm, the two arms
selected by the same env pair as above); the greedy 32-token sha must be
identical across arms.

## Companion legs

The cross-arm bit-identity leg on fleet Apple-GPU hardware (13-inch M1,
G13G, through its gpu-turn queue) lands in
`receipts/2026-11-23-g412-small-m-jwm1/`; it proves identity on hardware,
not the M4 Pro timing decision. This packet remains the J616s leg.

## Why these shapes

M = 17, 20, 23, 26, 31 covers the arm boundary the audit names (17..31 is
where M is too big for the 16-row arm and too small to fill a 32-row tile);
N/K follow the kit model's qmm prefill GEMMs (mlp up/down and attention
projections at K = 4096). The M = 23 x N = 12288 row is the attention
projection shape; N = 1024 is the smallest grid (the occupancy step-down's
own regime — after the fix this shape is exactly where the default picks
16 rows).
