# Prefill axes on G13: rasterization order + issue-quality twins of the qmm coopmat prefill kernel

Date: 2026-10-03/04. Chips: Apple M1 (G13G, 8 GPU cores, 16 GB) and
M1 Max (G13C). Kernel 7.1.12-2-7-ARCH (G13G) / 7.1.13-3-2-ARCH (G13C).
Hosts are redacted per the public-repo rule; the private lab notebook
holds unredacted transcripts and SHA256SUMS
(`entries/PrefillAxes/20261003T202947Z-jw16-jwm1-qmm-prefill-raster-issue.md`).
This continues receipts/2026-10-03-prefill-gap/README.md §5 (its next
axes); all per-op profile facts and the closed-axes list there carry
over. Provenance printed beside every measurement
(`scripts/mlx_provenance.py`, verified=match); A/Bs are same-build
env-selected arms (or the landed default vs its opt-out env), so the
"two builds must differ" rule is satisfied by construction; idle gates
(load1 < 0.5, PSI cpu some avg10 <= 0.05) recorded per row and only
green rows counted; `flock /tmp/m1-gpu.lock` per host.

## What landed

`group-of-GM=4` rasterization is now the **default** workgroup order of
the 32-row `QmmPrefillCoopmatBF16X32[FullN]` prefill route for
multi-row-tile grids (`matrix_m > 32`): dispatch becomes 1D over
`n_tiles * 4 * ceil(m_tiles/4)` slots and 4 consecutive workgroups share
one 32-column weight slab across 4 adjacent row tiles, so concurrently
resident workgroups reuse the dequantized weight slab in cache instead
of every row-tile wave re-streaming it. `MLX_OMARCHY_QMM_NO_RASTER=1`
opts back to the shipped x-major order; explicit
`MLX_OMARCHY_QMM_RASTER=swap|2|4|8`, `MLX_OMARCHY_QMM_TWON=1` and
`MLX_OMARCHY_QMM_PERSIST=<rows>` still override for experiments. The
per-output ascending-k f32 accumulation chain is untouched: every twin
is bit-identical to the shipped kernel by construction and pinned by
test (below). The M16 twins and non-FullN builds are out of scope
(low-occupancy grids, raster order irrelevant).

Commits (branch `agent/prefillaxes-qmm-raster`, based on origin/main
`7bcaeb1a4`): `87c64a9d0` (twins + env selection + doctest),
`4e96a8eb0` (landing default). Post-cut main content between
`22cdc6da5` and `7bcaeb1a4` includes the NormApple norm-kernel landing
(`ed83d941d`), which moved main-content generation digests — see
Digests.

## G13C A/B (G13C, diag wheel `0.32.4.dev202610032104+diag.87c64a9`,
5 alternating pairs ctl-first, 1 warmup + 3 timed reps per run,
medians of green-gate rows; n = green pairs per arm)

| model | cell | ctl | swap | g2 | g4 | g8 | twon | persist4 | persist8 |
|---|---|---|---|---|---|---|---|---|---|
| 2B | pf512 (n=5) | 1389.3 | -0.1% | **+1.2%** | +1.0% | +1.0% | -3.5% | -13.9% | -8.4% |
| 2B | pf1024 (n=3-5) | 1397.4 | -1.3% | +2.1% | **+2.4%** | +2.0% | -3.1% | -15.0% | -9.5% |
| 9B | pf512 (n=3) | 295.7 | +2.3% | +3.7% | +4.4% | **+5.0%** | -2.1% | -11.8% | -4.6% |
| 9B | pf1024 (n=1-4) | 294.4 | -22.4% | +3.1% | +3.6% | **+4.8%** | -2.8% | (not rerun) | (not rerun) |

tok/s wall prefill ratios vs ctl; absolute ctl medians in the private
notebook. One 2B window ran under ambient load (load1 up to 5.3): its
rows are excluded by the gate and the cell was rerun clean — the clean
rerun reproduced every ratio. One 9B pf1024 window expired mid-sweep
and was rerun, never spliced. The swap arm's 9B pf1024 -22.4% is
reproduced twice under green gates; reading (labeled inference): at
pf1024 the 9B x tensor is ~16 MB/layer and a full m-major order thrashes
cache re-reading A per column group, while at pf512 (8 MB) it fits —
consistent with swap +2.3% at 9B pf512 and ~0% on 2B.

## G13G confirmation (G13G, diag wheel
`0.32.4.dev202610032249+diag.4e96a8e`, qwen38 protocol
`--new-tokens 32 --limit 10 --passes 3 --warmup 2`, 5 alternating pairs
shipped (`MLX_OMARCHY_QMM_NO_RASTER=1`) first vs landed default,
n=5 per cell)

| model | cell | shipped tok/s | g4 tok/s | ratio |
|---|---|---|---|---|
| 2B | pf512 | 415.1 | 419.4 | +1.0% |
| 2B | pf1024 | 409.8 | 415.2 | +1.3% |
| 4B | pf512 | 157.4 | 160.7 | **+2.1%** |
| 4B | pf1024 | 150.2 | 151.3 | +0.8% |
| 9B | pf512 | 89.2 | 92.2 | **+3.4%** |
| 9B | pf1024 | 88.9 | 91.8 | **+3.2%** |

Sub-bar cells are reported as measured: the default never regressed a
cell on either chip (minimum +0.8%), and clears the 1.5% bar on 9B
(both cells, both chips) and 4B pf512 (both chips). Shipped-arm
absolute numbers reproduce the ledger within noise (4B pf512 157.4 vs
157.2 recorded; 9B pf512 89.2 vs 89.1-90.1). G13G gains are
cell-dependent (2B and 4B pf1024 land +0.8-1.3% where G13C measured
+2.3-2.4% on 2B pf1024) — the wide-matrix ops dominate the win.

## Bit-exactness

- Generation digests (qwen38 `ordered_records_sha256`) are identical
  across arms in every cell on both chips: 2B `509c1920…`, 4B
  `eac9fe32…` (the recorded release pin), 9B `d9892bdf…`.
- Absolute 2B/9B pins differ from the post-cut pins in the prefill-gap
  receipt (`a4ebce…`/`7fc97d…`) because main gained the shape-gated
  NormApple norm kernels (`ed83d941d`) between `22cdc6da5` and this
  branch's base `7bcaeb1a4`; 4B keeps its pre-NormApple value,
  consistent with the shape gate. This branch's backend diff touches
  only the qmm kernel family and its dispatch.
- Doctest (hardware, G13C): "qmm prefill axes twins are bit-identical
  to the shipped route" — baseline pins the SHIPPED order via the
  opt-out env, arms are the landed default plus each explicit env,
  across 8 K,N (2B/4B/9B layer shapes) x 12 odd M in {17..2047},
  bit-equality required.

## Kernel-routing proof (G13C, GPU_PROFILE dispatch census)

Each arm's profiled prefill dispatches its own kernel at n=300 (2
passes x 150 tiles): ctl `kernel_437`
(`QmmPrefillCoopmatBF16X32FullN`), swap 492, g2 493, g4 494, g8 495,
twon 496. The analyzer emits enum ordinals; 437 and 9 (222 casts/pass,
`CastBF16F32`) cross-check the prefill-gap census counts.

## Standing suites (hardware, this branch)

omarchy_runtime_tests 41/41, omarchy_fused_chain_tests 36/36,
omarchy_capability_sim_tests profile matrix 5/5 rc=0,
omarchy_primitive_tests 104/104, omarchy_matmul_family_tests 23/23
(including the new twins doctest running on device). Shader variants
compile under the device `glslc` (2026.3) in the wheel builds on both
chips.

## Closed axes (do not re-run)

TWO_N (two column tiles per A-tile load): -2.1..-3.5% everywhere.
PERSIST (m-axis launch cap 4/8): -4.6..-15.3%. Full swap (m-major): ~0
on 2B, +2.3% 9B pf512, **-22.4%** 9B pf1024 (A-operand cache thrash).
Remaining prefill-parity axes: unchanged from the prefill-gap receipt
(honeykrisp coopmat lowering, dependent-dispatch turnover), plus the
pf512 time breakdown re-profile.

## Artifacts

Private lab: `artifacts/PrefillAxes/` (sweep JSONLs, profiler NDJSON,
provenance, SHA256SUMS); scratch on the hosts cleaned after archiving.
The 9B snapshot had lost its weight shards on the G13C host since the
morning profile; restored byte-exact (shard sha256 verified both ends)
from the G13G host's pinned copy before its cells ran.
