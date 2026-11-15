# G16S 64-row qmm prefill tile (2026-11-15)

Default the landed G4+LdsPad bf16-hybrid qmm prefill route to a 64-row
tile on G16S when M ≥ 128 and the 64-row grid still fills the part.
One dequantized weight tile is reused across twice the output rows;
every output keeps the shipped ascending-k f32 chain.

## Setup

- Device: Apple M4 Pro (G16S B1)
- ICD: aurora-wip + Mesa #350 + #353 + #343 + #355
- Model: mlx-community/Qwen3-8B-4bit (bf16 activations, 4-bit group-64)
- Wheel: mlx-omarchy 0.32.4.dev202611150542+6ffa1e6a9
- 5 interleaved runs, 32 greedy tokens, `MESA_SHADER_CACHE_DISABLE=true`

## Numerics

Isolated `quantized_matmul` against the 32-row f32 path, bf16→f32 widen
(exact) compared for equality, shapes (M,N,K) in
{128,520,2060,16}×{4096,12288,1024}×4096: every 64-row, STEP_K=32,
CHEAP_DEQ and SG4 arm was bit-identical (`ALL_MATCH`).

32-token greedy sha identical in every run of every arm:
`a8bdfde11f24` (pf520), `1dd5a67f8afc` (pf2060).

CHEAP_DEQ (`(0x4B000000|nib) - 8388608.0`) is bit-identical to
`float(nibble)` for 0–15 and 0–255 on the host as well.

## Prefill medians (tok/s)

| arm | pf520 median (range) | pf2060 median (range) |
| --- | --- | --- |
| base (32-row G4+LdsPad) | 329.82 (325.56–331.33) | 317.78 (307.53–318.76) |
| TILE_ROWS=64 | 334.46 (329.77–336.16) **+1.41%**, overlap | 338.75 (338.17–339.94) **+6.60%**, disjoint |
| 64 + CHEAP_DEQ | 330.87 (329.29–331.77) +0.32% | 337.55 (337.27–339.62) +6.22%, disjoint |
| 64sg4 (128-lane) | 307.56 (304.40–312.11) **−6.75%**, disjoint | 317.64 (316.71–319.05) −0.04% |

Decode was flat to slightly up on the 64-row arm (med 40.34→40.96 at
pf520, 32.53→33.08 at pf2060).

STEP_K=32 was bit-identical but not an e2e win in the one-run smoke and
is left env-gated (`MLX_OMARCHY_QMM_COOPMAT_STEPK=32`). CHEAP_DEQ is
likewise env-gated; it did not add to the 64-row default. SG4 is
opt-in only (`=64sg4`).

Kill switch on G16S: `MLX_OMARCHY_QMM_COOPMAT_ROWS=32`. Other chips
keep the 32-row default unless `=64` is set.

## "Device gates pending"

`origin/experiment/main-qmm-step32-20260921` (2026-09-21) widened
STEP_K to 32 on the old staged f16 path and left "device gates pending":
an offline staging-index check only, not a production default. Those
gates on G16S are the GPU bit-identity checks and the A/B above.
STEP_K=32 passes identity and does not win e2e; the 64-row tile does
both.

CHUNK_DEQ (a 64-wide K tile, 9 KiB LDS) previously lost 16–28% on G14C
and was not re-defaulted.
