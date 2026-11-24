# G13G 64-row SG4 + STEPK=32 coopmat prefill default (2026-11-24)

Default the landed G4+LdsPad bf16-hybrid qmm prefill route on `G13G`
(Apple M1, 8-core) to the 4-subgroup 64-row tile (`64sg4`) paired with
`STEP_K=32` when `M >= 128` and the 64-row grid fills the part
(`>= 48` workgroups). Small grids that fall back to 32 rows, disallowed
shapes/formats, and other chips keep their prior defaults (`STEP_K=16`
where unset).

## Setup

- Device: Apple M1 (`G13G B1`, 8 GPU cores)
- Runtime: `mlx-omarchy` `v0.7.53` (`0.32.4.dev202611240009+197b068`)
- Control (`stock`): default unset (`32`-row `G4+LdsPad`, `STEP_K=16`)
- Candidate (`64sg4+k32`): `MLX_OMARCHY_QMM_COOPMAT_ROWS=64sg4`,
  `MLX_OMARCHY_QMM_COOPMAT_STEPK=32`
- Harness: interleaved balanced ABBA (`AB, BA, ...`), fresh subprocess per
  arm after one `ctx=64` warmup, `seed=292`, `max_tokens=16` greedy

## Measurements

### SiddhJagani/Qwen3.8-2B-mlx-4Bit (5 pairs, 10 scored runs)

| ctx | arm | pf median tok/s (range) | pf median ms/tok | decode tok/s | greedy ids sha256[:16] |
| --- | --- | --- | --- | --- | --- |
| 512 | stock | 427.47 (426.37–427.81) | 2.33937 | 50.45 | `b2a43f19aedd07bf` |
| 512 | cand | 459.59 (459.12–461.96) **+7.52%** | 2.17585 | 50.20 | `b2a43f19aedd07bf` |
| 2048 | stock | 434.31 (433.15–435.67) | 2.30249 | 49.49 | `4813b1a32844c01c` |
| 2048 | cand | 473.46 (473.21–473.83) **+9.01%** | 2.11209 | 49.39 | `4813b1a32844c01c` |

5/5 pairs won on both context lengths with disjoint intervals and
identical generated token IDs.

### mlx-community/Qwen3-4B-Instruct-2507-4bit (3 pairs, 6 scored runs)

| ctx | arm | pf median tok/s (range) | pf median ms/tok | decode tok/s | greedy ids sha256[:16] |
| --- | --- | --- | --- | --- | --- |
| 512 | stock | 178.45 (178.41–178.46) | 5.60387 | 24.50 | `05eb62e14aaba9cc` |
| 512 | cand | 197.93 (197.56–198.17) **+10.92%** | 5.05231 | 24.65 | `05eb62e14aaba9cc` |
| 2048 | stock | 152.39 (151.98–152.44) | 6.56190 | 22.30 | `f80f2037a7b89d5a` |
| 2048 | cand | 168.56 (168.54–168.61) **+10.61%** | 5.93254 | 22.30 | `f80f2037a7b89d5a` |

3/3 pairs won on both context lengths with disjoint intervals and
identical generated token IDs.

## Why STEP_K=32 stays gated to G13G + 64sg4

On the shipped 32-row tile (`Qwen3.8-2B`, single-screen comparison vs
stock median `430.66 tok/s` at `ctx=2048`), standalone `STEP_K=32`
measured `411.00 tok/s` (`-4.56%`), and `64sg4` without `STEP_K=32`
measured `445.29 tok/s` (`+3.40%`). Pairing `64sg4` + `STEP_K=32` hits
`464.36 tok/s` (`+7.83%`). `qmm_coopmat_stepk_requested` therefore
defaults to `32` only when `rows == 64 && sg4 && G13G`, so 32-row fallback
grids and other chips keep `STEP_K=16`.

## Kill switches

- `MLX_OMARCHY_QMM_COOPMAT_ROWS=32` restores the 32-row `K=16` path.
- `MLX_OMARCHY_QMM_COOPMAT_ROWS=64` selects the non-`SG4` 64-row `K=16` path.
- `MLX_OMARCHY_QMM_COOPMAT_STEPK=16` keeps 64-row `SG4` on `STEP_K=16`.
