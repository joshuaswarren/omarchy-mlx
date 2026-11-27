# G13C (Apple M1 Max) SdpaOnlineRope 2-Barrier Prologue and Preloaded Epilogue (`sdpa_2bar_epifast`)

Date: 2026-11-27
Device: 16-inch Apple M1 Max (G13C, 32 GPU cores, Vulkan / Honeykrisp)
Model: `mlx-community/Qwen3-4B-Instruct-2507-4bit` (`rev 50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`)
Runner contract: verbatim `p45_cell.py`, `MLX_OMARCHY_DEFER_TOKENS=0`, `MLX_DISABLE_COMPILE=1`, `HF_HUB_OFFLINE=1`

## Change

Single-file edit in `overlay/mlx/backend/omarchy/shaders/sdpa_decode_online_rope.comp` (HD=128 path, `+49/-35` lines; zero new kernel enums or flags):

- **2-barrier prologue (-36 full 1024-thread workgroup barriers per decode token):** In Prologue Phase B (`t = 0..63`), rotated pre-scaled Q is written into a disjoint `shared float s_q[128]` (512 B, total shared memory `17,536 B`, still 1024t max occupancy on G13C) while rotated K is packed directly into `s_k_words[64]` via `subgroupShuffleDown(..., 1u)` within subgroups 0 and 1. Because neither `s_q` nor `s_k_words` is written after `barrier()` #2, `barrier()` #3 is eliminated with zero WAR hazard against `s_o`.
- **Preloaded epilogue:** Preloads all four `s_o[rbase + 0..3] * w` values before the four `subgroupAdd` reductions and replaces four `acc / denom` divisions in `lane == 0` with one `inv_d = 1.0 / denom` and four multiplies (preserving `SDPA_OUTGATE`).

## Matched Wheel Artifacts

Built from merged `origin/main` (#1049 + #1050 + #1055) + 1-file shader diff (`overlay/mlx/backend/omarchy/shaders/sdpa_decode_online_rope.comp`):

- Base `libmlx.so` SHA-256: `c723b0a68c7f64336c1d81d6f2536679f6f6b306145327e8adec643523244f28`
- Candidate `libmlx.so` SHA-256: `f808d3556713a70d2efefa71d1f07ba70c69cd031c800adc07ceb7c422b40443`

## Correctness (0-ULP Bit-Exact)

Direct `SdpaOnlineRope` HD=128 output and written K-cache bit-exact check against base across 5 sequence lengths:

- `k_len=1`: `y_exact=True`, `k_exact=True`
- `k_len=17`: `y_exact=True`, `k_exact=True`
- `k_len=64`: `y_exact=True`, `k_exact=True`
- `k_len=512`: `y_exact=True`, `k_exact=True`
- `k_len=577`: `y_exact=True`, `k_exact=True`

All full-model decode runs matched the exact greedy token digest:
- `ctx=512`, `decode-128`: `efb521ae04afaad6f2309bc14d21462c72f0a5c836832c1a18d4ddb439e7c7f9`
- `ctx=2048`, `decode-128`: `542524d1f692f392f03221d79b8f8cc4345b3247072bf49e7572084bca2118e8`

Zero GPU timeouts (`GPU-TIMEOUTS delta=0`, `FOREIGN-GPU-USED count=0`).

## Measured Performance (5-Pair Interleaved ABBA, `DEFER_TOKENS=0`)

### `ctx=512`, `decode-128` (5 pairs × 5 timed samples/arm = 25 samples/arm)

| Pair | Order | Base (`tok/s`) | Cand (`tok/s`) | Delta (`%`) | Delta (`ms/tok`) |
|---:|---|---:|---:|---:|---:|
| 1 | base -> cand | `100.29` | `100.80` | `+0.50%` | `-0.050` |
| 2 | cand -> base | `100.30` | `100.55` | `+0.25%` | `-0.025` |
| 3 | base -> cand | `100.59` | `100.69` | `+0.11%` | `-0.011` |
| 4 | cand -> base | `100.33` | `100.72` | `+0.39%` | `-0.039` |
| 5 | base -> cand | `100.25` | `100.34` | `+0.09%` | `-0.009` |
| **Pair Median** | — | **`100.30`** | **`100.69`** | **`+0.39%`** | **`-0.039`** |
| **All-25 Pooled** | — | **`100.30`** | **`100.58`** | **`+0.28%`** | **`-0.028`** |

### `ctx=2048`, `decode-128` (3 pairs)

| Pair | Order | Base (`tok/s`) | Cand (`tok/s`) | Delta (`%`) | Delta (`ms/tok`) |
|---:|---|---:|---:|---:|---:|
| 1 | base -> cand | `90.79` | `91.14` | `+0.38%` | `-0.041` |
| 2 | cand -> base | `90.88` | `91.12` | `+0.27%` | `-0.029` |
| 3 | base -> cand | `90.62` | `90.64` | `+0.01%` | `-0.001` |
| **Median** | — | **`90.79`** | **`91.12`** | **`+0.36%`** | **`-0.040`** |
