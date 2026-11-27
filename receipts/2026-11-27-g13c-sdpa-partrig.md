# G13C (Apple M1 Max) SdpaOnlineRope Parallel Phase-A RoPE Trig (`sdpa_partrig`)

Date: 2026-11-27
Device: 16-inch Apple M1 Max (G13C, 32 GPU cores, Vulkan / Honeykrisp)
Model: `mlx-community/Qwen3-4B-Instruct-2507-4bit` (`rev 50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`)
Runner contract: verbatim `p45_cell.py`, `MLX_OMARCHY_DEFER_TOKENS=0`, `MLX_DISABLE_COMPILE=1`, `HF_HUB_OFFLINE=1`

## Change

Single-file edit in `overlay/mlx/backend/omarchy/shaders/sdpa_decode_online_rope.comp` (HD=128 path):

- Previously, Prologue Phase A ran Q/K RMSNorm on subgroups `0..1` while subgroups `2..31` sat idle at `barrier()`, and Prologue Phase B then evaluated all 64 `offset_data` loads + `exp(-t * beta)` + `omarchy_trig_cos` / `omarchy_trig_sin` values before rotating Q and K.
- Now, subgroups `2..3` (`t = 64..127`) compute the 64 `(cos, sin)` pairs in parallel during Phase A and stage them into existing unused `s_o[512..639]` shared memory (zero additional shared memory, zero new flags or kernel enums). Phase B reads `s_o[512 + t]` and `s_o[576 + t]` directly.
- Overlaps `offset_data` load latency and Cody-Waite trig ALU behind Q/K RMSNorm global loads and keeps trig register live ranges out of Phase B.

## Matched Wheel Artifacts

Built from identical `origin/main` + 1-file shader diff (`overlay/mlx/backend/omarchy/shaders/sdpa_decode_online_rope.comp`):

- Base `libmlx.so` SHA-256: `cb0afb161ff3f7437e94ec78777cec806e8d8ab36b299a8ddeb7119e721c6662`
- Candidate `libmlx.so` SHA-256: `58d51ede9ac845016ff912e01b02d9ff79d82160ffaca7df2ffae054a7c3a930`

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
| 1 | base -> cand | `95.42` | `96.60` | `+1.24%` | `-0.129` |
| 2 | cand -> base | `85.27`* | `96.49` | `+13.16%` | `-1.364` |
| 3 | base -> cand | `85.49`* | `96.25` | `+12.59%` | `-1.308` |
| 4 | cand -> base | `95.47` | `96.64` | `+1.23%` | `-0.127` |
| 5 | base -> cand | `95.28` | `85.46`* | `-10.31%` | `+1.206` |
| **Pair Median** | — | **`95.28`** | **`96.49`** | **`+1.27%`** | **`-0.132`** |
| **All-25 Pooled** | — | **`95.23`** | **`96.33`** | **`+1.15%`** | **`-0.120`** |

`*` Transient single-OPP GPU DVFS step-down (`~85.3 tok/s`) observed in 2 base runs (pairs 2, 3) and 1 candidate run (pair 5). Both clean pairs (1 and 4) agree tightly at `+1.23%` to `+1.24%` (`-0.127` to `-0.129 ms/tok`).

### `ctx=2048`, `decode-128` (3 pairs)

| Pair | Order | Base (`tok/s`) | Cand (`tok/s`) | Delta (`%`) | Delta (`ms/tok`) |
|---:|---|---:|---:|---:|---:|
| 1 | base -> cand | `87.23` | `87.77` | `+0.62%` | `-0.070` |
| 2 | cand -> base | `87.30` | `87.49` | `+0.22%` | `-0.025` |
| 3 | base -> cand | `87.15` | `87.33` | `+0.20%` | `-0.023` |
| **Median** | — | **`87.23`** | **`87.49`** | **`+0.30%`** | **`-0.034`** |

### Stacked Verification (atop PR #1049 + PR #1050, `ctx=512`, `DEFER_TOKENS=0`)

3-round interleaved screen atop `#1049 + #1050`:
- Base (`#1049 + #1050`): `99.20 tok/s` (`[99.11, 99.20, 99.35]`)
- Candidate (`+ sdpa_partrig`): **`100.35 tok/s`** (`[100.16, 100.35, 100.35]`, **`+1.16%`, `-0.115 ms/tok`, 3/3 disjoint wins**)
