# G13C (Apple M1 Max) SwiGLU Upfront Scale/Bias + ResNormVec Mid-Loop Scale/Bias (`sw_upfront + rn_midsb`)

Date: 2026-11-27
Device: 16-inch Apple M1 Max (G13C, 32 GPU cores, Vulkan / Honeykrisp)
Model: `mlx-community/Qwen3-4B-Instruct-2507-4bit` (`rev 50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`)
Runner contract: verbatim `p45_cell.py`, `MLX_OMARCHY_DEFER_TOKENS=0`, `MLX_DISABLE_COMPILE=1`, `HF_HUB_OFFLINE=1`
macOS matched reference: `104.53 tok/s` (`9.567 ms/tok`, `ctx=512`, `decode-128`, Qwen3-4B-Instruct-2507-4bit)

## Change

Strictly G13C-gated via the existing `*_fast` targets (`qmm_fast_only_default_part` selects `*_fast` on Apple M1 Max `G13C`; `G13G`, `G14C`, and all other parts are untouched and byte-identical):

1. **`qmm_vec_q4_swiglu_bf16_r2s8w2_fast` (`CMakeLists.txt`):** Drop `-DQMM_SW_SINKSB=1` so the G13C fast SwiGLU kernel issues its 8 scale/bias loads upfront before `Q4_QUAD_SWIGLU`. Under Honeykrisp's two-slot `AGX_SMART_WAITS` (`joshuaswarren/mesa-1#15`, commit `f6ffdc2b4a6f`), upfront scale/bias overlaps cleanly behind all four `Q4_QUAD_SWIGLU` FMA blocks (64 FMAs) instead of stalling after `xb`.
2. **`qmm_vec_q4_resnorm_bf16_r2s8w2_vec_fast` (`CMakeLists.txt`, `shaders/qmm_vec.comp`):** Add `-DQMM_Q4_MIDSB=1` (`Q4_ROWS_LOWREG_BODY`), hoisting `xa` and `xb` together and issuing `scale_r`/`bias_r` loads between `xa` and `xb` so `xb`'s 16 FMAs hide scale/bias latency on both shipped Mesa (`1650fc1a33`) and `mesa-1#15` (`f6ffdc2b4a6f`).

## Matched Wheel Artifacts

Built from merged `origin/main` (`92cdc173cb`, post-#1059/#1063) + this exact 2-file diff (`overlay/mlx/backend/omarchy/CMakeLists.txt`, `overlay/mlx/backend/omarchy/shaders/qmm_vec.comp`):

- Base `libmlx.so` SHA-256 (prefix verified via `/proc/self/maps`): `318c259103e59e66`
- Candidate `libmlx.so` SHA-256 (prefix verified via `/proc/self/maps`): `e3679d3512862643`
- Merged Mesa fork (`joshuaswarren/mesa-1#15`, commit `f6ffdc2b4a6f`, zero Mesa env vars): `2efcab53479ce266`
- Shipped Mesa (`1650fc1a33`, zero Mesa env vars): `e3b2c831489a1b4d`

## Correctness (0-ULP Bit-Exact)

- Software Vulkan (`lavapipe`) direct kernel comparison across 9 Qwen3-4B decode tensor shapes (`gate_up` SwiGLU and `o_proj`/`down_proj` ResNormVec): **0 ULP bit-identical** against base (`ALL_7_ARMS_0_ULP_IDENTICAL_ON_LAVAPIPE`).
- `glslangValidator -V --target-env vulkan1.3` + `spirv-val --target-env vulkan1.3`: clean on all 4 SwiGLU and ResNormVec variants (default + fast).
- `python3 scripts/check_compute_kernel_wiring.py`: clean (`807 declared, 803 referenced, 801 dispatch rows, 800 shader targets`).
- All full-model decode runs on Apple M1 Max matched the exact greedy token digest:
  - `ctx=512`, `decode-128`: `efb521ae04afaad6f2309bc14d21462c72f0a5c836832c1a18d4ddb439e7c7f9`
  - `ctx=2048`, `decode-128`: `542524d1f692f392f03221d79b8f8cc4345b3247072bf49e7572084bca2118e8`
- Zero GPU timeouts (`GPU-TIMEOUTS delta=0`, `FOREIGN-GPU-USED count=0`).

## Measured Performance — Merged Mesa Fork (`joshuaswarren/mesa-1#15`, `f6ffdc2b4a6f`)

### `ctx=512`, `decode-128` (5 pairs × ABBA, 5 timed samples/run = 50 timed samples/arm)

| Pair | Base ABBA (`tok/s`) | Cand ABBA (`tok/s`) | Delta (`%`) | Delta (`ms/tok`) |
|---:|---:|---:|---:|---:|
| 1 | `104.25` (`104.21, 104.30`) | `105.32` (`105.28, 105.36`) | `+1.03%` | `-0.098` |
| 2 | `104.13` (`104.26, 104.00`) | `105.29` (`105.15, 105.43`) | `+1.11%` | `-0.106` |
| 3 | `103.95` (`103.92, 103.99`) | `105.47` (`105.50, 105.43`) | `+1.46%` | `-0.138` |
| 4 | `104.01` (`103.91, 104.11`) | `105.46` (`105.38, 105.54`) | `+1.40%` | `-0.133` |
| 5 | `104.00` (`103.91, 104.08`) | `105.41` (`105.54, 105.28`) | `+1.36%` | `-0.129` |
| **Pair Median** | **`104.01`** (`9.615 ms`) | **`105.41`** (`9.487 ms`) | **`+1.35%`** | **`-0.128`** |
| **All-50 Pooled** | **`103.98`** | **`105.35`** (peak **`105.62`**) | **`+1.31%`** | **`-0.124`** |

- Disjointness: **5/5 disjoint wins** (`min(cand pairs) = 105.29 > max(base pairs) = 104.25`).
- vs. macOS matched reference (`104.53 tok/s`, `9.567 ms/tok`): **`105.41 tok/s` (`+0.88 tok/s`, `+0.84%`, `-0.080 ms/tok` ahead of macOS)**.

### `ctx=2048`, `decode-128` (3 pairs × ABBA, 5 timed samples/run = 30 timed samples/arm)

| Pair | Base ABBA (`tok/s`) | Cand ABBA (`tok/s`) | Delta (`%`) | Delta (`ms/tok`) |
|---:|---:|---:|---:|---:|
| 1 | `90.65` (`90.59, 90.72`) | `91.36` (`91.30, 91.42`) | `+0.78%` | `-0.085` |
| 2 | `90.61` (`90.51, 90.72`) | `91.38` (`91.49, 91.28`) | `+0.85%` | `-0.093` |
| 3 | `90.71` (`90.78, 90.65`) | `91.37` (`91.40, 91.34`) | `+0.72%` | `-0.079` |
| **Pair Median** | **`90.65`** (`11.031 ms`) | **`91.37`** (`10.945 ms`) | **`+0.79%`** | **`-0.086`** |
| **All-30 Pooled** | **`90.64`** | **`91.36`** (peak **`91.59`**) | **`+0.79%`** | **`-0.087`** |

- Disjointness: **3/3 disjoint wins**.

## Measured Performance — Shipped Mesa (`1650fc1a33`)

Both shipped-Mesa checks confirm net-positive / zero-regression behavior even prior to packaging `mesa-1#15` (`rn_midsb` offsets `sw_upfront` on older Mesa):

### `ctx=512`, `decode-128` (5 pairs × ABBA, 5 timed samples/run = 50 timed samples/arm)

| Pair | Base ABBA (`tok/s`) | Cand ABBA (`tok/s`) | Delta (`%`) | Delta (`ms/tok`) | Notes |
|---:|---:|---:|---:|---:|---|
| 1 | `102.46` (`102.39, 102.53`) | `102.73` (`102.71, 102.75`) | `+0.27%` | `-0.026` | Quiet |
| 2 | `97.25` (`102.42, 92.08*`) | `102.71` (`102.87, 102.56`) | `+5.62%` | `-0.547` | `*` bg CPU load on A2 |
| 3 | `96.36` (`90.50*, 102.21`) | `102.74` (`102.78, 102.69`) | `+6.62%` | `-0.645` | `*` bg CPU load on A1 |
| 4 | `102.41` (`102.34, 102.47`) | `91.43` (`90.78*, 92.08*`) | `-10.71%` | `+1.172` | `*` bg CPU load on B1/B2 |
| 5 | `102.30` (`102.32, 102.29`) | `102.73` (`102.70, 102.76`) | `+0.42%` | `-0.041` | Quiet |
| **Pair Median** | **`102.30`** (`9.775 ms`) | **`102.73`** (`9.735 ms`) | **`+0.42%`** | **`-0.041`** | 4/5 wins; quiet pairs `+0.27%`/`+0.42%` |
| **All-50 Pooled** | **`102.28`** | **`102.70`** | **`+0.41%`** | **`-0.040`** | — |

### `ctx=512`, `decode-128` quiet follow-up (3 pairs × ABBA, 30 timed samples/arm, zero background load, warm chassis)

| Pair | Base ABBA (`tok/s`) | Cand ABBA (`tok/s`) | Delta (`%`) | Delta (`ms/tok`) |
|---:|---:|---:|---:|---:|
| 1 | `100.16` (`100.15, 100.16`) | `100.19` (`100.24, 100.15`) | `+0.04%` | `-0.004` |
| 2 | `100.16` (`100.17, 100.15`) | `100.22` (`100.19, 100.26`) | `+0.06%` | `-0.006` |
| 3 | `100.11` (`100.21, 100.01`) | `100.23` (`100.25, 100.22`) | `+0.13%` | `-0.013` |
| **Pair Median** | **`100.16`** (`9.984 ms`) | **`100.22`** (`9.978 ms`) | **`+0.07%`** | **`-0.007`** |

- Disjointness: **3/3 disjoint wins** (`all30: 100.15 -> 100.23`, peak `100.41`).
