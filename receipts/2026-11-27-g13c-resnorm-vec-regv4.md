# G13C (Apple M1 Max) ResNormVec Zero-Shmem Single-Barrier Coalesced `uvec4` Epilogue + Pack32 + Dead `output0` Elision (`rn_regv4`)

Date: 2026-11-27
Device: 16-inch Apple M1 Max (G13C, 32 GPU cores, Vulkan / Honeykrisp)
Models:
- `mlx-community/Qwen3-4B-Instruct-2507-4bit` (`rev 50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`)
- `mlx-community/Qwen3.5-9B-MLX-4bit` (`rev 938d8919941c6e7efd3c7150eff7fe9d12afa631`)
Runner contract: verbatim `p45_cell.py` (`decode-128`, `MLX_OMARCHY_DEFER_TOKENS=0`, `MLX_DISABLE_COMPILE=1`, `HF_HUB_OFFLINE=1`)
macOS matched references:
- 4B `ctx=512`, `decode-128`: `104.53 tok/s` (`9.567 ms/tok`)
- 9B `ctx=512`, `decode-128`: `57.77 tok/s` (`17.310 ms/tok`)

## Change

Strictly G13C-gated via the existing `QmmVecQ4ResNormBF16R2S8W2VecFast` target (`qmm_fast_only_default_part` selects `*_fast` on Apple M1 Max `G13C`; non-G13C `ResNormVec`, `ResNormVec4 Fast`, and `SwiGLU Fast` SPIR-V binaries are byte-identical to `main`):

1. **`QMM_RESNORM_PACK32` + dead `output0` store elision (`0x1000000u`):** Extends `QMM_RESNORM_PACK32` to `QMM_RESNORM_VEC` (guarded in `primitives.cpp` by 4-byte alignment on `output0`/`addend0`, 16-byte alignment on `sum0`/`norm_weight`/`norm_output`, `lhs_size != 1`, and `(rhs_offset & 7) == 0`; falls back to `QmmVecQ4ResNormBF16R2S8W2Vec` otherwise). Halves epilogue `addend0`/`sum0` scalar accesses and skips dead `output0` stores when `EagerFusionScope` proves the QMM output has only the fused residual-add consumer (`dead_qmm_out`).
2. **`QMM_RESNORM_VEC_REG_V4` (zero-shmem, single-barrier coalesced 128-bit `uvec4` RMSNorm epilogue):** For `D in {2048, 2560, 4096}` (`n_v4 in {256, 320, 512}`), eliminates `shared uint s_resnorm_sum[2048]` (`8 KB` shared memory -> `0 KB`), drops three workgroup barriers per dispatch (`-216` workgroup barriers/token on 4B), and has threads `0..255` load 1 or 2 `uvec4` words each from `sum0_v4` and `norm_weight_v4` before the single reduction barrier, reduce via broadcast `subgroupAdd`, and write `norm_output_v4` as coalesced 128-bit `uvec4` stores (zero extra bindings, reusing bindings 7 and 11).

## Matched Build Artifacts

Built from `origin/main` (`9d89e4b80`, post-#1077) + this exact change:

- Base `core.cpython-314-aarch64-linux-gnu.so` SHA-256 (prefix verified via `/proc/self/maps`): `29fb2de2c7f365df`
- Candidate `core.cpython-314-aarch64-linux-gnu.so` SHA-256 (prefix verified via `/proc/self/maps`): `0eef791ec4c84236`
- Shipped Mesa (`1650fc1a33`, zero Mesa env vars): `e3b2c831489a1b4d`
- Merged Mesa fork (`joshuaswarren/mesa-1#15`, `f6ffdc2b4a6f`, zero Mesa env vars): `2efcab53479ce266`

## Correctness (0-ULP Bit-Exact)

- Live Vulkan GPU `omarchy_qmm_geometry_tests`: **`GEOM-TEST-PASS failures 0`**, including new G13C `ResNormVec Fast` 0-ULP checks (`resnorm2048_g13c_resnorm_vec_fast`, `resnorm2048_liveqmm_g13c_resnorm_vec_fast`, `resnorm4096_g13c_resnorm_vec_fast`).
- Live 32-lane Vulkan direct kernel check (`kernel=794` verified via `MLX_OMARCHY_TRACE_DISPATCH=1`): **22/22 0-ULP bit-identical** across 4B (`2560x4096`, `2560x9728`) and 9B (`4096x4096`, `4096x12288`) for both `dead_qmm_out=true` and `dead_qmm_out=false`.
- `glslangValidator -V --target-env vulkan1.3` + `spirv-val --target-env vulkan1.3`: clean; `cmp` verified **byte-identical SPIR-V** against `origin/main` for non-G13C `ResNormVec`, `ResNormVec4 Fast`, and `SwiGLU Fast`.
- `python3 scripts/check_compute_kernel_wiring.py`: clean (`815 declared, 811 referenced, 809 dispatch rows, 808 shader targets`).
- All full-model decode runs on Apple M1 Max matched exact greedy token digests:
  - 4B `ctx=512`: `efb521ae04afaad6f2309bc14d21462c72f0a5c836832c1a18d4ddb439e7c7f9`
  - 4B `ctx=2048`: `542524d1f692f392f03221d79b8f8cc4345b3247072bf49e7572084bca2118e8`
  - 9B `ctx=512`: `697272134f422788...` (exact match to macOS Metal)
- Zero GPU timeouts (`GPU-TIMEOUTS delta=0`, `FOREIGN-GPU-USED count=0`).

## Measured Performance — Shipped Mesa (`1650fc1a33`, Zero Mesa Fork)

### 4B `ctx=512`, `decode-128` (5 pairs × ABBA, 50 timed samples/arm)

| Pair | Base ABBA (`tok/s`) | Cand ABBA (`tok/s`) | Delta (`%`) | Delta (`ms/tok`) |
|---:|---:|---:|---:|---:|
| 1 | `102.60` (`102.55, 102.65`) | `104.80` (`104.98, 104.63`) | `+2.15%` | `-0.205` |
| 2 | `102.67` (`102.53, 102.82`) | `105.02` (`104.92, 105.12`) | `+2.29%` | `-0.218` |
| 3 | `102.40` (`102.36, 102.44`) | `104.87` (`104.94, 104.81`) | `+2.41%` | `-0.230` |
| 4 | `102.65` (`102.74, 102.55`) | `105.03` (`104.93, 105.12`) | `+2.32%` | `-0.221` |
| 5 | `102.69` (`102.66, 102.73`) | `104.99` (`104.92, 105.07`) | `+2.24%` | `-0.213` |
| **Pair Median** | **`102.65`** (`9.742 ms`) | **`104.99`** (`9.524 ms`) | **`+2.29%`** | **`-0.218`** |
| **All-50 Pooled** | **`102.59`** | **`104.92`** (peak **`105.20`**) | **`+2.27%`** | **`-0.216`** |

- Disjointness: **5/5 disjoint wins** (`min(cand pairs) = 104.80 > max(base pairs) = 102.69`).
- vs. macOS matched reference (`104.53 tok/s`, `9.567 ms/tok`): **`104.99 tok/s` (`+0.46 tok/s`, `+0.44%`, `-0.042 ms/tok` ahead of macOS on stock shipped Mesa)**.

### 4B `ctx=2048`, `decode-128` (3 pairs × ABBA, 30 timed samples/arm)

| Pair | Base ABBA (`tok/s`) | Cand ABBA (`tok/s`) | Delta (`%`) | Delta (`ms/tok`) |
|---:|---:|---:|---:|---:|
| 1 | `92.31` (`92.50, 92.12`) | `94.46` (`94.41, 94.50`) | `+2.33%` | `-0.247` |
| 2 | `92.73` (`92.84, 92.61`) | `94.37` (`94.29, 94.44`) | `+1.77%` | `-0.187` |
| 3 | `92.61` (`92.77, 92.45`) | `94.38` (`94.55, 94.22`) | `+1.91%` | `-0.203` |
| **Pair Median** | **`92.61`** (`10.798 ms`) | **`94.38`** (`10.595 ms`) | **`+1.91%`** | **`-0.203`** |
| **All-30 Pooled** | **`92.54`** | **`94.42`** (peak **`94.71`**) | **`+2.03%`** | **`-0.215`** |

- Disjointness: **3/3 disjoint wins**.

## Measured Performance — Merged Mesa Fork (`joshuaswarren/mesa-1#15`, `f6ffdc2b4a6f`)

### 4B `ctx=512`, `decode-128` (5 pairs × ABBA, 50 timed samples/arm)

| Pair | Base ABBA (`tok/s`) | Cand ABBA (`tok/s`) | Delta (`%`) | Delta (`ms/tok`) |
|---:|---:|---:|---:|---:|
| 1 | `105.14` (`105.15, 105.14`) | `107.26` (`107.10, 107.41`) | `+2.01%` | `-0.187` |
| 2 | `105.08` (`105.20, 104.95`) | `107.08` (`107.18, 106.98`) | `+1.91%` | `-0.178` |
| 3 | `105.27` (`105.15, 105.39`) | `106.98` (`107.04, 106.92`) | `+1.62%` | `-0.152` |
| 4 | `105.16` (`105.24, 105.08`) | `107.29` (`107.33, 107.25`) | `+2.03%` | `-0.189` |
| 5 | `105.15` (`105.07, 105.24`) | `107.24` (`107.25, 107.23`) | `+1.99%` | `-0.185` |
| **Pair Median** | **`105.15`** (`9.510 ms`) | **`107.24`** (`9.325 ms`) | **`+1.99%`** | **`-0.185`** |
| **All-50 Pooled** | **`105.15`** | **`107.13`** (peak **`107.49`**) | **`+1.88%`** | **`-0.176`** |

- Disjointness: **5/5 disjoint wins** (`+2.71 tok/s`, `+2.59%`, `-0.242 ms/tok` ahead of macOS `104.53`).

### 9B `ctx=512`, `decode-128` (3 pairs × ABBA, 30 timed samples/arm)

| Pair | Base ABBA (`tok/s`) | Cand ABBA (`tok/s`) | Delta (`%`) | Delta (`ms/tok`) |
|---:|---:|---:|---:|---:|
| 1 | `56.60` (`56.59, 56.60`) | `57.42` (`57.44, 57.41`) | `+1.46%` | `-0.254` |
| 2 | `56.59` (`56.59, 56.59`) | `57.47` (`57.50, 57.44`) | `+1.56%` | `-0.272` |
| 3 | `56.61` (`56.64, 56.57`) | `57.50` (`57.54, 57.46`) | `+1.57%` | `-0.274` |
| **Pair Median** | **`56.60`** (`17.668 ms`) | **`57.47`** (`17.399 ms`) | **`+1.55%`** | **`-0.269`** |
| **All-30 Pooled** | **`56.59`** | **`57.45`** (peak **`57.64`**) | **`+1.52%`** | **`-0.264`** |

- Disjointness: **3/3 disjoint wins** (`-0.51%` / `+0.089 ms/tok` vs macOS `57.77 tok/s`, exact greedy token digest match).
