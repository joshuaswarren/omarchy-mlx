# 2026-11-26: G13C (Apple M1 Max) 128-bit `uvec4` Coherent-L2 ResNorm Epilogue (`QmmVecQ4ResNormBF16R2S8W2Vec`)

## Change

- Added `ComputeKernel::QmmVecQ4ResNormBF16R2S8W2Vec` (`shaders/qmm_vec.comp`, `-DQMM_RESNORM_VEC=1`), using cooperative 128-bit `uvec4` coherent-L2 reads into `shared uint s_resnorm_sum[2048]` (16× fewer coherent L2 reads than 20 scalar `uint16_t` loads/thread: 320 `uvec4` loads at `D=2560`, 1.25 loads/thread) plus packed 32-bit `uint` stores to `norm_output` in the last-workgroup `R2S8W2` ResNorm epilogue, with zero extra QMV registers.
- Added `omarchy::resnorm_vec_default_part` in `overlay/mlx/backend/omarchy/device.h`: defaults ON on `G13C` (Apple M1 Max) for `2x8w2` `ResNorm` widths `{2048, 2560, 4096}` when `binding_limit >= 12`; `MLX_OMARCHY_RESNORM_VEC=1/0` overrides on any qualifying `2x8w2` part.
- Added unit test `resnorm vec default: the part rule and env override` in `overlay/tests/omarchy/test_vocab_tail.cpp`.

## Environment & Protocol

- **Hardware:** 16-inch MacBook Pro, Apple M1 Max (`G13C C0`), Linux `7.1.12`, Honeykrisp Vulkan ICD.
- **Model:** `mlx-community/Qwen3-4B-Instruct-2507-4bit` (`50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`).
- **Benchmark contract:** `MLX_OMARCHY_DEFER_TOKENS=0` (asserted in runner), `MLX_DISABLE_COMPILE=1`, `HF_HUB_OFFLINE=1`, `decode-128`, 1 discarded warmup per leg, alternating `base -> cand` / `cand -> base` pairs.
- **Measured commits & shared libraries:**
  - Base: `d21f3159b` (`libmlx.so` sha256 `ced0c3abf891a2da1d805d54a17e901ab63f4610fb61666c8fc70e905791e692`)
  - Candidate: `fe49cad1d` (`libmlx.so` sha256 `9b03f4cfc3c23b9b8daae0e9b5ea77e88a8f309d6fdce34418e7a8d5c27e10f7`; PR branch carries the byte-identical 7-file implementation ported onto `b05298c5d`)
- **Runner:** verbatim `p45_cell.py` (`sha256: 85bb63b5561153a260908d5e28054d9afdbe0ef345711ebb9851c82a74c7c85d`), 5 timed samples/leg at `ctx=512`, 3 timed samples/leg at `ctx=2048`.
- **Health counters:** `GPU-TIMEOUTS after=0 delta=0`, `FOREIGN-GPU-USED count=0`.

## 0-ULP Bit-Exact Parity Gate

All 5 shapes matched byte-for-byte (`0 ULP`) across `base`, `cand` (default ON), and `cand` with `MLX_OMARCHY_RESNORM_VEC=0`:
- `rn_2048x2048` (`N=2048, K=2048`)
- `rn_2560x4096_o` (`N=2560, K=4096`, `o_proj`)
- `rn_2560x9728_down` (`N=2560, K=9728`, `down_proj`)
- `rn_4096x4096` (`N=4096, K=4096`)
- `rn_2560x4096_scalar_w` (`N=2560, K=4096`, weightless `rms_norm` scalar weight path)

## 5-Pair Interleaved ABBA — `ctx=512`, `decode-128`, `DEFER_TOKENS=0`

| Pair | Order | Base median (tok/s) | Cand median (tok/s) | Delta (%) | Delta (ms/tok) |
|---|---|---|---|---|---|
| 1 | base -> cand | 95.50 (`[95.62, 95.63, 95.00, 95.50, 95.38]`) | 96.49 (`[96.42, 96.49, 96.15, 96.60, 96.56]`) | +1.04% | -0.108 |
| 2 | cand -> base | 95.22 (`[95.22, 95.00, 93.48, 95.42, 95.44]`) | 96.46 (`[96.46, 96.76, 96.75, 96.05, 96.38]`) | +1.30% | -0.134 |
| 3 | base -> cand | 95.53 (`[95.53, 95.59, 95.57, 95.23, 94.96]`) | 96.43 (`[96.59, 96.44, 96.30, 96.16, 96.43]`) | +0.93% | -0.097 |
| 4 | cand -> base | 95.37 (`[95.44, 95.26, 95.17, 95.37, 95.44]`) | 96.18 (`[96.06, 96.62, 94.55, 96.82, 96.18]`) | +0.85% | -0.088 |
| 5 | base -> cand | 95.31 (`[95.32, 95.31, 95.29, 94.97, 95.42]`) | 96.46 (`[93.63, 96.47, 96.46, 96.51, 95.57]`) | +1.20% | -0.125 |
| **Summary** | **5/5 wins** | **95.37** (all-25: **95.37**) | **96.46** (all-25: **96.44**) | **+1.14%** | **-0.118** |

- **Greedy token digest (all 50 runs identical):** `efb521ae04afaad6f2309bc14d21462c72f0a5c836832c1a18d4ddb439e7c7f9`

## 3-Pair Interleaved ABBA — `ctx=2048`, `decode-128`, `DEFER_TOKENS=0`

| Pair | Order | Base median (tok/s) | Cand median (tok/s) | Delta (%) |
|---|---|---|---|---|
| 1 | base -> cand | 86.73 (`[86.73, 87.10, 86.65]`) | 87.90 (`[87.72, 87.98, 87.91]`) | +1.36% |
| 2 | cand -> base | 87.07 (`[87.07, 86.83, 87.15]`) | 88.01 (`[88.01, 87.13, 88.06]`) | +1.09% |
| 3 | base -> cand | 86.70 (`[86.66, 86.85, 86.70]`) | 87.52 (`[87.52, 87.85, 87.45]`) | +0.95% |
| **Summary** | **3/3 wins** | **86.73** | **87.90** | **+1.36%** (**-0.155 ms/tok**) |

- **Greedy token digest (all 18 runs identical):** `542524d1f692f392f03221d79b8f8cc4345b3247072bf49e7572084bca2118e8`

## Local Unit & Wiring Checks

- `glslangValidator -V --target-env vulkan1.3` + `spirv-val --target-env vulkan1.3` clean (`41,608 B`).
- `python3 scripts/check_compute_kernel_wiring.py`: clean.
- `python3 -m unittest discover -s tests -p test_compute_kernel_wiring.py -v`: 8/8 OK.
- `omarchy_vocab_tail_tests -tc="*resnorm vec default*"`: 7/7 assertions passed.
