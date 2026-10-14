# Coopmat prefill and word decode for 4-bit group 32 and 8-bit affine weights

Runs on 2026-10-13, base main `a0a879b`, branch = base + this change. Each
measurement ran main and the branch back to back on the same boot, each
process in a user cgroup capped at 8.5 GiB (9 GiB for the perplexity runs,
which peak near 9 GiB).

## Question

Transposed affine `QuantizedMatmul` with 4-bit group 32, 8-bit group 64 or
8-bit group 32 weights ran the generic `QmmVec` and `QmmTileRb` kernels,
because every fast-route gate checked `bits == 4 && group_size == 64`.
Can those formats take the coopmat prefill and word decode routes without
changing 4-bit group 64, and with the arithmetic contract of each route?

## Setup

- MacBook Air 13-inch M2 2022 (J413, T8112), `Apple M2 (G14G B0)`, 8 GPU
  cores, 24 GB. NixOS, Linux 7.1.13 (Asahi), system firmware 15.6, OS
  firmware 13.5.
- Honeykrisp ICD from `joshuaswarren/mesa-1` at `6f6afc896844`, selected
  with `VK_DRIVER_FILES`.
- main: the omarchy-mlx package at `a0a879b` built by Nix
  (`0.32.4.dev198001020000+a0a879b`). Branch: `scripts/prepare-mlx.sh`
  tree with this overlay, Python bindings, `MLX_BUILD_CPU=OFF`.
- CPU governor schedutil, on battery (AC unplugged) for every run. Load
  average below 2 at start. The SoC die temperature is not exposed on this
  kernel; the SMC sensors read across all runs: NAND 40-58 C, charge
  regulator 38-48 C, WiFi module 42-61 C (`facts.jsonl`, before and
  after each run).
- Model: `mlx-community/K2-Horizon-7B-4bit` as published (dense 7B, bf16;
  q/k/v/o, embeddings and lm_head at 8-bit group 64, MLP at 4-bit group
  32). sha256: model-00001-of-00002.safetensors `134605fa293e6f56bb8862c4cc452028f1a134ac2bb842dce9ff1f09cfda933b`,
  model-00002-of-00002.safetensors `19b37d96f4afb339e367f1d13f5a4543c0745c5756dd5102b155cf57585c2248`,
  config.json `5e523194f1e6a64b5a3abd9b60406d4512f16fdd218a69ff67017a8d1b98bbca`.

## Results

### 4-bit group 64 is unchanged

`spirv-identity.sh a0a879b`: all 52 qmm_vec and qmm_coopmat builds of the
base compile to identical SPIR-V from the base and the branch shader
(33 qmm_vec, 19 qmm_coopmat).

### Correctness per operation

- `prefill_check.py`, K 4096, 40 cases (4 formats, f16 and bf16,
  M/N in 5/4096, 17/1024, 33/4096, 100/1000, 512/4096): the coopmat
  output is bit-identical to `MLX_OMARCHY_NO_COOPMAT=1` (the tile route)
  in 40 of 40 cases (`out_sha256` in `prefill-coopmat.jsonl` and
  `prefill-tile.jsonl`).
- `decode_check.py`, 32 cases at M = 1: relative L2 error against the
  fp32 dequantized reference, word kernel versus the generic kernel
  (`MLX_OMARCHY_QMM_VEC_Q4_WORD=0`), worst case per format:

| dtype | format | word | generic |
|---|---|---|---|
| bf16 | 4-bit g64 (shipped) | 2.23e-03 | 2.23e-03 |
| bf16 | 4-bit g32 | 2.29e-03 | 2.29e-03 |
| bf16 | 8-bit g64 | 2.44e-03 | 2.44e-03 |
| bf16 | 8-bit g32 | 2.39e-03 | 2.39e-03 |
| f16 | 4-bit g64 (shipped) | 9.40e-04 | 2.83e-04 |
| f16 | 4-bit g32 | 9.54e-04 | 2.75e-04 |
| f16 | 8-bit g64 | 2.90e-04 | 2.90e-04 |
| f16 | 8-bit g32 | 2.92e-04 | 2.92e-04 |

  The f16 4-bit ratio is the native qmv half-precision quad sum, the same
  for the shipped 4-bit group-64 word kernel. Metal's 8-bit qmv has no
  half sum, so 8-bit matches the generic kernel.

### Tests

- New: "q4g32 and q8 affine formats take the word decode and coopmat
  prefill routes" (`omarchy_matmul_family_tests`), 511272 assertions pass.
  With `MLX_OMARCHY_NO_COOPMAT=1` all 24 prefill pins fail. With
  `MLX_OMARCHY_QMM_VEC_Q4_WORD=0` the decode pin fails.
- On base `17bfdff` (qmm sources identical to `a0a879b`), the 47 suite binaries in `overlay/tests/omarchy` and the 6
  `omarchy_capability_sim_tests` profiles pass on the branch.
  `omarchy_ane_runtime_tests` does not link in a build without the ANE
  runtime and was not run.

### K2 as published

- Dispatch trace of a short chat (`k2-dispatch-counts-branch.txt`, kernel
  id counts): decode runs `QmmVecWordSubgroupBF16Q4G32` (604) and
  `QmmVecWordSubgroupBF16Q8G64` (606), prefill
  `QmmPrefillCoopmat*Q8G64` and `*Q4G32`. No `QmmVecSubgroupBF16` (319) or
  `QmmTileRb*` dispatch remains. The answer is unchanged
  ("17 x 23 = 391").
- Perplexity (`k2-ppl.py`, this repository's README at `a0a879b`):
  - 2048 tokens in 128-token chunks with KV cache: main 9.402129416962538,
    branch 9.402129416962538 (identical; prefill is bit-identical).
  - 300 tokens, one token per step (decode kernels): main
    14.020861864653401, branch 14.033564010302008 (+0.09%).
- Greedy ids (`k2-greedy-ids.py`, 256 tokens): the first 31 match, and
  position 31 differs. Decode follows qmv's summation order, so it is not
  bit-identical to the generic kernel. The shipped 4-bit group-64 word
  kernel has the same property.

### Speed

`k2-bench-mlx.py <model> 32,512 3`: one pp32 warm-up, then three runs of
each test (pp N: N prompt tokens into a fresh cache; tg128: 128 greedy
tokens after a 1-token prompt).

| test | main tok/s (3 runs) | main mean | branch tok/s (3 runs) | branch mean |
|---|---|---|---|---|
| pp32 | 9.47, 9.59, 9.56 | 9.54 | 58.96, 59.10, 59.13 | 59.06 |
| pp512 | 9.57, 9.68, 9.55 | 9.60 | 75.40, 74.87, 74.91 | 75.06 |
| tg128 | 2.02, 2.29, 2.29 | 2.20 | 8.16, 8.15, 8.24 | 8.18 |

Peak MLX memory: main 7.11 GiB, branch 7.12 GiB. Load: main 6.5 s, branch 6.6 s.

`qmm_micro.py`: K 4096, N 12288 (the K2 MLP up projection), random
weights quantized by `mx.quantize`. 3 warm-up calls, then 20 timed calls
(5 at M = 512), wall clock around `mx.eval`. The dense reference
matmul at M = 512 ran f16 1747.8 / 1708.4 GFLOP/s and bf16 913.6 / 895.2
(main / branch). Peak memory 0.38 GiB for both.

| dtype | format | M | main ms | branch ms | main GFLOP/s | branch GFLOP/s |
|---|---|---|---|---|---|---|
| float16 | 4-bit g64 | 1 | 0.883 | 0.758 | 113.9 | 132.8 |
| float16 | 4-bit g64 | 32 | 2.949 | 3.036 | 1092.3 | 1061.0 |
| float16 | 4-bit g64 | 512 | 36.697 | 36.696 | 1404.5 | 1404.5 |
| float16 | 4-bit g32 | 1 | 2.706 | 0.818 | 37.2 | 123.0 |
| float16 | 4-bit g32 | 32 | 20.363 | 3.340 | 158.2 | 964.3 |
| float16 | 4-bit g32 | 512 | 316.699 | 44.059 | 162.7 | 1169.8 |
| float16 | 8-bit g64 | 1 | 2.893 | 1.198 | 34.8 | 84.0 |
| float16 | 8-bit g64 | 32 | 21.092 | 2.811 | 152.7 | 1145.7 |
| float16 | 8-bit g64 | 512 | 317.581 | 39.198 | 162.3 | 1314.9 |
| float16 | 8-bit g32 | 1 | 2.748 | 1.238 | 36.6 | 81.3 |
| float16 | 8-bit g32 | 32 | 20.404 | 3.665 | 157.9 | 879.0 |
| float16 | 8-bit g32 | 512 | 317.155 | 49.314 | 162.5 | 1045.1 |
| bfloat16 | 4-bit g64 | 1 | 0.770 | 0.722 | 130.7 | 139.5 |
| bfloat16 | 4-bit g64 | 32 | 2.663 | 2.683 | 1209.4 | 1200.5 |
| bfloat16 | 4-bit g64 | 512 | 32.985 | 32.991 | 1562.5 | 1562.2 |
| bfloat16 | 4-bit g32 | 1 | 3.003 | 0.958 | 33.5 | 105.1 |
| bfloat16 | 4-bit g32 | 32 | 20.884 | 3.027 | 154.2 | 1064.1 |
| bfloat16 | 4-bit g32 | 512 | 321.676 | 41.570 | 160.2 | 1239.8 |
| bfloat16 | 8-bit g64 | 1 | 3.062 | 1.313 | 32.9 | 76.7 |
| bfloat16 | 8-bit g64 | 32 | 21.095 | 2.697 | 152.7 | 1194.5 |
| bfloat16 | 8-bit g64 | 512 | 321.405 | 35.344 | 160.4 | 1458.2 |
| bfloat16 | 8-bit g32 | 1 | 3.121 | 1.244 | 32.3 | 80.9 |
| bfloat16 | 8-bit g32 | 32 | 21.032 | 3.300 | 153.2 | 976.1 |
| bfloat16 | 8-bit g32 | 512 | 322.520 | 46.979 | 159.8 | 1097.1 |

The 4-bit group 64 rows run identical SPIR-V. Their spread (0.97x to
1.16x) is run-to-run noise.

## Limits

- One machine (base M2, G14G). No M1, M1 Max, M2 Max or M3 numbers are
  claimed. On other boards the gates, the batch cap and the default-route
  choice are the existing logic, and speed is unmeasured.
- Only the default coopmat route of each shape has a new-format build.
  The env-selected twins (TwoN, chunk dequant, the other g4 variants)
  stay 4-bit group 64.
- `gather_qmm` (MoE experts) still runs the scalar kernel for these
  formats.
- MoE router gates are 8-bit group 64 in current mlx-lm 4-bit
  conversions (gemma-4-26B-A4B-it-4bit, Qwen3-30B-A3B-Instruct-2507-4bit)
  and take the new routes. Their decode logits change at rounding level,
  so pinned token digests of those models may move. Not measured here.
