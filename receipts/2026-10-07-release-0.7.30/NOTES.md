# 2026-10-07 — mlx-omarchy v0.7.30 (batched bf16 coopmat QMM fix, batched GDN decode, routing default ON)

Release: v0.7.30, tag `v0.7.30` = `cf71ea276` (= origin/main tip 4f98f45d3 + the
gates-only line 5aaad7e89/cf71ea276: g7a packaged-ICD fixture + g10 primer
budget), annotated, pushed at the cut. aarch64-only. v0.7.29 was tagged but
superseded before release by this one: batched bf16 coopmat QMM rows >= 1
were wrong (fixed here by 6c4b9c489 with per-row pins d3ede9f4b).

Build provenance: lsdc3build VM, /opt/m2-tc glslc 2026.3 / shaderc 1.4.357,
SPIR-V pins recomputed: matmul_f32_coopmat_qk = bc6eb65b… (pin holds),
gather_qmm_sub_bf16 = 4a1db218… (unchanged from the v0.7.29 resolution — its
source did not move in this delta). Whole-encoder bundle pin-verified at
build time. VM env: pacman blas/lapack/lapacke, /usr/include/cblas.h symlink
into openblas/, CXXFLAGS=-I/usr/include/openblas (Arch ARM header layout).

## Fixes

- **Batched bf16 coopmat QMM: rows >= 1 were wrong** (the X_F32 batch-stride
  defect — halved x batch stride in the cast route). Fixed (6c4b9c489) with
  per-row correctness pins against the host reference (d3ede9f4b) and the
  M16/32-row build pins at the review shape (cf41e9b6f, 20bd64450). Any
  batched quantized_matmul at B>1 on the coopmat route was affected.
- **SDPA-VJP dK/dV zeros at rep=1**: the dkt/dvt tiles took kL rows instead
  of the score-plane qL rows (61f1de11f); post-fix battery: 8 suites green on
  hardware (7f9e0ffe1, b3ad332e0); residual last-key-head-1 legs stay
  may_fail.
- **Shared-buffer-view allocation guards restored** for dispatch_matmul,
  copy_gpu, dispatch_softmax (a092c24ae).
- **ssm-maskless**: exact extend mirror; all(p <= 0) guarded (c1adc33c5,
  bc3f3c109 + behavior test).
- **mlx-lm backports** #1910 (Qwen3 Coder untyped arguments) and #1935
  (max_tokens min_val 1) (4f98f45d3).
- **build-wheel**: clear guidance when lapacke.h is missing (9e163630d);
  LAPACK_INCLUDE_DIRS passed when lapacke.h lives only in /usr/include/openblas
  (aeef13ce7).

## Defaults ON in this release (batched decode stack)

- Batched GDN decode: native SDPA decode over the batch + maskless unpadded
  BatchKVCache decode (2dcd24ba9); greedy GenerationBatch steps take the
  pruned greedy head (6e1e8e13d).
- Host-built offsets + fused RoPE vector offsets (1a80bc79e).
- QMV batch: batch-shared bf16 decode GEMV (d762cb52f).
- Prefill: the block-causal prompt attention stopgap retired — full attention
  via SDPA (fe452b45c); MLX_OMARCHY_SDPA_CAUSAL_BLOCK=1 opts back in.
- Kill switches: MLX_OMARCHY_QMV_BATCH=0, MLX_OMARCHY_BATCH_GREEDY=1 (opt-in
  batch greedy, 640a981ef), MLX_OMARCHY_KV_HOST_OFFSET=1 (opt-in, f7c3168a6),
  MLX_OMARCHY_SDPA_DECODE_BATCH per call (4c0368949).

## Matmul direct-route table (G13C + G13G)

H8 measured rows: nn ws8, nt k4s8, tn ws8 (3ab07fbac); only rows that won on
both chips kept (6b9485cf0); chip- and shape-keyed table with a selection
test (6fef26219); n >= tile_n/2 enforced (a1a02c7d8); shipped-route rows
apply only on the measured chips (a89eaa975). Direct GEMM rows measured
neutral on the M2 Max G14C (7effedf2a); metal_baseline.py for the same cells
through MLX (1c51cf505).

## Fixed batched-prefill known issue from v0.7.29's investigation

The batched-prefill wrong-rows defect (BatchGenerator, 4 identical prompts,
rows 1-3 off 3.31/3.75/4.63 nats at step 0) is FIXED in this release by the
batched GDN decode/BatchKVCache work above — confirmed on the fixed wheel
(BatchGenerator 4 identical prompts: rows 0-3 identical token ids; the
v0.7.29 control still reproduces). Per the batched-prefill wrong-rows
investigation receipt (private lane, 2026-10-07).

## Gates

The battery + g16-qmm-batch (installed-wheel probe: batch-independence +
fp32 host reference, B {2,4} x T {16,17}, N 6144) and g16b (route probe:
M16 FullN, 32-row X32 FullN, TwoN qmm_tile_two) run on every host set.
Gate scripts come from MAIN (the runbook copies); the tag carries only
cf71ea276's scripts — gate scripts are not shipped code.

## Assets (draft)

- `mlx_omarchy-0.32.4.dev202610070939+cf71ea2-cp314-cp314-linux_aarch64.whl`
  sha256 `0a3ab728a801ab6576b98b94d8e2cf979d10f85b1d70be668708de4be243b4c3`
  (416,514,048 bytes)
- `omarchy-mlx-vendor-wheels-v0.7.30-cp314-aarch64.tar`
  sha256 `200c4481ca7b7bf7617e1fb1b52aa922f13d01dc9cbb03c574b39fbd1d7f9b6f`
  + `.sha256`
- `SHA256SUMS` (3-asset coverage)
