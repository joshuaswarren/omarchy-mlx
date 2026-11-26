# 2026-10-07 — mlx-omarchy v0.7.31 (batched bf16 coopmat QMM fix, kv-maskless series gate, batched GDN decode, routing default ON)

Release: v0.7.31, tag `v0.7.31` = `9b5c938fe` (origin/main tip at the FREEZE;
asserted ancestors: 599068f8e kv-maskless series gate, 3db5cc1ba g17, 72a62ea67/
892ea7e4e upstream backports, 1cc56037e g16b route probe). aarch64-only.
v0.7.29 and v0.7.30 were tagged but never published (superseded by this one;
one line each below). Build: lsdc3build VM, /opt/m2-tc glslc 2026.3 /
shaderc 1.4.357. ICD pin proposal for omarchy-pkgs: honeykrisp-omarchy-v3
bbbfa36dce7 (w7J green; the bf16 direct route additionally needs
VK_KHR_shader_bfloat16 = v3 6543eeb7df7 or newer — without it bf16 keeps the
staged kernel).

## Fixes

- **Batched bf16 coopmat QMM: rows >= 1 were wrong** (the X_F32 batch-stride
  defect that shipped in the unreleased v0.7.29/v0.7.30): fixed in
  qmm_coopmat (6c4b9c489) with per-row correctness pins (d3ede9f4b) and the
  M16/32-row build pins at the review shape (cf41e9b6f, 20bd64450).
- **g1 install blocker fixed**: the 0.32-only BatchKVCache patchers
  (kv-maskless, kv-host-offset, batch-greedy-head) are gated per mlx-lm
  series (599068f8e) — pristine mlx-lm 0.31.3 installs exit 0 again
  (g17-patch-series: both lines apply rc=0 + idempotent; the pre-fix script
  reproduced the g1 failure as the negative control, 9b5c938fe).
- **SDPA-VJP dK/dV zeros at rep=1**: the dkt/dvt tiles take kL rows
  (61f1de11f); post-fix battery 8 suites green on hardware (7f9e0ffe1,
  b3ad332e0); residual last-key-head-1 legs stay may_fail.
- **Shared-buffer-view allocation guards restored** (a092c24ae).
- **ssm-maskless**: exact extend mirror + all(p <= 0) guard (c1adc33c5,
  bc3f3c109).
- **mlx-lm backports** #1910, #1935 (4f98f45d3).
- **build-wheel**: lapacke.h guidance (9e163630d); LAPACK_INCLUDE_DIRS for
  Arch ARM openblas layout (aeef13ce7).

## Defaults ON (batched decode stack)

Batched GDN decode: native SDPA decode over the batch + maskless unpadded
BatchKVCache decode (2dcd24ba9); greedy head pruned per row (6e1e8e13d);
host-built offsets + fused RoPE vector offsets (1a80bc79e); QMV batch
(d762cb52f); prefill full attention via SDPA — the block-causal stopgap
retired (fe452b45c; MLX_OMARCHY_SDPA_CAUSAL_BLOCK=1 opts back in).
Kill switches: MLX_OMARCHY_QMV_BATCH=0, MLX_OMARCHY_BATCH_GREEDY=1 (opt-in),
MLX_OMARCHY_KV_HOST_OFFSET=0, MLX_OMARCHY_SDPA_DECODE_BATCH per call,
MLX_OMARCHY_SSM_MASKLESS=0, MLX_OMARCHY_KV_MASKLESS=0.

## Matmul direct-route table (G13C + G13G; needs VK_KHR_shader_bfloat16)

H8 rows nn ws8 / nt k4s8 / tn ws8 (3ab07fbac); rows that won on both chips
kept (6b9485cf0); chip- and shape-keyed table + selection test (6fef26219);
n >= tile_n/2 (a1a02c7d8); shipped rows apply only on the measured chips
(a89eaa975). Requires honeykrisp v3 6543eeb7df7+; without it bf16 keeps the
staged kernel. Direct GEMM rows neutral on G14C (7effedf2a).

## Routing

Automatic decision routing is ON by default (4cd3ef727 in the cut;
docs/serve.md). Kill switch MLX_OMARCHY_ROUTING=0.

## Batched-prefill known issue from v0.7.29's investigation — CLOSED

Fixed by the batched GDN decode/BatchKVCache work in this release;
confirmed on the fixed wheel (BatchGenerator 4 identical prompts: rows 0-3
identical token ids; the v0.7.29 control reproduces) per the
batched-prefill wrong-rows investigation receipt (private lane).

## Assets (draft)

- `mlx_omarchy-0.32.4.dev202610071347+9b5c938-cp314-cp314-linux_aarch64.whl`
  sha256 `a2f8c83e5c5f635d00702565a8557d87c40a9eccac885329dcec4692d85d300d`
  (416,666,133 B; built on lsdc3build, /opt/m2-tc glslc 2026.3/shaderc
  1.4.357, whole-encoder bundle pin-verified at build time)
- `omarchy-mlx-vendor-wheels-v0.7.31-cp314-aarch64.tar`
  sha256 `1cc390b1e25c55a80c8c21327a39d7e498cafb5144a75b0ff1fc9e6f8b54b074`
  (36 vendored wheels, verify-vendor twice) + `.sha256`
- `SHA256SUMS` (3-asset coverage)
