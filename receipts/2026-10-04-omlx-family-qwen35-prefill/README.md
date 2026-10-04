# oMLX qwen35_prefill classic — op-by-op coverage and M2 plan

Pins: oMLX jundot/omlx `v0.7.0` = `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`
(2026-10-01). omarchy-mlx `origin/main` `831f8fa03` (2026-10-04, includes
mx.fast.int8_matmul + the qmm g4 raster + the round-trip-diet GDN batch
landing). Models: `mlx-community/Qwen3.5-{2B,9B}-MLX-4bit` (4-bit affine,
group 64, dense, GDN hybrid with full-attn every 4 layers, head_dim=256,
Dk=Dv=128, H_linear=16/16 for 2B and 16/32 for 9B, ssm-dtype fp32).

This is the design + acceptance note for the v1.1 MATRIX row A25 lane
`FamQwen35Prefill`. The notebook thread
`entries/FamQwen35Prefill/2026-10-04T2236Z-devbox-jw14m2-qwen35-prefill-classic-design.md`
is the open entry.

## 1. oMLX classic prefill symbols and their fate on omarchy

Only the GPU-classic path is in scope. NAX (M5 tensor-unit) variants stay
n/a on M1/M2 by design; ANE symbols (`qwen35_ane_*`, `oq_a8_*` ANE) are
macOS-ANE experiments and out of scope here; oQ-A8 GPU symbols
(`qwen35_oq_a8_*`) are oQ-quantization-specific and only active when an
oQ model is loaded (the M2 cache holds 4-bit affine, not oQ); CPU/GPU
fp16 hybrid symbols (`qwen35_cpu_fp16_*`) are macOS-unified-memory tricks
and n/a on Linux; **NAX/ANE/CPU-hybrid/oQ-A8 are documented n/a**, not
implemented. The fast classic symbols that touch Qwen3.5-2B/9B prefill on
the 4-bit affine path are:

| oMLX classic symbol (NATIVE_SYMBOLS) | Algorithm | omarchy equivalent | Status |
|---|---|---|---|
| `qwen35_fa256_attention` (`qwen35_attention.metal`) | Steel-MMA flash-2 prefill, head_dim=256, GQA, causal, bf16/fp16, runtime q_block=32/k_block=8, dispatch budget for IOGPU preemption | **No equivalent.** omarchy's `ScaledDotProductAttention::use_fallback` returns false on inference, `force_fused=True` throws (`primitives.cpp:10904`), so prefill SDPA is the composed QK^T f32 coopmat graph (`MatmulF32CoopmatQkBF16` + `softmax_suffix` + `MatmulF32CoopmatPvBF16`) — O(L²) memory on the S intermediate, no flash tiling. Steel MMA is rejected by the omarchy MSL→GLSL translator | **GAP** — new `SdpaPrefillFlash256BF16` needed |
| `qwen35_q4_affine_qmm_t` (`qwen35_qmm.metal`) | Prefill-tuned q4 affine (per-group scale+bias) matmul, transposed RHS, M-heavy tile | `QmmPrefillCoopmatBF16X32FullNRasterG4` + `LdsPad` + `ChunkPad` arms (omarchy `qmm_coopmat.comp` + `primitives.cpp:7435+`); affine biases fully supported (`qmm_coopmat.comp:121-137`, `primitives.cpp:2991-3064` promote scales/biases); bf16 storage of scales+biases verified. g4 raster ship gives +2.4% 2B pf1024, +4.4% 9B pf512, +3.7% 9B pf1024 (QmmPeak receipt). Transposed-RHS dispatch path matches mlx-lm's affine mode | **COVERED** (op + tile) — speed parity claimed |
| `qwen35_q2/q5/q6/q8_affine_qmm_t` | Same as q4 with other bit widths | Same dispatch family; the mlx-lm 4-bit cache hits the q4 path; other widths covered by the same shader | **COVERED** (off the M2 cache's hot path; verified by qmm_coopmat's bits-1..8 support) |
| `qwen35_moe_weighted_sum` (`qwen35_qmm.metal`) | Fused combine: sorted-expert outputs weighted by router scores, single launch | Composed graph (multiply + sum). The M2 4-bit cache has no MoE models; this symbol is only on the Qwen3.6-35B-A3B omlx path, which the fleet does not have in cache. omarchy does have `QmmVecQ4MultiOutgateBF16` for a related out-gate prologue | **n/a hot path** (no MoE in the M2 cache; if it later shows up, ledger) |
| `qwen35_gather_qmm_rhs_t` (`qwen35_qmm.metal`) | gather_qmm with transposed RHS for MoE expert dispatch | `dispatch_gather_qmm` (omarchy `primitives.cpp:2986,6048,6246,6307`) | **COVERED** (off hot path; no MoE in cache) |
| `gated_delta_pipelined` (`gdn.py:878`) | JIT MSL exact sequential scan, Dk=128, 16-row value blocks, 256-thread workgroups, software-pipelined 12-token block | omarchy `GatedDeltaPrefillCoopmatBF16` (`gated_delta_prefill_coopmat.comp`) — chunked C=8, 128-thread workgroup per (head, Dv/32 slice), 4 simdgroups, 8x8 f32 coopmat state tiles. Defaults to `GatedDeltaPrefillCoopmatBatchBF16` (round-trip-diet, double-buffered) when the device's `max_compute_shared_memory_size ≥ 32000` (T6021 G14C: 32 KiB → engaged). `kGdnCoopmatMinTokens` gate ensures the chunked path engages for prefill T≥min. | **COVERED** (algorithm differs: oMLX pipelined exact-recurrence vs omarchy chunked coopmat WY — ledger decides) |
| `gated_delta_blocked_seq` (`gdn.py:560`) | JIT MSL blocked sequential, TB=16 (fp32) / TB=32 (bf16), register-resident state, fp32-exact | Falls into `GatedDeltaPrefillCoopmatBF16` when `Dk=Dv=128 && Hk=Hv && B==1 && bf16 && f32 state && scalar g && !mask && T≥kGdnCoopmatMinTokens` (`primitives.cpp:11550+`); TB=16 is fp32-blocked, omarchy's coopmat f32-accum is the equivalent on this exact shape | **COVERED** (omarchy engages the coopmat chunked; TB=16 fp32 path is the composed reference omarchy was equivalence-checked against) |
| `gated_delta_chunked_metal` (`gdn.py:338`) | FLA WY-representation chunked A/B (kt/U0/MU0/Qeff/lcg prep + chunk scan). "Accuracy-validated but slower than the stock kernel E2E" per the patch docstring (`patches/qwen35_gdn_chunked.py:9-11`) | `GatedDeltaPrefillBF16` (per-head token-scan, two-pass with snapshot checkpoints) + `GatedDeltaPrefillCoopmatBatchBF16` | **COVERED** (chunked coopmat batch is the "diet" of the same algorithm) |
| `qwen35_gdn_prework` (`patches/qwen35_gdn_prework.py`) | Fused conv1d + q/k norm + split + scalar scales. Comment: "this kernel also runs automatically for compatible FP16/BF16 B1/T1 decode through Qwen3_5GatedDeltaNet" — decode + verify, not prefill | `mlx-lm-qwen35-gdn-conv.patch` + `mlx-gdn-conv-decode.patch` + `gdn_conv_decode.comp` (omarchy decode) — the short conv is decode. Prefill short conv (causal depthwise over T): `conv_dw1d.comp` + `ConvDw1dF32/F16/BF16` (compute.h:761+) | **COVERED** for prefill (`ConvDw1dBF16`); the verify/decode prework fusion is off the cache's hot path (no MTP path cached) |
| `qwen35_moe_routed_decode`, `qwen35_moe_router`, `qwen35_moe_gate_up` | MoE router/gate-up + routed decode | Off the cache's hot path (no MoE in M2 cache); omarchy has the qmm/gather_qmm to back them if a MoE Qwen3.5 ever loads | **n/a hot path** (no MoE cached) |
| `qwen35_q4_mlp`, `qwen35_packed_linear` | Dense q4 MLP + packed linear | `QmmPrefillCoopmatBF16X32FullNRasterG4` + `Swiglu.comp` + `fast_rope_norm.comp` (omarchy has the SwiGLU fusion in `qmm_coopmat` itself — `QmmVecQ4MultiOutgateBF16` for the out-gate prologue) | **COVERED** (prefill MLP) |
| `qwen35_oq_a8_*`, `qwen35_ane_*`, `qwen35_cpu_fp16_*`, NAX `*_nax` | oQ-A8 W8A8 dynamic quant, ANE, CPU/GPU fp16 hybrid, NAX M5 tensor | n/a on the classic Qwen3.5 4-bit cache: oQ models are a separate (not-cached) family; ANE is macOS-only; CPU-fp16 hybrid is a macOS unified-memory optimization; NAX is M5-only by design | **n/a** (documented, not implemented) |

## 2. The implementation target

`qwen35_fa256_attention` is the only classic prefill symbol without an
omarchy equivalent. The omarchy backend has all the building blocks:
`MatmulF32CoopmatQkBF16` (QK^T into an f32 scores buffer with the
subgroup coopmat tile), `softmax_suffix.comp` (rowwise max + exp sum +
divide), `MatmulF32CoopmatPvBF16` (P·V with the same coopmat tile). The
*composed* path works and is bit-exact; the *fused* path saves the
materialized S intermediate (D_k × D_q = 256 × T bf16 scores in addition
to the f32 scores) and avoids the round trip, with the well-known
flash-attention two-pass outer/inner tiling.

The native omarchy kernel is a single `sdpa_prefill_flash256.comp` with:

- 2D grid `(q_block, num_heads)` where `q_block = ceil(T_q / Q_TILE)` and
  `num_heads = B * H_q`. Each workgroup produces Q_TILE output rows
  for one (B, H_q) pair.
- A shared-memory scratch that holds the (Q_TILE, D_k) bf16 Q block and
  rotates the K/V streams in K_TILE-wide chunks; f32 accumulators for
  the per-row m_i, l_i running stats and the (Q_TILE, D_v) output, with
  on-line softmax updates (flash-attention 2 algorithm).
- GQA: kv_head = h_q / gqa_factor; the K/V load is gqa_factor-shared.
- bf16 Q/K/V, f32 accumulators, fp16 optional in a follow-on.
- Causal: per-q-row upper-bound on the K column, exit early on the
  padding tail. Decode (q_len=1) and non-causal are out of scope.
- Dispatch-gate mirror of `ScaledDotProductAttention::use_fallback`:
  `head_dim==256 && GQA && causal && !has_mask && !has_sinks &&
   !output_logsumexp && bh*q_len <= 65535 && bh <= 65535` — every
  shape the composed graph would accept on this path, inverted.

Construct usage stays inside the translator-allowed subset: buffers,
threadgroup memory, barriers, subgroup reductions, bf16 loads, fma. No
simdgroup_matrix — the inner loop is the standard flash-2 outer/inner
tiling with the per-row running stats, no MMA. A subgroup-collective
`max`/`add` on the f32 accumulators is permitted by the translator
(omarchy's `softmax_suffix` and `reduce_*` shaders already use it).

The enum entry is `SdpaPrefillFlash256BF16`, append-only after
`SdpaDecodeNativeBF16Hd256` in `compute.h`. primitives.cpp gets one new
dispatch arm inside the existing `ScaledDotProductAttention::eval_gpu`
under the same conditions that select the composed path. The composed
graph stays in place and the env-flag `MLX_OMARCHY_SDPA_PREFILL_FLASH256=0`
is the kill switch back to it for any regression.

## 3. M2 ledger shape

h253 cell shape (M2Lane `2026-10-04T134800Z` convention): prefill cells
`--new-tokens 32 --prefill-tokens {512,1024} --limit 10 --passes 3
--warmup 2`; quiet gates load1<0.5, psi cpu avg10=0; >=6 min post-boot;
2 reps; per-cell `ordered_records_sha256` digest; provenance verified
match. Models: Qwen3.5-2B, Qwen3.5-9B.

Cells: pf512-2B, pf1024-2B, pf512-9B, pf1024-9B, each with the prefill
arm ON vs composed fallback OFF. Comparison legs: the omarchy fast-ON
arm (qmm g4 raster + GDN coopmat batch + the new prefill flash-256) vs
the composed-fallback arm (every prefill op as the composed graph,
forced by `MLX_OMARCHY_SDPA_PREFILL_FLASH256=0` +
`MLX_OMARCHY_QMM_NO_RASTER=1` + `MLX_OMARCHY_GDN_BATCH=0` to back out
each optimized path independently and report the per-op contribution).
Digest equality across arms on a 64-token teacher-forced window is the
numerics gate; the per-op rel-err gates are caught at op-level on
lavapipe before the M2 run.

The "oMLX path" comparison (oMLX serving with patches active vs current
omarchy path) depends on OmlxLinux landing their omarchy-mlx install
(`entries/OmlxLinux/20261004T221104Z` is pre-registration only as of
22:11Z). If they land before the M2 ledger window, the run adds a
third arm: omlx-on-omarchy with the `qwen35_*` patches active. If
not, the comparison is documented as deferred with the exact deferred
command and the omarchy fast-ON vs composed fallback is the reported
result for this lane (which is what the assignment's "current omarchy
path" requires in any case).

## 4. Out of scope

- NAX / M5-only paths. Documented n/a.
- oQ-A8 dynamic quantization paths. Not on the cache's hot path; the
  int8_matmul op omarchy just landed (`mx.fast.int8_matmul`,
  `int8_matmul.comp`) is the existing analog; new oQ-specific kernels
  are not justified by the fleet's current 4-bit affine loading.
- ANE POCs and the oMLX menubar app. n/a.
- mx.distributed cluster pieces. Blocked on OmarchyDistributed; not
  in this family.
- Decode-side GDN (`gated_delta_decode*`, `GatedDeltaDecodeBF16`,
  `gdn_conv_decode`). The M2 4-bit cache is prefill-bench-heavy for
  the 9B but decode uses the same family — decoded on the same branch
  if the perf ledger surprises.

## 5. References

- `receipts/2026-10-04-omlx-tensorfold-parity/MATRIX.md` rows A25–A28.
- `omlx/custom_kernels/qwen35_prefill/{fast.py,gdn.py}` at 4d4f5a28.
- `omlx/patches/qwen35_{gdn_chunked,fa256_attention,gdn_prework,...}.py` at 4d4f5a28.
- omarchy `overlay/mlx/backend/omarchy/compute.h` lines 484, 677, 731,
  10913–10960 (sdpa_kernel + GatedDeltaDecode/Prefill/Coopmat entries).
- omarchy `overlay/mlx/backend/omarchy/primitives.cpp` lines 10890–11010
  (ScaledDotProductAttention::use_fallback), 11190–11330 (GDN VJP),
  11540–11600 (GDN prefill T>1 dispatch), 7435–7610 (QmmPrefillCoopmat
  FullN + raster + LDS_PAD + ChunkPad).
- `receipts/2026-10-04-qmm-roofline/README.md` (qmm FullN+raster+pad
  evidence; +2-3% G14C / +0.9% G13G model prefill, bit-exact digests).
- `entries/M2Lane/2026-10-04T134800Z-jw14m2-v0726-lora-smoke-perf-ledger.md`
  (M2 ledger h253 cell shape + quiet gates + macOS same-machine
  reference search result: NONE for T6021 MLX decode/prefill; recorded
  accordingly).
- `entries/ParityMatrix/2026-10-04T2215Z-devbox-omlx-tf-parity-matrix.md`
  (closed sibling).
