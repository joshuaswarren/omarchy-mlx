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

## 6. Dev-box lavapipe results (2026-10-05, pre-M2)

Environment: x86_64 dev box, llvmpipe/lavapipe via a staged ICD prefix
(`OMARCHY_MLX_SYSTEM_PREFIX` fake tree + `MLX_OMARCHY_ALLOW_NON_APPLE=1`),
branch `agent/FamQwen35Prefill` at f289d60e3 (enum entry moved to the
append-only end after first landing mid-enum).

C++ numerics-gate doctest (`omarchy_fast_ops_tests`, test_fast_ops.cpp
"sdpa prefill flash-256 is no worse than the composed path on Qwen3.5
shapes"): **9,437,204 assertions, 0 failed**, three consecutive runs
(two before the enum move, one after). Gate contract implemented:
flash fp64-error no worse than the composed path (2x summation-order
slack, 1e-2 absolute floor), per-element arm-to-arm bound
(|f-c| <= 8 bf16 ULP + 1e-7), host fp64 reference consuming the same
bf16-rounded inputs the device arms load.

Python per-op A/B (`m2_gate_perop.py`, the M2 ticket payload):
GATE_PEROP **PASS** on all four shapes (2B GQA 8/2, 9B GQA 16/4,
qL=kL=512/1024, head_dim 256, causal bf16). composed.npz and flash.npz
sha256 IDENTICAL (1c227ecb...) — the two arms are byte-equal on
llvmpipe. max_abs vs fp64 ~1.95e-3 = bf16 output/probs quantization at
unit scale, equal on both arms. Artifacts (private notebook):
artifacts/FamQwen35Prefill/qwen35-prefill/.

Test-design notes carried for the M2 run: (1) the host reference MUST
consume bf16-rounded inputs — the fp32-original reference produced a
symmetric common-mode input-cast error (~96 ULP, both arms, same
elements) that is not a kernel defect; (2) per-element
ULP-vs-fp64 bounds are not the gate contract at cancellation-heavy
outputs (f32 accumulation vs f64, symmetric across arms); (3) on a
serialized software driver both summation orders coincide, so
engagement evidence on real silicon comes from a ULP-scale arm
difference plus the timing leg, not from the lavapipe A/B.

M2 work (incremental branch wheel from the od tree copy, private venv
with shebang-safe install, per-op + TF/free-run/PPL gates, then the
pf512/pf1024 ON-vs-OFF timing cells) is staged and queued for the
post-03:00Z window per Main's schedule.

## 6. M2 results (2026-10-05, T6021 real silicon)

Environment: jw14m2-linux, gpu-turn tickets per Main's queue discipline;
branch wheel built on the golden-clone recipe (rebase 690dc50ad on main
post-golden; wheel sha 1226e6d8, libmlx efb9ca45 — verified installed).

Numerics:
- Per-op A/B on the four Qwen3.5 shapes (2B GQA 8/2, 9B GQA 16/4,
  qL=kL=512/1024, bf16 causal head_dim 256): **GATE_PEROP PASS** —
  flash max-abs vs fp64 == composed max-abs (bf16 output quantization
  ~1.95e-3 at unit scale, equal both arms), arm diff <= 1 bf16 ULP.
- Shape sweep (10 classes: qL/kL 16..64 + ragged 53x117, tails and
  non-multiples): max arm diff <= 0.000977 (1 ULP) everywhere.
- Captured REAL model sdpa inputs (all 6 attention calls of one 2B
  forward: bf16 (1,8,5,256)/(1,2,5,256), scale 0.0625, mask causal,
  sinks None, contiguous): replayed outputs **bit-identical** between
  arms; live sdpa input tensors bit-identical between arms.

Performance (engagement probe, (1,8,2,1024,4096,256) causal):
- v1 layout (2 rows/lane, 512 f32 accumulators): 3646 ms/call —
  239x composed (register spill; the v1 register budget was past the
  255-allocation limit).
- v2 layout (D_v-split, 128 accumulators/lane): 1169-1172 ms/call —
  76x composed (15.2-15.6 ms). Remaining gap under diagnosis: the
  redundant per-lane softmax stats (32x exp amplification) and scalar
  staging are the measured-suspect stages; a v3 with row-lane stats +
  shared s_w weights is designed but not yet gated.

Model-level gate (Qwen3.5-2B, teacher-forced + greedy free-run):
**FAIL — OPEN DEFECT.** Flash-vs-composed TF top-1 agreement 40.6%,
last-logit diffs 4.3-9.6 points, free-run digests differ; flash-vs-
flash across processes is bit-deterministic (agreement 1.0), and the
sdpa op itself is exonerated at op level (above). The divergence
manifests ONLY in the live model forward with the flash arm engaged —
isolated replays of every captured call are bit-identical. Leading
hypothesis: a live-context-only hazard (adjacent-buffer corruption or
a driver-level resource interaction), not sdpa arithmetic. Debug
state: all capture/replay tooling + evidence in the notebook artifacts
(m2_capture_all.py, live_capture.py, sweep/bisect scripts, tf2b npz).

Per the assignment's honesty bar: the classic qwen35_prefill family on
omarchy is numerics-covered by the EXISTING kernels for everything
except fa256, and the new fa256 omarchy kernel is per-op correct and
engaged but NOT shippable as a default route until the model-level
gate passes; MLX_OMARCHY_SDPA_PREFILL_FLASH256=0 (or simply not
setting it — the default arm is ON only on the agent branch, not on
main) keeps the composed path.

## 7. v3 on T6021 + the open model-level defect (2026-10-05)

v3 (row-lane stats + shared s_w; commit ee37fd09d, wheel 142a54cc on
the golden-clone base): per-op GATE PASS on T6021 (4-shape battery),
engagement 1052 ms/call at the probe shape (v2 1169 — the 32x exp
elimination barely moved it; the dominant cost is the scalar staging
loop and/or dynamic-indexed accumulators in local memory; a v4 with
unrolled static-index accumulators + vectorized staging is the next
step).

Model-level gate: **FAIL, OPEN, signature IDENTICAL across v2 and v3**
(TF top-1 agreement 41.7%, p0 pos0 top2_gap 1.125 — same numbers both
generations). Full exoneration of the sdpa arithmetic: per-op sweep
(15 shape classes once extended) <= 1 ULP; all 6 captured real-model
calls replay bit-identically; live sdpa input tensors bit-identical
between arms; model forward deterministic across processes; and the
final discriminator — the first live call REPLACED with captured call0
inputs — produced output **bit-identical to composed**. The divergence
therefore appears only when the kernel runs inside the full live
forward with its own surrounding dispatch context: deterministic,
context-dependent, generation-independent. Remaining hypotheses are
backend-level (encoder barrier/descriptor-lifetime/buffer-aliasing
interaction specific to this kernel in a full-model command graph).

Status per the honesty bar: NOT shippable as a default route. The arm
exists only on agent/FamQwen35Prefill; main is unaffected;
MLX_OMARCHY_SDPA_PREFILL_FLASH256=0 is the documented kill switch for
anyone on the branch. All evidence and the full debug harness are
archived (receipts/.../tools/, notebook artifacts/) — a focused
backend session (encoder barriers/descriptor lifetime for this kernel
in the live command graph) is the named next step.
