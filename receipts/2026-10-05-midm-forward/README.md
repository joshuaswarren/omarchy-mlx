# MidM forward: multi-row (q_len 2..16) fused variants — census and design

**Status: in progress** (census complete; implementation on
`agent/midm-forward`, M2 verification pending). Perf and receipts land in
dated addenda; nothing here is a claim until it has a command receipt.

## Root cause (from A7 + this census)

The mid-M forward is dispatch-bound: q16 step 1,051 dispatches x ~118 us vs
decode 366 x ~45-66 us. The M=1-only fused decode kernels (grouped Q4 GEMV
`QmmVecQ4Multi*`, `SdpaDecodeNative*`) fence on `x.shape(-2) == 1` /
`q_len == 1`, so every linear and every attention composes at q_len 2..16.

## Census — one forward step, Qwen3-4B-Instruct-2507-4bit (dense, 36 layers,
bf16, hd128, GQA 32/8) on the M2, MLX_OMARCHY_TRACE_DISPATCH=1

Source: OmlxDflash captured traces (disp-{decode,q16}.err), submit-boundary
parse; kernel ids via the compute.h enum.

| role | M=1 dispatches/step | q16 dispatches/step | composed chain that replaced the fused kernel |
|---|---|---|---|
| qkv proj (wq,wk,wv + biases) | 1 (QmmVecQ4MultiSubgroupBF16 + 3 add epilogues) | 6 (3x CastBF16F32 + 3x QmmPrefillCoopmatM16) + 3 FEW:Add (bias adds eager) | grouped GEMV -> per-linear coopmat qmm + eager bias adds |
| wo + residual | 1 (grouped GEMV, add epilogue) | 2 (cast + coopmat) + 2 (FEW:Add + BinaryVec residual) | same |
| gate+up | 1 (grouped GEMV) | 2 (cast + coopmat x2... one pair per weight) | same |
| swiglu | 1 (SwigluBF16) | 1 (SwigluBF16) | already eager at both |
| down + residual | 1 (grouped GEMV, add epilogue) | 2 (cast + coopmat) + 2 (residual adds) | same |
| q/k norm+rope | 2 (FastRopeNormBF16) | 2 (FastRopeNormBF16) | none - already multi-row |
| norms (in/post) | ~2 (FastRmsNormBF16) | ~2 | none |
| attention (incl. KV cache write) | 1 (SdpaDecodeNativeBF16Hd128; KV write rides the qkv group's producer-direct window) | 6 (CopyGeneralBF16 x3 KV writes/view + MatmulF32CoopmatQkBF16 + SoftmaxF32 + MatmulF32CoopmatPvBF16; causal skips the mask array) | fused decode sdpa -> coopmat scores/softmax/PV composition |
| head (greedy) | 1 (QmmVecQ4WordSubgroupBF16) | 1 | none |
| **total** | **366** | **1,051** | surplus 685: qmm 10/layer, attention 5/layer, eager adds 4/layer |

Prefill32 step = 992 (same composed shape; 6.22 qmm pairs/layer).

### GDN (Qwen3.5 hybrid) — census by code read; live trace pending

`GatedDeltaUpdate` fused decode requires `T == 1` (primitives.cpp:
"decode only: the fast.cpp caller routes T > 1 to the precomputed-gates
scan"); q_len>1 rides the chunked prefill kernels
(gated_delta_prefill*.comp) plus the perrow_pf variant. Whether the chunked
scan is bit-identical to M sequential decode steps is FamQwen35Prefill's
lane and NOT re-derived here. Qwen3.5-2B/9B MLX 4bit are in the M2 HF cache;
a live trace ticket will append the GDN rows.

## Design

Per-row bit-identity is the invariant: each row of a multi-row dispatch is
instruction-for-instruction the single-row kernel for that row, so
speculative verify rows equal the same tokens decoded alone and
batched==single digest equality holds by construction.

### (a) QmmVecQ4Multi token variants (landed on the branch, M2 pending)

- `qmm_vec.comp` gains a QMM_VEC_TOKENS main: the token dimension sits
  INSIDE the k walk, so one weight-word load feeds all M<=16 rows
  (~M x weight-read amortization at the qmm level). Per token and slot
  column: same quad order into dot, same per-column input_sum chain, same
  block-epilogue fma, same subgroupAdd tree, same RNE stores as the
  single-token column. Blobs: token4 + token16, bf16 subgroup only.
- Fence: bf16, subgroup, rows 2..16, x row-contiguous whole (rows,k),
  k%64==0. Add epilogues fold at rows>1: full (rows,n) residual via
  in_strides row stride n, or the (n,) bias broadcast via stride 0 -
  both read-add-store exactly Q4_MULTI_STORE's rounding. Out-gate
  prologue and producer-direct KV windows stay single-row (planner never
  adopts them at rows>1; the shader #errors on the flag).
- Kill switch: MLX_OMARCHY_QMM_VEC_TOKEN_MULTI=0 (planner + dispatch).
- The added epilogue reach also deletes the q16 bias/residual FEW:Add +
  BinaryVec dispatches the single-row fence could not fold.

### (d) SdpaDecodeRowsBF16Hd128 (landed on the branch, M2 pending)

- One workgroup per (head, row); every bf16-arm scan is bounded by the
  row's causal key_end = k_len - q_len + row + 1. There is no -inf mask
  arithmetic anywhere: each row IS the single-query arm over its visible
  key prefix, so row m of the q16 verify equals q_len=1 decode with a
  (pos_m+1)-long cache, bit for bit - this is exactly the A7 divergence
  surface (batched verify numerics vs plain decode).
- Same shared layout, same 32-key... (bf16 arm: same composition-exact
  f32-score order - ascending-d 16-wide tiles, 256-lane trees, ascending-key
  16-wide PV) as the qualified single-query arm. hd128 blob only (Qwen3-4B);
  window k_len<=7168 unchanged.
- Fence: bf16, causal, batch 1, q_len 2..16, hd128, strides ok. Kill
  switch: MLX_OMARCHY_SDPA_DECODE_ROWS=0.
- The q16 cache writes (CopyGeneral x3/layer) stay composed in v1; folding
  them is a follow-up behind the same per-row discipline.

### Expected q16 step after (a)+(d)

4 grouped GEMV + 1 sdpa + 1 swiglu + 4 norms + 3 cache copies per layer
~ 13 x 36 + head ~ 470 dispatches (from 1,051), with qmm-level compute
amortized ~M x. At decode-class per-dispatch cost this is the ~2x-of-decode
target; the M2 ticket decides.

## Gates

1. Doctests (llvmpipe + M2): per-row bit-identity vs the M=1 route over the
   shape grid; kill-switch determinism (both routes byte-stable).
2. Greedy digests unchanged for plain decode (M=1 route untouched: same
   kernels, same fences).
3. DFlash 8x128 greedy set ON vs OFF: identity should not regress and is
   expected to improve (verify logits converge to plain-decode bits).
4. Perf (M2, quiet box, provenance lines): forward wall q1/q4/q16 before vs
   after; dispatch counts per step; DFlash ON/OFF tok/s per the A7
   procedure.
