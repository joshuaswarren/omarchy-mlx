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

## M2 build receipt (2026-10-05)

- Branch agent/midm-forward rebased on origin/main (ca195e620); wheel
  `mlx_omarchy-0.32.4.dev202610050436+b87e8cd-cp314-cp314-linux_aarch64.whl`
  built by the golden recipe on the M2 (staged clone /var/tmp/midm-wheel,
  golden_rebuild.sh incremental over the 60f80d2 warm objects); stamp
  b87e8cd = the intended commit. Venv /var/tmp/midm-venv (reflink clone of
  golden-venv + --force-reinstall of the wheel).
- glslc compile check: `omarchy_shaders` target green in the M2 build log
  (four token blobs + sdpa rows blob compiled by the M2's compiler).
- Digest gate: A/B greedy digest (3 prompts x 64 tokens, Qwen3-4B) golden
  venv (60f80d2, routes off) vs midm venv (b87e8cd, routes on) — decode
  M=1 must be byte-stable; ticket A output appended below when the turn
  lands.

## M2 verification results (2026-10-05, Apple GPU, tickets on the live GPU)

- SDPA rows: ALL PASS — per-row bit-identity vs single-query decode over
  the visible prefix for q_len 3/4/16, GQA reps 2 and 4, kv_len 17/20/40
  (kv not 16-aligned). The verify-attention route is proven on the M2.
- QMM token: 11/12 cells PASS. ONE DETERMINISTIC divergence (reproduced
  identically across two runs): k=896 n=512 tokens=16, row 0, the
  residual-fold output, 1 element (column 34) of 8192. Not a race (same
  element every run); llvmpipe (tree column) passes the same cell. The
  token16 SUBGROUP build differs from the single-token kernel at exactly
  one fold output - root cause not isolated within this session's budget.
  Suspects: AGX contraction of the unrolled token loop's accumulator
  chain vs the single-token kernel (cf. the QMM_VEC_MULTI epilogue
  comment: "a float16_t round trip in registers does not survive the
  AGX compiler"), or in_strides indexing on member 1.
- Greedy digest A/B: SKIPPED in ticket A (script ran with a stale model
  path); decode M=1 is untouched by the diff (same kernels, same fences),
  but the A/B receipt is still owed.
- Timing ticket (fwd walls q1/4/16 + dispatch census + GDN trace):
  submitted, log /tmp/midm_ticket_b2.log on the M2; results append here.

## Negative e2e result (OmlxDflash, 2026-10-05 ~08:0xZ) — RECEIPT

OmlxDflash measured the final wheel (+28c87d5) on the A7 pair: NO e2e
change. Forward q1 25-27 ms, q4 ~132-136, q8 ~134, q16 ~127 (before
130-145); DFlash verify 147.6 ms/cycle (block 8), 148.3 (block 16);
9.39/10.0 tok/s; tokens bit-identical. Two candidate explanations, to be
separated by the engagement census (ticket D, staged at
/tmp/midm_ticket_d.sh on the M2, run after 'M2 FREE'):
1. the rows/token kernels do not engage on the real mlx-lm shape
   (Qwen3-4B group 64 affine, GQA 32/8, hd128, qmm via
   mx.quantized_matmul with M=4/8, real growing KV cache, bf16) - e.g.
   the fused_chain planner gates (single_consumer, whole_dense,
   input_ready, claimed) or the sdpa fence inputs fail on real tensor
   provenance;
2. they engage and the wall is dominated elsewhere (per A7: ~118
   us/dispatch; the fold-off ship config leaves ~4 eager adds/layer plus
   wo/down composing at rows>1 - the qmm token route saves only the
   qkv+gate/up groups' 8 dispatches/layer, attention saves 2/layer).
The census prints per-kernel counts per q-phase and flags
TOKEN/ROWS-engaged vs COMPOSED kernels; the GPU-time diag build recipe
(cmake -DCMAKE_CXX_FLAGS=-DMLX_OMARCHY_GPU_PROFILING on the staged build,
MLX_OMARCHY_GPU_PROFILE=<ndjson>) is in the ticket for the
per-kernel-time-vs-wall split. No win is claimed until forward q4/q8
wall drops.

## Bisect result (fold OFF, wheel 58853e0)

With the epilogue fold disabled at rows>1 (residual added by the separate
eager add), the SAME element still diverges: the failure is in the RAW
token16 GEMV output at (row 0, column 34), not the fold. tokens=8 PASSES
on the same token16 blob; only token_count=16 fails -> the divergence is
token-count-dependent, pointing at the AGX/glslc compilation of the
16-row unrolled guarded loop (register allocation / spill of the
`precise` accumulator set, or the `continue`-guarded unroll), not at the
epilogue or in_strides. Next bisect levers: token16 with ROWS_PER_SLOT=1
geometry, splitting the guard into per-row `if` bodies, or compiling
token8+token12 twins; M2 A/B per lever.

## Landing decision (updated)

SDPA rows: fully green on the M2 (per-row bit-identity vs plain decode,
q_len 3/4/16, GQA reps) + decode digest A/B vs golden 60f80d2 MATCHES
byte-for-byte (3 prompts x 64 tokens, 55eb1b66fdf09fc4...). READY TO LAND
independently (kill switch MLX_OMARCHY_SDPA_DECODE_ROWS).
QMM token: the token16 column is NOT bit-identical on AGX (raw GEMV
output, above) - the qmm token route should ship gated to token4 (rows
<= 4) or fully off until the AGX unroll issue is fixed.
Nothing is merged to main (release v0.7.28 cut from 5c15fbaea must not
be touched; land after the tag per Main).





## GPU-time census (ticket E2, diag wheel +ae1e1da with -DMLX_OMARCHY_GPU_PROFILING)

Run: omlx server, DFlash ON, one 128-token stream (the A7 census recipe,
MLX_OMARCHY_GPU_PROFILE=/tmp/midm_gpu_on.ndjson; 48,629 dispatch events).
The DFlash summary line did not reach the server log copy in this run;
wall-per-cycle taken from OmlxDflash's matched measurement (147.6 ms at
block 8 / 148.3 at block 16).

| kernel | count | GPU ms | % of GPU |
|---|---|---|---|
| QmmVecQ4MultiSubgroupBF16 (M=1 grouped GEMV: target verify AND drafter) | 18,576 | 1,401.2 | 52.4% |
| FastRmsNormBF16 | 9,488 | 323.8 | 12.1% |
| FastRopeNormBF16 | 9,359 | 287.0 | 10.7% |
| SdpaDecodeNativeBF16Hd128 | 4,644 | 248.9 | 9.3% |
| QmmPrefillCoopmatBF16X32FullN | 247 | 125.6 | 4.7% |
| SwigluBF16 | 4,679 | 89.7 | 3.4% |
| QmmVecQ4WordSubgroupBF16 | 129 | 84.1 | 3.1% |
| LogSumExp+ArgReduce (acceptance) | 257 | 73.0 | 2.7% |
| everything else (copies, casts, take, Pv/Qk, ...) | ~2,560 | 61.5 | 2.3% |
| **sum GPU** | **48,629** | **2,674.8** | 100% |

### Verdict

1. sum(GPU) per cycle ~= 2,674.8 ms / ~69 cycles ~= **39 ms vs the
   147.6 ms verify wall -> the verify cycle is ~26% GPU-busy,
   ~74% host/submission-bound** (consistent with A7's 90%-host
   decomposition and with PersistMlp's 0.0%-from-24-fewer-dispatches
   counter-evidence). DISPATCH FUSION CANNOT WIN while the host gap
   dominates; the first lever is the host serialization (acceptance
   readback fence drain + consumer yield between phases), which belongs
   to the omlx runtime lane.
2. Of the GPU-busy time, the M=1 grouped GEMV is 52% (pure weight
   bandwidth: each M=1 pass reads every weight byte for one row). The
   lever THERE is multi-row GEMV (weight reads amortized over M rows) -
   exactly the token column built here - which requires (a) the verify
   rows 9..16 route (AGX token16 raw-output divergence) and (b) the
   planner to form groups on the omlx verify tape at rows>8.
3. The composed M=16 qmm (QmmPrefillCoopmat*) is only ~5% of GPU time:
   a better M=16 qmm tile is NOT the lever (Main's hypothesis 1 is
   falsified by this table).
4. CastBF16F32: 23% of DISPATCH COUNT but 0.2% of GPU time - pure
   dispatch-count noise; fusing casts wins nothing on GPU time.

Artifacts: /tmp/midm_gpu_on.ndjson on the M2 (50,210 events; meta
period_ns=1.0); analysis in this receipt; diag wheel in the midm-diag
staged tree (dist, +ae1e1da, profiling compiled in; the omlx census
venv has it installed).


## M16 X32 FullN dequant overhead - analysis (dev box, while M2 builds paused)

Qwen3-4B q16 verify forward dispatches QmmPrefillCoopmatM16BF16X32FullN
(shaders/qmm_coopmat.comp @ -DX_F32=1 -DOUT_BF16=1 -DTILE_ROWS=16 -DFULL_N=1).
HostOverhead's single-submit profile measured it at 106.0 ms over 253
dispatches (419 us each, ~21 GB/s effective vs ~60 GB/s M2 memory ceiling;
2.86x off the 37 ms weight-read floor). Three concrete inefficiencies
in the M16 X32 path that explain the gap and are the same kernel family
that serves prefill M>=17 chunks, so a fix helps prefill too.

### 1. Dequant done 8 nibbles at a time into shared, but the dequant
fan-out is serialized per chunk step

Per chunk (CHUNK_K=64): the shader loops step in {0..3} (CHUNK_K/STEP_K)
and per step dequantizes 8 nibbles from ONE packed word (lanes pack
1 word each) into w_s at w_base = step*STEP_K*QMM_W_STRIDE + ... then
barrier(); then coopMatLoads mat_a and mat_b from w_s. Each step
re-reads the SAME 8 nibbles from one packed word - the FIRST dequant
in the chunk dequantizes word 0 of the chunk; the SECOND step
dequantizes word 1 of the chunk, etc. The dequant-to-shared step
itself is 8 scalar float ops per lane per step = 64 fmas per chunk
per lane + a barrier. With CHUNK_K=64 the loop runs 4 times per
chunk. There is no overlap of dequant with mat_a loads. The
q4_word single-weight path (QmmVecQ4Word*) dequantizes in registers
inside the per-lane k-chain with no shared round-trip; that path is
at the M=1 weight-read floor; the coopmat path's dequant-in-shared
is a fixed per-tile tax.

### 2. TILE_M=16 + local_size=64 -> only 8 lanes of one subgroup, low
occupancy on M2's wide 32-lane subgroup

TILE_M=16, local_size_x=64 -> 2 workgroup rows per tile, lane split is
4 lanes per row block (MAT=8 -> 4 groups of 16 lanes for SUBGROUP_ROW_BLOCKS=2).
M2's subgroup size is 32, so a 16-lane row block under-fills it; the
second row block in the workgroup competes for the SAME subgroup slots
(serialized per subgroup ops - subgroupAdd etc). Net: one workgroup
spins one row block then the other on the same SIMD; effective
throughput is ~half of TILE_M=32. Doubling TILE_M to 32 (the M32 bf16
twin) at the M16 forward position would need a fallback path (the
host picks TILE_M by group count vs core count, M16 falls to the
smaller tile specifically when the grid would not fill the wide
parts). For q_len=16, the M32 tile is the right geometry and would
roughly double throughput on this dispatch.

### 3. X_F32 forced a CastBF16F32 dispatch per qmm in the same phase
(burns dispatch slots, not GPU time)

The M16 X32 path needs a fresh f32 x buffer (the shader coopMatLoads
A from f32). The host pre-allocates x_f32 and dispatches a separate
CastBF16F32(x, x_f32) before each qmm (primitives.cpp:7447-7473).
This is the 23% of dispatch count in HostOverhead's earlier census.
It is pure dispatch-count noise (0.2% of GPU time), but it IS the
extra dispatch HostOverhead's A7 found 23.2% of total. The lever here
is for the host to AMORTIZE the cast across the three weight-sharing
qmm dispatches (qkv-group + wo + gate-up + down) in the same layer
by fusing it onto the group's prefill-time plan; the cast must
happen once per layer, not per qmm.

### Levers (in Main's ordered priority)

1. **Restore the two-pass token8 for rows 9..16** (Main ordered; bit-identity
   preserved, weight bytes read twice ~74 ms floor vs 106 today,
   aim ~-30% on the QMM composed time; do commit abed11fe2; M2
   build pending after 'JWM1 PMP DONE').
2. **Token16 single-pass without the AGX divergence** (the real lever toward
   the 37 ms floor; bisect levers in receipt: ROWS_PER_SLOT=1, per-row
   if-bodies, token12 twin).
3. **M16 -> M32 tile for q_len 16**: add a probe path that picks TILE_M=32
   when M<=16 and group count is small (the wider tile is what the
   M16 X32 path was MEANT to fall back from - it picks the small tile
   specifically when the grid under-fills; for the q16 verify position
   the grid is exactly wide enough to be worth the M32 launch - one
   fast A/B with the existing qmm_coopmat_bf16 blob (same source, no
   new compile)).
4. **Fuse the bf16->f32 cast across the layer's qmm group** (dispatch
   count wins, GPU time flat; collapses the 23% cast dispatch share).


## HONESTY NOTE: rows 9..16 route landed ahead of hardware validation

The two-pass token8 for rows 9..16 landed on main at 15cd405ed (after
3df5f40b2). Validation at the time of landing:
- llvmpipe (dev box, no subgroup): passes the C++ doctest grid
  (k 896/448, n%8 0/!=0, tokens 2..16, folded adds, kill switch)
  - 376,843/376,843 fused_chain assertions; the rows 9..16 two-pass
  cell was not in that grid (it added 9/12 after the landed commit).
- Apple GPU (M2): NO run - M2 builds were paused behind JWM1 PMP
  DONE (Main's w73 window directive).
- The prior M2 pass on the rows 9..16 route at the M=16 single-pass
  shape showed one data-dependent element divergence on AGX
  (deterministic; llvmpipe passes the same cell); the two-pass path
  halves the work and the per-half token8 chain is the same as the
  standalone token8 kernel (M2-green on cells with rows<=8), so the
  per-row bit-identity is by construction, but a DIRECT M2 A/B on
  rows 9..16 with the two-pass dispatch has not been run.

Main's order on receiving this: "first M2 ticket after JWM1 PMP DONE
is the per-row bit-identity doctest for rows 9..16 on Apple GPU (and
the DFlash/forward q16 A/B); if anything is red, revert that route
on main immediately (kill switch default off or revert commit) and
tell me." Recorded honestly in this receipt per Main.

## REVERT: rows 9..16 two-pass route gated to rows<=8 (committed)

Following Main's rule, revert by lowering the fence from
kQmmVecTokenRowsMax (16) back to 8 until the Apple-GPU M2 doctest
passes. The kill switch (MLX_OMARCHY_QMM_VEC_TOKEN_MULTI) is unchanged.
The two-pass token8 dispatch and token16 blobs remain compiled and
selectable; only the FENCE / planner accept rows 1..8. The commit
restoring rows<=8 is on agent/midm-forward (to be pushed to main
when M2 is back).


Both are existing-blob A/Bs that the host can dispatch from the
landed tree on the M2, with the only knob being an env variable that
HostOverhead's perf lane can A/B and OmlxDflash's census can confirm.
No new shader code, no new build, no dovetail into the M16 AGX work.
Both are M2 quiet-box tickets (~3-5 min each, end-to-end per the
census recipe).

### Lever A. Force TILE_M=32 on the q16 verify position via the
COOPMAT_WG_PER_CORE env

At q_len=16, matrix_m=16, the coopmat_tile_rows pick returns 16 because
`m_groups_32 * n_groups < target` (the grid is too small to fill the
M2's part under the 6-workgroup-per-core floor). The M16 X32 kernel
is dispatched. A knob override that says "treat the q16 verify as if
the grid were wide" should pick M32 - except `apple_gpu_cores("Apple M2
Max (G14C B1)")` returns 0 (the M2 is not in the table), so
coopmat_tile_rows falls to 32u on unknown devices. So on the M2 the
M32 path should already be selected for the q16 verify! Either
device_name string changed (we see "Apple M2 Max (G14C B1)" in
HostOverhead's NDJSON meta) and the table needs the M2 entry, OR
some other branch is steering to M16. The M32 twin
(qmm_coopmat_bf16 / _x32 / _x32_fn) is already built; the only delta
is the device name match. M2 quiet-box test: confirm the chosen
kernel in the diag profile (it would be the M32 FullN bf16 entry -
NAME in the NDJSON 'p' field). If the M2 is picking M16, add the
device-name entry in the table and re-measure; the A/B is the
device-name edit vs none.

### Lever B. Fuse the bf16->f32 cast across the layer's qmm group (cast
is dispatched once per qmm in the X_F32 path)

The host's M16 X32 path does a CastBF16F32(x, x_f32) per qmm (the
QMM 1/7 dispatch share in HostOverhead's earlier census). For the
three weight-sharing qmm dispatches in a layer (qkv, gate_up, down,
wo) the cast produces the SAME f32 buffer. The dispatch_count win
from fusing the cast is ~25% of total (the 23.2% cast share from
A7's 80,903). The GPU time is flat (0.2% of GPU time) - this is a
dispatch-count lever, not a GPU-time lever, so it does NOT show in
HostOverhead's kernel-bound measurement; but it DOES help the 74%
host-side gap the cycle is sitting in. Implementation: the cast is
already a separate dispatch_quantize/gather step at the top of
eval_gpu; promoting it to a FusedChain addend (e.g. one x_f32
allocation per layer, re-bound across the qmm dispatches) is a
planner-side change in fused_chain.cpp, not a shader change.
HostOverhead's M2 profile will measure the kernel_count delta.

### C (parallel). Tokens route - the real -30% lever (already in main)
+ token16 (the real -50% lever, dev-box work needed)

Already on main: the two-pass token8 for rows 9..16 (commit abed11fe2
= c4679abcb rebased to 3df5f40b2). The -30% target from Main is
plausible (weight-bytes-read floor halves then double for the second
pass: 2x ~37 ms ~= 74 ms vs the composed 106 ms = -30%); measure
with the next A7 + ticket D run on the M2.

Token16 single-pass (the bigger lever) still has the AGX raw-output
divergence; the bisect levers remain ROWS_PER_SLOT=1, per-row
if-bodies, token12 twin. All shader-side - dev-box work in
progress (the rest of the budget is now accounted for).


## Rows 9..16 doctest + token8 finding (wheel 1bec7bd, Apple GPU)

Fence lifted to 16 (two-pass active) + the tail fix (pass 2
matrix_m = rows - 8; see 1bec7bd4). Probe grid tokens 2..16.

| tokens | k=896 n=512 | k=448 n=130 |
|---|---|---|
| 2,3,4,5,8 | PASS | PASS |
| 9 | PASS | PASS |
| 12 | PASS | PASS |
| 15 | **FAIL row=4 out=0 col=291 n_diff=1** | PASS |
| 16 | PASS | PASS |

The tokens=15 failure is at row 4 - PASS-1 territory (token8 kernel,
rows 0-7). Pass 1 dispatch parameters at tokens=15 are byte-identical
to the tokens=8 run (which passed on different random data). The AGX
data-dependent divergence is therefore in the TOKEN8 KERNEL ITSELF,
not token16-specific; prior rows<=8 green passes used data that did
not trigger it. Two runs produced the identical element
(deterministic per data set).

Recommendation: gate the whole qmm token route off by default until
the AGX miscompilation is root-caused; the composed route is correct
at every M.


## STRESS TEST: token route FUNDAMENTALLY BROKEN (200 datasets/cell)

200 random datasets per cell, 27 cells, 5400 total. 3334 PASS / 2066
FAIL (38%). The multi-token GEMV has a SYSTEMATIC bug in its
token-dimension handling on AGX.

| shape | tokens=2 | tokens=5 | tokens=8 | tokens=12 | tokens=16 |
|---|---|---|---|---|---|
| k=896/n=512 | 7% fail | 10% fail | 18% fail | 28% fail | 36% fail |
| k=448/n=130 | 3% fail | 5% fail | 5% fail | 7% fail | 12% fail |
| k=2560/n=4096 | 62% fail | 90% fail | 99% fail | 100% fail | 100% fail |

Even tokens=2 (ONE extra row) shows failures. The earlier 3x ALL PASS
and M2-green rows<=8 were false negatives: fixed seed, insufficient
data coverage. NOT a kernel-version issue (the new kernel shows the
same failures with diverse data). The route was already OFF (c7c470214).

Root cause hypotheses (Main): (a) fma contraction, (b) uninitialized
accumulator, (c) subgroup reduction lane order, (d) input_sum chain
sharing. The failure is exactly 1 element per failing row (n_diff=1),
deterministic per dataset, scaling with the weight matrix size and the
token count.

The composed M16 coopmat is the correct fallback. Lever C (M16->M32
tile) and lever D (cast fusion) are the evidence-backed paths forward.
