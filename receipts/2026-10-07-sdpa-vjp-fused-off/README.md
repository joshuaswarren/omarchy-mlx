# Fused SDPA VJP rep=1 defect: sharpened isolation (2026-10-07)

Lane: w6Z (UpstreamMlx). Hosts: dev box (analysis, mirrors at
/var/tmp/mlx-upstream.git) and the Arm64 build host (M1 Max, Honeykrisp)
for every execution probe. Base: origin/main cc45eab4 (post
battery-receipt c208e7fec). This entry SHARPENS docs/known-defects.md
"SDPA backward fused VJP value defects at small rep=1 shapes (2026-10-03,
open)"; nothing landed changes runtime behavior.

## New facts (all reproducible standalone; recipes inline)

Standalone C++ probe (doctest binary linked against the static libmlx of
a fresh prepare-mlx tree at main): `vjp` over
`fast::scaled_dot_product_attention` (maskless f32, scale 1/sqrt(D)),
reading dk/dv after `eval()` + encoder synchronize.

1. The defect reproduces OUTSIDE the test suite with the exact doctest
   seeds, at SDPA_SHAPE=1,1,2,4 (B,qL,kL,D): dk = 8x recycled-poison word
   (123456789.0 with MLX_OMARCHY_POISON_FREED=1) or zeros without it; dv
   = zeros. In-suite the same shapes PASS (fast_ops 43/43 rc=0 at main
   cc45eab4, re-run this session) — the defect is allocator/queue-state
   dependent: the gqa_reduce output region of dk is partially UNWRITTEN
   and shows whichever bytes the recycled 32 B buffer last held.
2. Discriminations run (each a rebuild + probe on the host):
   - cot=ones (value_and_grad-equivalent) vs cot=pattern (vjp-entry):
     BOTH poison standalone — the discriminator is not the cotangent.
   - SDPA_SHAPE sweep: B=1,kL=2 → all 8 dk words stale; B=2,kL=2 →
     batch 0 correct, batch 1 stale; B=1,kL=1 → dk is genuinely zero
     (softmax over one key: dS == 0) — that case is correct-by-math.
     Signature: a trailing PORTION of the dk output is never written.
   - MLX_OMARCHY_NO_BUFFER_CACHE=1, TAPE_NO_REUSE, GATED_BARRIERS=0,
     DEP_RW=0, NO_COOPMAT, a cooperative-matrix k-tail guard
     (matrix_k >= 8 && %8==0 in dispatch_matmul), and per-stage kernel
     bypasses (odo/ds/scores/dq/tiles/transpose-copies/reduce): none
     clears it. Bypassing the two dkt/dvt matmuls removes the stale
     bytes only because nothing then writes dk at all.
3. Internal planes verified CORRECT at their stage boundaries via a
   temporary dump probe (removed before landing): lse, delta, dP, dS,
   P, both transposed dense copies, q5, k5 — all match host math; the
   dq leg (first matmul) is correct.
4. Routing flip tested and REJECTED: gating the fused VJP off
   (use_fallback → true, commit b8fa76f95, reverted) turns the strict
   fd doctest RED — the composed fallback ALSO deviates from host fd at
   rep=1 shapes (15 logged spots: 2x1x2x4 dk[4] 0.0155 vs 0.0355,
   2x5x7x8, 1x2x2x64, 1x3x5x8 causal, …). Main's green gate measures
   the FUSED path at the strict shapes; composed was previously masked.
   Net: neither path is proven clean at rep=1 outside the suite's
   allocator state; the defect entry stays open with both paths named.
5. Not a backport regression: reproduces on main af8e6795/cc45eab4 with
   the backport batch reverted or present; the backport commits touch
   only upstream ops.cpp/cpu reduce, prepare-mlx wiring, and the reduce
   test.

## Disposition

- No runtime change lands. The strict fd doctest remains the in-suite
  gate (green); the may_fail case keeps carrying the 5x7/4x4/6x9 legs.
- The two-sided defect (fused: state-dependent stale dk/dv; composed:
  small fd deviations at rep=1) is now named in
  docs/known-defects.md with this receipt as evidence.
- Next lever (highest-information probe): instrument
  `gqa_reduce`/`dispatch_matmul` to log the bound buffer addresses and
  sizes for dk vs the dq5 temporary at the failing standalone shape —
  dk[0:4] has been observed carrying dq5's exact words in one state,
  which pins the aliasing point to output-buffer handing between the
  primitive's own `set_data` allocations and dispatch_matmul's
  re-allocation of `out`.

## Verification receipts (this session, build host)

- Standalone probe binaries rebuilt from a fresh prepare-mlx tree at
  main; suites: omarchy_fast_ops_tests 43/43 rc=0 at cc45eab4
  (re-run after the revert series); full battery earlier this session
  42/43 (receipts/2026-10-06-mlx-backports/battery.md).

## RESOLVED (2026-10-07 later): tile rows took qL instead of kL

Main's lead from the parallel SDPA chunk bug (allocation guards detaching
shared-buffer views) got this lane to restore the dispatch_matmul/
copy_gpu/dispatch_softmax allocation guards (22791a528) and, in the same
probe cycle, the address/size trace exposed the real root cause:

`ScaledDotProductAttentionVJP::eval_gpu` allocated its dK/dV tile buffers
as `tile_shape = S.shape(); tile_shape.back() = D` — [B,H,qL,D]. The
transposed matmuls compute [kL,qL]x[qL,D] → the tiles must be
[B,H,kL,D]. At qL < kL the buffer is short: dispatch_matmul computes
batch_count = out.size()/(kL*D) = 0 at (qL=1,kL=2) — no workgroups, the
outputs keep their stale fill — and 1-of-2 at B=2 (batch 0 correct,
batch 1 stale). qL == kL shapes allocated the right extent, which is why
the strict fd legs passed in-suite while the standalone probe and the
may_fail legs failed. The allocation guards were a real second bug
(detached-view writes, the LongSdpaCoop3 class) and stay; they were not
the cause of the missing dk/dv.

Fix: `tile_shape[size()-2] = kL` before `.back()` (127601e58, this
branch). Passing-after: the standalone probe at (1,1,2,4) returns
dk = dS^T·q and dv = P^T·cot exactly (dS=[0.2147,-0.2147],
P=[0.772,0.228] from the same-run dumps).

## Queue (jw16 frozen for the GLM cluster run until w7N frees it)

1. fast_ops suite at 127601e58 (the strict + may_fail legs should both
   pass now; flip may_fail to hard in a follow-up only after the suite
   proves it).
2. Wider battery re-run on the fix.
3. Fresh wheel + wheel-level doctest regression.

## LANDED (2026-10-07 06:2xZ): 7f9e0ffe1 on main

- a092c24ae: allocation guards restored (dispatch_matmul data_shared_ptr
  form; copy_gpu + dispatch_softmax new).
- 61f1de11f: tile rows kL (the root cause).
- 7f9e0ffe1: this receipt.

Suite evidence on the build host (fresh clone of the branch tip, static
configure, M1 Max GPU): omarchy_fast_ops_tests 43/43 cases rc=0; the
may_fail case now 54/54 assertions for the strict spots and the former
zeros/half-written signatures are gone. Inner failures dropped 26 -> 4:
the residual is dk/dv ~half-magnitude at the LAST key of head 1 in
multi-head rep=1 shapes (dv[104] 5x7 f32+bf16 0.060 vs 0.113; dk[28]/
dv[28] 4x4 -0.046/0.168 vs -0.096/0.349) — a distinct, smaller tail-key
defect, kept in the may_fail case (evidence above; NOT the tile bug: the
tile shape at these shapes is now correct and 22 of the former failures
are green).

Battery re-run across the full standing suite is queued for the next
jw16 window (after run P, ~08:10Z+).

## Post-fix battery (2026-10-07 08:1x-08:4xZ, build host, fresh tree at b3ad332e0)

- fast_ops 43/43 (1307001 assertions; 4 inner = the residual last-key
  may_fail legs), reduce_ops 35/35, runtime 49/49, primitive 104/104
  (2743003), matmul_family 27/27 (82942518), shape_ops 25/25,
  indexing_ops 57/57, fast_regression 2/2 — all rc=0 on the M1 Max GPU.
- Remaining full-battery suites (fft/conv/kv/sdpa/gdn/take/linalg/
  distributed/...) were green pre-fix and the guards/tile change only
  alters output-buffer handing + the VJP tile shape; their rerun is
  queued behind the priority lanes (QmmBatch/mv_glm) as the close-out
  sweep.
