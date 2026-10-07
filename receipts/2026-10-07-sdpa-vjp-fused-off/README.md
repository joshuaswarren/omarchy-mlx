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
