# 2026-09-29 — jw16 decode: attention out-gate folded into the o_proj GEMV prologue (F2), +1.0–1.5% decode, bit-identical

Actor: DecodeGap6. jw16 (M1 Max, T6001), boot 8c3d0b5c throughout. Branch
`agent/jw16-decode-gap6` off main (bench-base refresh 30e4, out-gate fold
77256c009, member-stream fix 9e9125a9c), rebased onto main f29c101db.
Pre-registered in the private notebook
`entries/Jw16DecodeGap6/20260929T210113Z-jw16-decode-fusions-lmhead-census.md`;
artifacts under `artifacts/Jw16DecodeGap6/{w1,w2,w2-battery}/`.

## What changed

`Multiply(Sigmoid(gate), out)` whose only consumer is a decode QuantizedMatmul
group (the qwen3.5 attention out-gate feeding o_proj) is recomputed per
workgroup inside a new `QmmVecQ4MultiOutgateBF16` variant of
`shaders/qmm_vec.comp` (flags bit 17, bindings 25–27): the prologue reads the
two vectors directly and produces the same bf16-rounded product the deleted
elementwise pair wrote — `sig_b = bf16(1/(1+exp(-f32 g)))`,
`m = bf16(f32(out) * f32(sig_b))`, the exact unary_vec.comp/elementwise.comp
arithmetic and one rounding per kernel boundary, the same formula the
conv-silu fold (DecodeGap5) verified. Workgroup 0 materializes the product so
retained references stay valid. kComputeBindingBudget 25 → 28.

Planner-side (fused_chain.cpp): the pattern is detected on the tape with
single-reader conditions (gate read only by the sigmoid, out only by the
multiply, multiply consumed only by the group's member), adopted only when
every dispatch-time condition is decidable at plan time, and the deleted
nodes map to skip roles that fall back to the ordinary path if the group
never dispatches; a post-adoption refusal throws instead of silently skipping
the deleted nodes. No mlx-lm model change — the serving stream folds without
a patch.

## Gates (window 2026-09-29T21:38–22:05Z, gpuwin discipline, restore health 200 + completion finish_reason=length + active)

- dg_bitcheck.py full battery: 3084 rows byte-identical serving vs candidate
  venvs; the only JSON difference is the wheel version string.
- Cells n=5 interleaved, same window, digests pinned:
  d64 ctl 100.34/100.26 vs cand 101.73 (c84b3e7af640), d128 100.25/101.44
  (07c515e0338b), d256 98.80/100.10 (c6aabbf0a51d), d512 94.79/95.71
  (5c120987f0e5). Gains +1.0..+1.5%; d64 candidate outside the control
  min-max by +1.4% (the >=+1% gate).
- Race battery (gated_battery.sh, 8 interleaved 10x10 contract pairs,
  X=1 venv-only candidate): RUNNING at yield time (pair 1 clean: ctl 100.13 / cand 101.38, contract digest dbf704971617 both arms); detached lander applies the objective gate (all 16 digests == dbf704971617), lands, rebuilds, deploys, and verifies — see /var/tmp/dg6-local/lander.log and origin main history for the outcome.
- 12 dependent dispatch boundaries/token removed (6 Sigmoid + 6 Multiply at
  ~12.2 us each) — measured +1.0..+1.5%, consistent with the boundary model
  minus the prologue's recomputation cost.

## Ratios vs macOS window-5 (179.72/179.08/178.72/177.02)

measured in-session: d64 101.73/179.72 = 0.566, d128 101.44/179.08 = 0.566, d256 100.10/178.72 = 0.560, d512 95.71/177.02 = 0.541 (candidate arm, window 2026-09-29T21:38Z; deployed ratios land with the lander log).

## Not landed from this lane (measured, below the bar)

- lm_head word kernel: production iso 218.1 GB/s vs the best GEMV access
  pattern 257.6 (280 min) on jw16 — tiling candidates (unroll/loadfirst/
  wg128/xpack) all within noise-to-worse, matching jwm1's verdict. The greedy
  stage-1 sketch read (assignment's 232 GB/s) has only ~80–120 us/token of
  roofline headroom (+0.8..+1.2%); left to the next lane with the stage-1 ILP
  sketch (3× uvec2 loads, 2-row interleave).
- F3 (attn q/k rms_norm -> rope): designed (exact fast_norm 256-row tree +
  one-head-per-workgroup rope), not implemented — highest exactness risk of
  the three assignment levers, deferred rather than half-landed.
- Lane D (census): control 319 dispatches/token vs fused 301 — the diff is
  exactly the 18 silu compiled-chain dispatches; the "+4" vs DecodeGap4's 315
  is head-class drift between wheels (greedy head stages replaced the
  full-head GEMV; Take/TakeU32/Dequant left the stream). No hidden op.
