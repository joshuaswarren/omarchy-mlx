# 2026-10-01 — jw16 decode: F6 producer-direct KV (norm-rope keys + raw GEMV values) landed and deployed

Actor: DecodeFuse6. jw16 (M1 Max, T6001), boot d8416e59 throughout the lane's
windows. Branch `agent/jw16-decode-fuse6` = F6 commit cherry-picked onto
origin/main (rebased twice over mid-flight main moves: a4894d5bc → 31af03eac;
final code commit 1e7cf5c45). Pre-registered in the private notebook
`entries/Jw16DecodeFuse6/20261001T045500Z-jw16-f6-land.md`; artifacts under
`artifacts/Jw16DecodeFuse6/` (reconcile, battery, cells, final, final-rebased,
deploy, each with SHA256SUMS). Continues Jw16DecodeFuse5 (gated wheel
fuse5.f6.6cab7ba54; dg_bitcheck 3084 identical, d64 +2.1% paired on e81ba3aa)
and Jw16DecodeFuse4.

## What changed (code commit 1e7cf5c45)

F6 (upstream commit 6cab7ba54, byte-identical gated files at the land):
- `plan_keys_window` accepts the 3-input rope_rms_norm RoPE (`has_norm`) —
  F3's fused q/k norm rides the same RoPE primitive with the norm weight as a
  third input; the redirected dispatch is norm- and bf16-aware and gates the
  input count itself.
- `plan_values_window` accepts a raw QuantizedMatmul/GemvGroup member output
  as the values terminal (v projections carry no Add epilogue); use_count==1
  enforced on the terminal.
- `dispatch_quantized_gemv_group` drops its epilogue requirement for a planned
  window and sets flags bit 18 (262144) so `qmm_vec.comp` Q4_MULTI_STORE
  stores the rounded output (not sum+addend) straight into the KV window.
- Kill switch `MLX_OMARCHY_KV_DIRECT=0` (planner + direct-write path default
  ON; the planner was provably inert before F6 because both sides refused).
- No default flips were needed at the land; the tree diff vs main is exactly
  fused_chain.cpp, primitives.cpp, qmm_vec.comp.

## The 18 [kv-plan] refusals, reconciled (the F5 gate's "expect 0" was wrong)

DecodeFuse5's gate script expected zero `[kv-plan]` lines ("F6 firing =
silence") and got 18. Reconciliation (pre-registered hypothesis, then proven):

- The trace is a per-process static counter capped at 6 that prints ONLY
  refused/unfirable pairs; pairs that plan print nothing.
- The 18 lines (6 pairs × 3) come from the process's FIRST graph eval — the
  warmup `mlx_lm.generate(prompt, max_tokens=8)` PROMPT PREFILL: the printed
  update shapes are [1,2,11,256] (an 11-row cache bulk update; measured prompt
  = corpus p001 = 12 tokens), which matches NEITHER a decode update
  ([1,2,1,256]) NOR the pure-prefill 512 leg. In the merged log the lines sit
  at 1879-1896, thousands of lines before "pass 0 done" (line 21625).
  Byte-identical lines appear on the PRE-F6 wheel's W1 dag run — the widening
  changed nothing for these pairs.
- The decode pairs FIRE: with MLX_OMARCHY_DAG_DUMP=1, dispatch count drops
  23412 → 22524 = 888 = exactly 12 × 74 kv SliceUpdate dispatches removed
  (12 per decode eval; 74 = 64 measured + warmup decode evals), digests
  bit-identical between KV_DIRECT on/off (cb3e87705c65, 1-pass protocol).

## Gates (all on boot d8416e59; serving venv untouched until deploy)

- **Race battery** (gated_battery.sh 8 pairs, ctl=serving vs cand): **16/16
  runs `dbf704971617`**; cand medians 105.16-105.66 vs ctl 103.68-103.97 —
  disjoint; every window restored health.
- **pf cells + logits** (candidate venv): pf512 records **100a61b62470 on
  5/5**; logits gates **f771c4265f88 (T=512) / ce24f3b4ce42 (T=1024) /
  b8c4e14f8f8a (T=2048)** all finite.
- **Whole-model bitcheck, final wheel**: GATE-FULL **PASS rows=3084** (twice:
  d54187e04 and 1e7cf5c45 wheels).
- **Paired cells, final rebased wheel** (1e7cf5c45, n=5 interleaved):
  d64 ctl 103.94 / ctl2 103.95 / **cand 105.76 (+1.75%)**, d128 103.57/105.63
  (**+1.99%**), d256 102.61/104.21 (**+1.56%**), d512 98.17/99.57
  (**+1.43%**) — cand outside ctl min-max at every length; **all four digests
  at the pins in every arm** (c84b3e7af640 / 07c515e0338b / c6aabbf0a51d /
  5c120987f0e5); **kill-switch arm inert** (103.98, at pin, inside ctl noise).
  The first rebased wheel (d54187e04) measured +1.71/+1.31/+1.68/+1.45 with
  the same gate results; DecodeFuse5's gated-wheel window on e81ba3aa
  measured d64 +2.1% — gain consistent, boot-to-boot spread ±0.5%.

## Deploy + deploy-verify (no env vars, defaults engage)

- deploy_wheel.sh: serving /var/tmp/v072-venv-fused
  0.32.3.dev202610010241+9db4d7d19 → **0.32.3.dev202610010525+1e7cf5c45**
  (wheel sha256 c1a7d67a283b5bef06e8996623155bbf5306157f4aa28d40e20aa03a4dccd849);
  rollback point `/var/tmp/v072-venv-fused.pre-20261001T003048`.
- Patch set byte-identical pre/post deploy: generate.py bc4903b5bda7d705,
  qwen3_5.py 88f803754e951f46 (greedy-prune, last-logits, conv-silu,
  rope-norm, qknorm intact).
- **Deploy-verify**: d64 **105.66 tok/s** at pin c84b3e7af640, d256 **104.36**
  at pin c6aabbf0a51d; pf512 records **100a61b62470 on 5/5**; logits gates
  f771c4265f88 / ce24f3b4ce42 / b8c4e14f8f8a all finite; restore health 200 +
  authenticated completion finish_reason=length + is-active.

## Ratios vs macOS window-5 (179.72/179.08/178.72/177.02)

Deployed stack: d64 105.66/179.72 = **0.588**, d128 105.63/179.08 = **0.590**
(final-wheel cand arm), d256 104.36/178.72 = **0.584**, d512 99.57/177.02 =
**0.562** (was 0.577/0.580/0.573/0.554 at the pre-F6 serving stack).

## Rollback

- Runtime: `MLX_OMARCHY_KV_DIRECT=0` (exact eager pair dispatch, digest-pinned
  and inert-verified) or `MLX_OMARCHY_ROPE_NORM_FUSE=0
  MLX_OMARCHY_GDN_QKNORM_FUSE=0` for the older folds.
- Deploy: swap `/var/tmp/v072-venv-fused.pre-20261001T003048` back and
  restart llm-inference.
- Code: revert 1e7cf5c45 (single commit).
