# 2026-10-04 DispatchFuse — 3-model decode dispatch census; 4B rope-norm fold lands default ON (+4.2..+4.9% bit-exact); 9B GDU raw-route enabled behind gates

Lane: DispatchFuse (worker). Host: jw16 (M1 Max, T6001). Base: origin/main
b8adbc966; land tip: aae57ea0e (rebased onto 4964dc9d4). Private notebook:
`entries/DispatchFuse/20261003T235500Z-jw16-3model-decode-dispatch-census-fuse.md`
(pre-registered before any window); artifacts
`artifacts/DispatchFuse/{w1-census,w1d,proof,ab-wA..wD}/` with SHA256SUMS.
Serving stack before this land: 0.32.4.dev202610031046+b581d5c, F6 active.

## Census (decode, per token, MLX_OMARCHY_GPU_PROFILE on diag wheels)

| model | arch | dispatches/tok (d512-d64 slope) | headline |
|---|---|---|---|
| 2B (Qwen3.8) | qwen3_next hybrid | 229.2 | post-F6; SliceUpdate gone; GDN fused kernels firing |
| 4B (Qwen3-4B-Instruct) | qwen3 dense 36L | 406.3 | q/k norms UNFOLDED: 72 norm dispatches/tok with plain rope; top class RMSNorm gx=1 58.5/tok |
| 9B (Qwen3.5-9B) | qwen3_5 hybrid 32L | 849.3 | ~700/tok composed-fallback F32 soup; fused GDU not firing (0.15/tok vs 2B 11.7) |

Serving baselines (unprofiled): 2B d64 109.30 / d512 102.51; 4B d64 64.34 /
d512 56.46; 9B d64 28.65 / d512 28.02 tok/s. New greedy digest pins
(1-pass protocol): 4B d64 e2c919be / d512 fff6d03b; 9B d64 26d569c8 / d512
0315217f; 2B unchanged (cb3e8770 matches the Fuse6-era pin).

## What landed (this receipt's commits)

1. **qwen3 dense rope-norm fold, default ON** (`scripts/patch-mlx-lm-qwen3-rope-norm.py`,
   env gate `MLX_OMARCHY_ROPE_NORM_FUSE`, default 1, kill switch =0): folds the
   per-head q/k RMSNorm into `mx.fast.rope_rms_norm` at the qwen3.py attention
   site (the F3 fold, ported to the dense-file code shape). Removes 2 norm
   dispatches per attention layer per token (72/tok on the 4B) plus the same
   again on prefill legs.
   - Correctness: iso bitcheck `tools/rope_norm_bitcheck4_qwen3.py` 24/24
     bit-identical (fenced legs raise); production greedy digests bit-identical
     to serving pins on every A/B arm at every depth (e2c919be / 1bb7ff66 /
     c9480d81 / fff6d03b), including the d512 pin taken a boot earlier.
   - Performance (paired alternating ctl/on, 5 pairs per cell, window gates
     uptime>=360 s / load<0.5 / PSI some avg10=0.00, serving wheel as ctl,
     0928f97a5 wheel + patcher as cand):

     | depth | ctl median | on median | paired delta | disjoint |
     |---|---|---|---|---|
     | d64 | 64.22 | 67.29 | **+4.71%** | yes |
     | d128 | 63.59 | 66.69 | **+4.86%** | yes |
     | d256 | 62.19 | 65.05 | **+4.60%** | yes |
     | d512 | 56.47 | 58.85 | **+4.18%** | yes |

     Neutral arm (patch applied, gate off): 63.44 at d64 — inert. This clears
     the pre-registered landing bar (>= +1.5% at d64, disjoint, digests pinned,
     non-negative everywhere) by roughly 3x.
2. **GDN raw-decode route enablement for Hk<Hv models (9B candidate, gated)**:
   - `scripts/patch-mlx-lm-gdn-raw-repeat.py` (env
     `MLX_OMARCHY_GDN_RAW_REPEAT`, default 0): expands q/k to the value-head
     count at decode so `GatedDeltaUpdate::use_fallback` (Hk==Hv) passes.
   - Kernel (commits 27b6904ed, d547f112b): raw-gates decode accepts an F32
     A_log (the Qwen3.5-9B conversion stores it F32; safetensors header
     verified) behind shader flag bit 9, read as two little-endian u16 words —
     the same f32 word the eager chain's astype(f32) produces; when bit 9 is
     set the in-kernel softplus stays f32, matching the C++ fallback reference.
     bf16-A_log models (Qwen3.8) take byte-identical original paths — bit 9
     unset — so 2B numerics and pins are untouched by construction.
   - Route proven: probe streams show the fused GatedDeltaUpdate dispatch on
     9B shapes (was: ~700/tok composed-fallback soup). NOT yet landed as a
     default: the fused kernel is inherently numerics-changing vs the composed
     fallback (reduction-order difference; measured out max_abs 6.24 on
     synthetic stress shapes) — the candidate goes through the numerics gate
     (greedy >=95% identical near-ties only, perplexity within 0.1%, re-pin)
     in its A/B before any default flip. Inert for every currently served
     model: the python gate defaults 0 and the kernel path requires an F32
     A_log checkpoint.

## Census follow-ups (ranked, not landed)

- 4B residual-add into the following RMSNorm (58.5 gx=1 norms/tok at
  1143 us/tok profiled): needs an `add_rms_norm` primitive (two outputs: the
  bf16 sum and its norm) — the same shape Qwen3.5/3.8/3 all share.
- 4B grouped qkv GEMV (three separate q/k/v QmmVec dispatches per layer).
- Refuted (do not re-run): sampling-tail argmax fusion (Fuse3 class 7),
  RMSNormGated epilogue (Fuse5, geometry), norm-prologue fold (Fuse5, -28.7%).

## Provenance

- mlx-omarchy branch `agent/jw16-decode-dispatchfuse` rebased onto
  origin/main 4964dc9d4 (main gained qmm_coopmat changes between the census
  base and the land; the final rebased wheel is re-gated below before
  deploy). Serving mlx_lm 0.31.3 + patchers applied at deploy.
- Host: jw16, kernel 7.1.13-3-2-ARCH; boots recorded per window in the
  notebook entry (the lane rode out 6 reboots from the Jw16Dvfs lane and
  interleaved one window each with AneGate3 and BarrierSched).
- Every window: `bash /var/tmp/appbar/gpuwin.sh` single command, <= 15 min,
  bash -n first, uptime/load/PSI gates inside, llm-inference restored
  (health probe finish_reason=length) and verified after every window.
- mlx_provenance: serving wheel b581d5c version_match=true (pre-land);
  candidate wheels stamped per commit (`+dfuse.<sha>`, diag builds carry the
  `+diag.` marker); stamps asserted distinct per A/B.

## Addendum 2026-10-04T06:0xZ — land executed; 9B A/B measured; gate verdict

- LANDED + DEPLOYED: main 15547ef8f (rebased tip aae57ea0e + receipt; ls-remote
  verified); serving venv /var/tmp/v072-venv-fused wheel
  0.32.4.dev202610040439+dfuse.15547ef8f (wheel sha256
  3990d3a478302b38...0910d8; rollback
  /var/tmp/v072-venv-fused.pre-20261004T000642) + patch set (qwen3 rope-norm
  fold default ON; gdn raw-repeat default OFF). Deploy-verify on the serving
  venv, NO env: 4B d64 e2c919be / d512 fff6d03b, 2B d64 cb3e8770 (all exact
  pins); llm-inference restored, health ok.
- 2B 1-pass d512 digest note: the 1-pass digest moved 619360bf -> 8bd69d6d
  with the land; the 512-token output_ids are IDENTICAL position-for-position
  (0 differing tokens, diff receipt in the notebook artifacts), and all four
  2B production 5-pass pins hold exactly (c84b3e7a / 07c515e0 / c6aabbf0 /
  5c120987) on the deployed wheel. The 1-pass canon difference is a
  record-metadata artifact, not numerics; open question only for the digest
  tooling, not the serving stack.
- 9B GDU raw route A/B (ctl = pre-land serving stack
  v072-venv-fused.pre-20261004T000642, on = deployed wheel +
  MLX_OMARCHY_GDN_RAW_REPEAT=1; 5 alternating pairs per cell):

  | depth | ctl median | on median | paired delta | greedy identical |
  |---|---|---|---|---|
  | d64 | 28.58 | 38.54 | **+34.8%** | 100% |
  | d128 | 28.53 | 38.51 | **+35.0%** | 100% |
  | d256 | 28.58 | 38.45 | **+34.5%** | 100% |
  | d512 | 28.08 | 37.56 | **+33.8%** | **72.66%** (first divergence at token 316) |

  ppl probe (49 teacher-forced tokens): mean NLL 11.31236 (composed) vs
  11.30684 (fused) = -0.049% — within the 0.1% bar.
  NUMERICS GATE VERDICT: the greedy >=95% bar FAILS at d512 (72.66%), so per
  the pre-registered rule the GDN_RAW_REPEAT default stays OFF (opt-in env).
  The measured +33.8..+35.0% greedy-identical-through-d256 result stands in
  the record for an owner decision on accepting the d512 near-tie divergence.
