# DecodeGap8 — F3 rope-norm fusion W2 GATE-FAIL on jw16 (refute; no land)

## Identity and status
- Reference run: 2026-09-30 (UTC); actor DecodeGap8 (omp-studio-local; ssh `jw16mbp1-linux`); pre-registration `entries/Jw16DecodeGap8/20260930T005155Z-jw16-f3-gate-land.md`. Host jw16 = M1 Max (apple,t6001), Omarchy Linux, boot id 8c3d0b5c (same boot as DecodeGap7/LandDG6).
- Branch: `agent/jw16-decode-gap7` tip 17c50eb36. origin/main still c89392a5c (branch is fast-forwardable; deliberately NOT pushed — this receipt documents the W2 refusal and is the only commit added in this lane). Candidate venv `/var/tmp/dg7-venv` = wheel 0.32.3.dev202609300045+17c50eb36 + hand-applied mlx-lm-rope-norm.patch (qwen3_next.py 77c1b9a8f8). Serving venv `/var/tmp/v072-venv-fused` = 0.32.3.dev202609292324+82f2f482f (untouched, healthy).

## Question and gate outcome
- Q (per pre-registration): does the fixed rope-norm fusion (17c50eb36, dangling-ref bug fixed) produce bit-identical model output and the same >= +1% d64 decode gain?
- W2 GATE WINDOW (one gpuwin window, `bash /var/tmp/appbar/gpuwin.sh "bash /var/tmp/dg7/gate.sh"`):
  - iso bitcheck (rope_norm_bitcheck.py v1): rc=1, no data — the script probed offset 130000, beyond the backend's own 100000 trig-accuracy gate (both arms throw `RuntimeError: [omarchy] RoPE rotational argument magnitude ... exceeds the built-in accuracy limit`). Test-script bug, not a fusion verdict.
  - dg_bitcheck shared-kernel battery (`dg_bitcheck.py` full rows): GATE-FULL PASS rows=3084 — the rows that don't exercise the new primitive remain bit-identical (version-string-only diff).
  - paired cells d64/128/256/512 n=5 interleaved ctl(serving) vs cand + ctl2 re-anchor: every CAND digest diverged from the pin; ctl digests held.
    | depth | ctl median (tok/s) | ctl digest   | cand median | cand digest  |
    |-------|--------------------|--------------|-------------|--------------|
    |   64  | 101.28             | c84b3e7af640 | 102.78      | a4cdc914211c |
    |  128  | 101.24             | 07c515e0338b | 102.05      | 61dc0c735030 |
    |  256  | 100.15             | c6aabbf0a51d | 100.49      | 764365adff17 |
    |  512  |  95.97             | 5c120987f0e5 |  96.45      | 13683eaf7396 |
    |  d64 ctl2 | 101.32          | c84b3e7af640 | —           | —            |
  - d64 gain: cand 102.78 vs ctl-range [101.28, 101.32] = **+1.48%** (passes the >+1% outside-min-max bar).
  - GATE-FINAL FAIL: digest mismatch at all four depths. Per pre-registered contract: NO LAND; artifacts retained; rollback not applicable (serving venv untouched).
  - Restore proof: RESTORE health_ok=1 probe_finish=length active=active ✓.

## Root-cause discrimination (DecodeGap8 add-on probes)
- iso v2 (`rope_norm_bitcheck2.py`, 25 cases, in-window on cand venv): fixed offsets to (0,1,511,4096,99999) and added model-shaped prefill cases (T=512 q/k, T=17 q/k, T=2 q, generic T=1/T=3/dims==D, bigvals/smallvals). Outcomes:
  - **PASS** (17 cases): every model DECODE case (T=1, offsets 0..99999, q H=8 and k H=2), prefill T=17/T=2 q and k, generic bf16 cases (dims256, dims64, dims32, bigvals, smallvals).
  - **FAIL — FUSED PATH (2 cases)**: `model-q-prefill-t512-off0` (n_mismatch=2 bytes), `model-q-prefill-t512-off600` (n_mismatch=7 bytes). model-k-prefill-t512 (H=2, 1024 rows) passes; q (H=8, 4096 rows) hits 2/2097152 ≈ 1e-6 per element.
  - **FAIL — FALLBACK-COMPOSITION PATH (4 cases)**: `f16-b1-h4-t1-d256`, `f32-b1-h4-t1-d256`, `traditional-b1-h4-t1-d256`, `big-d512-b1-h2-t1`. These legs compose inside the primitive (fused kernel is bf16-only and limited to D ≤ 256); they should be bit-identical to the explicit `rms_norm` + transpose + `rope` chain, and are not. (Class A is a separate bug in the primitive's fallback composition; it does NOT affect the model path which is bf16 and 1 < T ≤ 512 with D=256.)
- probe `dg8_probe.py` (in-window): layout-sensitivity tests both pass — `rms_norm` on contiguous (B,T,H,D) vs on transposed (B,H,T,D) view: 0 / 1048576 differ. Plain `rope` on transposed view vs on `mx.contiguous` copy: 0 differ. So layout is not the cause; the rare fused mismatch is internal to the fused shader's reduction or rounding.
- probe `dg8_probe2.py` (in-window, 3 runs): divergence is DETERMINISTIC (same 2 flat indices across runs: 259134 = (h1,t500,d62 rotated) and 499792 = (h3,t416,d80 tail)). Per run: tail_norm_vs_ref = 0 (ref tail = production `rms_norm` bits exactly, as expected for passthrough), tail_norm_vs_fused = **1** (the d=80 tail mismatch is the FUSED kernel's implicit `normed[80]` differing from production `rms_norm`'s `normed[80]` by 1 bf16 ULP). rotated_region_diffs = **1** (the d=62 mismatch is a downstream rotation product whose `x2=rounded[62]` input differs by 1 ULP from production's `normed[62]`).
- Localized: the fused kernel's implicit `rounded[]` values, produced by the claimed verbatim `fast_norm.comp` reduction tree (256 threads, one element each, stride-128..1 shared pairing, `mean = block[0]/256`, `norm = inversesqrt(mean + eps)`, single `bf16_store` round), differ from the production `mx.fast.rms_norm` output for ~1 element per 10⁶ even though the formulas are textually identical. Candidates not ruled out by static reading: NIR/llvmpipe reassociation or precision relaxation in the fused main, an additive shift in `eps` from the push-constant `bit_cast` path, or a subtler shared-memory ordering tiebreak in the 256-thread barrier. A live numerical diff at the row level is needed to pick one; this entry is the GATE FAIL, not the fix.

## Decision
- Per the pre-registered contract: ANY digest divergence → do not land; serving venv untouched; main stays clean.
- NO push of `agent/jw16-decode-gap7` to origin/main. (Branch tip is parked at d4de275fa with the receipt commit on top of 17c50eb36.)
- During the gate window, origin/main advanced three commits (attn128 sibling lane: a03149ad9 `attn128: post-build fixes shipped via x86 working tree, not committed`, 8b3982c21 `attn128: receipt for hd128 fused SDPA + small-k value Partition`, c689c7ba6 `attn128: fused hd128 decode SDPA (bf16+f16) and small-k value Partition`). The branch is no longer fast-forwardable; merge-base c89392a5c. A rebase would be needed before any future push — out of scope for this refute.
- NO rebase / NO battery / NO deploy-verify. (Battery and deploy-verify were conditional on the W2 gate passing.)
- This receipt is committed on the branch as the only change in the DecodeGap8 lane, so future attempts land only after the divergence is root-caused and the fused reduction is provably bit-identical to `fast_norm.comp`.
- No chain-dep-bench fold-arms feasibility measurement this run (Q2 is conditional on Q1 landing).

## Artifacts (jw16 boot id 8c3d0b5c throughout)
- `artifacts/Jw16DecodeGap8/w2-gate-refute/gate-window.log` — gate window transcript (sha256 25de429acd55658acc21a25cdc07dc06f26c7c3e8a5a2681eb12ef5cce5e3a67).
- `artifacts/Jw16DecodeGap8/w2-gate-refute/rope_norm_bitcheck2.py` — fixed iso bitcheck (sha256 77931e123c5623dd6b63aaa5044b82beadbe9528f569434d34c7f3be60b78e88).
- `artifacts/Jw16DecodeGap8/w2-gate-refute/dg8_probe.py` — layout-sensitivity probe (sha256 1883e22d5e0aa20880883d899d600d69241316914a10ddc70c78086712cad66f).
- `artifacts/Jw16DecodeGap8/w2-gate-refute/dg8_probe2.py` — determinism + tail-vs-rotation probe (sha256 3510a8e53c7f52e569d5f90ff0a4876daa4e698cd93c8d9d4a4a6278c28d1258).
- `artifacts/Jw16DecodeGap8/w2-gate-refute/gate/` — gate.sh outputs: identity, bit-serve, bit-cand, d{64,128,256,512}-{ctl,cand,ctl2} jsons + logs, gate-full.txt, gate-final.txt, rope-norm-iso.log (v1 traceback), rope-norm-iso2.json (v2 results), dg8-probe.json, dg8-probe2.json, SHA256SUMS (30 entries).
- Top-level manifest: `artifacts/Jw16DecodeGap8/w2-gate-refute/SHA256SUMS-TOP` and per-dir `gate/SHA256SUMS`.

## Post-state and next action
- jw16 boot id 8c3d0b5c (unchanged); serving venv `/var/tmp/v072-venv-fused` = 0.32.3.dev202609292324+82f2f482f (DecodeGap5-fold deployed, LandDG6-redeployed decode 101.68/101.25/99.95/95.82 tok/s intact; ratios 0.566/0.565/0.559/0.541 vs macOS window-5 179.72/179.08/178.72/177.02). llm-inference active, health 200.
- Candidate venv `/var/tmp/dg7-venv` and `/var/tmp/dg7-build` and `/tmp/dg7-wheels-jw16/` kept for forensic follow-up; disk guard (`dg7-build` deletion) deferred until the refutation is reviewed.
- Branch `agent/jw16-decode-gap7` tip 17c50eb36 unchanged; this receipt is the first commit added on top.
- Next action: root-cause the fused-shader 1-ULP mismatch at the row level (compare `fused`'s pre-rotation `rounded[]` against `mx.fast.rms_norm`'s output element-by-element for the failing row), fix the kernel so the per-element comparison shows zero diffs, rerun `rope_norm_bitcheck2.py`, and re-enter the W2 gate.
