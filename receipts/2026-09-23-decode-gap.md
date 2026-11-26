# 2026-09-23 — decode one-token profile + submit-cut fix (jwm1; jw16 pending recovery)

Lane: DecodeGap. Base: `agent/gdn-coopmat` a12b1aa15; branch `agent/decode-gap`
tip f99695093. jw16 went down mid-lane (m1n1 boot incident, physical recovery
pending — Main re-scoped to jwm1 first); all measurement on jwm1 (T8103).
Wheels built OFF-DEVICE in the macstudio ALARM chroot (docker container
dg-build3, image dg-alarm-py314:sep23 — no compiles on jwm1 during other
lanes' arms). Never installed anything over an installed path without a win.

## 1. One-token decode profile (diag wheel, jwm1)

Diag wheel `0.32.3.dev202609230808+diag.a12b1aa1`
(sha256 c77e26bd…) = a12b1aa15 + MLX_OMARCHY_GPU_PROFILING, same tree as
jw16's installed aae4dfc9 release. Driver
`.work/dg/profile_decode_driver.py` (mlx_lm 0.31.3 generate_step loop =
exact serving shape, one host sync per token), 32 greedy tokens, France
prompt, under flock /tmp/m1-gpu.lock. Artifacts: jwm1:/var/tmp/dg/out/
{prof.ndjson, windows.jsonl}, local .work/dg/out/{…, analysis.json}.
NOTE: the profiling build inserts 2 timestamps + a full barrier per dispatch
(62,996 barriers over the run), inflating wall ~2.6x vs release; counts are
exact and kernel times are relative shares.

Steady per-token (median over tokens 5..32):

| metric | value |
| --- | ---: |
| dispatches | **585** |
| submits | **3** |
| joins (encoder join path) | 0 (per-token read does not ride encoder synchronize; 1 join for the whole process) |
| wall (profiled) | 51.4 ms (release reference: 28.2 ms = 35.4 tok/s) |
| summed kernel GPU time (profiled) | 40.2 ms (**78.3% busy**) |

Top kernels by GPU time (steady totals, 28 tokens):

| kernel | n/tok | GPU ms | share |
| --- | ---: | ---: | ---: |
| QmmVecQ4MultiSubgroupBF16 n=1536 | 24 | 215.2 | 19.1% |
| QmmVecQ4MultiSubgroupBF16 n=256 | 48 | 169.3 | 15.0% |
| QmmVecQ4WordSubgroupBF16 n=248320 (lm_head) | 1 | 149.7 | 13.3% |
| QmmVecQ4WordSubgroupBF16 n=6144 | 18 | 78.7 | 7.0% |
| FastRmsNormBF16 n=2048 | 109 | 78.6 | 7.0% |
| GatedDeltaDecodeBF16 n=128 | 18 | 67.1 | 6.0% |
| casts (BF16F32 1176 + F32BF16 672 disp) | 66 | 47.2 | 4.2% |
| ElementwiseBF16 n=2048 | 36 | 30.1 | 2.7% |

**Attribution: decode on jwm1 is GPU-bound.** qmm GEMV family = 59.8% of
kernel time (in-model ~48.5 GB/s = ~82% of the M1's ~59 GB/s pattern roof —
occupancy lane 2026-09-22 verdict holds; closed). Next costs: the
norm/cast/elementwise soup (~15%, ~4 ms/tok release) and GDN decode (6%).
The 3 submits/token = kBatchNodeBudget 256 splitting the 585-node graph
(2 mid-graph commits at 256/512 nodes + finalize).

## 2. Fix: one batch per decode token (commit f99695093)

kBatchNodeBudget 256 → 4096 (encoder.h). Decode graphs now ride ONE submit;
the byte budget (limit/16) remains the real cap for large-tensor graphs, so
prefill flush behavior is unchanged (2026-09-08 8.11 GB incident stays
covered by the byte budget, not this constant).

Candidate wheel `0.32.3.dev202609230822+f996950` (sha256 55274e3b…),
control release wheel `0.32.3.dev202609230811+a12b1aa1` (sha256 76431039…),
both from the same chroot pipeline.

Compact gate (jwm1, 10 prompts x 1 pass x 32 tok, greedy, under lock):

| arm | decode tok/s | ordered digest |
| --- | ---: | --- |
| control (3 submits/tok) | 35.44 | 486872c4 |
| candidate (1 submit/tok) | 35.72 | **486872c4 (identical)** |

Digest equality = token stream bit-identical (scheduling-only change, as
required). Decode delta +0.8% is inside this chassis's run-to-run noise
(installed wheel measured 35.25 / 32.58 on consecutive reps). **Verdict:
the fix is correct and free, but NOT a measurable decode win on jwm1
because decode is GPU-bound** — exactly why the profile was measured before
claiming a win. NOT INSTALLED (install bar = win only; jwm1's installed
path remains b4757ac, untouched, rollback trivially = current state).

## 3. Installed-path baseline (jwm1, full contract protocol)

Installed wheel `0.32.3.dev202609230623+b4757ac` in /var/tmp/v072-venv-fused,
10 prompts x 3 passes, warmup 2, prefill512, 2 reps:

| rep | decode | prefill512 | ttft | e2e med | digest |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 35.25 | 139.56 | 51.45 | 1.135 s | bc519c03 |
| 2 | 32.58 | 138.61 | 49.80 | 1.205 s | bc519c03 |

Deterministic token stream (identical digest both reps); rate noise ~8%
between reps on this chassis. vs macOS jwm1 denominator (47.05 decode /
344 prefill512, receipt agent/m1-mac-denominator @1df33f6): Linux is
0.75x decode, 0.41x prefill512 on the installed wheel.

## 4. What remains (handoff)

1. **Fusion lane** (the real remaining lever per this profile): fuse the
   cast/norm/elementwise soup into neighbors — FastRmsNormBF16 x109/tok,
   CastBF16F32 x42/tok + CastF32BF16 x24/tok (n=2048, op 589835/720905),
   ElementwiseBF16 x36/tok. ~4 ms/tok release-side on jwm1; bigger on jw16.
   Shader work in overlay/mlx/backend/omarchy (primitives.cpp + .comp).
2. **jw16 validation** when the box is back: re-run the profile + the
   compact gate + full contract vs installed aae4dfc9 (digest pin dbf70497);
   jw16 has more GPU idle headroom so the submit-cut may measure there.
3. Full contract (3 passes x 2 reps + logits gate) for the f996950 line if
   the fusion lane wants it as a base; wheels + venvs are staged on jwm1
   (/var/tmp/dg/{wheels,venv-ctl2,venv-cand,scripts,logits_gate.py}).
4. Conv-ring default-on decision (patch in tree, −18/tok on the old wheel,
   needs identity + count re-run on this lineage) — still open, cheap to
   test in one window.

## 5. Receipts

- Profile NDJSON + windows + analysis: jwm1:/var/tmp/dg/out/,
  local ~/src/mlx-omarchy/.work/dg/out/ (analysis.json + this file's table).
- Wheels: jwm1:/var/tmp/dg/wheels/ (diag c77e26bd…, ctl 76431039…,
  cand 55274e3b…), built in dg-build3 (image dg-alarm-py314:sep23,
  committed with FetchContent caches).
- Branch: agent/decode-gap (local, f99695093 + lane tooling commit;
  push blocked by repo privacy hook as usual — transport by bundle).
- Baseline contract JSONs: jwm1:/var/tmp/dg/out/contract-installed-*.json,
  compact A/B: compact-{ctl,cand}.json.
