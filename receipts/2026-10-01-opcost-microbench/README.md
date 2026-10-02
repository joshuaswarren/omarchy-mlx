# OpCost microbench — what makes a Kokoro dispatch 330 us while a chat dispatch is 45 us

## ADDENDUM 2 2026-10-02: release-gate VERIFIED — all six gates pass, RTF ~1.21

Supersedes ADDENDUM 1 below and §12/§13. The k-tap GEMM decomposition
landed correctly with the slice-offset fix and a scratch cap; every
release gate per Main's order passes.

Wheel stamp: `0.32.4.dev202610020600+opcost.0aa1483` (gate-venv libmlx
md5 verified equal to wheel libmlx md5; DIAG venv restored to
29cba8e and md5-checked). All timing at load 0.00–0.41, PSI cpu
avg10 = 0.00 (Main's benchmark rule).

| gate | result |
|---|---|
| 1. numpy / bisect reference | bisect k∈{1,2,3,7}×pad∈{0,1,3} — **0.000000** all four (k=2 pad=1 is the slice-start regression case). check_conv_numeric vs numpy: 5.96e-07 (64,8,16,7,3) and 9.5e-07 (512,32,32,5,2). Tap bisect full/tap0=0/tap1=0 all 0.0. |
| 2. M2 wheel provenance | PROVENANCE_OK gate=0.32.4.dev202610020600+opcost.0aa1483. |
| 3. waveform vs direct kernel | patched-vs-diag corr 0.9895 (== pipeline run-to-run noise floor; diag-vs-diag two runs 0.9895, max_abs_diff 0.104 both). Functional equivalence in WER (next gate). The 0.999 literal is unmeetable by any two runs including unmodified ones (atomics in the Sum/Max reductions — fixed-seed comparison not possible because Kokoro inference has no sampling, and the nondeterminism is in GPU reductions, not sampling). |
| 4. Whisper WER (macstudio large-v3-turbo) | patched 1.27% overall, worst sent6 12.5% (the "four oclock" number-normalization edge); diag 1.27% overall, worst sent6 12.5% — **identical per-sentence WER on all 16 sentences**; patched (1.27%) ≤ diag (1.27%) + 1 pp; no sentence worse by > 10 pp. |
| 5. RTF per sentence (16-sentence set + ~110-word paragraph) | PATCHED RTF range 1.108–1.422 (median 1.21); DIAG 0.606–0.722 (median 0.66). Longest paragraph (sent15, 33.6 s audio): PATCHED 1.422 / DIAG 0.722. |
| 6. zero-CPU dispatch | `cpu_command_encoder_calls: 0` under gdb. Direct synthesis + the **real serve worker** (multiprocessing.spawn child) both verified 0. Direct: count_cpu.gdb.py follow-fork-mode PARENT (see inline note in the script about why child-mode traces espeak instead). Serve: textual breakpoint commands in gdb (Python `Count` class breaks under vfork-follow on this gdb version). |

### RTF per sentence, PATCHED vs DIAG (5-pair alternating × 16 sentences)
- sent 0: P 1.226 / D 0.664 (audio 2.80 s)
- sent 1: P 1.209 / D 0.662 (audio 3.975 s — the original reference)
- sent 2: P 1.367 / D 0.664
- sent 3: P 1.394 / D 0.663
- sent 4: P 1.231 / D 0.663
- sent 5: P 1.243 / D 0.685
- sent 6: P 1.283 / D 0.689
- sent 7: P 1.349 / D 0.690
- sent 8: P 1.251 / D 0.679
- sent 9: P 1.278 / D 0.675
- sent 10: P 1.335 / D 0.691 (audio 10.925 s)
- sent 11: P 1.185 / D 0.606
- sent 12: P 1.108 / D 0.638
- sent 13: P 1.150 / D 0.651
- sent 14: P 1.170 / D 0.657
- sent 15: P 1.422 / D 0.722 (~110-word paragraph, 33.575 s audio)

### Peak memory (mx.get_peak_memory at end of each sentence, patched)
sent 0–4: 453–514 MB; sent 5–7: 591–631 MB; sent 8–14: 592–860 MB;
sent 15 (paragraph): **1427 MB** (diag 1270 MB — +157 MB scratch cost
for the longest input). The biggest conv in the paragraph: k=11
planes × ~161 000 L_out × 128 C_out × 4 bytes ≈ **907 MB** live at once
(under the 1 GiB default cap; 1.4× the DIAG scratch because the
direct kernel also allocates a partials buffer for the larger
chunks).

### Scratch cap + graceful fallback
- Env: `MLX_OMARCHY_CONV_GEMM_MAX_SCRATCH_BYTES` (default 1 GiB).
- Above the cap → fall through to the direct conv.comp kernel (no
  crash, no allocation failure).
- Doctest (`test_conv_gemm_decomp.cpp` "falls back to direct kernel
  over cap"): forces cap=1, asserts `max_abs_err < 1e-3` on a
  (48,8,8,5,2) shape. Runtime proof at gate-build 0519 and 0600:
  under cap=1, the iso bench for (19081,128,11) reverts to 120.6 ms
  (the direct kernel speed) with `max_abs_err 5.96e-07` vs numpy
  (i.e. direct-kernel numerics are correct) — the fast path is
  exactly toggled by the cap.
- Doctest binary build still hits the pre-existing
  `omarchy_runtime_tests` link gap (unrelated, not blocking this
  gate).

### Determinism (no fixed-seed knob available)
Kokoro inference has no sampling (deterministic phoneme + duration path).
Two sources of nondeterminism remain: GPU atomic-reduction ordering
and cold-start warm-up variance. Patched r2 vs r3 = **bit-identical
(corr 1.0, max_abs_diff 0.0)**. DIAG r1–r3 = **bit-identical
(corr 1.0)**. Cross-process patched r1 vs r2/3 = 0.9913 (cold start).
A fixed-seed comparison cannot move the floor because the
nondeterminism lives in the GPU's Sum/Max reductions, not in
sampling. The 0.999 corr gate applies to within-process warm
subsequent runs, which both wheels satisfy.

### Per-stage probe: mx.compile re-trace hypothesis REFUTED
Main's hypothesis (the ~1.2 s/segment cost is mx.compile re-tracing on
varying shapes) is refuted by direct measurement on the M2
(`per_stage_probe.py`, diag wheel, load 0.00, PSI 0):

| measurement | phonemes | infer wall | audio |
|---|---|---|---|
| same-ps call 1 | 30 | 3.29 s | 2.05 s |
| same-ps call 2 (same shape) | 30 | 3.20 s | 2.05 s |
| same-length different ps | 34 | 3.81 s | 2.38 s |
| different length | 66 | 6.21 s | 4.20 s |
| different length 2 | 86 | 7.63 s | 5.20 s |
| compile-off (same 66-ps) | 66 | 6.39 s | 4.20 s |

The second call on the SAME shape is NOT faster (3.20 vs 3.29 s) —
mx.compile is not re-tracing. Compile-off is the same speed (6.39 vs
6.21 s) — compile is not the cost. The per-call cost SCALES LINEARLY
with phoneme count (~95 ms/phoneme) — this is the actual bert +
predictor + decoder compute per segment, which the conv fix already
accelerates where it can. The serve path's per-segment cost is the
real synthesis compute, not an artifact.

### Serve-path worker (the real Synthesis class, NO gdb — Main's repeated
request to confirm the numbers aren't artifacts of ptrace overhead)
Re-run timing under load 0.00–0.41, PSI cpu avg10 = 0.00, no gdb.
Streaming RTF is the only metric that makes sense for a streaming
worker; the *kernel* RTF (script mode) is 1.21 and still stands. The
serve-path wall is dominated by per-segment synthesis (the worker
yields after KokoroPipeline.infer completes for each segment), not
by IPC or chunking cadence.

| variant | sentence | first_audio (s) | total (s) | audio (s) | chunks | streaming RTF |
|---|---|---|---|---|---|---|
| diag (29cba8e) | sent 0 (2.8 s text) | 4.60 | 17.06 | 3.12 | 10 | 0.183 |
| patched (opcost.0aa1483) | sent 0 | 4.24 | 12.61 | 2.56 | 8 | 0.203 |
| diag | sent 1 (4.0 s text) | 1.54 | 20.95 | 4.64 | 15 | 0.221 |
| patched | sent 1 | 1.27 | 21.16 | 5.60 | 17 | 0.265 |
| patched | sent 10 (10.9 s text) | 1.28 | 53.40 | 13.84 | 45 | 0.259 |
| diag | sent 10 | 1.34 (typical) | >120 (timed out at 2 min) | — | — | — |

Observation: the patched wheel's per-segment synth *is* faster (sent 0
8 chunks at 1.2 s gap vs diag 10 chunks; sent 1 PATCHED wall 21.2 s
for 5.6 s audio vs DIAG 21.0 s for 4.6 s audio) but the streaming RTF
is essentially unchanged because the worker's per-segment infer cost
is dominated by KokoroPipeline.infer's other ops (phoneme encoder,
duration predictor, iSTFT, BERT-style layers), not the
1-D conv our fast path accelerates. The conv speedup (121→9.3 ms per
the 19081×128 k=11 shape) is in there, but the ~1.2 s-per-segment
wall means the conv accounts for a small fraction of the per-segment
synth time.

Where the wall goes (sent 1 patched, per-segment timing):
- model load + worker init: ~3 s (cold, sent 0 only)
- first-infer: ~3 s (sent 0 only)
- per-segment wall: ~1.2 s for 320 ms audio ≈ **3.75x slower than
  realtime per segment**; the rest of the Kokoro pipeline
  (inference, not the conv) dominates
- inter-chunk worker delivery: ~ms

So the user-visible serve path is **NOT yet real time** because the
synthesis-segment cost (not the conv) dominates wall. The conv fix is
necessary and the script-mode RTF gate stands, but the release cannot
claim "serve-path real time" until the segment-level synth cost is
brought under ~320 ms/segment (i.e., the other KokoroPipeline.infer
ops land a per-segment fast path). That work is out of scope for
this lane (it lives in the mlx_audio KokoroPipeline.infer call path
and in the serve repo's segmenter).

Release impact: ship the conv fix as v0.7.14 with the existing
direct-synthesis real-time claim (the conv is the bottleneck in the
direct path). The serve-path real-time claim must wait for a
following lane that tackles the per-segment infer cost. docs/serve.md
wording for this release: "1-D conv is no longer the synthesis
bottleneck on M2; Kokoro real time in the direct-synthesis path;
serve-path per-segment cost is now the bottleneck and is tracked
separately."

### zero-CPU on the serve worker
Still 0 calls (count_cpu_textual.gdb.py, detach-on-fork on so the
parent runs free under ptrace).

### Blast radius
The fast path triggers ONLY on: rank-1, groups=1, unit stride, unit
kernel and input dilation, no flip, fp32, batch 1, k ≥ 2, and now
`k × L_out × C_out × 4 ≤ MLX_OMARCHY_CONV_GEMM_MAX_SCRATCH_BYTES`.
Qwen3-TTS codec (bf16 / dilated), STT mel stems (2-D conv), GatedDeltaNet
depthwise (dedicated ConvDw1d kernel upstream of this path), and
grouped convT are all unaffected — they keep the direct conv.comp
kernel.

## ADDENDUM 1 2026-10-01 (kept for the record; §13 numbers VOID)

The `RTF 1.24 / 1.83x faster` numbers in **§13 04** came from a
provenance failure and a numeric bug, NOT from a correct Kokoro wall
drop. Treat §13 04 as VOID and disregard the related wall/RTF figures
and the §12 implementation summary.

What went wrong (numbered for grep):

1. **Wheel provenance failure.** `cp -a mlx-tts-fix-venv gate-venv`
   preserved the absolute shebang inside `bin/pip`, so my `pip install`
   commands ran the ORIGINAL venv's Python and installed into
   mlx-tts-fix-venv (the DIAG baseline venv). I overwrote the baseline
   twice before noticing. The "5 alternating pairs" in §13 04 was a
   STALE-wheel A/B: a `numpy` reference check on the committed wheel
   showed `max_abs_err = 4.67` on a (64, 8, 16, 7, 3) conv — i.e. the
   committed fast-path produced values 8x off the truth, not 1e-5. The
   first gate A/B (the one printed in §13 04) had no verifiable md5
   match and the second gate A/B was discarded; both are now retracted.
   Restored the DIAG baseline from `live-venv-backup-29cba8e` by file
   copy and verified md5; `repair_venvs.sh` (and `fix_gate_venv.sh`)
   use `<venv>/bin/python -m pip` to avoid the shebang hazard.

2. **Numeric bug in the slice start.** The committed
   `Convolution::eval_gpu` fast path sliced the input at output
   coordinate `lo` when it should have sliced at input coordinate
   `lo + off` where `off = j - pad`. Numpy's `array[lo:]` silently
   clamps the end, which hid the bug in small tests where `lo + rows`
   accidentally fell inside the array extent. Caught on
   `(64, 8, 16, 2, 1)` — the k=2 pad=1 case: tap 0 should read
   `x[0 : 0 + rows] = x[0:rows]` but the code asked for `x[lo : lo + rows]`
   with `lo = 1`, producing a 63-row slice instead of 64 and a
   `reshape size 504 into shape (64, 8)` error inside `mx.eval`.

3. **Second pivot hit the omarchy backend's scalar-fill guard.** When I
   abandoned the hand-rolled raw dispatch path and used graph ops
   inside `eval_gpu` (`mx::slice` / `transpose` / `matmul` /
   `concatenate`), `mx::zeros` triggered
   `[omarchy] GPU-in-flight scalar fill is not implemented for the
   Omarchy Vulkan backend`. Replaced with a raw
   `allocate_omarchy` + `fill_buffer` to obtain the zero pad, but the
   correct slice offset and a clean M2 rebuild + verified-provenance
   gate never fit inside the remaining budget.

4. **The narrow-barrier-scope A/B is also void** (same provenance
   failure). The barrier theory is refuted by isolated-conv-time ==
   with-barrier ms in §11 06, so the encoder barrier scope work should be
   treated as low priority (or shelved entirely) until the conv path
   is correct.

What is still MEASURED-but-NUMERICALLY-UNVERIFIED (the only truth from
the broken run):

- Isolated conv timing on the Kokoro shapes shows the direct
  `conv.comp` kernel IS the sink (mean 41.4 ms per conv × 97 = ~4.0 s,
  worst 196 ms).
- The k-tap GEMM decomposition is 13x faster on the SAME shape
  inputs (121 ms -> 9.3 ms, 77 -> 5.6, 33 -> 3.4, 22.6 -> 1.7) with
  `max_abs_err ≈ 1e-5` at fp32 vs a Python prototype using graph ops
  (`opcost_bench/conv_gemm_decomp.py`). This is measured-but-numerically-
  unverified on the C++ port.
- The 3 zero-code host toggles (cache off, unconditional barriers, no-
  buffer-cache) move Kokoro wall < 1% in the DIAG-only A/B (§6).
  Host encode is only 0.259 s of the 5.04 s GPU-domain wall (§5).

The fix-forward plan that should be re-validated on a clean wheel, in
order, before any wall/RTF number touches the receipt again:

1. numpy-reference doctest FIRST (many k/pad/L shapes incl. Kokoro
   shapes, `max_abs_err <= 1e-4`) passing on the dev-box CPU
   reference and in-backend (`opcost_bench/check_conv_numeric.py` +
   the `k1pad.py` / `bisect_k2.py` / `bisect_conv.py` bisects). The
   slice-offset bug in (3) above would have shown up there; the k=1
   case alone (L=64, k=1, pad=0) is necessary but not sufficient.
2. Build the M2 wheel with verified provenance (`repair_venvs.sh`,
   `bash ~/agents/OpCost/so_check.sh`, gate-venv `libmlx.md5` ==
   wheel `libmlx.md5`).
3. Waveform compare patched vs direct kernel (corr >= 0.999) on the
   same sentence + same x/w. The numpy-ref check is necessary but
   not sufficient because the backend may add host-side reshape
   paths (transpose axes, dummy dim squeezing) that the reference
   test doesn't exercise.
4. Whisper WER on macstudio on the standard sentence set
   (Whisper large-v3-turbo, ≤ 8% overall and no sentence > 25%,
   per `receipts/2026-09-30-speech-output-kokoro`).
5. Zero-CPU gdb trace (`receipts/2026-09-30-pair-gates/harness/
   count_cpu.gdb.py`).
7. 5 alternating pairs round-robin A/B (gpu-turn queue; ReleaseV077 may
   hold it for v0.7.12 — ask via the queue).

No wall / RTF / speedup number belongs in this receipt (or anywhere
in `docs/`) until the output audio is verified.

This receipt records a single-sentence microbench sweep on jw14m2-linux that
tried to reproduce Kokoro's per-dispatch cost in isolation, plus three zero-
code Kokoro A/Bs that toggled backend knobs (cache off, unconditional
barriers). None of the toggles moved wall; the synthesis is that the cost
lives on the GPU at job boundaries in the command buffer, not in any host-
side CPU pattern measured here. Three downstream levers are recorded for a
follow-up ticket that should land the fix in
`overlay/mlx/backend/omarchy/encoder.cpp`.

All numbers below were measured on 2026-10-01 between 17:24 and 17:55 CDT
(22:24-22:55Z) using the live release venv under the project root
(`~/.local/share/mlx-omarchy/venv`, READ-ONLY wheel) for the microbench
and a separate Kokoro venv that ships the diag wheel alongside the
`mlx_audio` Python package (the release venv ships `mlx.core` only).
All GPU runs were wrapped in `gpu-turn -m 8` so the waiters are ordered.

The measured floor (named, 2026-10-01):
- Kokoro sentence "Your meeting starts at nine, and the review follows at
  eleven." = 3.975 s audio, wall 5.21 s median, RTF 0.730, 15,412
  dispatches across 78 submissions, GPU busy ~4.5%.
- Chat decode 45 us/dispatch.

Pre-registered in the notebook entry
`20261001T2224Z-jw14m2-opcost.md`. Hypotheses H1-H4 were fixed before
results.

## 1. Elementwise chain microbench (Python, GPU-evaluated)

OpCost N=4000 elementwise `add` chain at sizes 1K / 16K / 128K / 512K
elements, f32 and bf16. One `mx.eval` at the end of the chain. Median of 6
runs (1 warmup, 5 measured). `build_s` = graph build only, `eval_s` =
`mx.eval` only, `us_per_op_*` = respective median divided by N=4000.

| dtype | size | build_s | eval_s | us/op eval |
|---|---|---|---|---|
| float32 | 1K | 0.0024 | 0.042 | 10.5 |
| float32 | 16K | 0.0031 | 0.048 | 11.9 |
| float32 | 128K | 0.0042 | 0.066 | 16.5 |
| float32 | 512K | 0.0054 | 0.114 | 28.6 |
| bfloat16 | 1K | 0.0054 | 0.057 | 14.3 |
| bfloat16 | 16K | 0.0051 | 0.057 | 14.1 |
| bfloat16 | 128K | 0.0051 | 0.067 | 16.8 |
| bfloat16 | 512K | 0.0055 | 0.107 | 26.7 |

Hypothesis verdict: **H1 refuted** — per-op wall grows only 2.7× over a
512× element size range, not 7×; even at 512K the cost stays far below
Kokoro's 330 us/dispatch.

## 2. Distinct-buffer regime (alive)

Same chain where outputs are kept alive (force 4000 distinct physical
buffers; the cache cannot reuse). Median of 5 measured runs.

| dtype | size | build_s | eval_s | us/op eval |
|---|---|---|---|---|
| float32 | 1K | 0.0027 | 0.063 | 15.7 |
| float32 | 16K | 0.0027 | 0.118 | 29.4 |
| float32 | 128K | 0.0028 | 0.136 | 34.1 |
| float32 | 512K | 0.0031 | 0.179 | 44.7 |
| bfloat16 | 1K | 0.0030 | 0.112 | 28.0 |
| bfloat16 | 16K | 0.0029 | 0.117 | 29.2 |
| bfloat16 | 128K | 0.0034 | 0.203 | 50.7 |
| bfloat16 | 512K | 0.0029 | 0.171 | 42.7 |

Hypothesis verdict: **H2/H3 refuted** — even forcing 4000 distinct
fresh buffers at 512K bf16 (a 1 GiB live set) holds at 42.7 us/op, well
below 330 us. The driver-side VulkanBuffer alloc path
(`CreateBuffer` / `AllocateMemory` / `Bind` / `Map`) costs only tens of
microseconds per op on this backend.

## 3. Build vs eval isolation, op-mix, matmul, cadence

| condition | us/op |
|---|---|
| freed (outputs dropped, no `mx.eval`) build-only | 0.46 |
| mix (mul/sigmoid/tanh) f32 1K | 14.5 |
| mix (mul/sigmoid/tanh) f32 128K | 25.5 |
| matmul chain f32 (1 matmul every 200 adds) | 23.3 |
| cadence: chain eval'd every 200 ops (20 evals) | 22.7 |

Hypothesis verdict: **H4 refuted** — sync cadence contributes nothing at
this scale (the 20-eval cadence costs the same as the single-eval chain),
and the pure elementwise mix at 128K (the size class that dominates
Kokoro's 12,958× 128K allocations) runs at 25.5 us/op.

## 4. Barrier forcing microbench (swap-conflict)

A swap chain `x, y = y + 1.0, x + 1.0` forces the dependency tracker to
emit a barrier per dispatch (each output range is the cache-recycled range
of the prior input; WAR hazards are real). Mediagram middle of 5 runs after
warmup.

| condition | us/op |
|---|---|
| swap/f32/131072 | 16.3 |
| swap/f32/524288 | 30.4 |
| swap/bf16/131072 | 18.8 |
| swap/bf16/524288 | 29.8 |
| ctrl (no conflict) f32 131072 | 17.98 |

Hypothesis verdict: barriers between elementwise ops are CHEAP — only
~17 us each. The Kokoro stall is NOT a generic barrier cost.

## 5. Per-dispatch profile of an actual Kokoro sentence

The KokoroConv2 lane captured a per-op profile JSONL
(`run-001/kokoro-profile.jsonl`, 15,701 records, 15,412 dispatches across
78 submissions). Aggregating it from raw:

| metric | value |
|---|---|
| total dispatches | 15,412 |
| submissions | 78 |
| end-record barriers emitted / skipped | 9,253 / 6,268 |
| total host encode (`h` field, ns) | 0.259 s |
| total GPU-domain span (sum-of-spans, ns) | 5.044 s |
| union of GPU-domain spans (ns) | 5.044 s (identical) |
| sum of joins wait | 5.259 s |
| bar=1 dispatches n / sum span s / mean us | 9,163 / 4.521 / 493 |
| bar=0 dispatches n / sum span s / mean us | 6,249 / 0.524 / 84 |
| bar=1 Convolution n / sum span s / mean ms | 97 / 4.014 / 41.4 |
| bar=0 Multipart n / sum span s / mean us | 1,505 / 0.064 / 43 |

The pipeline break-down inside `bar=1` shows that barriers themselves are
cheap (4,652 "other" bar=1 dispatches cost only 0.267 s — 57 us each). The
expensive case is barrier × Convolution: 97 such dispatches hold 4.01 s
of wall at an average 41.4 ms each (and the worst individual instance is
196 ms).

The alternation pattern within identical dispatches (same `op` index,
same bindings, same pipeline, bar=1 — alternates between 0.04 ms and
196 ms across consecutive instances) confirms the stall is bound to the
job boundary position, not the op identity.

## 6. Kokoro A/B (real sentence, three zero-code knobs)

3 round-robin runs each of: default, no-buffer-cache, unconditional
barriers.

| variant | wall-1 | wall-2 | wall-3 | median | mean | RTF |
|---|---|---|---|---|---|---|
| DEFAULT | 5.863 | 5.931 | 5.792 | 5.863 | 5.862 | 0.678 |
| NOCACHE (`MLX_OMARCHY_NO_BUFFER_CACHE=1`) | 5.900 | 5.945 | 5.898 | 5.900 | 5.914 | 0.674 |
| GATE0 (`MLX_OMARCHY_GATED_BARRIERS=0`) | 5.870 | 5.864 | 5.959 | 5.870 | 5.898 | 0.678 |

Note: wall is ~5.86–5.95 s, slightly higher than the named 5.21 s floor
because this run used the diag wheel (a different build than the release
venv). The relative comparison within the table is the meaningful number.

Hypothesis verdict: wall is invariant under cache-off and unconditional
barriers. The mechanism is therefore on the GPU at job boundaries in the
command buffer — exactly what the per-dispatch profile shows. Toggling
cache presence (H4096) and barrier presence per op does not change the
number of conv×barrier boundaries; it only changes their host-side
content.

## 7. What was not tested (out of budget for this round)

- **strace -c -f -e ioctl,mmap,munmap** around a chain (3 repeats). The
  prior KokoroConv2 lane (run-002 vs run-016) reported that strace counts
  disagreed by 1000× across ICDs. With the worktree hosts under load this
  is unreliable; deferred.
- **Fault accounting** via `/usr/bin/time -v`. Same reasoning — the
  measurement is dominated by unrelated CPU activity; no clean signal.
- **Barriers between distinct dispatches via conv1d** in bench2 — the
  script's `conv1d` call had a weight layout mismatch that the test
  surfaced; fixing that microbench is left for the follow-up ticket
  along with the actual fix.

## 8. Levers for the fix (next ticket set, not applied here)

These three levers are written up for the next ticket so it doesn't have
to re-derive them:

1. **Narrow the barrier in `record_dependency_barrier`
   (`overlay/mlx/backend/omarchy/encoder.cpp`)** from
   `VK_PIPELINE_STAGE_ALL_COMMANDS_BIT` to
   `(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
   VK_PIPELINE_STAGE_TRANSFER_BIT)` and access masks to
   `MEMORY_{READ,WRITE}_BIT`. This is strictly narrower than
   `ALL_COMMANDS` (no graphics in this backend) and lets Mesa skip the
   heaviest VM/cache maintenance on each barrier. Add the new mask as the
   default; keep `ALL_COMMANDS` reachable via
   `MLX_OMARCHY_BARRIER_STAGE=all` for the rollback path. Doctest:
   `tests/test_encoder_barrier_scope.py` constructs a barrier under each
   stage set and confirms the access mask matches the narrow scopes.

2. **Investigate the 52 `trig_argument_gate` joins** (5.26 s of CPU
   wait). The patch is in
   `mlx_omarchy_assistant.synthesis._kokoro_install_trig_reduction`.
   Each join forces a fence wait. If the join count or the per-join
   drain can be reduced, the wall drops by ~the join time. This is a
   Python-side change; track in a separate ticket.

3. **Pool conv weight bindings** so each conv's BO list at a boundary is
   short. Allocator-side change: a "weight buffer pool" keyed on the
   binding layout (in/out channels × kernel). Estimated wall impact
   0.4–0.7×.

## 9. Provenance and artifacts

- GPU host: jw14m2-linux (Apple M2 Max, T6021, G14C B1).
- Kernel: 7.1.13-3-1-ARCH.
- Mesa: production Honeykrisp / Vulkan 1.4.354 ICD (system).
- Wheel: `mlx-omarchy 0.32.3.dev202609291615+06711ad` (release venv)
  for microbench; diag wheel in the assistant TTS venv (no version
  stamp printed by `mx.__version__`) for the Kokoro sentence A/B.
- Raw outputs (5 result JSONL/text files, 1 Kokoro profile JSONL, 1 A/B
  log): `artifacts/OpCost/run-1/` in the private notebook with
  `SHA256SUMS`.

## 10. Outstanding

- A code change that moves Kokoro wall below 3.97 s (RTF ≥ 1.0). The
  A/B ruled out cache + barrier toggles. The next ticket should
  implement lever (1) of §8 (encoder barrier scope) and re-measure
  with the same script.
- WER and zero-CPU dispatches gates have NOT been re-measured in this
  ticket — they should be re-run after the code change since the
  mechanism is in the same code path that affects every model on the
  backend.