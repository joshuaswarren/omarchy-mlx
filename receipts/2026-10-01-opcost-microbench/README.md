# OpCost microbench — what makes a Kokoro dispatch 330 us while a chat dispatch is 45 us

## ADDENDUM 2026-10-01 (post-mortem, supersedes §13 RTF claim)

The `RTF 1.24 / 1.83x faster` numbers in **§13 04** came from a
prompt-injection-flavoured provenance failure and a numeric bug, NOT
from a correct Kokoro wall drop. Treat §13 04 as VOID and disregard
the related wall/RTF figures and the §12 implementation summary.

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