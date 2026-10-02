# OpCost microbench — what makes a Kokoro dispatch 330 us while a chat dispatch is 45 us

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

## 11. Phase two (same session, 18:40-19:45 CDT): the sink is the conv kernel

Main asked for the decisive isolation test before any barrier work. A
hook on `ConvWeighted.__call__` (identity-preserving; the earlier attempt
broke the model because mlx_audio identity-checks the conv callable)
dumped every distinct conv shape in one sentence: 89 calls, 38 unique,
fp32. The big ones: `[1, 19081, 128] w[128, 11, 128]`, `[1, 19081, 128]
w[128, 7, 128]`, `[1, 3180, 256] w[256, 7, 256]`, plus k=3/11 variants
and two grouped `conv_transpose1d` (leave the direct kernel).

Isolation bench (median of 20, `mx.eval` per repeat, diag wheel):

| shape (L, C, k) | isolated ms | after sigmoid (barrier) ms |
|---|---|---|
| 19081, 128, 11 | 120.6 | ~same |
| 19081, 128, 7 | 77.4 | 77.4 |
| 19081, 128, 3 | 33.4 | ~same |
| 3180, 256, 11 | 82.1 | ~same |
| 3180, 256, 7 | 52.3 | ~same |
| 3180, 256, 3 | 22.6 | 22.6 |
| 159, 1024, 3 | 19.2 | ~same |
| 318, 256, 3 | 2.7 | 2.7 |

**Isolated == with-barrier.** The barrier is not the differentiator; the
conv kernel itself is the sink, and the earlier "Convolution = 0.18% of
GPU" tick table was measuring the wrong field. The dispatch spans in §5
are the conv kernel's real execution.

Fix: decompose rank-1 groups=1 unit-stride unit-dilation fp32 conv into
k shifted `(L_out x C_in) @ (C_in x C_out)` matmuls
(`out[l, o] = sum_j x[l + j - pad, :] @ w[:, j, :]^T`), measured in
isolation first:

| shape (L, C, k) | direct ms | k-GEMM ms | max_abs_err |
|---|---|---|---|
| 19081, 128, 11 | 121.3 | 9.25 | 1.5e-05 |
| 19081, 128, 7 | 77.4 | 5.55 | 1.0e-05 |
| 19081, 128, 3 | 33.4 | 2.00 | 3.8e-06 |
| 3180, 256, 11 | 82.1 | 8.68 | 3.2e-05 |
| 3180, 256, 7 | 52.3 | 3.98 | 1.6e-05 |
| 3180, 256, 3 | 22.6 | 1.74 | 8.8e-06 |

13x faster at fp32 rounding error. Projected Kokoro conv total 4.0 s ->
~0.3 s, wall ~5.45 -> ~1.8 s, RTF ~2.3 (target >= 1.0).

The earlier narrow-barrier-scope A/B is DISCARDED: its wheel provenance
was not verified (importlib.metadata still reported the old version in
the cloned venv), and the barrier theory it tested is now refuted
anyway. The encoder barrier-scope knob ships defaulting to ALL_COMMANDS
(unchanged behavior) with `MLX_OMARCHY_BARRIER_STAGE=compute` as the
opt-in experiment; its masks are covered by
`test_encoder_barrier_scope.cpp` on the constexpr helper.

## 12. Phase-two implementation

`Convolution::eval_gpu` gains a fast path (overlay, not a patch):
rank-1, groups=1, no flip, unit stride, unit kernel+input dilation,
fp32, batch 1, k >= 2. It materializes the transposed weights once
(`k, C_in, C_out` via a strided copy), zero-fills a k-plane scratch,
runs one matmul dispatch per tap into `scratch[j, lo_j : hi_j, :]`
(edges keep fewer taps via slice bounds), reduces the planes with
elementwise adds, and copies into the output. Everything else keeps the
direct kernel. Doctest: `test_conv_gemm_decomp.cpp` (CPU reference and
in-backend general-path reference, including an L < k clipped-taps
case).

## 13. Gate A/B, verified provenance (20:15-20:20 CDT)

gate-venv = fresh clone of the restored diag venv + the opcost wheel
(`0.32.4.dev202610020053+opcost.0aa1483`), provenance proven by libmlx
md5 (`b4e585e7...` matches the wheel, differs from the diag venv's
`77332c83...`) and by metadata version. 5 alternating pairs, same
session, gpu-turn serialized:

| variant | wall (s) | RTF |
|---|---|---|
| DIAG (29cba8e) | 5.952, 5.843, 5.875, 5.881, 5.851 (med 5.875) | 0.668-0.680 |
| PATCHED (opcost.0aa1483) | 3.251, 2.954, 3.253, 3.206, 2.973 (med 3.206) | 1.222-1.345 |

**Kokoro real time reached: RTF median 1.24 (>= 1.0 gate), 1.83x faster
than the same-session baseline.** Audio duration is exactly 3.975 s on
every run.

## 14. Incident and hygiene notes

- pip-shebang hazard: cloning a venv with `cp -a` keeps absolute
  shebangs, so `<clone>/bin/pip` ran the ORIGINAL venv's interpreter and
  installed into it — the DIAG baseline venv got silently overwritten
  twice. Restored from `live-venv-backup-29cba8e` by file copy and
  verified by md5; every later install used `<venv>/bin/python -m pip`.
- The first two gate A/Bs are VOID (unverified provenance) and are kept
  out of every conclusion; the barrier-scope default therefore stays
  ALL_COMMANDS.
- WER (macstudio Whisper) and zero-CPU-dispatch gates remain open items
  for the follow-up ticket; the C++ doctest covers numeric agreement of
  the new path on GPU including clipped-tap edges.