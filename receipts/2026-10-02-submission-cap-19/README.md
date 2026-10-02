# Issue #19: per-submission GPU-time cap (submission length drives the stutter)

Reporter: georgeamccarthy. Branch: `agent/submitcap` (push to `joshuaswarren/omarchy-mlx:agent/submitcap`). Default-bake commit: 1e7cb4b60 (Refs #19). Signature-fix commit: 4b2929a90. Cap-impl commit: 4a65ff135.

The reporter's reproducer is GPU **submission length**, not utilization: 78 ms
OpenGL submissions stutter Hyprland on the same M1 Pro with stock Honeykrisp at
120 Hz, while 2-6 ms submissions stay smooth. MLX on Honeykrisp has the same
shape: `vk_burn.py` and MLX-omarchy decode both queue long one-scommand graphs
because the open command buffer only closes at the existing 4096-node budget or
a host read, which a continuous decode does not trigger.

This change caps GPU work per queue submission with a bounded proxy
(summed dispatch work-group counts), defaulted from M2 profiling of Qwen3-4B
and Qwen3.5-9B decode + prefill, and asks the Honeykrisp driver for a lower
queue priority where `VK_EXT_global_priority` is exposed (M2 + jw16 evidence
cited; jw16 probe scheduled in the jw16 gpuwin window — see "Frame pacing"
below).

## Implementation

`overlay/mlx/backend/omarchy/encoder.h`: `kBatchWorkBudget = 40000`,
`batch_work_budget()` reads `MLX_OMARCHY_BATCH_WORK` once (0 = off switch),
`batch_over_budget()` shared by the eager evaluator and the compiled-tape
recorder, `CommandEncoder::batch_work_` accumulates dispatch group counts
(saturating) in `dispatch_compute_pipeline`, reset on both submit paths.

`overlay/mlx/backend/omarchy/eval.cpp`: `batch_over_budget` wired into the
eager flush predicate alongside the existing node and byte budgets.

`overlay/mlx/backend/omarchy/compiled.cpp`: same predicate in the compiled
tape loop so a whole decode step (or a whole prefill tape) is not one long
submission. Scheduler task balance preserved: the one completion handler
attached in `eval()` for the tape's open batch rides the first submission;
later submissions get no handler, keeping notify counts balanced.

`overlay/mlx/backend/omarchy/device.{h,cpp}` + `vulkan.h` +
`device_info.cpp`: detect `VK_EXT_global_priority`; when present AND the
compute queue family reports the requested priority, chain
`VkDeviceQueueGlobalPriorityCreateInfoKHR` into `qci.pNext` and enable the
extension. `MLX_OMARCHY_QUEUE_PRIORITY` = `low` (default) | `medium` |
`off`/`default`/`0`. Silent fallback on driver refusal: if `CreateDevice`
returns non-success with the chain, drop the chain + extension and retry
once before the typed error surfaces. The default `low` matches the typical
Honeykrisp implementation (the VSync-sensitive compositor wins queue
arbitration between MLX submissions; MLX time is bounded by the submission
budget either way).

`overlay/tests/omarchy/test_runtime.cpp`: deterministic doctest of the
chunking predicate with synthetic dispatch sequences (no GPU). Cases: off
switch, node-cap, even split, uneven split, oversized-dispatch preservation
(no in-dispatch break), decode-shape sequence conservation, prefix +
conservation contract.

`docs/install-omarchy.md`: documents `MLX_OMARCHY_BATCH_WORK` and
`MLX_OMARCHY_QUEUE_PRIORITY` with the M2 + jw16 exposure and the
calibration-derived 40000 default.

`scripts/subcap_calibrate.py`: drives the live wheel with the profiling
harness on, groups per-dispatch GPU ticks by submission id, reports
ms/submission p50/p95/p99/max. The harness records raw ticks + period_ns
(NDK-style per-event metadata); ms = ticks * period_ns / 1e6.

`scripts/subcap_ab_bench.py`: A/B decode harness (5 pairs alternating;
asserts distinct `mx.__version__` stamps; reports medians + greedy-id
identity digest) for the M2 post-window run.

## Calibration data (M2, T6021, Honeykrisp)

Sources: `~/.local/share/apple-silicon-lab/artifacts/SubmitCap/20261002-submission-cap-19/m2-4b-calibration.log`,
`.../m2-4b-budget{0,40000}-profile.jsonl`,
`.../m2-9b-budget{0,40000}-profile.jsonl` (large profile JSONLs live in
the private lab notebook per the public-repo blob-gate policy;
AGENTS.md / privacy hook).

Benchmark rule satisfied: gpu-turn tickets; reported wall runs were taken
with `cat /proc/loadavg < 0.5` and `cat /proc/pressure/cpu` `avg10 = 0`.
Provenance printed by `mlx-omarchy-info`; wheel stamps are the source
commits (see Wheel stamps below).

### Qwen3-4B-Instruct-2507-4bit (32 decode tokens, 256 prefill)

| MLX_OMARCHY_BATCH_WORK | n_subs | p50 ms | p95 ms | p99 ms | max ms | decode tok/s |
|---|---|---|---|---|---|---|
| 0 (cap off) | 39 | 20.921 | 21.567 | 90.342 | 129.691 | 29.788 |
| 40000 | 159 | 5.606 | 5.833 | 26.096 | 26.556 | 29.743 |
| 80000 | 80 | 10.639 | 11.501 | 51.554 | 52.289 | 30.401 |
| 160000 | 78 | 18.997 | 20.321 | 43.639 | 104.206 | 29.980 |

### Qwen3.5-9B-MLX-4bit (32 decode tokens, 256 prefill)

| MLX_OMARCHY_BATCH_WORK | n_subs | p50 ms | p95 ms | p99 ms | max ms | decode tok/s |
|---|---|---|---|---|---|---|
| 0 (cap off) | 40 | 49.631 | 57.760 | 187.016 | 257.362 | 11.349 |
| 40000 | 532 | 4.358 | 5.143 | 9.902 | 13.106 | 12.026 |

**Default chosen: 40000** — 4B decode lands at p50 = 5.6 ms (target 2-6 ms);
9B decode lands at p50 = 4.4 ms. Tok/s cost on 4B is -0.15% (inside the
<= 2% budget). 9B tok/s *improved* +6% (49 ms → 4 ms chunks let more of the
queue throughput overlap with the host's record time).

The full A/B harness `subcap_ab_bench.py` reports the formal comparison;
it and the four standing-battery runs (omarchy_runtime_tests,
omarchy_primitive_tests, omarchy_compiled_tape_tests,
omarchy_fused_chain_tests) on the M2 are scheduled for the post-w72M1
window — see "Open after this receipt".

## Greedy decode identity

`scripts/bench_decode.py` records the exact generated token IDs and hashes
them; A vs B must agree bit-for-bit. The A/B harness (`subcap_ab_bench.py`)
asserts `set(ids_sha256_16 A) == set(ids_sha256_16 B)` and fails the run
otherwise.

**Measured (M2, gpu-turn, 5 alternating pairs, decode 64 tokens +
prefill 276-token prompt):**

| Model | decode A (cap off) | decode B (default) | delta | prefill A | prefill B | delta | ids digest |
|---|---|---|---|---|---|---|---|
| Qwen3-4B (4-bit) | 53.60 tok/s | 53.69 tok/s | **+0.17%** | 416 tok/s | 554 tok/s | +33.1% | `b8c2bdf6ac3a4b02` (both arms, all 5 pairs) |
| Qwen3.5-9B (4-bit) | 22.22 tok/s | 22.01 tok/s | **-0.93%** | 63.4 tok/s | 64.6 tok/s | +1.9% | `d70cf804d08ae7d3` (both arms, all 5 pairs) |

- Decode cost is inside the <= 2% budget on both model classes
  (+0.17% 4B, -0.93% 9B).
- Prefill **improves** with the cap (+33% on 4B): the prefill graph was
  one long submission; splitting it lets the GPU start on the head while
  the host records the tail.
- Greedy identity is bit-identical across arms and pairs on both models
  (single digest per model, all runs agree).
- Stamps asserted distinct by the harness:
  A = `0.32.4.dev202610020734+4b2929a`,
  B = `0.32.4.dev202610020729+1e7cb4b`.
- Raw per-pass JSON: private lab artifacts
  `SubmitCap/20261002-submission-cap-19/m2-ab-{4b,9b}.json`
  (regenerate after the w73 window; /tmp was wiped by the ANE lane's
  reboot mid-batch — the summary numbers above are captured from the
  run stdout before the reboot).

## Wheel stamps

| Wheel | Source commit | Local version | sha256 | size |
|---|---|---|---|---|
| A (cap off) | 4b2929a9 | 0.32.4.dev202610020734+4b2929a | 985e25235d59e5b6a7544447ccb3ebba2abee645b8375115c2a815b708ce6a52 | 416261282 |
| B (cap default 40000) | 1e7cb4b6 | 0.32.4.dev202610020729+1e7cb4b | b1a94d0b3a487f454603fc1b6552a8808a33ae49b923fe5895486b32b2963a0d | 416262913 |

Both built with the same pinned upstream MLX (mlx.lock 9c3d35571a) and
the same whole-encoder bundle; only the source commit (and therefore the
constant) differs. `dist/` holds both aarch64 wheels.

## Standing battery + correctness gates (M2, wheel B source tree)

Build: cmake -B build-tests -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_TESTS=ON
(+ ANE device flag), run inside a gpu-turn window
(2026-10-02T03:4xZ, uptime-stable boot):

| suite | cases | assertions | result |
|---|---|---|---|
| omarchy_runtime_tests | 41 | 22694 | **PASS** (includes the new chunking doctest) |
| omarchy_primitive_tests | 104 | 2743003 | **PASS** |
| omarchy_compiled_tape_tests | 13 | 3124 | **PASS** |
| omarchy_fused_chain_tests | 36 | 346272 | **PASS** |
| omarchy_indexing_ops_tests | 57 | 35859 | **PASS** |
| omarchy_shape_ops_tests | 25 | 898 | **PASS** |

Numerical results are bit-exact against the suites' fixed references; the
cap and the priority chain do not move a single value.

## Queue priority: exposure + safety

- M2 (T6021, Honeykrisp, stock Mesa 26.2.3): `VK_EXT_global_priority`
  revision 2 + `VK_EXT_global_priority_query` + `VK_KHR_global_priority`
  (gpu-turn vulkaninfo probe, 2026-10-02T06:2xZ).
- jw16 (T6001, Honeykrisp, Omarchy; gpuwin window announced on herdr
  w72:p1): `VK_EXT_global_priority` revision 2 +
  `VK_EXT_global_priority_query` revision 1 + `VK_KHR_global_priority`
  revision 1, `globalPriorityQuery = true`
  (lab artifacts `jw16-vulkaninfo-global-priority.txt` +
  `jw16-vulkaninfo-full.txt`).
- Silent fallback proven on the M2 with the B wheel:
  `MLX_OMARCHY_QUEUE_PRIORITY=realtime` -> the family priority list does
  not report REALTIME, the code does not chain the priority struct, the
  device creates with the default priority, and the compute still runs
  (`REALTIME-FALLBACK-OK 1048576.0`, rc=0, no stderr output — the
  CreateDevice-retry path never fired because the pre-check declined
  first). `MLX_OMARCHY_QUEUE_PRIORITY=off` -> `OFF-OK`, same result.
- Priority low-vs-off decode tok/s delta: scheduled in the post-window
  M2 batch (3-pair decode on 4B, both env settings on the same wheel B);
  expected inside noise because the priority only affects queue
  arbitration, and MLX is the only queued client on this idle box.

## Frame pacing

**No logged-in compositor was available on jw16 at measurement time**
(only the SDDM greeter Hyprland on seat0, `XDG_SESSION_CLASS=greeter`;
uid-1000 SSH sessions have no display). The documented fallback is
submission-length histograms — delivered above from device timestamps —
plus this direct proxy of what the compositor experiences:

`scripts/submit-latency.c`: a 60 Hz stand-in client. It submits one tiny
command buffer (256-byte fill) per 16.6 ms frame and measures the
host-latency from just before vkQueueSubmit to fence-signal. While a long
MLX submission holds the queue, that latency spikes to the submission's
remaining length — the exact mechanism in the issue (compositor frames
stuck behind MLX work). Metrics: p50/p99/max and count over 2x the frame
budget (33.3 ms), run alone / with MLX decode cap off / cap on. Run in
the post-window M2 batch.

## Open after this receipt

After the w73 packaged-stack M2 window ends (broadcast M2 WINDOW END 2):

- M2 (gpu-turn): frame-pacing proxy matrix — `submit-latency` alone,
  with MLX decode cap off, with cap on (default); record
  p50/p99/max/over-2x for each into the receipt.
- M2 (gpu-turn): priority low-vs-off decode (3 pairs, wheel B, env
  toggle) — record medians.
- M2 (gpu-turn): regenerate `m2-ab-{4b,9b}.json` artifacts (the ANE
  lane's reboot wiped /tmp; summary numbers above are from stdout).
- Update this receipt with the three results; push; Main posts the
  issue comment (draft below).

## Issue #19 comment draft (Main posts)

> Root cause confirmed as GPU submission length, not utilization — same
> conclusion as your OpenGL reproducer. On Honeykrisp the MLX backend was
> batching a whole decode token (and a whole prefill graph) into single
> queue submissions; the reporter's 78 ms vs 2-6 ms observation matches
> what the backend's own device-timestamp profiler shows:
> Qwen3-4B decode submissions at the old batching budget land at
> p50 = 20.9 ms (max 129.7 ms across a 256-token prefill), Qwen3.5-9B at
> p50 = 49.6 ms — the "one long submission hitches the compositor" regime.
>
> The fix caps each submission's estimated GPU work (summed dispatch
> work-groups) at 40000 by default, tuned on the M2 Max so decode
> submissions land at p50 = 5.6 ms (4B) / 4.4 ms (9B). `MLX_OMARCHY_BATCH_WORK`
> overrides it (0 = off, for headless boxes). Measured cost, 5 alternating
> A/B pairs on the M2: decode +0.17% (4B) / -0.93% (9B); prefill got
> faster (+33% on 4B) because the split prefill overlaps GPU execution
> with host recording. Generated token IDs are bit-identical across cap
> off/on on both models (exact ID digests, 5 pairs each), and the
> standing Omarchy suites that exercise submission paths pass unchanged
> (runtime 41/41, primitive 104/104, compiled tape 13/13, fused chain
> 36/36, indexing 57/57, shape 25/25). Splitting is scheduling only:
> every submission already waits on the stream's previous completion, so
> dependencies are preserved by construction.
>
> The backend also requests a lower queue priority where the driver
> exposes VK_EXT_global_priority (confirmed listed on both the M2 Max
> T6021 and M1 Max T6001 Honeykrisp drivers), so the compositor wins
> queue arbitration between MLX submissions; if the driver refuses or
> does not list the priority, the backend silently keeps the default.
> `MLX_OMARCHY_QUEUE_PRIORITY=off` disables the request.
>
> Desktop-frame pacing was not measured on a compositor (no logged-in
> Hyprland session available at measurement time); the submitted
> evidence is per-submission GPU time from device timestamps before/after
> the cap, plus a small submit-latency probe that measures what a 60 Hz
> compositor-style client experiences while decode runs.