# TDT decode dispatch cut — speculative joint window (2026-09-23)

Branch `agent/tdt-defer-commit` (worktree
`~/.config/superpowers/worktrees/mlx-omarchy/TdtFused`, off
`agent/parakeet-session` tip `ef7d139e0`). Tip `b4757ac4`. Push to origin
remains blocked by the privacy hook on the pre-existing ancestor blob
(ed91550d9); branch travels as `/var/tmp/tdtfused-land/tdtfused7.bundle`
(on jwm1) and lives in the jwm1 clone `/var/tmp/pk-sess-b1`.

## What was wrong with the "1130 dispatches" model

Backend trace counters (`mlx_omarchy_trace_snapshot` via ctypes, deltas
around `tdt_decode`) showed the golden pass's TDT stage at
**1996 compute dispatches in only 277 submissions** (~7 dispatches per
submit): `mx.fast.metal_kernel` dispatches already batch into one command
buffer per `run_step`, and the one `mx.eval` per step is the single sync.
The cost is the **per-submission round trip** (~1.3-1.5 ms: submit,
cross-submission timeline wait, host wake) plus per-dispatch
pre+post memory barriers — not a submit per dispatch.

`MLX_OMARCHY_DEFER_COMMIT` (knob from `8bba36b21`) was measured first:
277 submits either way, TDT within noise, only `commit_calls_noop`
bookkeeping changed. The knob is a no-op for a loop that syncs once per
step anyway; the product flip was dropped.

## Change (commit `b4757ac4`)

The emitting frame pays **two** round trips: the frame-entry joint
(blank check, no preceding decode → never in the fused cache) and the
decoder step (whose fused joint the next joint call reuses). The decoder
step's submit now also evaluates the joint head for the next 6 frames
against the new state (`_JOINT_WINDOW_SOURCE`, one extra dispatch), and
the joint callback resolves frame-entry joints host-side from the
already-synced window rows (`spec["host"]`, one eager 6x8198 fp32 copy
per decode submit — a lazy per-row `np.asarray` would pay its own slice
dispatch + submit per hit, which the first measurement caught).

- The window kernel streams the 640x8198 fp16 joint weights **once for
  all six slots** (the joint head is weight-bandwidth-bound: a naive
  per-slot window re-read 10.5 MB x 12 per decode and measured ~+180 ms
  TDT). Each thread owns one output lane and applies every loaded weight
  element to six per-slot scalar accumulators; slot relu vectors live in
  threadgroup memory (declared with a literal size — the translator's
  shared-memory regex rejects expressions).
- Per-element arithmetic is identical to `_JOINT_SOURCE` (fp16 relu,
  fp32 ascending k chain, one fp16 rounding, fp16 bias), so window rows
  are bit-identical to what a joint-only step computes.
- Rows past `valid_frames` zero-fill and are never queried (the control
  loop ends at `valid_frames`; durations are [0..4], so blank hops reach
  at most base+4 < base+6).
- Joint-only fallback (window exhausted): pays one submit, re-speculates
  from its frame, and **skips the projector dispatch** — jmode 1 rebuilds
  the relu from `dec_in`, so the projector output was never read.
- 0 joint-only runs across all 16 measurement passes: every
  frame-entry joint resolved from the window.

## Before/after — real jwm1 runs, golden fixture, one process per block,
flock /tmp/m1-gpu.lock, warm passes (runs 2-4 of each block), interleaved
control/candidate blocks in one window

Control = `bf8793f` wheel in `/var/tmp/tdtfused-venv` (untouched copy of
the installed venv). Candidate = `b4757ac` wheel installed in
`/var/tmp/v072-venv-fused`. ANE exec in warm passes shows the known
session-wake penalty (233-279 ms), identical in both arms; fresh passes
(~142 ms) are the idle signature.

| arm | pass | tdt ms | total ms | mel ms | checks |
| --- | --- | --- | --- | --- | --- |
| ctrl bf8793f | warm x6 | 588.6 mean (589.0 med) | 1031.7 mean | 60.3-62.9 | 0 fail |
| cand b4757ac | warm x6 | **503.4 mean (499.6 med)** | **932.2 mean** | 59.8-62.8 | 0 fail |

Warm TDT **-85.2 ms mean (-14.5%)**, total **-99.5 ms (-9.6%)**.
Submissions inside the TDT stage: 277 -> 145. Compute dispatches:
1996 -> 1735 (projector skip -130, joint-only runs -131, window +145,
slice copies removed). All passes: status `match`, 104 emissions,
transcript sha `db501a8c...`, `decode_control=host`.

## Install (landed)

Candidate wheel
`mlx_omarchy-0.32.3.dev202609230623+b4757ac-cp314-cp314-linux_aarch64.whl`
sha256 `ef450bf5c57b8e13c1bbef71d47f7f80a0d985677b692ad70b4c7b8e0648827d`
installed into `/var/tmp/v072-venv-fused` (via `bin/python -m pip
--force-reinstall --no-deps`; NOTE `bin/pip` in the venv copies has a
shebang to the original venv — `python -m pip` is the safe form). Share
content `share/mlx-omarchy/parakeet-1` including the whole-encoder bundle
`bundles/parakeet-encoder-whole` survived every reinstall (untracked by
pip's RECORD); pin checks green on every pass. Golden on the installed
path: the 8 candidate-arm passes above.

Rollback wheel (bf8793f source, rebuilt because `build-wheel.sh` wipes
`dist/`): `/var/tmp/tdtfused-rollback-dist/mlx_omarchy-0.32.3.dev202609230545+bf8793f-...whl`
sha256 `a1361b58eceefb5302d72ffdc85cc23b392b63cd75382412311a0de548d48bbf`
(source-identical to the previously installed `51e24c0c...` wheel; these
builds are not byte-reproducible). The 8e050c6 wheel remains at
`/var/tmp/pk-rollback-b1/dist/`. Extra insurance: the candidate wheel +
branch bundle copied to `/var/tmp/tdtfused-land/`; share backup at
`/var/tmp/tdtfused-share-backup` (438 MB).

## Notes for the next lane

- The remaining TDT floor is 145 round trips (one decoder submit per
  host token decision) at ~2-3 ms each under this machine's wake/latency
  profile: further cuts need either cheaper submissions (honeykrisp/mesa
  lane, `joshuaswarren/mesa-1` per the migration rule) or relaxing host
  control — both out of scope here.
- `MLX_OMARCHY_DEFER_COMMIT` stays off by default; measured no-op for
  this loop (kept for the LLM-generation lanes that tuned it).
- Machine-state caveat: host-side stages still swing with background
  load; the A/B above interleaved arms inside one window so the contrast
  holds, but absolute numbers should not be compared across hours.
