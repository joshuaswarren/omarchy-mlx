# Parakeet persistent session + derived-channel cache (2026-09-23)

Scope: remove the repeated setup cost from the installed Parakeet path on
jwm1 (T8103). Two changes landed off `agent/integrated-e2e` (tip
`8e050c6d5`), worktree `~/src/mlx-omarchy-parakeet-session-wt`, branch
`agent/parakeet-session`:

- `0a4e65bb2` ane: identity-keyed derived-role-channel cache
  (`overlay/mlx/backend/omarchy/ane/bundle.cpp`). `validate_program_contract`
  re-read the full ANEC payload (458 MB encoder) on every open per program
  only to re-derive the (src, dst) surface map. The map is now cached under
  the same cheap file identity the TOFU digest cache uses
  (`path|dev|ino|size|mtime_ns`), process map + sidecar
  (`~/.cache/mlx-omarchy/ane-channel-cache.txt`), kill switch
  `MLX_OMARCHY_ANE_CHANNEL_CACHE` / override `..._PATH`. Per-binding
  validation still runs every open; an implausible sidecar entry falls
  through to a full re-derivation, so a bad cached map fails the open
  loudly.
- `bf8793f74` parakeet: process-level resident ANE session singleton,
  `--repeat` (`overlay/tools/coreml/vulkan_encoder.py`,
  `overlay/tools/coreml/ane_resident.py`,
  `overlay/tools/mlx-omarchy-parakeet/mlx_omarchy_parakeet.py`). Repeated
  transcriptions in one process reuse the spawned worker and resident
  bundles; each pass opens a fresh batch scope; the atexit hook releases
  the worker outside measured passes. Submit/batch-close failures drop the
  held session; `ANE_ISLAND_PRIVATE_SESSION=1` restores private sessions.
  The report carries a `session` block (shared/reused/open_ms/
  batch_open_ms/close_ms) so startup lands separately from per-call
  latency.

Wheels (built on jwm1, `scripts/build-wheel.sh`, DEV_RELEASE=1):

| artifact | commit | sha256 |
| --- | --- | --- |
| candidate `mlx_omarchy-0.32.3.dev202609230300+bf8793f-cp314-cp314-linux_aarch64.whl` | `bf8793f` | `51e24c0c8a99c64252f5a5af36f68cb9bbd30e5c7fb7febe188caf14107c8f20` |
| rollback (pre-change tip) `/var/tmp/pk-rollback-b1/dist/` | `8e050c6` | `ff890352aa6c10a9354d3776bd8b893cede7f7227e91e5ed4a9ab1e437cf3dea` |

## Stage table — real jwm1 runs, golden fixture, warm caches, flock /tmp/m1-gpu.lock

Arm A = installed wheel `8e050c6` (`/var/tmp/v072-venv-fused`, untouched).
Arm B = candidate wheel in a fresh venv (`/var/tmp/pk-sess-venv`), same
share content (whole-encoder bundle copied byte-identical, sha-verified).
Driver: `/var/tmp/pk-sess-driver.py` (in-process repeats through
`_transcribe`; per-run reports under `/var/tmp/pk-sess-out/<arm>/`).

Every run below: status `match`, all pin checks pass, 104 emissions,
transcript sha `db501a8c…`, `decode_control=host`.

| arm | process/pass | total ms | encoder ms | of which exec ms | tdt ms | mel ms |
| --- | --- | --- | --- | --- | --- | --- |
| A r1 | fresh / fresh session | 1732.6 | 873.8 | 141.6 | 569.1 | 187.2 |
| A r2 | same process / fresh session | 1647.4 | 880.1 | 140.9 | 590.4 | 60.5 |
| A r3 | same process / fresh session | 1654.4 | 880.6 | 142.8 | 593.7 | 61.5 |
| B r1 | fresh / fresh session, first-ever channel derive | 1805.4 | 944.5 | 141.3 | 568.9 | — |
| B r2 | same process / reused session | 1038.7 | 274.1 | 272.9 | 585.6 | — |
| B r3 | same process / reused session | 1012.2 | 275.2 | 273.9 | 574.2 | — |
| B-fresh2 | fresh / fresh session, channel sidecar hit | 1416.4 | 520.6 | 141.4 | 594.4 | — |
| B-nocache | fresh / fresh session, `MLX_OMARCHY_ANE_CHANNEL_CACHE=0` | 1600.1 | 683.6 | 141.3 | 609.5 | — |
| B-prof r2 | same process / reused session, cProfile on | 1055.0 | 232.8 | 231.6 | 613.9 | 61.2 |

Session blocks (new report schema): B r1 open 802.0 / reused false;
B-fresh2 open 378.0; B-nocache open 541.0; B r2-r3 open 0.0,
batch_open 0.2, close 0.1, reused true.

## Reading

- In-process second call: 1647.4 -> 1012.2-1055.0 ms total (-37%), encoder
  stage 880.1 -> 232.8-275.2 ms. The pass no longer pays worker spawn +
  register + device program load (~564 ms) or session teardown (~150 ms).
- Fresh process (CLI-per-utterance): open 541.0 ms with the channel cache
  disabled vs 378.0 ms with a warm channel sidecar (-163 ms; the 458 MB
  payload read + task-stream walk is skipped). First-ever open after a
  content change pays 802.0 ms once and writes the sidecar.
- The 131 ms gap between B r1 open (802.0) and the no-cache open (541.0)
  plus B-fresh2 (378.0) is the derive+sidecar-write cost, paid once per
  bundle content generation.

## Findings from the same window

1. First-exec-after-idle wake penalty: a fresh session's whole-encoder
   submit is ~141 ms; the first submit on a reused session after ~1-2 s
   of idle (decode/report phase) is ~232-273 ms (+90-131 ms, device wake).
   Even paying it, reuse wins by ~600 ms per call. Related: the
   `ane_exec failed for program 4` flake (below) is the failure-shaped
   cousin of the same device-state effect.
2. TDT host control loop (570-614 ms across all arms today): cProfile on
   the reused-session run shows 276 `run_step` calls (145 decoder mode-0
   ~2.7 ms each, 131 joint mode-1 ~1.5 ms each); Python-side overhead
   (flags, argmax, numpy) is ~15-20 ms of the stage. The remainder is
   per-dispatch submit+sync through the omarchy backend (~1130 kernel
   dispatches). No host-side hotspot worth fixing: the gpu-loop
   alternative is measured slower (717-722 vs 593-609 ms, commit
   `8e050c6d5`), and the remaining lever is fusing the two LSTM
   layers' chains+fold dispatches into fewer launches while staying
   bit-exact - separate kernel work, not attempted here.
3. Mel frontend: fresh-process 184-194 ms splits into ~125 ms per-process
   pipeline creation (omarchy/Vulkan; macOS Metal restores pipelines
   faster - this is the 1.6x gap Main flagged) + ~61 ms in-process GPU
   exec (profiled: `stage_mel` tottime 61.2 ms with all sub-functions at
   ~0 ms, i.e. the time is GPU wait inside `mx.eval`). The driver-side
   fix belongs to the mesa disk-cache lane.

## ane_exec program-4 flake (characterization; bounded repro attempt negative)

MelShaderCache run 1, ~23:00 CDT 2026-09-22: first whole-encoder exec in a
fresh worker ~10 min after M1Finish's heavy Qwen ANE corpus ended on the
same device. `resident submit whole-encoder failed: job status=1 ...
elapsed_ms=1227 detail=ane_exec failed for program 4`; worker stderr
empty; immediate retry 3/3 clean in equally fresh workers. Shape:
intermittent first-exec failure after heavy ANE load + idle,
device/firmware state, not client state. Bounded repro attempt 2026-09-23
00:00 CDT (`/var/tmp/pk-pocket.log`): after QwenFuseHw's heavy parity
smoke + a 7-minute idle, a fresh-worker first exec passed cleanly -
exec 141.9 ms, identical to the immediate control (REPRO2 141.7) - so the
flake did not reproduce under the matched shape. Not reproduced across 15
golden first-execs today. The failure mode is loud (run fails, never
silent) and the resident client's no-retry safety model stays as is - any
retry policy is a design decision for Main, not this lane.

## Install (landed)

Candidate wheel `bf8793f` (sha256 `51e24c0c…`) installed into
`/var/tmp/v072-venv-fused` 2026-09-22 23:53 CDT with the whole-encoder
bundle preserved and restored sha-identical
(`WHOLE_BUNDLE_RESTORED_IDENTICAL`, `/var/tmp/pk-pocket.log`). Golden
smoke on the installed path: 2/2 match, transcript `db501a8c…`, 104
emissions, all checks:

| pass | pass shape | total ms | encoder ms | open ms | exec ms | tdt ms |
| --- | --- | --- | --- | --- | --- | --- |
| INSTALL 1 | fresh process, one-time channel derive on this path | 1557.7 | 986.9 | 841.7 | 143.9 | 305.7 |
| INSTALL 2 | same process, reused session | 567.3 | 142.8 | 0.0 | 141.9 | 304.7 |

INSTALL 2 is the reuse end state on an idle machine: the encoder stage
equals the bare exec (141.9 ms) - worker spawn, register, device program
load and teardown are all gone from the per-call path.

Rollback: `8e050c6` wheel, sha256 `ff890352…`, kept at
`/var/tmp/pk-rollback-b1/dist/`; the pre-change wheel is reproducible from
`/var/tmp/integ-e2e-8e050c6.bundle` as well. Candidate venv
`/var/tmp/pk-sess-venv` retained until QwenFuseHw's contract runs finish.

## Machine-state caveat for the stage table

Host-side stages (tdt_decode, mel_frontend, detokenize) swing up to ~2x
with background machine load: tdt_decode ranged 304.7-613.9 ms across
today's arms while the ANE-side exec stayed rock-steady at ~141 ms. The
A/B arms were measured minutes apart under similar load, so the contrast
(open/teardown removal, channel-cache hit) stands; absolute tdt/mel
numbers from contended windows should not be compared across hours. The
TDT profile (575 ms in `run_step`) was taken in a contended window and
cProfile-inflated; its hotspot conclusion (device dispatch+sync bound,
Python overhead ~15-20 ms) is unchanged.
