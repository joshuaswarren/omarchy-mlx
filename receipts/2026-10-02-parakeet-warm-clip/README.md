# 2026-10-02 — `mlx-omarchy-parakeet transcribe` warm-clip latency, lever shipped (L1: stamp-aware verify)

Source: branch `ParakeetWarm` at `b4ff86d13` (origin/main), worktree
`~/.config/superpowers/worktrees/mlx-omarchy/ParakeetWarm`. Notebook entry
`~/.local/share/apple-silicon-lab/entries/ParakeetWarm/20261002T231221Z-jwm1-parakeet-warm-timeline-h000.md`.

## Question and outcome

The shipped `mlx-omarchy-parakeet transcribe` paid ~239 ms per invocation
re-hashing the ~1.5 GB reference cache (`fetch.verify_cache`) even
after `download` / `verify` had already validated the bytes. The
opt-in `MLX_OMARCHY_PK_TRUST_CACHE=1` env var short-circuits that
re-hash against a sidecar written at the last successful verify.
Measured speed-up on jwm1 (T8103, MesaParity venv
`0.32.4.dev202610012048+6cff5ea`, libane `d06222a8…`):

| arm                                | cell (median) | source                         |
|------------------------------------|---------------|--------------------------------|
| cold / warm pass (default)         | 238-246 ms    | `cold-pass/probe.log`, `warm-pass/probe.log` |
| trust-on pass (`MLX_OMARCHY_PK_TRUST_CACHE=1`) | **0.173 ms** | `trust-on-pass/probe.log` |

The 1386× drop translates to ~239 ms off every CLI invocation on the
warm-clip use case (a process where `verify` / `download` ran at least
once and the cache has been unchanged since). All other measured cells
in the timeline (python interpreter boot ~350 ms, the ANE resident
session open ~806 ms on rep 1 of each invocation, decoder_load ~44 ms,
mel / encoder_ane / tdt_decode / detokenize) are unchanged and dominate
the remaining ~1.7 s of warm-clip end-to-end.

## Files changed

- `overlay/tools/coreml/reference.py` —
  - new `CACHE_VERIFIED_NAME = ".verified-hashes"` sidecar constant,
  - new `trust_cache(cache_dir, lock)` reads the sidecar and refuses
    on missing stamp / size drift / recorded-vs-lock hash drift /
    file-mtime newer than the sidecar (catches post-verify writes),
  - new `record_verified_hashes(cache_dir, lock)` writes the sidecar
    after every successful hash (called from `record_stamps`,
    refreshes after any verify fall-through),
  - new `verify_cache_with_stamp(cache_dir, lock, *, trust_stamp)`
    is the wrapper `transcribe` calls when `MLX_OMARCHY_PK_TRUST_CACHE`
    is set; `trust_stamp=False` is a straight `verify_cache` so the
    default behavior is byte-identical.
- `overlay/tools/coreml/fetch_parakeet_reference.py` — re-exports
  `verify_cache_with_stamp` for the CLI.
- `overlay/tools/mlx-omarchy-parakeet/mlx_omarchy_parakeet.py` —
  `_transcribe` consults `MLX_OMARCHY_PK_TRUST_CACHE` and calls
  `verify_cache_with_stamp` instead of `verify_cache`. The opt-in is
  string-typed (`off / 0 / false / no` disable; everything else
  enables) so a typo does not silently enable the fast path.
- `overlay/tests/omarchy/coreml/test_trust_cache.py` — 11 cases
  covering: trust pass on fresh stamp, fail on missing stamp, fail
  on size drift, fail on recorded-vs-lock hash drift, fail on a
  same-size post-verify tamper (mtime guard), `trust_stamp=False` is
  strict, `trust_stamp=True` falls through to verify on drift and
  refreshes stamps. 11/11 PASS on workstation python3.11.
- `docs/parakeet.md` — documents the env var and the sidecar.

## Outputs preserved (golden contract)

The trust-on path does NOT alter the transcribe report. Across the
two measured arms and the trust-on arm, the four reports
(`warm-pass/r1-report.json`, `warm-pass/r11-report.json`,
`trust-on-pass/r1-report.json`, `trust-on-pass/r11-report.json`)
hash to the same transcript:

```
db501a8c080380ea027ffa50a4b4956c39df77cb692c4fb78e556311a11a0790
```

Per-report summary (median cells from the same runs):

| report                   | status | emissions | checks_failed | ane.exec_ms | ane.session.open_ms | total_pipeline_ms |
|--------------------------|--------|-----------|---------------|-------------|---------------------|-------------------|
| warm-pass/r1             | match  | 104       | []            | 138.768     | 806.437             | 1182.303          |
| warm-pass/r11            | match  | 0         | []            | 143.219     | 0.0 (reused)        | 336.435           |
| trust-on/r1              | match  | 104       | []            | 162.554     | 1126.948            | 1636.499          |
| trust-on/r11             | match  | 0         | []            | 161.123     | 0.0 (reused)        | 410.754           |

(`trust-on/r1` runs under `load1=0.90` (other-lane CPU contention at
the time of measurement); `warm-pass` cells are from an idle gate
window.)

## Stage timeline (median ms, warm pass; --repeat 11, reps 2-11)

| stage             | median | notes                                          |
|-------------------|--------|------------------------------------------------|
| python boot       | ~350   | inferred: 5.94 s wall − 5.547 s `transcribe_to_python` |
| import.cli_module | 30-34  | probe                                          |
| check_runtime_deps| 29.5   | numpy + google.protobuf import probe            |
| verify_cache      | 239-246| CPU-bound SHA-256 over ~1.5 GB reference cache |
| pipeline rep 1    | 1182-1321 | decoder_load 44 + audio 7 + mel 33 + encoder_ane 947 (session open 806 inside) + tdt 137 + detok 15 |
| pipeline rep 2-11 | 395-456 | session reused; sum of stages ≈ the report's total_pipeline_ms |

## L3: persistent ANE worker daemon across CLI invocations

Opt-in with `MLX_OMARCHY_PK_KEEP_WORKER=1` and
`MLX_OMARCHY_ANE_SOCK=PATH`; default off. The daemon,
`mlx-omarchy-ane-worker --daemon --socket PATH --idle-time-ms N`, opens
the same sealed resident session `--serve` opens (libane pin and bundle
hashes checked at daemon start), binds `PATH` in mode 0600, refuses
cross-uid peers (SO_PEERCRED), holds a single-instance `flock` on
`PATH.lock`, and serves one client at a time. It exits on the idle
timer, SIGTERM/SIGINT, `--stop`, or a lost session. The client
attaches with a 1 s connect bound and falls back to a private worker on
any connect failure.

### Measured on jwm1 (T8103), same boot, interleaved A/B, n=10 per arm

Boot `92c5b211-0b3b-4840-80c6-f6dc0062aae4`, uptime 1:12–1:16, load1
0.00–0.39 and PSI cpu some avg10 0.00 before every call. Venv rebuilt
from `mlx_omarchy-0.32.4.dev202610012048+6cff5ea` (provenance:
core/libmlx match the wheel RECORD), plus the L1/L3 Python files and a
worker built from this tree (sha256 `fa6c856a…`). Each call is a fresh
`transcribe` process; wall is launch to exit. Order DPDP… (D = daemon,
P = private).

| median / p95, ms | private (default) | daemon attached |
|---|---|---|
| per-call wall | 1938.8 / 1970.0 | **852.7 / 860.2** |
| `ane.session.open_ms` | 827.9 / 830.7 | **0.3 / 0.4** |
| `encoder_ane` stage | 968.2 / 970.8 | **141.4 / 142.3** |
| `ane.exec_ms` | 138.7 / 139.0 | 139.7 / 140.6 |
| `total_pipeline_ms` | 1204.8 / 1216.1 | **372.1 / 376.6** |
| audio_load | 6.4 / 6.6 | 6.4 / 7.1 |
| decoder_load | 43.3 / 45.5 | 43.3 / 44.6 |
| mel_frontend | 33.1 / 34.0 | 33.6 / 35.0 |
| tdt_decode | 139.0 / 141.6 | 136.5 / 137.9 |
| detokenize | 14.3 / 22.1 | 11.6 / 11.9 |

With the daemon attached, the encoder stage sits at the ANE exec floor
(141.4 ms stage vs 139.7 ms exec). Per-call wall drops by 1086 ms
(−56%). Both arms: 10/10 `status=match`, transcript sha
`db501a8c080380ea027ffa50a4b4956c39df77cb692c4fb78e556311a11a0790`, 104
emissions, 0 failed checks, `cpu_tensor_events=0`, report `transport`
`daemon` vs `private` as intended. A separate run of 20 sequential daemon
calls on one daemon: 20/20 match, all `transport=daemon`, wall median
848.6 ms (max 876.5), open median 0.31 ms.

About 480 ms of the daemon-arm wall falls outside the pipeline: client
process start, imports, and the strict cache rehash. A second interleaved
run on the same boot set `MLX_OMARCHY_PK_TRUST_CACHE=1` (L1) in both arms
(n=10 per arm; load1 ≤ 0.44, PSI avg10 ≤ 0.18 and 0.00 on most calls):

| median / p95, ms | private + L1 | daemon + L1 |
|---|---|---|
| per-call wall | 1727.2 / 1769.1 | **623.0 / 652.8** |
| session open | 830.9 | 0.3 |
| encoder_ane | 970.9 | 140.5 |
| total_pipeline | 1208.8 | 378.4 |

Both arms 10/10 match with the same sha, 104 emissions, no failed checks
and `cpu_tensor_events=0`. With both levers on, a warm `transcribe` takes
623 ms instead of 1939 ms (−68%). What remains is about 245 ms of
process start and imports, plus the pipeline. The pipeline is the ANE
encoder at its exec floor (~140 ms) plus GPU TDT decode (~137 ms),
decoder load (~43 ms), mel (~34 ms), detokenize (~12 ms) and audio load
(~6 ms).

### Resilience and device state (same boot)

| case | result |
|---|---|
| kill -9 a client at 0.55 / 0.65 / 0.75 s, then a new call | next call `match`, `transport=daemon`: session survives |
| kill -9 the daemon at 0.50 / 0.55 / 0.60 s (inside the ANE submit) | client exits rc=1 within ~1 s with a named refusal, no report written, no hang |
| after the daemon kill | 0 worker processes, `/dev/accel/accel0` has no holder; the supervised child exits on channel EOF |
| next call while the stale socket file exists | connect refused, falls back to private, `match` |
| restarted daemon | stale socket replaced, next two calls `match` over `daemon` |
| idle timer (3 s) | `idle exit`, `daemon released programs=1`, socket and lock removed, device free |
| dmesg, ANE device `26bc04000.ane` and DART, whole boot | 0 fault/error/timeout lines; last ANE lines are boot-time (6.7 s) |

### jw16 (M1 Max, T6001), one gpuwin window, interleaved n=10 per arm

Release wheel `0.32.4.dev202610021752+539d870e` (sha256 `721020b0…`,
v0.7.19), libane `d06222a8…` (pin), worker built from origin/main with
the same sha256 as jwm1 (`fa6c856a…`). gpuwin stopped llm-inference for
the window and restored it afterwards (`RESTORE health_ok=1
probe_finish=length`). Load1 was 0.63–1.55 and PSI avg10 ≤ 0.32 during
the calls, because the serving host had just stopped. The idle gate was
not met: the A/B is interleaved, so the delta is comparable, but the
absolute times are not idle-gated.

| median / p95, ms | private (default) | daemon attached |
|---|---|---|
| per-call wall | 2213.3 / 2229.7 | **1082.7 / 1107.8** |
| session open | 889.1 / 898.8 | **0.3 / 0.3** |
| encoder_ane | 1330.7 / 1340.4 | **442.1 / 445.9** |
| `ane.exec_ms` | 440.3 / 440.4 | 440.4 / 441.0 |
| total_pipeline | 1515.5 / 1521.9 | **627.3 / 634.7** |
| tdt_decode | 98.6 / 101.6 | 98.8 / 101.6 |

Both arms 10/10 match: sha `db501a8c…`, 104 emissions, no failed checks,
`cpu_tensor_events=0`. Per-call wall −1130.6 ms (−51%); the encoder stage
reaches T6001's whole-encoder ANE exec floor (~440 ms). The resilience
script passed in the same window. Client kill -9 left the session intact.
Daemon kill -9 mid-run gave client rc=1, left 0 workers and accel0 free,
and the next call fell back to private and matched. The restarted daemon
matched, and the idle exit released programs. dmesg showed 0 ANE/DART
fault lines.

### Correction (2026-10-03): the earlier "20/20 shipped" claim

The previous version of this section (commits `be30c0143`/`b933d8124`)
reported 20/20 daemon-attached matches with `total_pipeline_ms ~1216`
and `session.open_ms ~840`. Those numbers are the private-worker cost.
INFERENCE: those calls fell back silently to the private worker, because
the daemon they reached could not forward a request. The report had no
transport field, so the fallback was invisible. Two defects remained in
that code:

1. The pump used `splice(2)`, which requires one end to be a pipe. Both
   daemon ends are sockets, so every splice failed with EINVAL, and the
   pump read that as client EOF.
2. The client's detach sent the wire `close`, which `resident_child_loop`
   handles as "release the device and exit". That ended the shared
   session after the first client.

The current pump moves bytes with read/write and tracks the wire
framing in both directions. A client `close` at a cycle boundary is
answered `released` by the daemon and never forwarded. Client EOF
between cycles keeps the session. Client EOF after `run` drains and
discards the in-flight reply, bounded by the deadline. Client EOF inside
a partly written request, a `failed:` reply, or resident EOF loses the
session, and the daemon exits non-zero instead of serving a stale or
corrupt reply. The report now carries `ane.session.transport`, and a
daemon attach no longer counts as a worker start.

### Files

- `overlay/tools/mlx-omarchy-ane-worker/main.cpp`: `--daemon`,
  `--stop`, the framed pump (`run_framed_pump`), single-instance lock,
  peer-uid check.
- `overlay/tools/coreml/ane_resident.py`: daemon attach with fallback,
  `transport`, `alive` covers the socket transport.
- `overlay/tools/coreml/vulkan_encoder.py`: `session_transport`.
- `overlay/tools/mlx-omarchy-parakeet/mlx_omarchy_parakeet.py`:
  `ane.session.transport` in the report.
- `overlay/tests/omarchy/coreml/test_ane_resident.py`: attach serves
  consecutive clients without spawning; missing socket and stale socket
  fall back; the daemon stays opt-in when only the socket env is set.
- `scripts/` in this receipt: the A/B, 20-call, resilience and kill-sweep
  harnesses used above. They hard-code the jwm1 scratch paths of this
  run.

## Hardware / safety

- Repo branch: `ParakeetWarm` at `b4ff86d13` (origin/main).
- Modified files in the worktree (paths relative to the worktree):
  `overlay/tools/coreml/reference.py`,
  `overlay/tools/coreml/fetch_parakeet_reference.py`,
  `overlay/tools/mlx-omarchy-parakeet/mlx_omarchy_parakeet.py`,
  `overlay/tests/omarchy/coreml/test_trust_cache.py`,
  `docs/parakeet.md`,
  `receipts/2026-10-02-parakeet-warm-clip/README.md` (this file).
- Artifact manifest with sha256:
  `~/.local/share/apple-silicon-lab/artifacts/ParakeetWarm/warm-clip-timeline-h000/SHA256SUMS`
  (10 files, ~46 KB).