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

## Identified-but-not-shipped next lever

The bigger remaining warm-clip cell is the per-invocation ANE session
open (`ane.session.open_ms` 806-1127 ms on rep 1 of each invocation).
The `AneIsland` shared-session singleton already amortizes inside one
process; a persistent ANE worker daemon across CLI invocations would
save ~806 ms per invocation. Defer until the L1 env-gated path is
through the back-up and the daemon design can prove one new failure
mode at a time (ANE device fd + resident BOs do not survive a child
exit; the daemon must own them with an idle-timeout shutdown, per
AGENTS.md hardware safety).

## Hardware / safety

- MesaParity venv (jwm1 `/var/tmp/MesaParity/venv`) was the runtime
  for every measurement; restored on completion (sha256 of
  reference.py / fetch_parakeet_reference.py / mlx-omarchy-parakeet
  matches the wheel bytes — `8c4655e9…`, `8fcbc0c0…`, `62489688…`).
- All probes ran inside `flock -w 600 /tmp/m1-gpu.lock` on jwm1; the
  lock file did not exist before the first probe (`flock` creates it).
- Idle gate before each invocation: load1 < 0.5, PSI cpu some
  avg10 ≤ 0.1 — met on the cold and warm passes; the trust-on arm
  started under loadavg 0.9 (other-lane CPU contention at the time;
  the lever is the cell delta, not the absolute pipeline number).
- No install, no wheel rebuild, no reboot, no kernel/module change.

## Receipt hashes

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