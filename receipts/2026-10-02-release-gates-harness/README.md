# 2026-10-02 — release gate harness lands in the repo (`scripts/release-gates/`) + new gate 7d (fresh-image parakeet transcribe)

Status: Phase 1 of the v0.7.15 release work. The gate battery that lived as
ad-hoc scripts in `/tmp` (v0.7.11–v0.7.14 releases) is now a parametrised
harness in the repo, and the coordination lane's requested new gate —
**g7d, fresh-image parakeet transcribe end to end** — is added and
demonstrated both ways: it FAILS on the published v0.7.14 assets with the
real defect and PASSES with the fix that is on main (`94a6699bc`).

Nothing in `/tmp` was deleted; the working scripts were adapted, not
discarded. The v0.7.15 cut (phase 2) runs this harness against the DRAFT
release and waits for Main's `CUT GO`.

## What landed

`scripts/release-gates/`:

| file | role |
|---|---|
| `env.sh` | shared parameterisation; every host-specific value arrives via env with placeholder defaults (`M2_SSH`/`JW16_SSH` ssh aliases, `GATE_ROOT`, `ASSETS_DIR`, `TTS_PACK_HOME`, `SERVING_VENV`, …). No real hostnames, addresses, or personal paths are committed. |
| `g1-clean-install.sh` | clean install into a throwaway HOME from the DRAFT assets; post-fix wheels must stage the `mlx-omarchy-parakeet` launcher (`PARAKEET_LAUNCHER staged` is required) |
| `g2-online-9b.sh` | fresh HOME + EMPTY HF cache, online 9B chat+compare; listener cleanup verified inside the gate |
| `g3-online-4b-card.sh` | fresh 4B home; card visibility via the corrected SSE runner (default since v0.7.12) |
| `g4-offline.sh` | app AND probe in one `unshare -n -r` namespace, loopback up |
| `g5-laya.sh` | both assistant manifests pin the Laya revision |
| `g6-codec.sh` | TTS codec regression against the INSTALLED wheel, zero skips |
| `g7a-packaged-icd.sh` | packaged-ICD fixture contract via `OMARCHY_MLX_SYSTEM_PREFIX` |
| `g7b-system-install.sh` | offline `--system` install from the vendor tar via the tag worktree; staged-tree readback now includes `usr/bin/mlx-omarchy-parakeet`; launcher path-leak check |
| `g7c-ane-worker-verify.sh` | jw16: private venv from draft wheel+vendor lock; packaged golden e2e `verify` through the venv's python (the data-file shebang resolves the system interpreter); `SERVING_VENV` guard refuses to touch the serving venv |
| `g7d-fresh-transcribe.sh` | NEW — see below |
| `g8-kokoro.sh` + `g8-kokoro-driver.py` | Kokoro smoke (first-load RTF; explicitly NOT real-time qualification) |
| `gate-probe.py`, `gate3-card-runner.py` | the chat/compare/card probe helpers, parametrised by argv (no baked paths) |
| `run-all.sh` | runs g1→g8 in order on the M2, stages + announces + runs g7c/g7d on jw16 in gpuwin windows, writes `gates.done`; ≥25 G free-disk preflight; a skipped jw16 leg fails the run loudly |

Ordering, inputs/outputs, cleanup, and the past harness pitfalls are
documented in `scripts/release-gates/README.md` (gate-home naming split,
shebang/pip traps, disk-full admission refusal, gpuwin windows).

Tests: `tests/test_release_gate_contract.py` keeps the
`MLX_OMARCHY_PAIR_DEV_QUALIFICATION` gate (which already covers the new dir
recursively) and gains `ReleaseGateHarnessTests`: harness files present, no
real hosts/paths committed, placeholder defaults, run-all executes the
gates in order and refuses a skipped jw16 leg, g7d pins the golden
transcript sha + exercises both user-style entry forms, g7d cpu mode stops
after the dependency probe, g7c guards the serving venv. 8/8 green.

## Gate 7d — fresh-image parakeet transcribe end to end

The defect it guards: on a fresh Omarchy image the installed
`mlx-omarchy-parakeet transcribe` refused with `missing runtime
dependencies: numpy, google.protobuf` because the wheel ships the CLI as a
data file with an `env python3` shebang and no launcher was staged — the
entry bound the system interpreter. Fixed on main by `94a6699bc` (staged
launcher + CLI self-heal); background in
`receipts/2026-10-02-parakeet-deps/README.md`.

What the gate does (`G7D_MODE=ane`, the default, on the ANE host):

1. installs from the DRAFT assets exactly like a user (private venv from
   wheel + vendor lock, fresh home, no manual pip);
2. asserts the SYSTEM interpreter is dep-less — otherwise nothing is proven;
3. invokes the installed entry as a user would, twice: the packaged data
   file via its shebang (or a staged launcher via `G7D_LAUNCHER`) and the
   verbatim `python3 -S` form;
4. transcribes the packaged golden fixture; PASS requires exit 0 on both
   forms, a report written BY THIS RUN with `status=match`, every pin check
   green, `ane_mode=true`, `cpu_tensor_events=0`, transcript sha256 ==
   `db501a8c080380ea027ffa50a4b4956c39df77cb692c4fb78e556311a11a0790`
   (cross-checked against the wheel's own `parakeet-reference.lock`), and
   the string `missing runtime dependencies for transcribe` ABSENT
   everywhere.

`G7D_MODE=cpu` runs on the dev box with no ANE: it stages the installed
layout under a throwaway prefix (owning venv python carries numpy+protobuf)
and drives the CLI's own dependency probe under `python3 -S`, stopping right
after the dependency probe — no device open.

## Proof: FAIL on v0.7.14, PASS on main (all captured)

Assets: the PUBLISHED v0.7.14 wheel (sha `60f175e0afde9568…`) + vendor tar
(sha `24a5bef313eeb6b2…`), verified against the release `SHA256SUMS`.
Fix CLI = origin/main `94a6699bc`, sha
`15213009d5831dea9f1d09f4b429ed2bacfd2737adad87cae41dde56fcc4cf21`
(source and installed bytes identical). v0.7.14 CLI sha
`62489688090151f628c8cd2929212292db81e2d0e6a6fb5ba966be568eb6634d`.

1. **Dev box, CPU-only, v0.7.14 CLI bytes** →
   `TranscribeRefusal: missing runtime dependencies for transcribe: numpy,
   google.protobuf` → **g7d RC=1, OLD_FAILURE_STRING PRESENT — DEFECT**.
2. **Dev box, CPU-only, main CLI** → probe re-execs into the owning venv,
   passes, process reaches the next refusal downstream of the deps probe →
   **g7d RC=0, DEP_BOUNDARY_CROSSED** (zero device interaction).
3. **jw16, private venv from the published v0.7.14 assets, no overlay**
   (boot `6b8b6035-c610-4348-9ef4-40194280d2d6`):
   entry form → `missing runtime dependencies for transcribe: google.protobuf`,
   `python3 -S` form → `missing runtime dependencies for transcribe: numpy,
   google.protobuf` (the verbatim jwm1 defect); both TRANSCRIBE_EXIT 1,
   `GOLDEN_FAIL no transcribe-report.json written by THIS run` →
   **g7d RC=1 — the gate catches the shipped defect**. CPU-only; the refusal
   precedes any device/worker work.
4. **jw16, same assets + `G7D_OVERLAY_CLI`=<main CLI>, inside a gpuwin
   window (announced to the coordination pane first)** → INSTALL_EXIT 0,
   overlay shas recorded (62489688… → 15213009…), TRANSCRIBE_EXIT[entry] 0,
   TRANSCRIBE_EXIT[bare-S] 0, old failure string absent, fresh report
   `transcriptions/20261002T123005Z`: `status=match`, `ane_mode=True`,
   `cpu_tensor_events=0`, all 7 pin checks pass, transcript sha
   `db501a8c…` == golden pin == wheel-lock pin →
   **GOLDEN_MATCH PASS, GATE7D_EXIT 0**. gpuwin restore:
   `RESTORE health_ok=1 probe_finish=length active=active`.

Full logs: private notebook
`apple-silicon-lab/entries/Release0715/20261002T074000Z-jw16-g7d-fresh-transcribe-proof.md`
(raw transcripts with per-line output). jw16 logs remain at
`/tmp/g7d-jw16-{defect2,fix}/v0.7.14-*/…g7d-fresh-transcribe.log` until the
tmpfs clears.

### Harness bug found and fixed during the proof

The first jw16 FAIL run printed `GOLDEN_MATCH PASS` from the STALE
Close0714 report in the shared cache while the gate's own transcribes wrote
nothing. g7d now scopes the golden check to reports with mtime after the
gate's own start; the clean FAIL rerun shows the honest
`GOLDEN_FAIL no transcribe-report.json written by THIS run` with the final
RC still 1. The released gate cannot read a previous run's evidence as its
own pass.

## Known limits

- The PASS run uses the v0.7.14 wheel with the fixed CLI overlaid (the
  ParakeetDeps method). That is exactly what the task asked to demonstrate;
  the v0.7.15 draft wheel will carry the fix natively, so phase 2 runs g7d
  with no overlay.
- g7d `ane` mode reuses the staged parakeet reference cache
  (`~/.cache/mlx-omarchy/parakeet-reference`, staged by `download`/g7c);
  on a cold host the gate stages it via `download --json` first (hash
  check + fetch of what is missing).
- The dev-box cpu mode exercises probe + re-exec + the next refusal; it
  cannot reach a real ANE machine check (x86), so end-to-end golden
  evidence only exists from the ANE host.
- Gates 1–8 were NOT run now (no draft release exists yet); they run at the
  v0.7.15 cut. The harness's phase-1 verification is the contract test
  suite (8/8) plus the g7d proof above; syntax of every runner is
  `bash -n` clean.
