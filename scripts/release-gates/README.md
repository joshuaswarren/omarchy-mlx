# Release gate harness

Parametrised gate runners for a draft-first release. Every gate installs from
the DRAFT release assets and refuses to run against anything else. Adapted
from the ad-hoc v0.7.11–v0.7.14 gate scripts (the working versions; nothing
in /tmp was deleted) with one convention set and every host-specific value
passed by environment variable.

## Parameters (env; no real hosts or paths are committed)

| variable | meaning | default |
|---|---|---|
| `TAG` | DRAFT release tag (**required**) | — |
| `ASSETS_DIR` | draft assets: `*cp314*linux_aarch64.whl`, `omarchy-mlx-vendor-wheels-*.tar[.sha256]`, `*cp311*linux_x86_64.whl`, `SHA256SUMS` | `/tmp/$TAG-assets` |
| `GATE_ROOT` | parent for throwaway gate state | `$HOME` |
| `M2_SSH` / `JW16_SSH` | ssh **aliases** (never raw addresses) of the two gate hosts | placeholders |
| `GPU_LOCK` | M2 serialization flock | `/tmp/m2-gpu.lock` |
| `TTS_PACK_HOME` | pinned voice-pack home (g6/g8) | `$HOME/mlx-tts-home` |
| `INSTALL_TREE` | worktree with the tag's `install.sh` (g7b) | `$GATE_ROOT/$TAG-worktree` |
| `SERVING_VENV` | jw16 serving venv path; g7c/g7d refuse to touch it | unset (guard off) |
| `PY_AARCH64` | aarch64 host python | `python3.14` |
| `RUN_JW16` | `1` = run g7c/g7d on jw16 in gpuwin windows | off (run fails loudly) |
| `GATE_INSTALL_SH`, `GATE_TAG_TARBALL` | offline overrides for the tag install.sh / tests tarball | fetched from the tag |

Gate homes (all under `$GATE_ROOT`, one naming scheme): `$TAG-gate-home`
(g1 install; reused by g2–g8), `$TAG-gate7d-home` (g7d's own fresh install),
`$TAG-fresh-hf` (empty HF cache for g2/g3/g4), `$TAG-assist-{9b,4b,off}`,
`$TAG-gate-logs/` (all gate logs + `gates.done` summary).

## Ordering

`run-all.sh`: **g1** clean install → **g2** online 9B (empty HF cache) →
**g3** online 4B + SSE card → **g4** offline (unshare namespace) → **g5**
Laya manifest pin → **g6** TTS codec regression (installed wheel) → **g7a**
packaged-ICD fixture contract → **g7b** offline system install + staged-tree
readback → **g8** Kokoro smoke → **g9** read-aloud playout (headless
Chromium, no GPU) → on the jw16 ANE host, each in its own
gpuwin window announced to the jw16 coordination pane first: **g7c** packaged
ANE worker verify, **g7d** fresh-image parakeet transcribe.

Publish + Latest only if every gate exits 0 and
`python3 scripts/verify-release-assets.py <tag>` prints `VERIFIED` on the
DRAFT. Gate runners must NEVER set `MLX_OMARCHY_PAIR_DEV_QUALIFICATION`
(enforced by `tests/test_release_gate_contract.py`).

## Gate inputs/outputs

| gate | reads | writes / asserts |
|---|---|---|
| g1 | draft assets, tag install.sh | `$TAG-gate-home` install; `INSTALL_EXIT 0`; chat + parakeet launcher help exit 0 |
| g2 | g1 home, empty `$TAG-fresh-hf` | one chat + one compare turn complete; `OUTBOUND_DENIED False`; listener port released |
| g3 | g1 home, HF cache | chat complete; SSE card events with a visible component; `CARD_CHECK PASS`; listener released |
| g4 | g1 home, HF cache | same turns with `OUTBOUND_DENIED True` inside `unshare -n -r` |
| g5 | g2/g3 assistant homes | both manifests pin the Laya revision |
| g6 | installed venv, tag tests tarball, voice pack | codec unittest OK, zero skips |
| g7a | system ICD + `$TAG-sysinst` prefix | `icd_source=packaged`, expected sha == driver sha |
| g7b | vendor tar, `$INSTALL_TREE/install.sh` | offline `--system` install exit 0; staged tree incl. `usr/bin/mlx-omarchy-parakeet`; no launcher path leaks |
| g9 | g1 home's installed `static/js`, a Chromium-family browser (`GATE_CHROMIUM` or PATH) | four sentences against a serialising fake `/api/speak` on the real Web Audio clock: `SPEAK_QUEUE_SMOKE PASS` = in order, 0 overlaps, 0 gaps, 0 busy retries, one audio-done after the last chunk; no browser = exit 2 |
| g7c | draft wheel + vendor tar on jw16 | private venv; packaged `verify` golden e2e: status match, all pin checks pass, `ane_mode=true`, `cpu_tensor_events=0` |
| g7d | draft wheel + vendor tar + staged reference cache on jw16 (or a CLI file in cpu mode) | user-style `transcribe` of the pinned fixture: exit 0 twice (entry + `python3 -S`), golden transcript sha match, `ane_mode=true`, old failure string ABSENT |

### g7d details (`g7d-fresh-transcribe.sh`)

Exists because a real fresh Omarchy image shipped `mlx-omarchy-parakeet
transcribe` that bound the system python (no numpy/protobuf) and refused.
The gate reproduces the user surface exactly: fresh install from the draft
assets, no manual pip, system interpreter asserted dep-less, then the entry
invoked twice (packaged data file via its shebang — or a staged launcher via
`G7D_LAUNCHER` — and the verbatim `python3 -S` form). PASS needs exit 0,
`status=match` with every pin check green, `ane_mode=true`,
`cpu_tensor_events=0`, transcript sha256 equal to the pinned golden sha
(`db501a8c…`, cross-checked against the wheel's own
`parakeet-reference.lock`), and the string
`missing runtime dependencies for transcribe` ABSENT everywhere.

`G7D_MODE=cpu` (or `G7D_CPU_ONLY=1`) runs on the dev box with no ANE: it
stages the installed layout under a throwaway prefix (owning venv python
with numpy+protobuf) and drives the CLI's own dependency probe under
`python3 -S`, stopping right after the dependency probe — no device open.
PASS = the failure string is absent and the probe crossed into the owning
venv. Point `G7D_CLI`/`G7D_OVERLAY_CLI` at a pre-fix CLI to see the gate
FAIL (that is the v0.7.14 demonstration).

Proving the gate catches the defect: run it against the published v0.7.14
assets (`ASSETS_DIR` pointing at those wheel+tar) — it MUST fail with the old
string — then rerun with `G7D_OVERLAY_CLI=<origin/main CLI>` — it must pass.

## Cleanup

- Gate homes/caches are throwaway: `rm -rf $GATE_ROOT/$TAG-gate-home
  $GATE_ROOT/$TAG-gate7d-home $GATE_ROOT/$TAG-fresh-hf
  $GATE_ROOT/$TAG-assist-* $GATE_ROOT/$TAG-gate-logs
  $GATE_ROOT/$TAG-tag-tests` and on jw16 `/tmp/omarchy-release-gates`,
  `/tmp/$TAG-vendor-*`, `/tmp/$TAG-gate-logs`, the gate homes.
- g2/g3 kill the chat server's process group and FAIL if `fuser` still shows
  a listener on the assistant port — never skip this check.
- jw16: g7c/g7d only run inside `/var/tmp/appbar/gpuwin.sh` windows (it stops
  llm-inference, takes the GPU flock, restores + health-probes on exit);
  announce to the jw16 pane before each window; `SERVING_VENV` must name the
  serving venv so the private-venv guard is live. Windows are serialized by
  the gpuwin mutex; keep each under 15 minutes.
- M2: every GPU-touching step holds `$GPU_LOCK`. Check the boot id before a
  battery (a reboot mid-battery invalidates timing claims; treat fresh-boot
  machines accordingly).

## Past harness pitfalls (why the conventions exist)

0. **Fresh-dir wheel builds only** — every release wheel builds in a FRESH
   work dir (`MLX_OMARCHY_WORK_DIR` pointing at a never-before-used path)
   from a clean checkout of the cut sha. The in-source cmake build state
   of a previous sha's build must never be reused: a mixed-object
   `libmlx.so` looks like a working wheel until a GC-heavy process dies
   in it (2026-10-04: a reused build dir produced a wheel whose failure
   mode was blamed on the code for an hour before the pristine rebuild
   settled it).

1. **Gate-home naming split** — early batteries mixed `$TAG-gate-home` and
   per-gate homes, so g2/g3 ran against whatever install g1 last left and a
   "pass" proved nothing. One `$TAG`-prefixed scheme, every gate refuses
   non-fresh paths (`gate_refuse_existing`).
2. **Shebang/pip traps** — the packaged CLIs are data files with
   `#!/usr/bin/env python3` shebangs; on the gate hosts that resolves to the
   SYSTEM interpreter, which has none of the venv's packages (the exact g7d
   defect). g7c invokes the packaged CLI with the venv's python; g7d exists
   to prove the fixed launcher/self-heal makes the user surface work anyway.
   Also: `pip install <wheel>` does NOT bring the vendor deps — install via
   `pip install --no-index --find-links $VDIR -r requirements-lock.txt`.
3. **Disk-full admission refusal** — g3 once failed because /home was 100%
   full and the app's own admission refused the 4B setup (spent gate scratch
   from old releases). `run-all.sh` checks ≥25G free BEFORE the battery.
4. **gpuwin windows on jw16** — never run ANE/GPU work outside a gpuwin
   window (it owns the stop → flock → restore → health-probe cycle); two
   gates = two windows, announced to the coordination pane; a window whose
   RESTORE line is not `health_ok=1 probe_finish=length` counts as failed.
5. **Kokoro smoke scope** — the g8 RTF includes the first model load; it
   proves the pipeline is wired, not real-time qualification. Do not quote
   it as a steady-state number.
