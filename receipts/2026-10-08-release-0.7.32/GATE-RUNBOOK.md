# v0.7.32 gate-run RUNBOOK (generated 2026-10-08 from the v0.7.31 receipts:
# receipts/2026-10-07-release-0.7.31/gates-run/*). Goal: any host window is
# minutes from firing. Nothing here runs until Main says the wheel exists.

## 0. Inputs to stage BEFORE any window (per host, as FILL)
- Draft assets for v0.7.32 into `~/v0732-assets/` on the target host:
  wheel `mlx_omarchy-0.32.4.dev*-cp314-cp314-linux_aarch64.whl`,
  `omarchy-mlx-vendor-wheels-v0.7.32-cp314-aarch64.tar` (+ `.sha256`),
  `SHA256SUMS`. Verify on arrival: `cd ~/v0732-assets && sha256sum -c SHA256SUMS`.
- Gate scripts from main tip:
  `rsync -a <worktree>/scripts/release-gates/ <host>:~/v0732-gates/release-gates/`
  plus the standalone probes (they live in receipts/, not scripts/):
  `g16-qmm-batch.py`, `g16b-qmm-route-probe.py`, `g-g13c.sh`,
  `g-hold-g13g.sh`, `g17-patch-series.py`, `apply-mlx-lm-patches.sh`,
  `patches/` (both series), `70-omarchy-mlx-cpu-dma-latency.rules`
  (source: receipts/2026-10-06-release-0.7.29/gates/ — unchanged since).
- mlx-lm pristine wheels for g17: `pip download mlx-lm==0.31.3 mlx-lm==0.32.0 --no-deps -d ~/v0732-dl/`.

## 1. Env contract (placeholders to fill per host — from scripts/release-gates/env.sh)
- `TAG=v0.7.32` (or the draft tag Main cuts)
- `ASSETS_DIR=$HOME/v0732-assets`
- `GATE_ROOT=$HOME` (state lands in `~/v0.7.32-gate-home`, logs `~/v0.7.32-gate-logs`)
- `TAG_SHA=<40-char sha of the tag commit>` (REQUIRED: gate_ensure_install_tree asserts it)
- `EXPECTED_WHEEL_SHA256=<sha>` / `EXPECTED_VTAR_SHA256=<sha>` (REFUSE on mismatch)
- M2 only: `GPU_LOCK=/tmp/m2-gpu.lock`; jw16 legs: `RUN_JW16=1 JW16_SSH=16m1mbp SERVING_VENV=~/v0731-venv` (fresh path per host)
- g16/g16b: `VK_DRIVER_FILES` leg B overrides to the coreglass ICD; leg P uses the packaged ICD (no var).
- g-g13c: runs INSIDE a gpu-turn ticket (`GPU_TURN_TICKET=1` from the driver); expects `EXPECTED_WHEEL_SHA256` set.
- g-hold: `HOLD_WHEEL=~/v0732-assets/<wheel> HOLD_VENDOR_TAR=~/v0732-assets/<vtar> HOLD_RULE_SRC=~/v0732-gates/70-omarchy-mlx-cpu-dma-latency.rules` (HOLD_MODEL defaults /var/tmp/MesaParity/model); udev expectation root:video 0660.

## 2. Gate order and measured durations (v0.7.31 evidence, M2 12.2)
From gates.done (receipts/2026-10-07-release-0.7.31/gates-run/m2-gates-done-final.log):
| gate | wall (v0.7.31 measured) |
|---|---|
| g1-clean-install | 43 s |
| g2-online-9b | 203 s |
| g3-online-4b-card | 62 s |
| g4-offline | 16 s |
| g5-laya | 0 s |
| g6-codec | 11 s |
| g7a-packaged-icd | 0 s |
| g7b-system-install | 17 s (after INSTALL_TREE auto-staged; driver now does this — 97d31bbf0) |
| g8-kokoro | 10 s |
| g9-speak-queue | 12 s |
| g10-kokoro-primer | 123 s |
| g11-card-9b | 21 s |
| g12-kokoro-stream | 89 s |
| g14-routing | 23 s |
| M2 total | ~10.5 min |

jw16 ANE legs (from jw16-gates-receipt-jw16.log, v0.7.31):
build-wheel 169 s, g13-build 258 s, g15-build 8 s, g7c 148 s, g7d 43 s,
g13-run 135 s, g15-run 1 s → ~13 min + the g7c/g7d windows.

Standalone probes (not in run-all):
- g16 legs P+B: ~2-3 min each (12/12 rows bit-identical check)
- g16b legs P+B: ~16 s each
- g-g13c legs P+B: ~2-5 min each (inside gpu-turn tickets)
- g-hold legs P/B: 81 s / 110 s (v0.7.31 measured on jwm1)
- g17 patch-series: ~2 min (dev box measured 4/4 in 2.1 s per apply pair)

## 3. Exact commands per host

### M2 (G14C) — full battery + Bonsai leg
```
ssh <M2_SSH>
export TAG=v0.7.32 ASSETS_DIR=$HOME/v0732-assets GATE_ROOT=$HOME
export TAG_SHA=<tagsha> EXPECTED_WHEEL_SHA256=<wheelsha> EXPECTED_VTAR_SHA256=<vtarsha>
bash ~/v0732-gates/release-gates/run-m2-battery.sh   # wraps run-all.sh; PASS rule: 14 LOCAL gates RC=0, jw16 gates SKIPPED-only
# then the Bonsai G14C leg ticket via ~/bin/gpu-turn (per its queue entry)
```

### jwm1 (G13G) — main battery + probes
```
ssh jwm1
export TAG=v0.7.32 ASSETS_DIR=$HOME/v0732-assets GATE_ROOT=$HOME
export TAG_SHA=<tagsha> EXPECTED_WHEEL_SHA256=<wheelsha>
bash ~/v0732-gates/release-gates/g1-clean-install.sh
for g in g2-online-9b g3-online-4b-card g4-offline g5-laya g6-codec g7a-packaged-icd g7b-system-install g8-kokoro g9-speak-queue g10-kokoro-primer g11-card-9b g12-kokoro-stream g14-routing; do bash ~/v0732-gates/release-gates/$g.sh || echo "FAIL $g"; done
python3 ~/v0732-gates/g16-qmm-batch.py            # leg P (packaged ICD)
VK_DRIVER_FILES=<coreglass-icd> python3 ~/v0732-gates/g16-qmm-batch.py   # leg B
python3 ~/v0732-gates/g16b-qmm-route-probe.py     # legs P+B same pattern
bash ~/v0732-gates/g-hold-g13g.sh                  # with HOLD_* env above
python3 ~/v0732-gates/g17-patch-series.py          # or the jwm1-g17-run2.sh pattern (repo apply script, layout 2)
```

### jw16 (G13C) — restage first (home wiped), then battery
```
ssh 16m1mbp   # plain alias, ControlMaster (NEVER ControlPath=none)
mkdir -p ~/v0732-assets ~/v0732-gates
# stage assets + scripts as FILL (section 0), sha256sum -c, then:
export TAG=v0.7.32 ASSETS_DIR=$HOME/v0732-assets GATE_ROOT=$HOME
export TAG_SHA=<tagsha> EXPECTED_WHEEL_SHA256=<wheelsha> SERVING_VENV=$HOME/v0732-venv
# ANE legs via jw16-gates.sh (stages to /tmp, runs g7c/g7d/g13/g15)
bash ~/v0732-gates/jw16-gates.sh
# g-g13c legs P+B inside gpu-turn tickets (GPU_TURN_TICKET=1 contract):
bash ~/v0732-gates/g-g13c.sh    # leg P; repeat with VK_DRIVER_FILES override for leg B
```

### dev box (CPU) — g17 only
```
bash /tmp/g17-0732-run.sh   # pattern: pristine 0.31.3/0.32.0 wheels + main's apply script; expect ok=4/4
```

## 4. Receipt rules (unchanged from v0.7.31)
- Every gate logs `WHEEL_IDENTITY` + `uname_r` (feb57d020, 2811cf901).
- SUMMARY generated from logs by script, never hand-edited.
- Land under receipts/2026-10-08-release-0.7.32/gates-run/ via fetch+rebase.
- scrub /home/<user>/ paths before commit (privacy-check blocks them).
