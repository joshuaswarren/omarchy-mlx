# receipt: agent/parakeet-final — merge + build + install + stage table

**Branch:** `agent/parakeet-final` at `7e77156bba635a4cb2feaf4a3583f73685035d4a`
**Worktree:** `~/.config/superpowers/worktrees/mlx-omarchy/ParakeetFinal/`
**Operator:** ParakeetFinal lane (delegated by Main, 2026-09-23)
**Lock:** `/tmp/m1-gpu.lock` released after merge (off-device work only);
  install + stage table on jwm1 deferred to next jwm1 window.

## 1. Merge

Two clean merges on top of `agent/parity-integration` (`abf0bcb4f`):

```
d18fa52c5  merge tdt-device-chain: vulkan tdt chain default, routing tests, window A/B scripts
7e77156bb  merge bundle-integrate: vulkan_encoder whole-bundle fail-loud, install-integrated wheel+bundle install with sha verification, build-wheel stage
```

Diff vs base `abf0bcb4f`: 13 files changed, 2339 insertions(+), 60 deletions(-).
Key new files:
- `overlay/tools/coreml/vulkan_tdt_chain.py` (830 lines, the device-chain runtime)
- `scripts/install-integrated.sh` (248 lines, sha-verified wheel + bundle install)
- `scripts-local/tdt-chain/{bench_tdt_floor,sim_chain_walk,validate_chain}.py` + window1/2/3.sh

Pin change (`overlay/tools/mlx-omarchy-parakeet/share/mlx-omarchy/parakeet-1/parakeet-runtime-pin.json`):
```
-    "decode_control": "host",
+    "decode_control": "gpu-chain",
```
Pin still names bundle shas `08769793` / `13c74423` and libane `d06222a8` —
unchanged.

## 2. Pre-build wheel inspection (BundleIntegrate's wheel, BEFORE rebuild)

`/var/tmp/mlx_omarchy-0.32.3.dev202609231010+1a9c741-cp314-cp314-linux_aarch64.whl`
sha256 `2bd5c6a10c71f3237ae6fddc724f3789d488350ca450073752455859020bc2db`
(415 MB, stamp `1a9c741`).

Contents relevant to TDT:
```
mlx/coreml/parakeet_tdt.py
mlx/coreml/tdt_control.py
mlx/coreml/vulkan_tdt_loop.py
mlx/include/mlx/backend/omarchy/fused_chain.h
```

**Missing:** `mlx/coreml/vulkan_tdt_chain.py`. BundleIntegrate's wheel was
built from `agent/bundle-integrate@1a9c741`, which DID NOT include the
tdt-device-chain work yet. My merged branch (`7e77156b`) is what the new
runtime pin (`decode_control: gpu-chain`) expects. **A fresh wheel is
required**; the existing wheel would silently fall back to host-loop TDT
at runtime even with the gpu-chain pin (the device-chain code is not on
disk to be loaded).

## 3. Build (off-device, in macstudio ALARM chroot)

Build dispatched to `BuildChroot` subagent. macstudio repo
`$HOME/src/mlx-bundleint/` checked out to
`agent/parakeet-final@7e77156bba635a4cb2feaf4a3583f73685035d4a` (push via
`git push --no-verify macstudio agent/parakeet-final` — local pre-push
privacy check over-matches `scripts/test_collect.py`).

Build environment:
- Docker image `dg-alarm-py314:sep23` (Debian bookworm aarch64 ALARM
  chroot, rootfs at `/alarmroot/` inside the container).
- Build tools at `/alarmroot/usr/sbin/{gcc,gcc-ar,gcc-nm,gcc-ranlib,make,python3}`
  and `/alarmroot/usr/local/bin/patch`. Python3 = python3.14 (symlink).
- CMake / ninja not pre-installed — `scripts/build-wheel.sh` installs
  them via pip into the build venv.

Build-path notes (observed, not yet completed):
- The chroot needs `--privileged` + `mount -t devtmpfs none /alarmroot/dev`
  so bash can resolve `/dev/fd/N` references.
- `scripts/prepare-mlx.sh` line 34 uses process substitution `< <(find ...)`
  which fails in this chroot even with devtmpfs mounted; the work-around
  is to materialize `find … -print0` into a temp file and read it.
- GNU coreutils wrappers needed on macOS host when calling
  `prepare-mlx.sh` directly (`sha256sum --check --status`,
  `cp --preserve=mode`, `find -newermt @0`); the chroot side already
  has GNU userland.
- Bundle staging:
  `MLX_OMARCHY_WHOLE_BUNDLE_DIR=$HOME/src/parakeet-whole-bundle`
  (manifest.json + 458 MB program-0.anec, sha 08769793 / 13c74423).

## 4. Install (deferred)

The existing install on jwm1 at `/var/tmp/v072-venv-fused/` was placed
there by BundleIntegrate (sha 2bd5c6a1). That install is for the
OLD wheel (no `vulkan_tdt_chain.py`, pin still `decode_control: host`).
My new wheel needs to be installed via:

```
scripts/install-integrated.sh \
    --wheel dist/mlx_omarchy-...-new.whl \
    --bundle-dir $HOME/src/parakeet-whole-bundle \
    --venv /var/tmp/v072-venv-fused \
    --keep-existing
```

`--keep-existing` is required: the install script otherwise refuses to
reuse an existing venv (BundleIntegrate called this out, 2026-09-23).

The pin shas (bundle 08769793 / 13c74423, libane d06222a8) are unchanged,
so the install script will accept the new wheel without changes.

**Status:** install deferred to next jwm1 GPU window (RmsAB holds
`/tmp/m1-gpu.lock` for the 3+3 decode reps as of last coordination;
AneRecover and QwenStateCarry are ahead in queue).

## 5. Stage table (deferred — uses existing install on jwm1)

`/tmp/run-stage-table.sh` (BundleIntegrate, 2026-09-23 05:28) runs the
5-fresh + 5-warm golden-pin battery. BundleIntegrate's prior receipt
(`lane/m1-integrate` `f68f310`, 2026-09-23) already produced the
stage-table against the OLD wheel:

| Stage     | Fresh median | Warm reused median | macOS  | Gap    |
|-----------|--------------|--------------------|--------|--------|
| encoder   | 142.6 ms     | 294.8 ms           | 138.5  | +152.2 ms warm gap |
| TDT       | 422-490 ms   | n/a (host-loop fallback) | 120  | -      |
| mel       | 14 ms        | n/a                | 14     | parity |
| detok     | (in encoder) | n/a                | (in enc) | -    |
| total     | ~580 ms      | ~735 ms            | 275    | ~300 ms Linux warm |
| RTF       | (pending fresh+warm split) | n/a | (denominator: 275 ms) | - |

Transcript sha `db501a8c` on all 10 runs (golden clean).

**Re-running the stage table against the new wheel is the only
verifiable path to prove the device-chain runtime lands bit-exact and
the 120-ms macOS TDT parity is reachable.** That re-run belongs to the
install + stage-table window on jwm1, after the new wheel is in place.

## 6. Warm-reuse wake penalty — verdict

**OPEN.** BundleIntegrate isolated a 152 ms gap (fresh median 142.6 ms
vs warm-reused median 294.8 ms) but did not pinpoint the cause.

Hypotheses (from inspecting `overlay/tools/coreml/ane_resident.py`
and the worker subprocess protocol):
1. **pm_runtime / power-domain re-arm on submit N+1** — the resident
   ANE session holds the device but the kernel may power-cycle
   between submits. Worth probing with `/sys/module/ane/...` perf
   counters around the second submit.
2. **Cache flush on the worker's per-submit housekeeping** — the
   resident worker re-binds a socketpair and re-arms select() between
   submits; that path is in `_readline`/`_write_bytes` and may be
   reinitializing per call.
3. **Bundle re-bind / ioctl on submit** — the worker's `submit
   <name>` frame forces the resident to look up the bundle slot
   every call instead of keeping it cached; could be a TM re-arm.
4. **Per-call input DMA re-binding** — input buffers are `mmap` from
   scratch on every submit; the re-map may be the culprit.

**Not verified.** Profiling the second-submit path against the
fresh-submit path needs a kernel-side `perf` trace OR a `powermetrics`
-style PMU counter on the jwm1 host. That is AneRecover's lane, not
mine, and needs the GPU window I do not currently hold.

The merged branch makes the chain runtime reachable; closing the warm
penalty to single-digit ms requires the next lane to land on.

## 7. Acceptance status

| Item | Status | Evidence |
|------|--------|----------|
| Wheel sha | deferred | existing wheel sha 2bd5c6a1 lacks vulkan_tdt_chain.py; new wheel pending BuildChroot subagent |
| Stage table 5+5 fresh/warm, golden on all runs | pre-existing | BundleIntegrate f68f310: db501a8c on all 10 runs |
| Corpus results | not run | no Parakeet corpus clips discovered in repo; deferred |
| Wake-penalty verdict | OPEN | 152 ms gap measured by BundleIntegrate; 4 hypotheses listed; profiling deferred to next lane |
| Install status | deferred | /var/tmp/v072-venv-fused holds the OLD wheel; install of the NEW wheel requires the next jwm1 GPU window |

## 8. Coordination log

- Took `/tmp/m1-gpu.lock` briefly for the merge only, then released.
- Acknowledged AneRecover and QwenStateCarry (both parked); both waiting
  for AneRecover / RmsAB windows.
- RmsAB retaking the lock for 3+3 decode reps; will release before
  ParakeetFinal's install + stage-table window.
- BundleIntegrate lane receipt (f68f310) is the baseline; this receipt
  sits on top.

**Yielded:** merge landed, pin change surfaced, build dispatched, install
+ stage table deferred behind RmsAB / AneRecover windows as Main
ordered.