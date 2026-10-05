# Golden base build — one consistent build base for every M2 lane

Date: 2026-10-05. Actor: GoldenBuild. Host: the T6021 machine (aarch64,
Python 3.14, btrfs root on /var/tmp). Public receipt; the private twin lives
in the lab notebook under entries/GoldenBuild/ and
artifacts/GoldenBuild/2026-10-05-golden-wheel/.

## What exists on the build host after this run

| Path | What |
|---|---|
| `/var/tmp/golden-wheel` | pristine `git clone --branch main` of origin/main + `.work` (staged upstream tree, build venv, build objects) |
| `/var/tmp/golden-wheel/dist` | exactly one wheel: the golden wheel |
| `/var/tmp/golden-venv` | golden venv: wheel + deps + mlx-lm 0.31.3 + `apply-mlx-lm-patches.sh` |
| `/var/tmp/golden-clone-venv.sh <dest>` | reflink-clone the venv, rewrite shebangs/activate*, verify import + libmlx.so sha |
| `/var/tmp/golden-clone-tree.sh <dest>` | reflink-clone the build tree, TEXT-ONLY path transplant with mtime preservation, no-op probe (`make -q`, and make-plan equality with the golden tree) |
| `/var/tmp/golden-wheel/golden_rebuild.sh` | incremental rebuild: sync overlay delta into the staged tree, touch, re-wheel |
| `/var/tmp/golden-wheel.base.tar` | cold baseline of the post-build tree (dist excluded) |
| `/var/tmp/golden-build-summary.txt` | stage-by-stage receipts (shas, timings) |

Build-time sha of origin/main: `60f80d2a066c46441bf9b47170ec951b33a5feb5`
Golden wheel: `mlx_omarchy-0.32.4.dev202610050436+60f80d2-cp314-cp314-linux_aarch64.whl` (sha256 `902128978b7508533bec23c4637a68d7af267a716d381c6d5388acff9ff94866`)
Base tar sha256: `20179f4d63728671f8d8de962b3d34ce287876b5611a1c82be50ec0641618056`
Full build wall: `141` s · incremental rebuild (1 overlay
file): `21` s · tree clone + transplant: `2`
s · venv clone: `0-1` s · reflink `cp -a` of the venv:
`<0.1 (venv clone via cp --reflink=always measured 0-1 s end-to-end; /usr/bin/time absent on the build host, timing line logged as garbage — capability evidenced by the clone wall times)` s. Smoke: device_info keys + 2B d64 greedy digest
`76ec87cb6ceda7113ac15b6e3be364f89e7aef76c559bd1f1925083d86cf55f3` (pin `76ec87cb…` from receipts/2026-10-04-m2-perf-ledger) —
`PASS (108.23 tok/s median vs ledger 108.0/108.2; model loaded from the local SiddhJagani/Qwen3.8-2B-mlx-4Bit snapshot — the path the ledger itself used; `mlx-community/Qwen3.8-2B-mlx-4Bit` does not resolve)`.

## The recipe (what the orchestrator ran)

```bash
# 0. preflight: builders idle, gpu-turn quiet, >=25G free
pgrep -af '[c]c1plus|[c]make|[n]inja' && echo BUSY
gpu-turn --status

# 1. fresh clone, pinned
git clone --branch main https://github.com/joshuaswarren/omarchy-mlx.git /var/tmp/golden-wheel
git -C /var/tmp/golden-wheel rev-parse HEAD   # recorded before building

# 2. whole-encoder bundle: /var/tmp/encoder-whole/bundle or ~/bundles/parakeet-whole,
#    verified against overlay/.../parakeet-runtime-pin.json BEFORE building
#    (manifest 08769793…, program 13c74423…). NEVER the overlay pin dir itself —
#    staging the reference copy aborts on manifest-sha mismatch by design.

# 3. build (build-wheel.sh owns CMAKE_ARGS; -j8, niced, HOME explicit)
cd /var/tmp/golden-wheel && nice -n 10 env HOME=~ DEV_RELEASE=1 \
  CMAKE_BUILD_PARALLEL_LEVEL=8 \
  MLX_OMARCHY_WHOLE_BUNDLE_DIR=/var/tmp/encoder-whole/bundle \
  bash scripts/build-wheel.sh     # rc must be 0 BEFORE any venv work

# 4. golden venv (install.sh contract: python -m pip everywhere, never cp -a)
python3 -m venv --clear /var/tmp/golden-venv
/var/tmp/golden-venv/bin/python -m pip install -U pip
/var/tmp/golden-venv/bin/python -m pip install dist/mlx_omarchy-*.whl
/var/tmp/golden-venv/bin/python -m pip install --no-deps mlx-lm==0.31.3   # pulls upstream mlx otherwise
/var/tmp/golden-venv/bin/python -m pip install \
  "transformers[sentencepiece]==5.16.1" numpy protobuf pyyaml jinja2 huggingface_hub soundfile
MLX_OMARCHY_CONV_RING=0 bash scripts/apply-mlx-lm-patches.sh /var/tmp/golden-venv
```

## How a lane builds incrementally (the only supported paths)

**Path A — own a clone of the golden tree (recommended):**

```bash
/var/tmp/golden-clone-tree.sh /var/tmp/<lane>-wheel     # reflink + text-only transplant
cd /var/tmp/<lane>-wheel
# edit overlay/... (your branch's files; `git diff --name-only origin/main...branch`)
bash golden_rebuild.sh overlay/mlx/backend/omarchy/your_file.cpp   # explicit-touch form
# or, with no args: re-syncs overlay/ into .work/mlx and re-wheels
```

**Path B — apply a branch delta onto the golden tree itself:** copy the
branch's overlay files into the staged tree and touch them — what
`golden_rebuild.sh` automates. rsync with preserved mtimes alone is NOT
enough (2026-09-02 incident: freshly edited overlay landed older than object
files; ninja skipped the recompile and green suites tested stale binaries).

**Venv per lane:** `/var/tmp/golden-clone-venv.sh /var/tmp/<lane>-venv`.
NEVER `cp -a` a venv by hand: shebangs and `activate*` keep the source path
and `pip` then writes into the wrong venv (the shared one — KernelBattery
did this twice on 2026-10-05). The clone script rewrites regular files
only, skips symlinks (never `sed -i` a symlinked interpreter), and refuses
success unless `mx.__file__` is under the destination and the installed
`libmlx.so` sha equals the wheel's.

## Hazards baked into the kit (each one bit a lane already)

1. **`prepare-mlx.sh` re-extracts the pinned tarball and `rm -rf`s the staged
   tree.** On a warmed tree it discards the object store and any non-overlay
   delta. It is for fresh trees only; incremental work goes through
   `golden_rebuild.sh` (or build-wheel.sh with the skip-prepare guard).
2. **`sed -i` on binaries corrupts them** (01:34Z: `libgguflib.a` "file too
   short" after a path transplant). Every transplant here is
   `grep -rlI` text-only; the clone scripts restore nothing — they never
   touch binaries.
3. **Build failures must abort before venv work** (01:31Z: a venv step ran on
   a failed build and surfaced a stale dist wheel). Orchestrator and
   rebuild script both gate on rc.
4. **Bundle provenance:** verify manifest+program shas against the runtime
   pin BEFORE the build; never point `MLX_OMARCHY_WHOLE_BUNDLE_DIR` at the
   overlay pin dir.
5. **Time math on a CDT host:** `-newermt 'UTC string'` matches local time —
   use `-mmin`.
6. **HOME explicit** in every non-interactive env (pip, HF cache, gpu-turn).
7. **`cp -a --reflink=always` can fail partway** on filesystems without
   reflinks; the clone scripts copy into a temp dir and `mv` only after a
   clean copy, so a dest is never half-populated.
8. **setup.py builds with the DEFAULT CMake generator (Unix Makefiles)** —
   there is no build.ninja anywhere even though ninja is installed. The
   cmake dir is `build/temp.*/mlx.core/` (never `build/` itself), and
   FetchContent `_deps/*-subbuild` dirs carry their own CMakeCache.txt, so
   locate the real one by `-path '*mlx.core/CMakeCache.txt'`, not depth.
9. **`make -q` is 1 on the golden tree itself** (cmake's self-regen edges
   are pending on every first `make`). The clone no-op criterion is
   therefore plan equality: `make -n` output, path-normalized, must be
   identical to the golden tree's — which the clone script asserts.
10. **`sed -i` bumps mtimes on transplanted build files** (Makefile,
    flags.make, .d): without restoring the original mtime, make stales the
    whole clone. The clone script stats before sed and `touch -d` after.
11. **rsync's default size+mtime check re-copies overlay files whose only
    difference is mtime** (checkout time vs prepare-mlx touch time): a
    no-arg incremental rebuild touched 420 files and would have recompiled
    the world. Use `rsync -a -c` and touch only what transferred.
12. **The 2B digest pin loads from the local
    `models--SiddhJagani--Qwen3.8-2B-mlx-4Bit` snapshot** (that is the path
    the ledger used); the `mlx-community/` hub id does not resolve and
    burns ticket time on a 401.

## Rebuilding a corrupted lane tree

```bash
rm -rf /var/tmp/<lane>-wheel && /var/tmp/golden-clone-tree.sh /var/tmp/<lane>-wheel
# nuclear option (works even if the golden tree itself is corrupted):
tar -xf /var/tmp/golden-wheel.base.tar -C /var/tmp   # restores golden-wheel
```

If the GOLDEN tree is corrupted: restore from the base tar, delete the
affected `.golden-status` markers, and rerun `golden_build.sh` — completed
stages keep their markers, so only the damaged stages rerun.

## Reproducing from zero (no golden artifacts at all)

Clone → verify bundle → build-wheel.sh as in the recipe above; then
`tar -cf` the post-build tree (minus dist) before the first lane clones.
The whole run is scripted: `golden_build.sh` (marker-gated stages) +
`golden_smoke_2b.sh` (one gpu-turn ticket, 2B d64 digest against the ledger
pin).
