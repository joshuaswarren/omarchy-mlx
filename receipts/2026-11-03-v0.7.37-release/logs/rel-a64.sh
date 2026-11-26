#!/usr/bin/env bash
# v0.7.37 aarch64 assets from a clean detached checkout at the pin: the cp314 wheel (scripts/build-wheel.sh), the fixed-name copy, the vendored
# wheel set tar and its sha256 sidecar. Adapted from w7Q's v0.7.36 build and vendor scripts (two steps there, one row here). Needs network.
# usage: [REL_BUNDLE=dir] rel-a64.sh [B]    (B = output dir, default ~/scratch/w72/rel-0737-a64; it must not exist)
set -euo pipefail
PIN=06d1038d2aa26ed235e65d2aa68677b9e904d65b
TAG=v0.7.37
B="${1:-$HOME/scratch/w72/rel-0737-a64}"
D="omarchy-mlx-vendor-wheels-$TAG-cp314-aarch64"
BUNDLE="${REL_BUNDLE:-$HOME/scratch/w7q-sp/bundle}"
[ ! -e "$B" ] || { echo "refusing: $B already exists (start from a fresh directory)"; exit 2; }
[ -f "$BUNDLE/manifest.json" ] || { echo "pinned ANE bundle missing at $BUNDLE"; exit 2; }
command -v python3.14 > /dev/null || { echo "python3.14 not found"; exit 2; }
mkdir -p "$B"
echo "start $(date -u +%T) load $(cut -d' ' -f1 /proc/loadavg) free $(df -BG --output=avail "$HOME" | tail -1)"
git clone -q --no-checkout https://github.com/joshuaswarren/omarchy-mlx.git "$B/wt"
cd "$B/wt"
git checkout -q --detach "$PIN"
[ "$(git rev-parse HEAD)" = "$PIN" ] || { echo "head is not the pin"; exit 2; }
[ -z "$(git status --porcelain)" ] || { echo "worktree is not clean"; exit 2; }
echo "checkout ok $(git rev-parse --short=8 HEAD)"
export MLX_OMARCHY_WHOLE_BUNDLE_DIR="$BUNDLE" DEV_RELEASE=1 MLX_OMARCHY_SOURCE_COMMIT="${PIN:0:8}"
nice -n 19 ./scripts/build-wheel.sh > "$B/build-wheel.log" 2>&1 || { echo "build-wheel failed"; tail -8 "$B/build-wheel.log" | cut -c1-170; exit 3; }
echo "build-wheel ok $(date -u +%T)"
set -- dist/*cp314*linux_aarch64.whl
[ "$#" -eq 1 ] && [ -f "$1" ] || { echo "expected exactly one cp314 aarch64 wheel in dist"; exit 3; }
WHEEL="$B/$(basename "$1")"
python3 scripts/check-version-stability.py .work/mlx
cp "$1" "$WHEEL"
cp "$WHEEL" "$B/omarchy-mlx-wheel-cp314-aarch64.whl"
(cd "$B" && sha256sum "$(basename "$WHEEL")" omarchy-mlx-wheel-cp314-aarch64.whl | tee wheel.sha256)
# the system python3.14 has no pip, and vendor-wheels.sh runs `python3.14 -m pip`: a fresh venv goes first on PATH
python3.14 -m venv "$B/vvenv"
PATH="$B/vvenv/bin:$PATH" nice -n 19 packaging/vendor-wheels.sh --out "$B/vendor/$D" --wheel "$WHEEL"
tar -cf "$B/$D.tar" -C "$B/vendor" "$D"
(cd "$B" && sha256sum "$D.tar" > "$D.tar.sha256" && cat "$D.tar.sha256")
ls -l "$B/$D.tar" "$B/$D.tar.sha256" | cut -c1-160
echo BUILD_VENDOR_OK > "$B/BUILD_VENDOR_OK"
echo "done $(date -u +%T)"
