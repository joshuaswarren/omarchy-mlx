#!/bin/bash
# build-wheels.sh SHA... : on the 16-inch (the ledger's build host), build one omarchy-mlx wheel per commit into build/dist-bis/. CPU only.
# Same recipe and work dir as the daily ledger (DEV_RELEASE=1, shared ccache), so a rebuild after a small diff takes minutes.
set -euo pipefail
B=$HOME/.local/share/coreglass
S=$B/build
src=$S/omarchy-mlx-bis
mkdir -p "$S/dist-bis"
[ -d "$src/.git" ] || git clone -q --filter=blob:none https://github.com/joshuaswarren/omarchy-mlx "$src"
git -C "$src" fetch -q origin main
for sha in "$@"; do
  full=$(git -C "$src" rev-parse "$sha^{commit}")
  short=$(git -C "$src" rev-parse --short=7 "$full")
  if ls "$S/dist-bis/"mlx_omarchy-*"+$short-"*.whl >/dev/null 2>&1; then echo "have $short"; continue; fi
  git -C "$src" reset -q --hard "$full"
  echo "building $short ($(git -C "$src" log -1 --format=%s | cut -c1-70)) start $(date -u +%FT%TZ)"
  (cd "$src" && nice -n 10 env DEV_RELEASE=1 MLX_OMARCHY_WHOLE_BUNDLE_DIR="$B/whole-bundle" MLX_OMARCHY_WORK_DIR="$S/work" \
    bash scripts/build-wheel.sh) > "$S/wheel-bis-$short.log" 2>&1 || { echo "build $short FAILED"; tail -15 "$S/wheel-bis-$short.log"; exit 1; }
  mv -f "$src"/dist/mlx_omarchy-*.whl "$S/dist-bis/"
  ls "$S/dist-bis/"mlx_omarchy-*"+$short-"*.whl
  echo "done $short $(date -u +%FT%TZ)"
done
sha256sum "$S"/dist-bis/*.whl | cut -c1-80
