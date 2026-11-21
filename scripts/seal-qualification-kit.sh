#!/usr/bin/env bash
# Seal the qualification kit as a standalone archive.
#
# The archive holds the kit, the vendored mlx-lm patch series and the
# script that applies it, so a Linux run works from the archive alone:
# the kit looks for the series beside itself and finds it here. Without
# the series the run would fail in setup with
# mlx_lm_patch_series_missing.
#
# Usage: seal-qualification-kit.sh <out-dir> [tag]
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out=${1:?usage: seal-qualification-kit.sh <out-dir> [tag]}
tag=${2:-$(git -C "$root" describe --tags --always --dirty 2>/dev/null || echo untagged)}
name="omarchy-mlx-qualification-kit-$tag"
stage="$out/$name"
mkdir -p "$stage/scripts" "$stage/patches"
cp -R "$root/qualification-kit" "$stage/qualification-kit"
cp "$root/scripts/apply-mlx-lm-patches.sh" "$stage/scripts/"
cp -R "$root/patches/." "$stage/patches/"
cp "$root/LICENSE" "$stage/LICENSE" 2>/dev/null || true
cd "$stage"
find . -type f -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
find . -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
cd "$out"
tar -cf "$name.tar" "$name"
sha256sum "$name.tar" > "$name.tar.sha256"
printf 'archive: %s\n' "$out/$name.tar"
cat "$out/$name.tar.sha256"
