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
self="${BASH_SOURCE[0]}"
while [ -L "$self" ]; do
  link=$(readlink "$self")
  case "$link" in
    /*) self=$link ;;
    *) self=$(dirname "$self")/$link ;;
  esac
done
root="$(cd -- "$(dirname -- "$self")/.." && pwd)"
if [ ! -d "$root/qualification-kit" ]; then
  top=$(git -C "$(dirname -- "$self")" rev-parse --show-toplevel 2>/dev/null || true)
  [ -n "$top" ] && root="$top"
fi
out=${1:?usage: seal-qualification-kit.sh <out-dir> [tag]}
tag=${2:-$(git -C "$root" describe --tags --always --dirty 2>/dev/null || echo untagged)}
name="omarchy-mlx-qualification-kit-$tag"
stage="$out/$name"
mkdir -p "$stage/scripts" "$stage/patches"
cp -R "$root/qualification-kit" "$stage/qualification-kit"
cp "$root/scripts/apply-mlx-lm-patches.sh" "$stage/scripts/"
# The series calls these helpers by name; an archive without them fails on
# Linux at the first patch that needs one.
for helper in "$root"/scripts/patch-mlx-lm-*.py; do
  [ -e "$helper" ] || continue
  cp "$helper" "$stage/scripts/"
done
cp -R "$root/patches/." "$stage/patches/"
cp "$root/LICENSE" "$stage/LICENSE" 2>/dev/null || true
cd "$stage"
find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find . -name '*.pyc' -type f -delete 2>/dev/null || true
find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
cd "$out"
tar -cf "$name.tar" "$name"
sha256sum "$name.tar" > "$name.tar.sha256"
printf 'archive: %s\n' "$out/$name.tar"
cat "$out/$name.tar.sha256"
