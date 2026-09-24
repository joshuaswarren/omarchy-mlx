#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/mlx.lock"

WORK_DIR="${MLX_OMARCHY_WORK_DIR:-$ROOT/.work}"
ARCHIVE="$WORK_DIR/mlx-$MLX_VERSION-$MLX_COMMIT.tar.gz"
SOURCE_DIR="$WORK_DIR/mlx"
mkdir -p "$WORK_DIR"

if [[ ! -f "$ARCHIVE" ]]; then
  curl --fail --location --retry 3 --retry-all-errors \
    --output "$ARCHIVE" "$MLX_ARCHIVE_URL"
fi
printf '%s  %s\n' "$MLX_ARCHIVE_SHA256" "$ARCHIVE" | sha256sum --check --status

STAGING_DIR="$(mktemp -d "$WORK_DIR/.mlx.XXXXXX")"
cleanup() {
  rm -rf "$STAGING_DIR"
}
trap cleanup EXIT

tar --extract --gzip --file "$ARCHIVE" --strip-components=1 \
  --directory "$STAGING_DIR"

while IFS= read -r -d '' file; do
  relative="${file#"$ROOT/overlay/"}"
  if [[ -e "$STAGING_DIR/$relative" ]]; then
    echo "overlay path already exists upstream: $relative" >&2
    exit 1
  fi
done < <(find "$ROOT/overlay" -type f -print0)

# Copy modes and contents but NOT timestamps. `cp -a` preserved overlay
# mtimes, which silently dropped edits from builds: a freshly edited
# overlay file could land in the staging tree older than an existing
# object file, so ninja considered the object up to date and never
# recompiled it. That produced green suites testing code that was not in
# the binary, and cost several hours of misattributed failures on
# 2026-09-02. Staged files must always look newer than prior build output.
cp -r --preserve=mode "$ROOT/overlay/." "$STAGING_DIR/"
find "$STAGING_DIR" -newermt '@0' -exec touch {} +
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-build.patch"
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-eager-fusion.patch"
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-rope-settle-tape.patch"
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-python-package.patch"
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-io-device.patch"
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-export-dense-constants.patch"
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-linalg-gpu.patch"
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-python-buffer.patch"
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-gated-delta-mask.patch"
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-omarchy-quantize-errors.patch"
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-distributed-reduce-scatter-assert.patch"
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-fast-bool-mask-floor.patch"
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-omarchy-metal-kernel.patch"
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-gated-delta-raw-gates.patch"
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-version-time.patch"
# GDN decode chain: fused RMSNorm+SwiGLU-gate and RMSNorm+scalar-mul
# fast primitives (FastNormGatedBF16 backend kernel).
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-fast-rms-norm-gated.patch"
# GDN decode chain: conv1d with the state concatenation folded into
# the read (GdnConvDecodeBF16 backend kernel).
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-gdn-conv-decode.patch"

rm -rf "$SOURCE_DIR"
mv "$STAGING_DIR" "$SOURCE_DIR"
trap - EXIT
printf '%s\n' "$SOURCE_DIR"
