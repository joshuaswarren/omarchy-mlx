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

find "$ROOT/overlay" -type f -print0 > /tmp/overlay_files.tmp
while IFS= read -r -d '' file; do
  relative="${file#"$ROOT/overlay/"}"
  if [[ -e "$STAGING_DIR/$relative" ]]; then
    echo "overlay path already exists upstream: $relative" >&2
    exit 1
  fi
done < /tmp/overlay_files.tmp
rm -f /tmp/overlay_files.tmp

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
  < "$ROOT/patches/mlx-no-silent-cpu-fallback.patch"
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-eager-fusion.patch"
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-rope-settle-tape.patch"
# rope_rms_norm: the attention q/k RMSNorm folded into the rope dispatch
# (bit-identical; the omarchy backend reproduces the fast RMSNorm reduction
# inside the rope kernel).
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-rope-rms-norm.patch"
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
  < "$ROOT/patches/mlx-fast-bool-mask-floor.patch"
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-omarchy-metal-kernel.patch"
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-gated-delta-raw-gates.patch"
# B > 1 GDN prefill: run the batch row by row through the B = 1 fused route
# (a batched call used to take the per-token composed fallback).
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-gated-delta-prefill-rows.patch"
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
# Greedy vocab head (QmmVecGreedyBF16 backend kernel). The hunk sits far
# from the other patches' edits, but their insertions shift its line
# numbers, so this one is applied with fuzz 3 (context still verified).
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=3 \
  < "$ROOT/patches/mlx-fast-greedy-argmax.patch"
# Symmetric-int8 matmul with runtime-quantized activations
# (fast::Int8Matmul; the TensorFold MiniMax-H3 W8A8 path). Composed
# fallback in fast.cpp, Vulkan kernel via shaders/int8_matmul.comp.
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-fast-int8-matmul.patch"
# Bonsai 1-bit / 2-bit affine decode kernels
# (fast::BonsaiQ1AffineQmv, fast::BonsaiQmvWide, fast::BonsaiQ1Dequantize):
# omarchy-native Vulkan equivalents of oMLX custom_kernels/bonsai. The
# 1-bit uint8 Bonsai pack is not addressable through the qmm_vec.comp
# uint32 word reader, so these are dedicated shaders
# (shaders/bonsai_qmv_q1.comp, shaders/bonsai_qmv_wide.comp,
# shaders/bonsai_dequant_q1.comp). Composed fallback in fast.cpp.
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-fast-bonsai-qmv.patch"

# Backend-generic upstream fixes, applied in upstream first-parent order.
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-view-offset.patch"
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-eval-cleanup-deadlock.patch"
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-shared-buffer-reshape-contiguity.patch"
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-view-last-axis-stride.patch"
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-view-contiguity-flags.patch"
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-multioptimizer-empty-group.patch"
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-as-strided-contiguity.patch"
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-aligned-array-pointer.patch"
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-vmap-scatter-axis.patch"
# logcumsumexp: promote integer/bool inputs to a float dtype before the
# scan. The integer scan accumulated logaddexp in integer arithmetic (a
# cumulative maximum) and the GPU had no integer kernel; every other
# log-space op promotes.
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-logcumsumexp-float-promote.patch"
# CPU reductions over large arrays: widen offsets and sizes to int64 so a
# reduction past 2^31 elements reads the right rows.
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-reduce-large-offsets.patch"
# Sort/argsort on transposed GPU views picked rows from the wrong input
# row: collapse_contiguous_dims kept a leading size-1 axis as its own
# collapsed dim, so a transposed view never collapsed to the contiguous
# sort kernel and the sort ran over the wrong stride plan (upstream
# #4366; shared backend/common part).
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-sort-collapse-unsorted-axes.patch"
# vmap of an inverse real FFT to an odd length: the output length is not
# recoverable from the input shape, so carry an odd_out flag through the
# FFT primitive's vmap (upstream #4637).
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-fft-vmap-odd-irfft.patch"
# CPU compile cache: build under a unique temp name and rename into
# place, so concurrent processes never load a partly written lib*
# (upstream #4638).
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-cpu-compile-cache-race.patch"
# Dynamic-slice start arrays: read the indices contiguously (upstream
# #4599 CPU part; the cuda/metal hunks are out of scope).
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-slice-noncontig-start-cpu.patch"
# clip_grad_norm: sum float16 squares in float32 and cast after scaling,
# so fp16 overflow zeroes every gradient (upstream #4626).
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-clip-grad-norm-fp16.patch"
# CPU quantized matmul: accumulate the scalar (non-SIMD) full-row and
# k-outer kernels in float32 instead of the activation dtype. A bfloat16
# accumulator drifts several percent from the f64 dequant reference at
# K=16384 while each group dot stays correct; the SIMD variants already
# accumulate Simd<float, S>.
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-cpu-qmm-fp32-accum.patch"
# Pickle of a bfloat16 array passed strides = nullptr, so a non-contiguous
# (e.g. F-contiguous transposed) array unpickled with wrong strides
# (upstream #4649).
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-pickle-bf16-strides.patch"
# Python -> mlx conversion narrowed through float for int64/bool/float64
# list and scalar targets: 2**24+1 lost its odd bit going to int32/int64,
# 1e-50 became False for bool, and int lists to float64 cast via float32
# (upstream #4656).
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-python-dtype-narrowing.patch"
# Compile scalar merging merged a scalar that is also a compiled output:
# the output_map lookup then threw std::out_of_range (unordered_map::at)
# and compiling a function returning both a value and a constant scalar
# crashed (upstream #4658).
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-compile-scalar-output.patch"
# The .npy loader read the header shape without validation: a negative or
# overflowing dim from an untrusted file computed a wild total size
# (upstream #4657).
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-npy-shape-validate.patch"
# Integer floor_divide became one FloorDivide primitive instead of a
# Divide/Remainder/Subtract subgraph (upstream #4642); the omarchy backend
# implements FloorDivide::eval_gpu. Drop at the first pin at or past 2654664a3.
patch --directory="$STAGING_DIR" --strip=1 --forward --fuzz=0 \
  < "$ROOT/patches/mlx-floor-divide.patch"

rm -rf "$SOURCE_DIR"
mv "$STAGING_DIR" "$SOURCE_DIR"
trap - EXIT
printf '%s\n' "$SOURCE_DIR"

python3 - "$SOURCE_DIR" <<'PY'
import re
import sys
from pathlib import Path

source = Path(sys.argv[1])
header = (source / "mlx/fast.h").read_text()
bindings = (source / "python/src/fast.cpp").read_text()
refs = set(re.findall(r"&\s*(?:mx::)?fast::([A-Za-z_]\w*)|(?:mx::)?fast::([A-Za-z_]\w*)\s*\(", bindings))
names = {name for pair in refs for name in pair if name}
missing = sorted(name for name in names if not re.search(r"MLX_API\s+(?:array|std::vector<array>|CustomKernelFunction)\s+" + re.escape(name) + r"\s*\(", header))
if missing:
    print("prepare-mlx: missing mlx/fast.h declarations for: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
print("prepare-mlx: Python fast bindings have matching mlx/fast.h declarations")
PY
