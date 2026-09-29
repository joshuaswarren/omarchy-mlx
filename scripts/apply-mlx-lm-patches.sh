#!/usr/bin/env bash
# Apply the vendored mlx-lm serve patches to a venv's mlx_lm package.
#
# GDN fast route + GDN raw route: ON by default. The fast route sends
# gated-delta updates to mx.fast.gated_delta_update; the raw route adds the
# T==1 decode dispatch to mx.fast.gated_delta_update_raw (measured on
# t8103: decode 17.46 -> 36.37 tok/s, pin dbf704971617fdfc identical to
# t6001). Both self-guard on hasattr, falling back to the upstream kernel
# when the entry point is absent.
# Greedy vocab prune: ON by default. Tied 4-bit/g64 lm_head decode steps
# go to mx.fast.greedy_quantized_argmax; the patch itself no-ops on any
# other head and MLX_OMARCHY_NO_GREEDY_PRUNE=1 restores the upstream step.
# GDN q/k scaled norm: ON by default (mx.fast.rms_norm_scaled, decode rows only).
# Conv-ring: OFF by default (decode-only experimental optimization);
# set MLX_OMARCHY_CONV_RING=1 to enable.
#
# Idempotent: an already-applied patch is reported and skipped.
set -euo pipefail
VENV="${1:?usage: apply-mlx-lm-patches.sh /path/to/venv}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Layout 1 (installed by install.sh): script and patches/ in $PREFIX.
# Layout 2 (repo checkout): script in scripts/, patches/ one level up.
if [[ -d "$SCRIPT_DIR/patches" ]]; then
  ROOT="$SCRIPT_DIR"
elif [[ -d "$SCRIPT_DIR/../patches" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
  echo "error: patches/ not found beside $SCRIPT_DIR or in $SCRIPT_DIR/.." >&2
  exit 4
fi
SITE="$(dirname "$(ls -d "$VENV"/lib/python3.*/site-packages/mlx_lm 2>/dev/null | head -n 1 || true)")"
[[ -d "$SITE/mlx_lm" ]] || { echo "mlx_lm not found under $VENV" >&2; exit 3; }
apply() {
  local name="$1"
  if [[ ! -f "$ROOT/patches/$name" ]]; then
    echo "error: patch file missing: $ROOT/patches/$name" >&2
    exit 5
  fi
  if patch --dry-run --directory="$SITE" --strip=1 --forward --fuzz=0 \
      < "$ROOT/patches/$name" >/dev/null 2>&1; then
    patch --directory="$SITE" --strip=1 --forward --fuzz=0 \
      < "$ROOT/patches/$name"
    echo "applied: $name"
  elif patch --dry-run --directory="$SITE" --strip=1 --reverse \
      < "$ROOT/patches/$name" >/dev/null 2>&1; then
    echo "already applied: $name"
  else
    echo "patch does not apply (mlx-lm version mismatch?): $name" >&2
    return 1
  fi
}
apply mlx-lm-gated-delta-fast-route.patch
apply mlx-lm-gated-delta-raw.patch
apply mlx-lm-greedy-prune.patch
# GDN q/k rms_norm + scalar multiply -> mx.fast.rms_norm_scaled (decode-sized rows,
# bf16, self-guarded on hasattr; bit-identical to the composed pair on jwm1: 7fe6badf
# digest unchanged, decode +2.1%). The gated-norm site is NOT shipped: it diverges.
apply mlx-lm-qwen35-qk-scaled.patch
# GDN decode conv -> mx.fast.gdn_conv_update (state concat + carry folded into the conv
# kernel; decode S==1, bf16, self-guarded on hasattr): bit-identical on jwm1 (decode64/128/256
# digests 7fe6badf/da5568ee/7d0523ae unchanged), decode +1.5-2.0%.
apply mlx-lm-qwen35-gdn-conv.patch
# GDN gated norm -> mx.fast.rms_norm_gated (rms_norm + silu(gate) * x in one dispatch, decode-sized
# rows only, bf16, self-guarded on hasattr). Requires the wheel's bit-exact FastNormGatedOnly kernel
# (precise product associations; verified 0 mismatches vs the composed chain over an exhaustive bf16
# gate sweep). Measured on jwm1: decode64/128 +2.4%/+1.7% with identical digests.
apply mlx-lm-qwen35-gated-norm.patch
# Early first submit around prompt processing + the first token (sets
# MLX_OMARCHY_BATCH_FIRST=128 for those graphs only, cleared after the first token;
# backend ignores nothing else). Scheduling only: digests identical; jwm1 TTFT -12%,
# pipelined decode unchanged. MLX_OMARCHY_NO_TTFT_EARLY_SUBMIT=1 disables at runtime.
apply mlx-lm-ttft-early-submit.patch
if [[ "${MLX_OMARCHY_CONV_RING:-0}" == 1 ]]; then
  apply mlx-lm-convring.patch
else
  echo "conv-ring: OFF (set MLX_OMARCHY_CONV_RING=1 to enable)"
fi
