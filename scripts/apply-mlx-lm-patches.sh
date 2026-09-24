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
if [[ "${MLX_OMARCHY_CONV_RING:-0}" == 1 ]]; then
  apply mlx-lm-convring.patch
else
  echo "conv-ring: OFF (set MLX_OMARCHY_CONV_RING=1 to enable)"
fi
