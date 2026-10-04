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
# Two series: patches/ for mlx-lm 0.31.3 (every vendor lock), patches/mlx-lm-0.32/ for the 0.32 API
# line (mlx-lm 94cdcae, the commit oMLX pins). StopSequences is the 0.32 API marker.
SERIES="patches"
if grep -q "^class StopSequences" "$SITE/mlx_lm/generate.py"; then
  SERIES="patches/mlx-lm-0.32"
fi
echo "mlx-lm patch series: $SERIES"
if [[ ! -d "$ROOT/$SERIES" ]]; then
  # install.sh pins mlx-lm 0.31.3 and downloads only that series; a venv later moved to the 0.32 line needs a repo
  # checkout's copy of this script.
  echo "error: $SERIES is not in $ROOT; run scripts/apply-mlx-lm-patches.sh from a repo checkout" >&2
  exit 5
fi
if [[ "${MLX_OMARCHY_CONV_RING:-0}" == 1 && "$SERIES" != "patches" ]]; then
  echo "error: conv-ring has no mlx-lm 0.32 port (experimental, off by default); unset MLX_OMARCHY_CONV_RING" >&2
  exit 6
fi
apply() {
  local name="$1"
  if [[ ! -f "$ROOT/$SERIES/$name" ]]; then
    echo "error: patch file missing: $ROOT/$SERIES/$name" >&2
    exit 5
  fi
  if patch --dry-run --directory="$SITE" --strip=1 --forward --fuzz=0 \
      < "$ROOT/$SERIES/$name" >/dev/null 2>&1; then
    patch --directory="$SITE" --strip=1 --forward --fuzz=0 \
      < "$ROOT/$SERIES/$name"
    echo "applied: $name"
  elif patch --dry-run --directory="$SITE" --strip=1 --reverse \
      < "$ROOT/$SERIES/$name" >/dev/null 2>&1; then
    echo "already applied: $name"
  else
    echo "patch does not apply (mlx-lm version mismatch?): $name" >&2
    return 1
  fi
}
apply mlx-lm-gated-delta-fast-route.patch
# GDN fast-route Hk!=Hv repeat: Qwen3.5-9B (Hk=16, Hv=32) hit the composed
# per-token fallback without this; A/B 42 -> 317 tok/s prefill 512 on M2.
apply mlx-lm-gated-delta-fast-route-repeat.patch
apply mlx-lm-gated-delta-raw.patch
# Serve-path GDN prefill fallback fix: the batched route's ArraysCache sets
# left_padding=[0]*B even unpadded, so make_mask hands an all-True mask into
# gated_delta_update and the fused coopmat prefill loses its maskless gate
# (two-pass scan fallback, ~11x/layer on the 27B serve path). make_mask now
# returns None when the mask is provably all-valid (host-known mirror);
# padded batches keep the mask. Kill switch MLX_OMARCHY_SSM_MASKLESS=0.
python3 "$ROOT/scripts/patch-mlx-lm-ssm-maskless.py" "$VENV"
apply mlx-lm-greedy-prune.patch
# GDN q/k rms_norm + scalar multiply -> mx.fast.rms_norm_scaled (decode-sized rows,
# bf16, self-guarded on hasattr; bit-identical to the composed pair on jwm1: 7fe6badf
# digest unchanged, decode +2.1%). The gated-norm site is NOT shipped: it diverges.
apply mlx-lm-qwen35-qk-scaled.patch
# GDN decode conv -> mx.fast.gdn_conv_update (state concat + carry folded into the conv
# kernel; decode S==1, bf16, self-guarded on hasattr): bit-identical on jwm1 (decode64/128/256
# digests 7fe6badf/da5568ee/7d0523ae unchanged), decode +1.5-2.0%.
apply mlx-lm-qwen35-gdn-conv.patch
# GDN conv + silu epilogue: the decode fast route's nn.silu folds into the
# gdn_conv_update dispatch (activate=True; the wheel's GdnConvDecode kernel
# rounds conv out, sigmoid, and the product exactly like the composed
# sigmoid -> multiply chain). Removes the standalone Silu dispatch per GDN
# layer (18/token). Requires the activate kwarg (landed with the same
# wheel); older wheels fail loudly at the first decode step.
apply mlx-lm-conv-silu.patch
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
# Last-logits prefill: cached (incremental) calls compute the quantized head only
# for the final prompt position (448 ms of the T=2048 prefill on the M1 Max was
# the head over 2048 positions no consumer reads). Decode steps are untouched
# (greedy fast path never routes through this call). Whole-sequence calls
# (cache=None: scoring/training) and MLX_OMARCHY_FULL_LOGITS=1 keep full logits.
# Bit-exact: records digest and per-digest gates unchanged (Jw16PrefillGap3).
apply mlx-lm-last-logits.patch
# Attention q/k RMSNorm -> mx.fast.rope_rms_norm (the q_norm/k_norm + rope
# chain folds into one dispatch per tensor; the wheel's FastRopeNorm kernel
# reproduces the fast RMSNorm reduction and rounds to bf16 before the
# rotation, bit-identical to the composed chain). Default ON since the
# DecodeFuse4 combined land (kill switch =0). Python patcher:
# GNU patch 2.8 on this host fails byte-verified hunks (DecodeFuse3).
python3 "$ROOT/scripts/patch-mlx-lm-rope-norm.py" "$VENV"
# GDN q/k rms_norm_scaled -> gdn_conv_update epilogue (F4): the conv
# dispatch carries the per-head norm pair (fast_norm_gated mode-1
# semantics in the gdn_conv_decode kernel epilogue), bit-identical to the
# composed chain. Default ON since the DecodeFuse4 combined land
# (kill switch =0). Python patcher (GNU patch 2.8 on this host
# fails byte-verified hunks, DecodeFuse3).
python3 "$ROOT/scripts/patch-mlx-lm-qknorm.py" "$VENV"
# Qwen3 dense rope-norm fold (qwen3.py attention site; the F3 fold ported
# to the dense file code shape). Default ON since 2026-10-04: bit-exact,
# 4B decode +4.18..+4.86% across d64-d512 (receipts/
# 2026-10-04-jw16-decode-dispatchfuse.md). Kill switch
# MLX_OMARCHY_ROPE_NORM_FUSE=0.
python3 "$ROOT/scripts/patch-mlx-lm-qwen3-rope-norm.py" "$VENV"
# GDN raw-decode GQA repeat (Hk<Hv models, e.g. Qwen3.5-9B): expands q/k to
# the value-head count so the fused GatedDeltaUpdate kernel fires instead of
# the composed per-token fallback (~700 small dispatches/token). Default ON
# since 2026-10-04: 9B decode +31..35% with the order-matched kernel
# (free-run greedy identity 100%, TF agreement 99.39%, per-op fp64 error
# identical to composed; receipts/2026-10-04-jw16-decode-dispatchfuse.md
# addenda 2-3). Kill switch MLX_OMARCHY_GDN_RAW_REPEAT=0.
python3 "$ROOT/scripts/patch-mlx-lm-gdn-raw-repeat.py" "$VENV"
