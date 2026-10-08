#!/usr/bin/env bash
# Apply the vendored mlx-lm serve patches to a venv's mlx_lm package.
#
# GDN fast route and raw 9B decode route are ON by default. Set
# MLX_OMARCHY_GDN_RAW_REPEAT=0 to opt out of raw decode. The fast route sends
# gated-delta updates to mx.fast.gated_delta_update; the raw route adds T==1
# decode dispatch to mx.fast.gated_delta_update_raw. Both self-guard on
# hasattr, falling back to the upstream kernel when the entry point is absent.
# Greedy vocab prune: ON by default. Tied 4-bit/g64 lm_head decode steps
# go to mx.fast.greedy_quantized_argmax; the patch itself no-ops on any
# other head and MLX_OMARCHY_NO_GREEDY_PRUNE=1 restores the upstream step.
# GDN q/k scaled norm: ON by default (mx.fast.rms_norm_scaled, decode rows only).
# Conv-ring: OFF by default (decode-only experimental optimization);
# set MLX_OMARCHY_CONV_RING=1 to enable.
#
# Idempotent: an already-applied patch is reported and skipped.
set -euo pipefail
# The series applies with patch(1). A missing binary used to surface as a
# misleading "mlx-lm version mismatch" (patch's exit 127 fails every probe).
if ! command -v patch >/dev/null 2>&1; then
  echo "error: patch is not installed; install it (e.g. pacman -S patch) and rerun." >&2
  exit 6
fi
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
# A patch whose own reverse no longer matches can still be proven applied:
# a LATER patch of this series reverse-matching is content evidence the
# series already ran on this tree (later patches insert into the same
# regions and rewrite earlier patches' added lines, so exact reverse
# matching only survives for the last patch per region). No later reverse
# match means this tree never got that far and the loud error stands,
# which is what catches a wrong mlx-lm version.
SERIES_PATCHES=(
  mlx-lm-tool-call-arguments.patch
  mlx-lm-qwen3-coder-untyped-args.patch
  mlx-lm-max-tokens-min-one.patch
  mlx-lm-gated-delta-fast-route.patch
  mlx-lm-gated-delta-fast-route-repeat.patch
  mlx-lm-gated-delta-raw.patch
  mlx-lm-greedy-prune.patch
  mlx-lm-qwen35-qk-scaled.patch
  mlx-lm-qwen35-gdn-conv.patch
  mlx-lm-conv-silu.patch
  mlx-lm-qwen35-gated-norm.patch
  mlx-lm-ttft-early-submit.patch
  mlx-lm-convring.patch
  mlx-lm-last-logits.patch
  mlx-lm-last-logits-qwen3.patch
)
series_already_applied() {
  local name="$1" other i=0 j
  for j in "${SERIES_PATCHES[@]}"; do
    [[ "$j" == "$name" ]] && break
    i=$((i + 1))
  done
  for ((j = i + 1; j < ${#SERIES_PATCHES[@]}; j++)); do
    other="$ROOT/$SERIES/${SERIES_PATCHES[$j]}"
    [[ -f "$other" ]] || continue
    patch --dry-run --directory="$SITE" --strip=1 --reverse \
      < "$other" >/dev/null 2>&1 && return 0
  done
  return 1
}
apply() {
  local name="$1"
  if [[ ! -f "$ROOT/$SERIES/$name" ]]; then
    echo "error: patch file missing: $ROOT/$SERIES/$name" >&2
    exit 5
  fi
  local forward_rc=0
  patch --dry-run --directory="$SITE" --strip=1 --forward --fuzz=0 \
      < "$ROOT/$SERIES/$name" >/dev/null 2>&1 || forward_rc=$?
  if (( forward_rc == 127 )); then
    echo "error: patch(1) is not installed; install it (e.g. pacman -S patch) and rerun." >&2
    exit 6
  elif (( forward_rc == 0 )); then
    patch --directory="$SITE" --strip=1 --forward --fuzz=0 \
      < "$ROOT/$SERIES/$name"
    echo "applied: $name"
  elif patch --dry-run --directory="$SITE" --strip=1 --reverse \
      < "$ROOT/$SERIES/$name" >/dev/null 2>&1; then
    echo "already applied: $name"
  elif series_already_applied "$name"; then
    echo "already applied: $name (hunks rewritten by later patches in this series)"
  else
    echo "patch does not apply (mlx-lm version mismatch?): $name" >&2
    return 1
  fi
}
apply mlx-lm-tool-call-arguments.patch
# Qwen3 Coder untyped tool arguments (upstream #1910): without a top-level
# "type", only JSON objects/arrays parse; text and scalars stay strings.
apply mlx-lm-qwen3-coder-untyped-args.patch
# max_tokens min_val 1 (upstream #1935): the handler answers 400 for a
# max_tokens: 0 request instead of 200-then-generator-reject.
apply mlx-lm-max-tokens-min-one.patch
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
# Batched serve-path SDPA: BatchKVCache.make_mask returns an array even when no
# row is padded, which fails the native SDPA decode gate (no array mask), so
# every full-attention layer of a batched decode step composed. make_mask(1)
# now returns None when the host mirror proves the mask all-True; padded
# batches, prefill and outside writes keep the mask. Needs the ssm patch's
# `import os`. Kill switch MLX_OMARCHY_KV_MASKLESS=0.
if [[ "$SERIES" == "patches/mlx-lm-0.32" ]]; then
  python3 "$ROOT/scripts/patch-mlx-lm-kv-maskless.py" "$VENV"
  # Batched decode host joins: BatchKVCache.offset is a lazy device array, so the
  # omarchy RoPE gate synchronizes before every RoPE call to bound it. A host
  # mirror tracks every in-class update and the offset is stored as a host-built
  # array with the same values. Kill switch MLX_OMARCHY_KV_HOST_OFFSET=0.
  python3 "$ROOT/scripts/patch-mlx-lm-kv-host-offset.py" "$VENV"
else
  # Both patchers anchor on the mlx-lm 0.32 BatchKVCache API (prepare/finalize
  # right-padding mirrors, host-list filter); 0.31.3's older BatchKVCache
  # predates them (its filter even syncs via .min().item()) and upstream
  # reworked the class in 0.32, so there is nothing to port onto. The served
  # mlx-lm route is single-concurrency; a batched 0.31.3 user falls back to
  # the upstream array mask and lazy offset (slower, never wrong). See
  # docs/kernel-flags.md.
  echo "kv-maskless/kv-host-offset: 0.32 series only; skipped on $SERIES"
fi
apply mlx-lm-greedy-prune.patch
# Greedy GenerationBatch steps (BatchGenerator / oMLX) can take the pruned
# greedy head per row and leave logprobs lazy instead of projecting every row
# onto the full vocabulary. Opt-in (MLX_OMARCHY_BATCH_GREEDY=1): on jw16 it
# cost B=1 and in-process B=4 time; only oMLX c4 gained. Sampled batches and
# logits processors always keep the full step. 0.32 line only.
if [[ "$SERIES" == "patches/mlx-lm-0.32" ]]; then
  python3 "$ROOT/scripts/patch-mlx-lm-batch-greedy-head.py" "$VENV"
else
  # GenerationBatch does not exist on the 0.31 line (the patcher self-guards
  # too); the gate keeps the series log honest about what shipped.
  echo "batch-greedy-head: 0.32 series only (no GenerationBatch); skipped on $SERIES"
fi
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
# Same last-logits change for the dense qwen3 family (qwen3.py; the 4B is model_type qwen3):
# cached prefill computes the head for the final position only. 4B pf512 +9.0% on the M1
# Max (178.3 vs 163.5 tok/s, 3 interleaved pairs); token digests equal; last-position logits
# within one bf16 ULP and identical fp64-reference error (receipts/2026-10-08-qwen3-last-logits).
# MLX_OMARCHY_FULL_LOGITS=1 restores the full head; cache=None calls always keep it.
apply mlx-lm-last-logits-qwen3.patch
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
# the value-head count so the fused GatedDeltaUpdate kernel can fire instead
# of the composed per-token fallback (~700 small dispatches/token). Default
# ON on every chip; set MLX_OMARCHY_GDN_RAW_REPEAT=0 to opt out. The 9B
# route passes the updated bf16-ULP numerics gate; see docs/numerics-gate.md.
python3 "$ROOT/scripts/patch-mlx-lm-gdn-raw-repeat.py" "$VENV"
