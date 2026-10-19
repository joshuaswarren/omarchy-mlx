#!/usr/bin/env bash
# test_audio_gate_skip.sh: the audio branches of apply-mlx-lm-patches.sh against
# minimal fake venvs built from the licensed 0.5.6 fixture: (1a) a comment that
# mentions the gate on a pristine module must NOT satisfy the skip (the guard
# lands); (1b) a drifted patched module (marker present, neither hunk fits)
# must exit nonzero and leave no completion record; (2) a genuinely patched
# module skips; (3) a pristine module takes the patch and a second run is
# idempotent. Needs MLX_LM_TREE pointing at a site-packages with a stock
# mlx-lm 0.31.3 (the series the script applies without version mismatch).
set -euo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"
F="$R/tests/fixtures/mlx_audio_056/depthwise_conv1d_kernel.py"
P="$R/patches/mlx-audio-mossformer-depthwise-metal-gate.patch"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
fail=0
mk() {
  rm -rf "$T/v"
  mkdir -p "$T/v/lib/python3.11/site-packages/mlx_audio/sts/models/mossformer2_se"
  SRC="${MLX_LM_TREE:?set MLX_LM_TREE to a site-packages with stock mlx-lm 0.31.3}"
  cp -a "$SRC/mlx_lm" "$T/v/lib/python3.11/site-packages/"
  cp -a "$SRC"/mlx_lm-*.dist-info "$T/v/lib/python3.11/site-packages/"
  M="$T/v/lib/python3.11/site-packages/mlx_audio/sts/models/mossformer2_se/depthwise_conv1d_kernel.py"
  case $1 in
    comment) cat "$F" > "$M"; printf '# TODO: add mx.metal.is_available() gate upstream\n' >> "$M" ;;
    drifted) cat "$F" > "$M"
             patch --silent --batch --forward -p1 --fuzz=0 \
               -d "$T/v/lib/python3.11/site-packages" < "$P"
             sed -i 's/and mx.metal.is_available()/and mx.metal.is_available()  # drifted/' "$M" ;;
    patched) cat "$F" > "$M"
             patch --silent --batch --forward -p1 --fuzz=0 \
               -d "$T/v/lib/python3.11/site-packages" < "$P" ;;
    pristine) cat "$F" > "$M" ;;
  esac
}
run() { set +e; bash "$R/scripts/apply-mlx-lm-patches.sh" "$T/v" >"$T/log" 2>&1; RUN_RC=$?; set -e; }
REC="$T/v/lib/python3.11/site-packages/mlx_lm/.omarchy-serve-patches"
GUARD="$T/v/lib/python3.11/site-packages/mlx_audio/sts/models/mossformer2_se/depthwise_conv1d_kernel.py"

echo "== 1a: comment marker on a pristine module (guard absent)"
mk comment; run
[ "$RUN_RC" = 0 ] || { echo "FAIL: rc $RUN_RC (the forward patch should apply)"; fail=1; }
if grep -q 'Metal gate applied' "$T/log" && grep -q 'and mx.metal.is_available()' "$GUARD"; then
  echo "ok: the comment did not satisfy the skip; the real guard landed"
else
  echo "FAIL: comment-only marker satisfied the skip or the guard is absent"; fail=1
fi

echo "== 1b: drifted patched module (marker present, neither hunk fits)"
mk drifted; run
if [ "$RUN_RC" = 0 ]; then echo "FAIL: rc 0 on a drifted module"; fail=1
elif [ -e "$REC" ]; then echo "FAIL: completion record written on failure"; fail=1
elif grep -q 'did not apply exactly' "$T/log"; then echo "ok: loud refusal, no record"
else echo "FAIL: wrong failure text"; tail -4 "$T/log"; fail=1; fi

echo "== 2: genuinely patched module (exact upstream hunk)"
mk patched; run
[ "$RUN_RC" = 0 ] || { echo "FAIL: rc $RUN_RC on the patched module"; fail=1; }
if grep -q 'already present (exact upstream hunk)' "$T/log"; then echo "ok: exact-hunk skip"
else echo "FAIL: no exact-hunk skip line"; fail=1; fi
[ -e "$REC" ] || { echo "FAIL: no record after the clean run"; fail=1; }

echo "== 3: pristine module takes the patch; second run idempotent"
mk pristine; run
[ "$RUN_RC" = 0 ] || { echo "FAIL: rc $RUN_RC on pristine"; fail=1; }
if grep -q 'Metal gate applied' "$T/log" && grep -q 'and mx.metal.is_available()' "$GUARD"; then
  echo "ok: patch applied"
else echo "FAIL: patch not applied"; fail=1; fi
run
if [ "$RUN_RC" = 0 ] && grep -q 'already present (exact upstream hunk)' "$T/log"; then
  echo "ok: idempotent second run"
else echo "FAIL: second run not idempotent (rc $RUN_RC)"; fail=1; fi

[ "$fail" = 0 ] && echo "ALL OK" || { echo "FAILURES"; exit 1; }
