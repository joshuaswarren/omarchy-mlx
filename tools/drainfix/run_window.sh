#!/usr/bin/env bash
# DrainFix outer driver: gates, gpuwin window, service check, digest report. Not run inside a window.
# usage: run_window.sh w1|w2|w3|w4
set -euo pipefail
PLAN_NAME=${1:?usage: run_window.sh w1|w2|w3|w4}
ROOT=/var/tmp/drainfix
TOOLS=$ROOT/tools
BASE_PY=/var/tmp/v072-venv-fused/bin/python
CAND_PY=$ROOT/venv-cand/bin/python

case "$PLAN_NAME" in
  w1) INNER=w1_inner.sh; DF_PLAN="" ;;
  w2) INNER=ab_inner.sh; DF_PLAN="w2-pin-d128-ctl:128:ctl:1;w2-pin-d256-ctl:256:ctl:1;w2-n1-d64-neutral:64:neutral:1;w2-d64-r1-ctl:64:ctl;w2-d64-r1-on:64:on;w2-d64-r2-ctl:64:ctl;w2-d64-r2-on:64:on;w2-d64-r3-ctl:64:ctl;w2-d64-r3-on:64:on;w2-d64-r4-ctl:64:ctl;w2-d64-r4-on:64:on;w2-d64-r5-ctl:64:ctl;w2-d64-r5-on:64:on;w2-d128-r1-ctl:128:ctl;w2-d128-r1-on:128:on;w2-d128-r2-ctl:128:ctl;w2-d128-r2-on:128:on;w2-d128-r3-ctl:128:ctl;w2-d128-r3-on:128:on;w2-d256-r1-ctl:256:ctl;w2-d256-r1-on:256:on;w2-d256-r2-ctl:256:ctl;w2-d256-r2-on:256:on;w2-d256-r3-ctl:256:ctl;w2-d256-r3-on:256:on" ;;
  w3) INNER=ab_inner.sh; DF_PLAN="w3-d64-r1-on:64:on;w3-d64-r1-ctl:64:ctl;w3-d64-r2-on:64:on;w3-d64-r2-ctl:64:ctl;w3-d64-r3-on:64:on;w3-d64-r3-ctl:64:ctl;w3-d128-r1-on:128:on;w3-d128-r1-ctl:128:ctl;w3-d128-r2-on:128:on;w3-d128-r2-ctl:128:ctl;w3-d128-r3-on:128:on;w3-d128-r3-ctl:128:ctl;w3-d256-r1-on:256:on;w3-d256-r1-ctl:256:ctl;w3-d256-r2-on:256:on;w3-d256-r2-ctl:256:ctl;w3-d512-r1-on:512:on;w3-d512-r1-ctl:512:ctl;w3-d512-r2-on:512:on;w3-d512-r2-ctl:512:ctl;w3-d512-r3-on:512:on;w3-d512-r3-ctl:512:ctl;w3-d512-r4-on:512:on;w3-d512-r4-ctl:512:ctl;w3-d512-r5-on:512:on;w3-d512-r5-ctl:512:ctl;w3-d512-stack:512:stack:3" ;;
  w4) INNER=ab_inner.sh; DF_PLAN="w4-d64-r1-ctl:64:ctl;w4-d64-r1-on:64:on;w4-d64-r2-ctl:64:ctl;w4-d64-r2-on:64:on;w4-d128-r1-ctl:128:ctl;w4-d128-r1-on:128:on;w4-d128-r2-ctl:128:ctl;w4-d128-r2-on:128:on;w4-d256-r1-on:256:on;w4-d256-r1-ctl:256:ctl;w4-d256-r2-on:256:on;w4-d256-r2-ctl:256:ctl;w4-d256-r3-on:256:on;w4-d256-r3-ctl:256:ctl;w4-d512-r1-ctl:512:ctl;w4-d512-r1-on:512:on;w4-d512-r2-ctl:512:ctl;w4-d512-r2-on:512:on;w4-n-d64-neutral:64:neutral:1;w4-n-d512-neutral:512:neutral:1;w4-d512-stack20:512:stack20:3" ;;
  *) echo "unknown plan $PLAN_NAME"; exit 2 ;;
esac

test ! -e /var/tmp/JW16_MAINTENANCE
uptime_s=$(cut -d. -f1 /proc/uptime)
test "$uptime_s" -ge 360 || { echo "boot settle gate: ${uptime_s}s"; exit 1; }
load1=$(cut -d' ' -f1 /proc/loadavg)
load_int=${load1%%.*}; load_frac=${load1#*.}00
(( $((10#$load_int * 100 + 10#${load_frac:0:2})) < 50 )) || { echo "load gate: $load1"; exit 1; }
psi=$(cat /proc/pressure/cpu)
case "$psi" in *"some avg10=0.00"*) ;; *) echo "PSI gate: $psi"; exit 1;; esac
for f in /var/tmp/appbar/gpuwin.sh "$BASE_PY" \
         ${HOME}/bench-scripts/qwen38-mlx-bench.py \
         ${HOME}/bench-scripts/qwen38-2b-prompts.jsonl \
         ${HOME}/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e7c97b86b34b381/config.json; do
  test -e "$f" || { echo "missing prerequisite: $f"; exit 1; }
done
if [ "$PLAN_NAME" != w1 ]; then test -e "$CAND_PY"; fi

OUT=$ROOT/$PLAN_NAME-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
printf 'host=%s load1=%s psi=%s uptime_s=%s plan=%s\n' "$(hostname)" "$load1" "$psi" "$uptime_s" "$PLAN_NAME" > "$OUT/env.txt"
printf 'DF_PLAN=%s\n' "$DF_PLAN" >> "$OUT/env.txt"

if [ "$PLAN_NAME" = w1 ]; then
  CMD="timeout 840 env OUT=$OUT bash $TOOLS/w1_inner.sh"
else
  CMD="timeout 840 env OUT=$OUT DF_PLAN='$DF_PLAN' DF_CAND_PY=$CAND_PY bash $TOOLS/ab_inner.sh"
fi
/var/tmp/appbar/gpuwin.sh "$CMD" 2>&1 | tee "$OUT/gpuwin.log"

# service restore proof
svc=$(systemctl is-active llm-inference || true)
printf 'post-window llm-inference: %s\n' "$svc" | tee -a "$OUT/env.txt"
if [ "$svc" != active ]; then /var/tmp/appbar/gpuwin.sh true; svc=$(systemctl is-active llm-inference || true); printf 'restored: %s\n' "$svc" | tee -a "$OUT/env.txt"; fi

"$BASE_PY" "$TOOLS/check_digests.py" "$OUT" | tee "$OUT/summary.txt"
sha256sum "$OUT"/*.json > "$OUT/SHA256SUMS" 2>/dev/null || true
printf 'results=%s\n' "$OUT"
