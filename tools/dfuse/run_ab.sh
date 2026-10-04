#!/usr/bin/env bash
# DispatchFuse A/B outer driver: gates, gpuwin window, service check.
# usage: run_ab.sh wA|wB|wN
set -euo pipefail
PLAN_NAME=${1:?usage: run_ab.sh wA|wB|wN}
TOOLS=/var/tmp/dfuse
case "$PLAN_NAME" in
  wA) DF_PLAN="a-n-d64:64:neu4b:1;a-d64-r1-ctl:64:ctl4b;a-d64-r1-on:64:on4b;a-d64-r2-ctl:64:ctl4b;a-d64-r2-on:64:on4b;a-d64-r3-ctl:64:ctl4b;a-d64-r3-on:64:on4b;a-d64-r4-ctl:64:ctl4b;a-d64-r4-on:64:on4b;a-d64-r5-ctl:64:ctl4b;a-d64-r5-on:64:on4b" ;;
  wB) DF_PLAN="b-d128-r1-on:128:on4b;b-d128-r1-ctl:128:ctl4b;b-d128-r2-on:128:on4b;b-d128-r2-ctl:128:ctl4b;b-d128-r3-on:128:on4b;b-d128-r3-ctl:128:ctl4b;b-d128-r4-on:128:on4b;b-d128-r4-ctl:128:ctl4b;b-d128-r5-on:128:on4b;b-d128-r5-ctl:128:ctl4b" ;;
  wC) DF_PLAN="c-d256-r1-on:256:on4b;c-d256-r1-ctl:256:ctl4b;c-d256-r2-on:256:on4b;c-d256-r2-ctl:256:ctl4b;c-d256-r3-on:256:on4b;c-d256-r3-ctl:256:ctl4b;c-d256-r4-on:256:on4b;c-d256-r4-ctl:256:ctl4b;c-d256-r5-on:256:on4b;c-d256-r5-ctl:256:ctl4b" ;;
  wD) DF_PLAN="d-d512-r1-on:512:on4b;d-d512-r1-ctl:512:ctl4b;d-d512-r2-on:512:on4b;d-d512-r2-ctl:512:ctl4b;d-d512-r3-on:512:on4b;d-d512-r3-ctl:512:ctl4b;d-d512-r4-on:512:on4b;d-d512-r4-ctl:512:ctl4b;d-d512-r5-on:512:on4b;d-d512-r5-ctl:512:ctl4b" ;;
  w9A) DF_PLAN="n9-d64:64:neu9b:1;9-d64-r1-ctl:64:ctl9b;9-d64-r1-on:64:on9b;9-d64-r2-ctl:64:ctl9b;9-d64-r2-on:64:on9b;9-d64-r3-ctl:64:ctl9b;9-d64-r3-on:64:on9b;9-d64-r4-ctl:64:ctl9b;9-d64-r4-on:64:on9b;9-d64-r5-ctl:64:ctl9b;9-d64-r5-on:64:on9b" ;;
  w9B) DF_PLAN="9-d128-r1-on:128:on9b;9-d128-r1-ctl:128:ctl9b;9-d128-r2-on:128:on9b;9-d128-r2-ctl:128:ctl9b;9-d128-r3-on:128:on9b;9-d128-r3-ctl:128:ctl9b;9-d128-r4-on:128:on9b;9-d128-r4-ctl:128:ctl9b;9-d128-r5-on:128:on9b;9-d128-r5-ctl:128:ctl9b" ;;
  w9C) DF_PLAN="9-d256-r1-on:256:on9b;9-d256-r1-ctl:256:ctl9b;9-d256-r2-on:256:on9b;9-d256-r2-ctl:256:ctl9b;9-d256-r3-on:256:on9b;9-d256-r3-ctl:256:ctl9b;9-d256-r4-on:256:on9b;9-d256-r4-ctl:256:ctl9b;9-d256-r5-on:256:on9b;9-d256-r5-ctl:256:ctl9b" ;;
  w9D) DF_PLAN="9-d512-r1-on:512:on9b;9-d512-r1-ctl:512:ctl9b;9-d512-r2-on:512:on9b;9-d512-r2-ctl:512:ctl9b;9-d512-r3-on:512:on9b;9-d512-r3-ctl:512:ctl9b;9-d512-r4-on:512:on9b;9-d512-r4-ctl:512:ctl9b;9-d512-r5-on:512:on9b;9-d512-r5-ctl:512:ctl9b" ;;
  wX1) DF_PLAN="x-d64-off:64:off9b;x-d64-r1-ctl:64:ctl9b;x-d64-r1-neu:64:neu9b;x-d64-r2-neu:64:neu9b;x-d64-r2-ctl:64:ctl9b;x-d64-r3-ctl:64:ctl9b;x-d64-r3-neu:64:neu9b;x-d64-r4-neu:64:neu9b;x-d64-r4-ctl:64:ctl9b;x-d64-r5-ctl:64:ctl9b;x-d64-r5-neu:64:neu9b;x-d128-r1-ctl:128:ctl9b;x-d128-r1-neu:128:neu9b;x-d128-r2-neu:128:neu9b;x-d128-r2-ctl:128:ctl9b;x-d128-r3-ctl:128:ctl9b;x-d128-r3-neu:128:neu9b;x-d128-r4-neu:128:neu9b;x-d128-r4-ctl:128:ctl9b;x-d128-r5-ctl:128:ctl9b;x-d128-r5-neu:128:neu9b;x-d128-off:128:off9b" ;;
  wX2) DF_PLAN="x-d256-off:256:off9b;x-d256-r1-ctl:256:ctl9b;x-d256-r1-neu:256:neu9b;x-d256-r2-neu:256:neu9b;x-d256-r2-ctl:256:ctl9b;x-d256-r3-ctl:256:ctl9b;x-d256-r3-neu:256:neu9b;x-d256-r4-neu:256:neu9b;x-d256-r4-ctl:256:ctl9b;x-d256-r5-ctl:256:ctl9b;x-d256-r5-neu:256:neu9b" ;;
  wX3) DF_PLAN="x-d512-off:512:off9b;x-d512-r1-ctl:512:ctl9b;x-d512-r1-neu:512:neu9b;x-d512-r2-neu:512:neu9b;x-d512-r2-ctl:512:ctl9b;x-d512-r3-ctl:512:ctl9b;x-d512-r3-neu:512:neu9b;x-d512-r4-neu:512:neu9b;x-d512-r4-ctl:512:ctl9b;x-d512-r5-ctl:512:ctl9b;x-d512-r5-neu:512:neu9b" ;;
  w2A) DF_PLAN="n2-d64:64:neu2b:1;2-d64-r1-ctl:64:ctl2b;2-d64-r1-on:64:on2b;2-d64-r2-ctl:64:ctl2b;2-d64-r2-on:64:on2b;2-d64-r3-ctl:64:ctl2b;2-d64-r3-on:64:on2b;2-d64-r4-ctl:64:ctl2b;2-d64-r4-on:64:on2b;2-d64-r5-ctl:64:ctl2b;2-d64-r5-on:64:on2b" ;;
  *) echo "unknown plan $PLAN_NAME"; exit 2 ;;
esac
test ! -e /var/tmp/JW16_MAINTENANCE
uptime_s=$(cut -d. -f1 /proc/uptime); test "$uptime_s" -ge 360 || { echo "boot settle gate: ${uptime_s}s"; exit 1; }
load1=$(cut -d" " -f1 /proc/loadavg)
li=${load1%%.*}; lf=${load1#*.}00
(( $((10#$li * 100 + 10#${lf:0:2})) < 50 )) || { echo "load gate: $load1"; exit 1; }
psi=$(cat /proc/pressure/cpu)
case "$psi" in *"some avg10=0.00"*) ;; *) echo "PSI gate: $psi"; exit 1;; esac
for f in /var/tmp/appbar/gpuwin.sh /var/tmp/v072-venv-fused/bin/python \
         ${DF_CAND_PY:-/var/tmp/dfuse-cand/bin/python} \
         ${HOME}/bench-scripts/qwen38-mlx-bench.py \
         ${HOME}/bench-scripts/qwen38-2b-prompts.jsonl; do
  test -e "$f" || { echo "missing prerequisite: $f"; exit 1; }
done

OUT=/var/tmp/dfuse/ab-$PLAN_NAME-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
printf 'host=%s load1=%s psi=%s uptime_s=%s plan=%s\n' "$(hostname)" "$load1" "$psi" "$uptime_s" "$PLAN_NAME" > "$OUT/env.txt"
printf 'DF_PLAN=%s\n' "$DF_PLAN" >> "$OUT/env.txt"

CMD="timeout 840 env DF_OUT=$OUT DF_PLAN='$DF_PLAN' DF_CAND_PY=${DF_CAND_PY:-/var/tmp/dfuse-cand/bin/python} bash $TOOLS/ab_inner.sh"
bash /var/tmp/appbar/gpuwin.sh "$CMD" 2>&1 | tee "$OUT/gpuwin.log"

svc=$(systemctl is-active llm-inference || true)
printf 'post-window llm-inference: %s\n' "$svc" | tee -a "$OUT/env.txt"
if [ "$svc" != active ]; then bash /var/tmp/appbar/gpuwin.sh true; svc=$(systemctl is-active llm-inference || true); printf 'restored: %s\n' "$svc" | tee -a "$OUT/env.txt"; fi
sha256sum "$OUT"/*.json > "$OUT/SHA256SUMS" 2>/dev/null || true
printf 'results=%s\n' "$OUT"
