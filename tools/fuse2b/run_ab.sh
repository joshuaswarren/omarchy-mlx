#!/usr/bin/env bash
# Fuse2B A/B outer driver: gates, gpuwin window, service check.
# usage: run_ab.sh wA|wA2|wB|wB2|wC|wC2|wD|wD2|wN
#   wA/wA2  d64   (wA leads with the neutral cell, wA2 is the counterbalance)
#   wB/wB2 d128   wC/wC2 d256   wD/wD2 d512
set -euo pipefail
PLAN_NAME=${1:?usage: run_ab.sh wA|wA2|wB|wB2|wC|wC2|wD|wD2|wN}
TOOLS=/var/tmp/fuse2b
PLAN=""
pair() { # depth round
  PLAN="${PLAN:+$PLAN;}$1-d$1-r$2-ctl:$1:ctl2b;$1-d$1-r$2-on:$1:on2b"
}
case "$PLAN_NAME" in
  wA)  PLAN="n2-d64:64:neu2b:1" ; pair 64 1; pair 64 2; pair 64 3; pair 64 4; pair 64 5 ;;
  wA2) pair 64 1; pair 64 2; pair 64 3; pair 64 4; pair 64 5 ;;
  wB)  pair 128 1; pair 128 2; pair 128 3; pair 128 4; pair 128 5 ;;
  wB2) pair 128 1; pair 128 2; pair 128 3; pair 128 4; pair 128 5 ;;
  wC)  pair 256 1; pair 256 2; pair 256 3; pair 256 4; pair 256 5 ;;
  wC2) pair 256 1; pair 256 2; pair 256 3; pair 256 4; pair 256 5 ;;
  wD)  pair 512 1; pair 512 2; pair 512 3; pair 512 4; pair 512 5 ;;
  wD2) pair 512 1; pair 512 2; pair 512 3; pair 512 4; pair 512 5 ;;
  *) echo "unknown plan $PLAN_NAME"; exit 2 ;;
esac
DF_PLAN="$PLAN"
test ! -e /var/tmp/JW16_MAINTENANCE
uptime_s=$(cut -d. -f1 /proc/uptime); test "$uptime_s" -ge 360 || { echo "boot settle gate: ${uptime_s}s"; exit 1; }
load1=$(cut -d" " -f1 /proc/loadavg)
li=${load1%%.*}; lf=${load1#*.}00
(( $((10#$li * 100 + 10#${lf:0:2})) < 50 )) || { echo "load gate: $load1"; exit 1; }
psi=$(cat /proc/pressure/cpu)
case "$psi" in *"some avg10=0.00"*) ;; *) echo "PSI gate: $psi"; exit 1;; esac
for f in /var/tmp/appbar/gpuwin.sh /var/tmp/v072-venv-fused/bin/python \
         /var/tmp/fuse2b-cand/bin/python \
         ${HOME}/bench-scripts/qwen38-mlx-bench.py \
         ${HOME}/bench-scripts/qwen38-2b-prompts.jsonl; do
  test -e "$f" || { echo "missing prerequisite: $f"; exit 1; }
done

OUT=$TOOLS/ab-$PLAN_NAME-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
printf 'host=%s boot=%s load1=%s psi=%s uptime_s=%s plan=%s\n' "$(hostname)" \
  "$(cat /proc/sys/kernel/random/boot_id)" "$load1" "$psi" "$uptime_s" "$PLAN_NAME" > "$OUT/env.txt"
printf 'DF_PLAN=%s\n' "$DF_PLAN" >> "$OUT/env.txt"

CMD="timeout 840 env DF_OUT=$OUT DF_PLAN='$DF_PLAN' bash $TOOLS/ab_inner.sh"
bash /var/tmp/appbar/gpuwin.sh "$CMD" 2>&1 | tee "$OUT/gpuwin.log"

svc=$(systemctl is-active llm-inference || true)
printf 'post-window llm-inference: %s\n' "$svc" | tee -a "$OUT/env.txt"
if [ "$svc" != active ]; then bash /var/tmp/appbar/gpuwin.sh true; svc=$(systemctl is-active llm-inference || true); printf 'restored: %s\n' "$svc" | tee -a "$OUT/env.txt"; fi
sha256sum "$OUT"/*.json > "$OUT/SHA256SUMS" 2>/dev/null || true
printf 'results=%s\n' "$OUT"
