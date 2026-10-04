#!/usr/bin/env bash
# Fuse2B: retry wrapper for the W1c window. Pre-checks the window gates
# OUTSIDE gpuwin so a gated abort never stops/starts llm-inference; each
# attempt is at least 6 minutes apart; 15 attempts max (~2h span).
set -uo pipefail
BUILD=/var/tmp/fuse2b-build
LOG=/var/tmp/fuse2b/w1c.log
for attempt in $(seq 1 15); do
  test ! -e /var/tmp/JW16_MAINTENANCE || { echo "attempt $attempt: maintenance"; sleep 300; continue; }
  uptime_s=$(cut -d. -f1 /proc/uptime); test "$uptime_s" -ge 360 || { sleep 120; continue; }
  load1=$(cut -d" " -f1 /proc/loadavg)
  li=${load1%%.*}; lf=${load1#*.}00
  psi=$(cat /proc/pressure/cpu)
  case "$psi" in *"some avg10=0.00"*) ;; *) echo "attempt $attempt: PSI $psi"; sleep 360; continue;; esac
  if (( $((10#$li * 100 + 10#${lf:0:2})) >= 500 )); then echo "attempt $attempt: load $load1"; sleep 300; continue; fi
  echo "attempt $attempt: gates pass (load $load1, psi ok) - launching window $(date -u +%FT%TZ)"
  bash "$BUILD/tools/fuse2b/w1c.sh" >> "$LOG" 2>&1
  rc=$?
  echo "attempt $attempt: window rc=$rc $(date -u +%FT%TZ)"
  if grep -q "GDN-CONV-DELTA-BITCHECK PASS" "$LOG" 2>/dev/null; then echo "BITCHECK PASS - done"; exit 0; fi
  if grep -q "GDN-CONV-DELTA-BITCHECK FAIL" "$LOG" 2>/dev/null; then echo "BITCHECK FAIL - real result, stop retrying"; exit 2; fi
  sleep 360
done
echo "exhausted 15 attempts"
exit 1
