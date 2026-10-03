#!/usr/bin/env bash
# CardLatency lane: CONFIRMATION of the shipped build (promotion + wrap,
# original prompt) on the frozen held-out v4, once per pair.
# Usage: lane_ship.sh   (nohup me via setsid; log to lane_ship.log)
set -u
cd /tmp/CardLatency
LOG=/tmp/CardLatency/lane_ship.log
say() { echo "ship $(date -u +%FT%TZ) $*" >> "$LOG"; }
recorded() { python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['prompts']))" "$1" 2>/dev/null || echo 0; }
T=14

say "lane start boot $(cat /proc/sys/kernel/random/boot_id) kernel $(uname -r)"

for combo in "compact4b compact" "everyday9b everyday"; do
  set -- $combo
  for attempt in $(seq 1 10); do
    [ "$(recorded /tmp/CardLatency/v4_ship_$1.json)" -ge 36 ] && break
    say "v4 ship $1 attempt $attempt ($(recorded /tmp/CardLatency/v4_ship_$1.json)/36)"
    ./lane_wait.sh $T ./ticket_v4_suite.sh "$1" after3 ship >> "$LOG" 2>&1
    sync; sleep 15
  done
done
say "lane done"
