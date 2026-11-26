#!/bin/bash
# One M2 slot for one suite step, one gpu-turn ticket at a time, never past STOP.
# usage: mc_slot.sh HH:MM <checkout> <suite file> <tag or -> <model>
set -u
H=/home/joshuawarren
MC=$H/agents/MarkdownCards
PY=$H/.local/share/mlx-omarchy/venv/bin/python
STOP=$(date -d "$1" +%s); REPO=$2; SUITE=$3; TAG=$4; MODEL=$5
[ "$TAG" = "-" ] && TAG=""
NAME="$(basename "$SUITE" .json | sed 's/^cards_//')${TAG}_${MODEL}"
while pgrep -f "[g]pu-turn .*MARKCARDS_HOME" > /dev/null; do sleep 15; done
echo "slot $NAME start $(date -Is) stop $1 kernel $(uname -r) boot $(cat /proc/sys/kernel/random/boot_id) load $(cat /proc/loadavg)" >> $MC/logs/lane.log
recorded() { python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['prompts']))" "$MC/results/$NAME.json" 2>/dev/null || echo 0; }
while [ "$(recorded)" -lt 36 ]; do
  [ $(( STOP - $(date +%s) )) -lt 900 ] && { echo "slot $NAME closing $(date -Is) at $(recorded)/36" >> $MC/logs/lane.log; exit 0; }
  m=$(( (STOP - $(date +%s)) / 60 )); [ $m -gt 15 ] && m=15
  (cd $MC/$REPO && $H/bin/gpu-turn -m $m -- env MARKCARDS_HOME=$H $PY \
     receipts/2026-09-30-card-promotion/run_suite.py --suite "$SUITE" --tag "$TAG" \
     --model $MODEL --budget-s 750 --stop-at $STOP) >> $MC/logs/$NAME.log 2>&1
  tail -2 $MC/logs/$NAME.log | grep -q "STOP-AT" && { echo "slot $NAME stop-at $(date -Is) at $(recorded)/36" >> $MC/logs/lane.log; exit 0; }
  sync
done
echo "=== SLOT $NAME DONE $(date -Is) ===" >> $MC/logs/lane.log
