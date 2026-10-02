#!/usr/bin/env bash
# CardLatency lane: HELD-OUT v4, one timed run per pair per build, in the
# declared order.  Resumable: each ticket continues from the results
# checkpoint; a suite is complete at 36 recorded prompts.
# Usage: lane_v4.sh [base|cand ...]   (nohup me; log to lane_v4.log)
# Legs default to both; pass "base" to run only the baseline column and
# leave the candidate leg gated on dev-set validation.
set -u
cd /tmp/CardLatency
LOG=/tmp/CardLatency/lane_v4.log
LEGS=${*:-"base cand"}
recorded() { python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['prompts']))" "$1" 2>/dev/null || echo 0; }

suite() {  # pair build tag out
  local pair=$1 build=$2 tag=$3 out=$4
  for attempt in $(seq 1 16); do
    [ "$(recorded "$out")" -ge 36 ] && return 0
    echo "v4 ticket attempt $attempt $tag $pair ($(recorded "$out")/36)" >> "$LOG"
    ./lane_wait.sh 24 ./ticket_v4_suite.sh "$pair" "$build" "$tag" >> "$LOG" 2>&1
    sync
  done
  echo "$tag $pair stopped at $(recorded "$out")/36" >> "$LOG"
}

echo "v4 lane start $(date -u +%FT%TZ) boot $(cat /proc/sys/kernel/random/boot_id) kernel $(uname -r) legs: $LEGS" >> "$LOG"
case " $LEGS " in
  *" base "*)
    suite everyday9b base  base       /tmp/CardLatency/v4_base_everyday9b.json
    suite compact4b base   base       /tmp/CardLatency/v4_base_compact4b.json
    ;;
esac
case " $LEGS " in
  *" cand "*)
    suite everyday9b after cand       /tmp/CardLatency/v4_cand_everyday9b.json
    suite compact4b after  cand       /tmp/CardLatency/v4_cand_compact4b.json
    ;;
esac
echo "v4 lane done $(date -u +%FT%TZ)" >> "$LOG"
