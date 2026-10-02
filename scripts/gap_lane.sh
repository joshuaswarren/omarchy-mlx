#!/usr/bin/env bash
# CardLatency gap lane: sized for the ~75 min M2 gap.
# Order (all resumable; every stage skipped when its checkpoint is complete):
#   1. dev-subset suites, both pairs (validity gate for everything below)
#   2. v4 BASE compact4b (timed)
#   3. v4 CAND compact4b (timed) -- only if dev validity passed
# The 9B v4 legs do not fit in the window; they are queued separately.
# Usage: gap_lane.sh   (nohup me; log to gap_lane.log)
set -u
cd /tmp/CardLatency
LOG=/tmp/CardLatency/gap_lane.log
say() { echo "gap $(date -u +%FT%TZ) $*" >> "$LOG"; }
recorded() { python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['prompts']))" "$1" 2>/dev/null || echo 0; }

say "lane start boot $(cat /proc/sys/kernel/random/boot_id) kernel $(uname -r)"

# 1. dev subsets (both pairs)
for pair in compact4b everyday9b; do
  ./lane_wait.sh 22 ./ticket_dev_suite.sh "$pair" >> "$LOG" 2>&1
done

# 2. v4 base compact4b
for attempt in $(seq 1 8); do
  [ "$(recorded /tmp/CardLatency/v4_base_compact4b.json)" -ge 36 ] && break
  say "v4 base compact4b attempt $attempt ($(recorded /tmp/CardLatency/v4_base_compact4b.json)/36)"
  ./lane_wait.sh 22 ./ticket_v4_suite.sh compact4b base base >> "$LOG" 2>&1
done

# 3. v4 cand compact4b, gated on dev validity
DEV_OK=$(python3 - <<'EOF'
import json, os
ok = True
for pair in ("compact4b", "everyday9b"):
    path = f"/tmp/CardLatency/dev_cand_{pair}.json"
    try:
        rows = json.load(open(path))["prompts"]
    except Exception:
        ok = False; break
    if len(rows) < 15:
        ok = False; break
    cards = [r for r in rows if r["category"] == "card-worthy"]
    others = [r for r in rows if r["category"] != "card-worthy"]
    if sum(1 for r in cards if r["pass"]) < 0.8 * len(cards):
        ok = False
    if any(not r["pass"] for r in others):
        ok = False
print("yes" if ok else "no")
EOF
)
say "dev validity: $DEV_OK"
if [ "$DEV_OK" = "yes" ]; then
  for attempt in $(seq 1 8); do
    [ "$(recorded /tmp/CardLatency/v4_cand_compact4b.json)" -ge 36 ] && break
    say "v4 cand compact4b attempt $attempt ($(recorded /tmp/CardLatency/v4_cand_compact4b.json)/36)"
    ./lane_wait.sh 22 ./ticket_v4_suite.sh compact4b after cand >> "$LOG" 2>&1
  done
fi
say "lane done"
