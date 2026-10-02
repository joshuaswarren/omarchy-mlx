#!/usr/bin/env bash
# CardLatency gap lane (M2 GAP, ~3-4 h). Tickets <= 19 min (hard cap 20).
# Order: dev subsets (both pairs) -> v4 BASE (4B then 9B) -> v4 CAND
# (4B then 9B, gated on dev validity). All stages resumable checkpoints;
# each stage is skipped once complete. One ticket at a time.
# Usage: gap_lane.sh   (nohup me; log to gap_lane.log)
set -u
cd /tmp/CardLatency
LOG=/tmp/CardLatency/gap_lane.log
say() { echo "gap $(date -u +%FT%TZ) $*" >> "$LOG"; }
recorded() { python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['prompts']))" "$1" 2>/dev/null || echo 0; }
T=19

say "lane start boot $(cat /proc/sys/kernel/random/boot_id) kernel $(uname -r)"

for pair in compact4b everyday9b; do
  [ "$(recorded /tmp/CardLatency/dev_cand_${pair}.json)" -ge 15 ] && continue
  ./lane_wait.sh $T ./ticket_dev_suite.sh "$pair" >> "$LOG" 2>&1
done

suite() {  # pair build tag out
  local pair=$1 build=$2 tag=$3 out=$4
  for attempt in $(seq 1 12); do
    [ "$(recorded "$out")" -ge 36 ] && return 0
    say "v4 $tag $pair attempt $attempt ($(recorded "$out")/36)"
    ./lane_wait.sh $T ./ticket_v4_suite.sh "$pair" "$build" "$tag" >> "$LOG" 2>&1
    sync
    sleep 15
  done
  say "v4 $tag $pair stopped at $(recorded "$out")/36"
}

suite compact4b base base /tmp/CardLatency/v4_base_compact4b.json
suite everyday9b base base /tmp/CardLatency/v4_base_everyday9b.json

DEV_OK=$(python3 - <<'EOF'
import json
ok = True
for pair in ("compact4b", "everyday9b"):
    try:
        rows = json.load(open(f"/tmp/CardLatency/dev_cand_{pair}.json"))["prompts"]
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
  suite compact4b after cand /tmp/CardLatency/v4_cand_compact4b.json
  suite everyday9b after cand /tmp/CardLatency/v4_cand_everyday9b.json
else
  say "cand legs skipped: dev validity failed"
fi
say "lane done"
