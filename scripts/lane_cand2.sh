#!/usr/bin/env bash
# CardLatency lane: CANDIDATE 2 (lead-in sentence before the fence).
# Order: dev subsets on after2 (both pairs) -> v4 after2 cand2 (4B, 9B).
# Dev gate before the held-out legs: first-text p95 <= 2.0 s both pairs,
# >= 80% card-worthy promoted, 0 spurious. Tickets <= 14 min.
# Usage: lane_cand2.sh   (nohup me via setsid; log to lane_cand2.log)
set -u
cd /tmp/CardLatency
LOG=/tmp/CardLatency/lane_cand2.log
say() { echo "cand2 $(date -u +%FT%TZ) $*" >> "$LOG"; }
recorded() { python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['prompts']))" "$1" 2>/dev/null || echo 0; }
T=14
export BUILD=after2 TAG=cand2-dev

say "lane start boot $(cat /proc/sys/kernel/random/boot_id) kernel $(uname -r)"

for combo in "compact4b compact" "everyday9b everyday"; do
  set -- $combo
  [ "$(recorded /tmp/CardLatency/dev_cand2_$1.json)" -ge 15 ] && continue
  OUT=/tmp/CardLatency/dev_cand2_$1.json ./lane_wait.sh $T ./ticket_dev_suite.sh "$1" >> "$LOG" 2>&1
done

DEV=$(python3 - <<'EOF'
import json
out = {}
for label, pair in (("compact4b", "compact"), ("everyday9b", "everyday")):
    try:
        rows = json.load(open(f"/tmp/CardLatency/dev_cand2_{label}.json"))["prompts"]
    except Exception:
        print("no"); raise SystemExit
    if len(rows) < 15:
        print("no"); raise SystemExit
    cards = [r for r in rows if r["category"] == "card-worthy"]
    others = [r for r in rows if r["category"] != "card-worthy"]
    if sum(1 for r in cards if r["pass"]) < 0.8 * len(cards):
        print("no"); raise SystemExit
    if any(not r["pass"] for r in others):
        print("no"); raise SystemExit
    text = sorted(r["first_text_s"] for r in rows if r["first_text_s"] is not None)
    if not text or text[int(0.95 * (len(text) - 1))] > 2.0:
        print("no"); raise SystemExit
print("yes")
EOF
)
say "dev gate: $DEV"

if [ "$DEV" = "yes" ]; then
  for attempt in $(seq 1 10); do
    [ "$(recorded /tmp/CardLatency/v4_cand2_compact4b.json)" -ge 36 ] && break
    say "v4 cand2 compact4b attempt $attempt ($(recorded /tmp/CardLatency/v4_cand2_compact4b.json)/36)"
    ./lane_wait.sh $T ./ticket_v4_suite.sh compact4b after2 cand2 >> "$LOG" 2>&1
    sync; sleep 15
  done
  for attempt in $(seq 1 12); do
    [ "$(recorded /tmp/CardLatency/v4_cand2_everyday9b.json)" -ge 36 ] && break
    say "v4 cand2 everyday9b attempt $attempt ($(recorded /tmp/CardLatency/v4_cand2_everyday9b.json)/36)"
    ./lane_wait.sh $T ./ticket_v4_suite.sh everyday9b after2 cand2 >> "$LOG" 2>&1
    sync; sleep 15
  done
fi
say "lane done"
