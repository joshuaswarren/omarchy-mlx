#!/usr/bin/env bash
# CardLatency ticket: dev-subset lever run for ONE pair (candidate build).
# Usage: ticket_dev_suite.sh <pair> <minutes>
set -euo pipefail
LABEL=${1:?pair label compact4b|everyday9b}
case "$LABEL" in
  compact4b) PAIR=compact ;;
  everyday9b) PAIR=everyday ;;
  *) echo "unknown label $LABEL" >&2; exit 64 ;;
esac
cd /tmp/CardLatency
{
  echo "== dev-suite $PAIR start $(date -u +%FT%TZ)"
  echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
  echo "kernel=$(uname -r)"
  echo "uptime=$(uptime)"
  echo "loadavg=$(cat /proc/loadavg)"
  echo "psi_cpu=$(cat /sys/fs/cgroup/cpu.pressure | head -1)"
} >> window_dev.txt
UP=$(cut -d' ' -f1 /proc/uptime); UP=${UP%.*}
QUIET=0
for attempt in 1 2 3 4 5 6; do
  L=$(cut -d' ' -f1 /proc/loadavg)
  P=$(awk '/^some/{split($2,a,"=");print a[2]}' /proc/pressure/cpu)
  if awk -v u="$UP" 'BEGIN{exit !(u>=360)}' && awk -v l="$L" 'BEGIN{exit !(l<0.5)}' && awk -v p="$P" 'BEGIN{exit !(p+0==0)}'; then
    QUIET=1; break
  fi
  sleep 30
  UP=$(cut -d' ' -f1 /proc/uptime); UP=${UP%.*}
done
if [ "$QUIET" -ne 1 ]; then
  echo "WINDOW NOT QUIET — aborting $LABEL" >> window_dev.txt
  exit 42
fi

V=~/.local/share/mlx-omarchy/venv/bin/python
$V receipt/run_card_latency_suite.py \
  --pair "$PAIR" \
  --home ~/agents/PairGates/homes/"$LABEL" \
  --repo-serve /tmp/CardLatency/after/serve \
  --suite /tmp/CardLatency/dev_subset.json \
  --tag cand-dev \
  --out /tmp/CardLatency/dev_cand_${LABEL}.json \
  --budget-s 1380
echo "== dev-suite $PAIR done $(date -u +%FT%TZ)"
