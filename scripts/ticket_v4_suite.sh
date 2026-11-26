#!/usr/bin/env bash
# CardLatency ticket: HELD-OUT v4 suite slice for ONE pair and ONE build.
# Usage: ticket_v4_suite.sh <pair> <build:base|after> <tag> [minutes]
set -euo pipefail
LABEL=${1:?pair label compact4b|everyday9b}
case "$LABEL" in
  compact4b) PAIR=compact ;;
  everyday9b) PAIR=everyday ;;
  *) echo "unknown label $LABEL" >&2; exit 64 ;;
esac
BUILD=${2:?base or after}
TAG=${3:?tag}
cd /tmp/CardLatency
{
  echo "== v4 $PAIR $BUILD start $(date -u +%FT%TZ)"
  echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
  echo "kernel=$(uname -r)"
  echo "uptime=$(uptime)"
  echo "loadavg=$(cat /proc/loadavg)"
  echo "psi_cpu=$(cat /sys/fs/cgroup/cpu.pressure | head -1)"
} >> window_v4.txt
UP=$(cut -d' ' -f1 /proc/uptime); UP=${UP%.*}
QUIET=0
for attempt in 1 2 3 4 5 6; do
  L=$(cut -d' ' -f1 /proc/loadavg)
  P=$(awk '/^some/{split($2,a,"=");print a[2]}' /proc/pressure/cpu)
  if awk -v u="$UP" 'BEGIN{exit !(u>=360)}' && awk -v l="$L" 'BEGIN{exit !(l<0.5)}' && awk -v p="$P" 'BEGIN{exit !(p+0==0)}'; then
    QUIET=1; break
  fi
  echo "guard retry $attempt (up=$UP load=$L psi=$P)" >> window_v4.txt
  sleep 30
  UP=$(cut -d' ' -f1 /proc/uptime); UP=${UP%.*}
done
if [ "$QUIET" -ne 1 ]; then
  echo "WINDOW NOT QUIET — aborting v4 $PAIR $BUILD" >> window_v4.txt
  exit 42
fi

V=~/.local/share/mlx-omarchy/venv/bin/python
$V $BUILD/scripts/mlx_provenance.py > provenance_v4_$BUILD.txt 2>&1 || true
cat provenance_v4_$BUILD.txt

$V receipt/run_card_latency_suite.py \
  --pair "$PAIR" \
  --home ~/agents/PairGates/homes/"$LABEL" \
  --repo-serve /tmp/CardLatency/$BUILD/serve \
  --suite /tmp/CardLatency/$BUILD/scripts/cards_held_out_v4.json \
  --tag "$TAG" \
  --out /tmp/CardLatency/v4_${TAG}_${LABEL}.json \
  --budget-s 1500
echo "== v4 $PAIR $BUILD done $(date -u +%FT%TZ)"
