#!/usr/bin/env bash
# CardLatency ticket: card timing with the LEVER build (after state).
# Usage: ticket_card_after.sh   (paths fixed for /tmp/CardLatency layout)
set -euo pipefail
cd /tmp/CardLatency
{
  echo "== after-ticket start $(date -u +%FT%TZ)"
  echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
  echo "kernel=$(uname -r)"
  echo "uptime=$(uptime)"
  echo "loadavg=$(cat /proc/loadavg)"
  echo "psi_cpu=$(cat /sys/fs/cgroup/cpu.pressure | head -1)"
} > window_after.txt
UP=$(cut -d' ' -f1 /proc/uptime); UP=${UP%.*}
QUIET=0
for attempt in 1 2 3 4 5 6; do
  L=$(cut -d' ' -f1 /proc/loadavg)
  P=$(awk '/^some/{split($2,a,"=");print a[2]}' /proc/pressure/cpu)
  if awk -v u="$UP" 'BEGIN{exit !(u>=360)}' && awk -v l="$L" 'BEGIN{exit !(l<0.5)}' && awk -v p="$P" 'BEGIN{exit !(p+0==0)}'; then
    QUIET=1; break
  fi
  echo "guard retry $attempt (up=$UP load=$L psi=$P)" >> window_after.txt
  sleep 30
  UP=$(cut -d' ' -f1 /proc/uptime); UP=${UP%.*}
done
if [ "$QUIET" -ne 1 ]; then
  echo "WINDOW NOT QUIET — aborting" | tee -a window_after.txt
  exit 42
fi
echo "window quiet" >> window_after.txt

V=~/.local/share/mlx-omarchy/venv/bin/python
$V base/scripts/mlx_provenance.py > provenance_after.txt 2>&1 || true
cat provenance_after.txt

for combo in "compact compact4b" "everyday everyday9b"; do
  set -- $combo
  $V after/scripts/card_decompose.py \
    --pair "$1" \
    --home ~/agents/PairGates/homes/"$2" \
    --repo-serve /tmp/CardLatency/after/serve \
    --out "/tmp/CardLatency/decompose_${2}_after.json" \
    --label "after-$2"
done
echo "== after-ticket done $(date -u +%FT%TZ)"
