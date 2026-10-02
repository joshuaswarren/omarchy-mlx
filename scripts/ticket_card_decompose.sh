#!/usr/bin/env bash
# CardLatency ticket 1: baseline decomposition, one card turn per pair.
# Requires a quiet window: >= 6 min uptime, loadavg < 0.5, PSI cpu avg10 = 0.
set -euo pipefail
cd /tmp/CardLatency
{
  echo "== ticket1 start $(date -u +%FT%TZ)"
  echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
  echo "kernel=$(uname -r)"
  echo "uptime=$(uptime)"
  echo "loadavg=$(cat /proc/loadavg)"
  echo "psi_cpu=$(cat /sys/fs/cgroup/cpu.pressure | head -1)"
} > window.txt
UP_S=$(awk '{print int($1)}' /proc/uptime)
QUIET=0
for attempt in 1 2 3 4 5 6; do
  LOAD=$(cut -d' ' -f1 /proc/loadavg)
  PSI=$(head -1 /sys/fs/cgroup/cpu.pressure | grep -o 'avg10=[0-9.]*' | cut -d= -f2)
  if [ "$UP_S" -ge 360 ] && awk "BEGIN{exit !($LOAD < 0.5)}" && awk "BEGIN{exit !($PSI + 0 == 0)}"; then
    QUIET=1; break
  fi
  echo "guard retry $attempt (up=$UP_S load=$LOAD psi=$PSI)" >> window.txt
  sleep 30
  UP_S=$(awk '{print int($1)}' /proc/uptime)
done
if [ "$QUIET" -ne 1 ]; then
  echo "WINDOW NOT QUIET — aborting" | tee -a window.txt
  exit 42
fi
echo "window quiet" >> window.txt

V=~/.local/share/mlx-omarchy/venv/bin/python
H=~/.local/share/mlx-omarchy
$V base/scripts/mlx_provenance.py > provenance.txt 2>&1 || true
cat provenance.txt

# Baseline serve code (origin/main 29100916f): /tmp/CardLatency/base/serve
# Server pair ids are compact/everyday; the home dirs carry the model label.
for combo in "compact compact4b" "everyday everyday9b"; do
  set -- $combo
  $V base/scripts/card_decompose.py \
    --pair "$1" \
    --home ~/agents/PairGates/homes/"$2" \
    --repo-serve /tmp/CardLatency/base/serve \
    --out "/tmp/CardLatency/decompose_${2}_base.json" \
    --label "base-$2"
done
echo "== ticket1 done $(date -u +%FT%TZ)"
