#!/usr/bin/env bash
# CardLatency lane waiter: wait for a quiet window, then run one gpu-turn
# ticket.  Usage: lane_wait.sh <minutes> <script.sh> [script args...]
set -u
MIN=${1:?minutes}
SCRIPT=${2:?script}
shift 2
for i in $(seq 1 80); do
  UP=$(cut -d' ' -f1 /proc/uptime)
  UP=${UP%.*}
  L=$(cut -d' ' -f1 /proc/loadavg)
  P=$(awk '/^some/{split($2,a,"=");print a[2]}' /proc/pressure/cpu)
  if [ "${UP:-0}" -ge 360 ] && awk -v l="$L" -v p="$P" 'BEGIN{exit !(l<0.5 && p+0==0)}'; then
    echo "quiet window met after ${i} polls (up=${UP}s load=${L} psi=${P})"
    exec ~/bin/gpu-turn -m "$MIN" -- bash "$SCRIPT" "$@"
  fi
  sleep 20
done
echo "no quiet window within 80 polls" >&2
exit 43
