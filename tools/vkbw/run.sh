#!/usr/bin/env bash
# Build vkbw and run the full sweep REPS times (default 3) on one device.
# Usage: ./run.sh [LABEL]            results go to ./results/LABEL-runN.csv
# Env:   DEVICE=<index from ./vkbw --list>, REPS=<n>, RUN_TIMEOUT=<seconds per run, default 900>,
#        VKBW_ARGS="--quick" (extra vkbw flags)
set -euo pipefail
cd "$(dirname "$0")"
label="${1:-$(uname -m)}"
make -s vkbw
mkdir -p results
./vkbw --list
echo "kernel: $(uname -sr)"
for n in $(seq 1 "${REPS:-3}"); do
  # shellcheck disable=SC2086
  timeout "${RUN_TIMEOUT:-900}" ./vkbw ${DEVICE:+--device "$DEVICE"} ${VKBW_ARGS:-} --csv "results/$label-run$n.csv" >/dev/null
  echo "run $n done: results/$label-run$n.csv"
done
python3 summarize.py "results/$label"-run*.csv
