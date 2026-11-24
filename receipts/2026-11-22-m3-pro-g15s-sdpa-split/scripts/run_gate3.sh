#!/usr/bin/env bash
# run_gate3.sh: the docs/numerics-gate.md reference captures (numerics_gate.py cpuref: stock MLX, CPU stream, float32) in venv-ref, both
# models side by side in one hold of the GPU lock (CPU-heavy; stock MLX runs this on about one core, so the two do not compete).
set -uo pipefail
L=${LANE}
R=${REPO}
M=${MODELS}; TXT=$L/scripts/context-text.txt
exec 9>/tmp/m1-gpu.lock; flock -w 5400 -x 9 || exit 97
for m in qwen3-4b-2507-4bit qwen3.8-2b-4bit; do
  { echo "== before $(date -u +%FT%TZ) loadavg $(cat /proc/loadavg)"; free -m | sed -n 2p; } > $L/evidence/NG-cpuref-$m.state
  ( timeout --kill-after=30s 21600 env HF_HUB_OFFLINE=1 $L/venv-ref/bin/python $L/scripts/numerics_gate.py $M/$m cpuref $L/runs/gate/NG-cpuref-$m.npz none $R $TXT < /dev/null > $L/logs/NG-cpuref-$m.log 2>&1
    echo "== after $(date -u +%FT%TZ) exit $? loadavg $(cat /proc/loadavg)" >> $L/evidence/NG-cpuref-$m.state ) &
done
wait
for m in qwen3-4b-2507-4bit qwen3.8-2b-4bit; do [ -s $L/runs/gate/NG-cpuref-$m.npz ] || { echo "MISSING NG-cpuref-$m.npz"; exit 1; }; done
echo GATE3_DONE
