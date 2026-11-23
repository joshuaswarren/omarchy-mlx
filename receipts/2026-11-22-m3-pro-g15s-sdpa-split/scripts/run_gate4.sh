#!/usr/bin/env bash
# run_gate4.sh: the Qwen3-4B docs/numerics-gate.md reference (numerics_gate.py cpuref) as two halves of the same scored range run side by
# side (stock MLX runs it on one core, about 4.7 s per step): steps 0-2559 after the 128-token prefill (cache 128-2687) and steps 2560-5119
# after a chunked prefill of 2688 tokens (cache 2688-5247), under /tmp/m1-gpu.lock. The recorded run used an earlier version that took
# the lock in a keeper started after the halves (the Qwen3.8-2B reference held it then) and logged exit codes without failing.
set -uo pipefail
L=${LANE}
R=${REPO}
M=${MODELS}/qwen3-4b-2507-4bit; TXT=$L/scripts/context-text.txt
exec 8>/tmp/m1-gpu.lock; flock -w 5400 -x 8 || exit 97  # the lock is taken before any half starts and held until both end
pids=()
for seg in "a 128" "b 2688"; do set -- $seg
  n=NG-cpuref-qwen3-4b-2507-4bit-$1
  { echo "== before $(date -u +%FT%TZ) loadavg $(cat /proc/loadavg)"; free -m | sed -n 2p; echo "GATE_PREFIX=$2 GATE_STEPS=2560"; } > $L/evidence/$n.state
  ( timeout --kill-after=30s 21600 env HF_HUB_OFFLINE=1 GATE_PREFIX=$2 GATE_STEPS=2560 $L/venv-ref/bin/python $L/scripts/numerics_gate.py $M cpuref $L/runs/gate/$n.npz none $R $TXT < /dev/null > $L/logs/$n.log 2>&1
    echo "== after $(date -u +%FT%TZ) exit $? loadavg $(cat /proc/loadavg)" >> $L/evidence/$n.state ) &
  pids+=($!)
done
wait "${pids[@]}"
for h in a b; do [ -s $L/runs/gate/NG-cpuref-qwen3-4b-2507-4bit-$h.npz ] || { echo "MISSING NG-cpuref-qwen3-4b-2507-4bit-$h.npz"; exit 1; }; done
echo GATE4_DONE
