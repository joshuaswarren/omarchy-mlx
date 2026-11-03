#!/bin/bash
# #316 only, resumable: waits for a granite #316 ab.py already running, then the
# remaining #316 arms, the #316 per-op check, waits for the granite and Qwen
# references, and analyzes. Re-running skips finished outputs.
set -uo pipefail
L=$HOME/gpu/upr/logs; G=$HOME/gpu/upr/gate; O=$G/out
step() { echo "== $1 start $(date -u +%FT%TZ)"; bash $G/run_gate.sh "$@" >> $L/gate-$1.log 2>&1; local rc=$?; echo "== $1 rc=$rc $(date -u +%FT%TZ)"; return $rc; }
while pgrep -f "[a]b.py .*ab-316-" > /dev/null; do sleep 15; done
step arms316 || { echo RUNNER_FAILED arms316; exit 3; }
step perop316 || { echo RUNNER_FAILED perop316; exit 4; }
if [ ! -s $O/ref-granite-4.0-1b-4bit.json ] || [ ! -s $O/ref-Qwen2.5-0.5B-Instruct-4bit.json ]; then
  pgrep -f "[g]ate_tf.py measure --cpu" > /dev/null || step ref "granite-4.0-1b-4bit Qwen2.5-0.5B-Instruct-4bit" || { echo RUNNER_FAILED ref; exit 5; }
fi
while [ ! -s $O/ref-granite-4.0-1b-4bit.json ] || [ ! -s $O/ref-Qwen2.5-0.5B-Instruct-4bit.json ]; do
  pgrep -f "[g]ate_tf.py (measure --cpu|merge)" > /dev/null || { echo RUNNER_FAILED ref-missing; exit 5; }
  sleep 15
done
step analyze316 || { echo RUNNER_FAILED analyze316; exit 6; }
echo "RUNNER316_DONE $(date -u +%FT%TZ)"
