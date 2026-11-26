#!/bin/bash
# Chains the gate phases; the CPU reference runs beside the GPU arms (correctness
# runs, no timing). Each phase logs to ~/gpu/upr/logs/gate-<phase>.log.
set -uo pipefail
L=$HOME/gpu/upr/logs; G=$HOME/gpu/upr/gate
step() { echo "== $1 start $(date -u +%FT%TZ)"; bash $G/run_gate.sh "$@" > $L/gate-$1.log 2>&1; local rc=$?; echo "== $1 rc=$rc $(date -u +%FT%TZ)"; return $rc; }
mkdir -p $G/out; rm -f $G/out/ref.rc
step facts || { echo RUNNER_FAILED facts; exit 1; }
step validate || { echo RUNNER_FAILED validate; exit 2; }
( step ref; echo $? > $G/out/ref.rc ) &
step arms || { echo RUNNER_FAILED arms; wait; exit 3; }
step perop || { echo RUNNER_FAILED perop; wait; exit 4; }
wait
[ "$(cat $G/out/ref.rc 2>/dev/null)" = 0 ] || { echo RUNNER_FAILED ref; exit 5; }
step analyze || { echo RUNNER_FAILED analyze; exit 6; }
echo "RUNNER_DONE $(date -u +%FT%TZ)"
