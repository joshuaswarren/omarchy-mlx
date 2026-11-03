#!/bin/bash
# The whole gate in Joshua's order, resumable: #316, then #317, then #321. Every
# phase skips finished outputs; nothing new starts at or after 20:00Z (a phase
# already running is left to finish or is stopped by hand, losing at most the run
# in flight). Waits for an earlier runner or ab.py first, so no two runners overlap.
set -uo pipefail
L=$HOME/gpu/upr/logs; G=$HOME/gpu/upr/gate; O=$G/out
GR=granite-4.0-1b-4bit; QW=Qwen2.5-0.5B-Instruct-4bit; LL=Llama-3.2-1B-Instruct-bf16
step() {
  if [ "$(date -u +%H%M)" -ge "${STOP_AT:-2000}" ]; then echo "DEADLINE_STOP before $1 $(date -u +%FT%TZ)"; exit 0; fi
  echo "== $1 start $(date -u +%FT%TZ)"; bash $G/run_gate.sh "$@" >> $L/gate-$1.log 2>&1; local rc=$?; echo "== $1 rc=$rc $(date -u +%FT%TZ)"; return $rc
}
need_ref() {  # need_ref <model...>: start the missing references unless one is running, then wait
  local m
  for m in "$@"; do
    while [ ! -s $O/ref-$m.json ]; do
      if ! pgrep -f "[g]ate_tf.py (measure --cpu|merge)" > /dev/null; then step ref "$m" || return 1; fi
      [ -s $O/ref-$m.json ] || sleep 15
    done
  done
}
while pgrep -f "[r]unner316.sh|[a]b.py .*ab-3" > /dev/null; do sleep 15; done
step arms316 && step perop316 && need_ref $GR $QW && step analyze316 || { echo RUNNER_FAILED 316; exit 3; }
echo "DONE 316 $(date -u +%FT%TZ)"
step arms317 && step perop317 && need_ref $LL && step analyze317 || { echo RUNNER_FAILED 317; exit 4; }
echo "DONE 317 $(date -u +%FT%TZ)"
step arms321 && step perop321 && need_ref $GR && step analyze321 || { echo RUNNER_FAILED 321; exit 5; }
echo "DONE 321 $(date -u +%FT%TZ)"
echo "RUNNER_ALL_DONE $(date -u +%FT%TZ)"
