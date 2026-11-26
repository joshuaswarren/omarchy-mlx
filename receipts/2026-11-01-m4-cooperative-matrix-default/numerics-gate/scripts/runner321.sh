#!/bin/bash
# #321 gate, resumable: arms321 (5+5, HK_APPLE9_COOPMAT=0 on both arms), perop321, analyze321.
# Every phase skips finished outputs. Started 2026-11-02 evening round.
set -uo pipefail
L=$HOME/gpu/upr/logs; G=$HOME/gpu/upr/gate; O=$G/out
step() { echo "== $1 start $(date -u +%FT%TZ)"; bash $G/run_gate.sh "$@" >> $L/gate-$1.log 2>&1; local rc=$?; echo "== $1 rc=$rc $(date -u +%FT%TZ)"; return $rc; }
[ -s $O/ref-granite-4.0-1b-4bit.json ] || { echo "no granite ref"; exit 2; }
step arms321 && step perop321 && step analyze321 || { echo RUNNER_FAILED 321; exit 5; }
echo "DONE 321 $(date -u +%FT%TZ)"
