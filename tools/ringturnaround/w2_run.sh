#!/usr/bin/env bash
# RingTurnaround W2 runner: chain slopes (per-gap us/op) + landing decode cells in ONE window.
set -euo pipefail
test -n "${OUT:?missing OUT}"
RCW=${RT_CHAIN_WORDS:-}
if [ -n "$RCW" ]; then
  OUT=$OUT RT_CHAIN_WORDS=$RCW bash /var/tmp/ringturn/w2_inner.sh
fi
if [ -n "${RT_PLAN:-}" ]; then
  OUT=$OUT RT_PLAN=$RT_PLAN bash /var/tmp/ringturn/w1_inner.sh
fi
echo "== w2_run done =="
