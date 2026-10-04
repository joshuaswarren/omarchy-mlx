#!/usr/bin/env bash
set -euo pipefail
/var/tmp/gdubar-v0726-venv/bin/python "$1" /var/tmp/dfuse/bitexact/failing-operands.npz /var/tmp/dfuse/bitexact/stage-call0.json > /var/tmp/dfuse/bitexact/stage.log 2>&1 || { echo STAGE-FAILED; tail -6 /var/tmp/dfuse/bitexact/stage.log; exit 1; }
echo STAGE-OK
