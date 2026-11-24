#!/usr/bin/env bash
# chain_builds.sh MAIN HEAD: the A/B builds, one lock hold per build step.
set -uo pipefail
L=${LANE}
lk() { flock -w 5400 /tmp/m1-gpu.lock timeout --foreground 5400 "$@" < /dev/null; }
lk bash $L/scripts/build_wheel.sh main $1 > $L/logs/build-main.log 2>&1; echo "main rc=$?"
lk bash $L/scripts/build_wheel.sh head $2 > $L/logs/build-head.log 2>&1; echo "head rc=$?"
lk bash $L/scripts/build_tests.sh head > $L/logs/build-tests-head.log 2>&1; echo "tests rc=$?"
lk bash $L/scripts/build_nocpu.sh > $L/logs/build-nocpu.log 2>&1; echo "nocpu rc=$?"
echo BUILDS_DONE
