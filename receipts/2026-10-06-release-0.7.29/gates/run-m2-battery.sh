#!/usr/bin/env bash
# M2 battery — wraps the tag's parametrised gate harness on the T6021 gate
# host. Stages the draft assets, then runs scripts/release-gates/run-all.sh
# exactly as v0.7.28 did (jw16 legs OFF; they run separately as g7c/g7d on
# the G13C host once its window opens).
#
# Usage (on the gate host, from a clone of the tag):
#   TAG=v0.7.29 ASSETS_DIR=/tmp/v0.7.29-assets GATE_ROOT=/tmp/m2-gates \
#     bash receipts/2026-10-06-release-0.7.29/gates/run-m2-battery.sh
# Expected PASS:
#   - every gate line "RC=0" in the run-all output,
#   - $GATE_ROOT/v0.7.29-gate-logs/gates.done lists every gate green,
#   - no FAIL lines.
set -euo pipefail
: "${TAG:=v0.7.29}"
: "${ASSETS_DIR:=/tmp/${TAG}-assets}"
: "${GATE_ROOT:=/tmp/m2-gates}"
REPO="${REPO:?set REPO to a checkout of tag $TAG (scripts/release-gates must exist)}"

[[ -f "$REPO/scripts/release-gates/run-all.sh" ]] || { echo "FAIL: run-all.sh not found under $REPO"; exit 1; }
mkdir -p "$ASSETS_DIR"
echo "== assets staged in $ASSETS_DIR =="
ls -l "$ASSETS_DIR"
echo "== battery =="
cd "$REPO"
TAG="$TAG" ASSETS_DIR="$ASSETS_DIR" GATE_ROOT="$GATE_ROOT" RUN_JW16= \
  bash scripts/release-gates/run-all.sh
echo "== summary =="
cat "$GATE_ROOT/$TAG-gate-logs/gates.done"
echo "M2_BATTERY_PASS"
