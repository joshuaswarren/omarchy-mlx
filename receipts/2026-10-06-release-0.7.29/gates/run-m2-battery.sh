#!/usr/bin/env bash
# run-m2-battery — the M2 gate battery with the explicit pass rule (w7K B1):
#
#   PASS means: run-all.sh exits 2 (its by-design INCOMPLETE code when the
#   jw16 legs are skipped), gates.done holds <g>_RC=0 for each of the 14
#   LOCAL_GATES, and the ONLY other lines are the 4 jw16 gates as SKIPPED.
#   Anything else is FAIL.
#
# Also (w7K should): verify the staged assets against SHA256SUMS before the
# battery. The jw16 legs (g7c, g7d, g13, g15) run separately — see
# jw16-gates.sh; the release is complete only when BOTH pass.
#
# Usage (on the gate host, REPO = extracted tag tree with scripts/release-gates):
#   TAG=v0.7.29 ASSETS_DIR=$HOME/v0.7.29-assets GATE_ROOT=$HOME/v0.7.29-gates \
#   REPO=$HOME/v0.7.29-repo \
#   GATE_INSTALL_SH=$REPO/install.sh GATE_TAG_TARBALL=$ASSETS_DIR/omarchy-mlx-0.7.29.tar.gz \
#     bash run-m2-battery.sh
set -uo pipefail
: "${TAG:=v0.7.29}"
: "${ASSETS_DIR:=/tmp/${TAG}-assets}"
: "${GATE_ROOT:=/tmp/m2-gates}"
REPO="${REPO:?set REPO to the extracted tag tree (scripts/release-gates must exist)}"
GATES_DIR="$REPO/scripts/release-gates"
LOG_DIR="$GATE_ROOT/$TAG-gate-logs"
DONE="$LOG_DIR/gates.done"

[[ -f "$GATES_DIR/run-all.sh" ]] || { echo "FAIL: run-all.sh not found under $REPO"; exit 1; }
mkdir -p "$ASSETS_DIR" "$LOG_DIR"

echo "== assets (SHA256SUMS) =="
cd "$ASSETS_DIR"
if sha256sum -c --ignore-missing --quiet SHA256SUMS; then
  echo "ASSETS_OK"
else
  echo "FAIL: staged assets do not match SHA256SUMS"; exit 1
fi

echo "== battery =="
cd "$REPO"
rc=0
TAG="$TAG" ASSETS_DIR="$ASSETS_DIR" GATE_ROOT="$GATE_ROOT" RUN_JW16= \
  bash "$GATES_DIR/run-all.sh"
rc=$?

echo "== gates.done =="
cat "$DONE"

echo "== explicit pass rule (w7K B1) =="
LOCAL_GATES=(g1-clean-install g2-online-9b g3-online-4b-card g4-offline g5-laya
             g6-codec g7a-packaged-icd g7b-system-install g8-kokoro
             g9-speak-queue g10-kokoro-primer g11-card-9b g12-kokoro-stream
             g14-routing)
JW16_GATES=(g7c-ane-worker-verify g7d-fresh-transcribe g13-gdn-maskless g15-trig)
fail=0
for g in "${LOCAL_GATES[@]}"; do
  grep -q "^${g}_RC=0\$" "$DONE" || { echo "FAIL: $g is not RC=0"; fail=1; }
done
expected_skips=0
for g in "${JW16_GATES[@]}"; do
  if grep -q "^${g}_RC=SKIPPED\$" "$DONE"; then
    expected_skips=$((expected_skips+1))
  else
    echo "FAIL: $g missing its SKIPPED line"; fail=1
  fi
done
total_lines=$(grep -c "_RC=" "$DONE")
expected_lines=$(( ${#LOCAL_GATES[@]} + ${#JW16_GATES[@]} ))
if (( total_lines != expected_lines )); then
  echo "FAIL: gates.done has $total_lines lines, expected $expected_lines"; fail=1
fi
if (( rc != 2 )); then
  echo "FAIL: run-all.sh exited $rc, expected the by-design 2 (INCOMPLETE)"; fail=1
fi
if (( fail == 0 )); then
  echo "M2_BATTERY_PASS (14 local RC=0; only the 4 jw16 gates SKIPPED; run-all rc=2)"
  exit 0
fi
echo "M2_BATTERY_FAIL"
exit 1
