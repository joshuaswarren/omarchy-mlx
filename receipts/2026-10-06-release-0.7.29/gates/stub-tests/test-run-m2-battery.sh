#!/usr/bin/env bash
# Stub harness for run-m2-battery.sh (w7K B1): the explicit pass rule.
#   1 positive: run-all stub exits 2 with 14 local _RC=0 + 4 SKIPPED -> PASS
#   2 negative: one local gate RC=1                            -> FAIL
#   3 negative: a jw16 gate ran locally (no SKIPPED line)      -> FAIL
# Usage: bash stub-tests/test-run-m2-battery.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE="$HERE/../run-m2-battery.sh"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/batt-stub.XXXX")"
FAILED=0

make_repo() { # $1 = case dir, $2 = g5 rc, $3 = emit jw16 SKIPPED lines (0/1)
  local repo="$1/repo"
  mkdir -p "$repo/scripts/release-gates" "$1/assets" "$1/gates/v0.7.29-test-gate-logs"
  local done_f="$1/gates/v0.7.29-test-gate-logs/gates.done"
  : >"$done_f"
  for g in g1-clean-install g2-online-9b g3-online-4b-card g4-offline g5-laya \
           g6-codec g7a-packaged-icd g7b-system-install g8-kokoro g9-speak-queue \
           g10-kokoro-primer g11-card-9b g12-kokoro-stream g14-routing; do
    if [[ "$g" == g5-laya && "$2" != 0 ]]; then
      echo "${g}_RC=$2" >>"$done_f"
    else
      echo "${g}_RC=0" >>"$done_f"
    fi
  done
  if (( $3 )); then
    for g in g7c-ane-worker-verify g7d-fresh-transcribe g13-gdn-maskless g15-trig; do
      echo "${g}_RC=SKIPPED" >>"$done_f"
    done
  fi
  cat >"$repo/scripts/release-gates/run-all.sh" <<'EOF'
#!/usr/bin/env bash
DONE="$GATE_ROOT/$TAG-gate-logs/gates.done"
for g in g1-clean-install g2-online-9b g3-online-4b-card g4-offline g5-laya \
         g6-codec g7a-packaged-icd g7b-system-install g8-kokoro g9-speak-queue \
         g10-kokoro-primer g11-card-9b g12-kokoro-stream g14-routing; do
  if [[ "$g" == g5-laya && "${STUB_G5_RC:-0}" != 0 ]]; then
    echo "${g}_RC=${STUB_G5_RC}" >> "$DONE"
  else
    echo "${g}_RC=0" >> "$DONE"
  fi
done
if [[ "${STUB_JW16_SKIPPED:-1}" == 1 ]]; then
  for g in g7c-ane-worker-verify g7d-fresh-transcribe g13-gdn-maskless g15-trig; do
    echo "${g}_RC=SKIPPED" >> "$DONE"
  done
fi
exit 2
EOF
  chmod +x "$repo/scripts/release-gates/run-all.sh"
  echo stub >"$1/assets/stub-asset.bin"
  printf '%s  stub-asset.bin\n' "$(sha256sum "$1/assets/stub-asset.bin" | awk '{print $1}')" >"$1/assets/SHA256SUMS"
}

run_case() { # $1 name, $2 g5 rc, $3 jw16 skipped
  local name="$1"
  local case_dir="$WORK/$name"
  make_repo "$case_dir" "$2" "$3"
  export STUB_G5_RC="$2" STUB_JW16_SKIPPED="$3"
  (
    export TAG=v0.7.29-test
    export ASSETS_DIR="$case_dir/assets"
    export GATE_ROOT="$case_dir/gates"
    export REPO="$case_dir/repo"
    set -euo pipefail
    bash "$GATE"
  ) >"$case_dir/out.log" 2>&1
  echo $? >"$case_dir/rc"
}

check() { if (( $2 == 0 )); then echo "ok: $1"; else echo "FAIL: $1"; FAILED=1; fi; }

run_case positive 0 1
rc=$(cat "$WORK/positive/rc")
check "positive rc=0" $([[ $rc == 0 ]]; echo $?)
grep -q "M2_BATTERY_PASS" "$WORK/positive/out.log"; check "positive M2_BATTERY_PASS" $?

run_case one_fail 1 1
rc=$(cat "$WORK/one_fail/rc")
check "one-fail rc=1" $([[ $rc == 1 ]]; echo $?)
grep -q "M2_BATTERY_FAIL" "$WORK/one_fail/out.log"; check "one-fail M2_BATTERY_FAIL" $?
grep -q "g5-laya is not RC=0" "$WORK/one_fail/out.log"; check "one-fail names the gate" $?

run_case no_skip 0 0
rc=$(cat "$WORK/no_skip/rc")
check "no-skip rc=1" $([[ $rc == 1 ]]; echo $?)
grep -q "M2_BATTERY_FAIL" "$WORK/no_skip/out.log"; check "no-skip M2_BATTERY_FAIL" $?

run_case rerun_hygiene 0 1
mkdir -p "$WORK/rerun_hygiene/gates/v0.7.29-test-gate-home/.local/bin" "$WORK/rerun_hygiene/gates/v0.7.29-test-gate-logs"
echo stale >"$WORK/rerun_hygiene/gates/v0.7.29-test-gate-home/.local/bin/mlx-omarchy-chat"
echo stale >"$WORK/rerun_hygiene/gates/v0.7.29-test-gate-logs/gates.done"
(
  export TAG=v0.7.29-test
  export ASSETS_DIR="$WORK/rerun_hygiene/assets"
  export GATE_ROOT="$WORK/rerun_hygiene/gates"
  export REPO="$WORK/rerun_hygiene/repo"
  set -euo pipefail
  bash "$GATE"
) >"$WORK/rerun_hygiene/out2.log" 2>&1
echo $? >"$WORK/rerun_hygiene/rc2"
check "rerun rc=0" $([[ $(cat "$WORK/rerun_hygiene/rc2") == 0 ]]; echo $?)
[[ -e "$WORK/rerun_hygiene/gates/v0.7.29-test-gate-home" ]] && r=1 || r=0
check "rerun cleared stale gate-home" $r

echo "== raw outputs =="
for c in positive one_fail no_skip; do echo "--- $c (rc=$(cat "$WORK/$c/rc"))"; tail -4 "$WORK/$c/out.log"; done
if (( FAILED )); then echo "STUB_HARNESS run-m2-battery: FAIL"; exit 1; fi
echo "STUB_HARNESS run-m2-battery: PASS"
rm -rf "$WORK"
