#!/usr/bin/env bash
# run-all.sh — execute the full release gate battery against a DRAFT release
# tag, in order, and write a per-gate exit-code summary.
#
# Usage (on the M2 gate host, assets staged under $ASSETS_DIR):
#   TAG=v0.7.15 RUN_ANE_GATES=1 ANE_GATE_HOST_SSH=<alias> SERVING_VENV=<serving venv path> \
#     bash run-all.sh
#
# Order: g1 install, g2 online-9B, g3 online-4B card, g4 offline, g5 laya,
#        g6 codec, g7a packaged-ICD, g7b system install, g8 kokoro,
#        g9 read-aloud playout (headless Chromium, no GPU),
#        then on the ANE gate host: g7c packaged-ANE-worker verify,
#        g7d fresh-image parakeet transcribe (each in its own gpuwin window,
#        announced to the coordination pane first).
# Exit code: 0 only when every gate passed. ANE-host gates without RUN_ANE_GATES=1
# are recorded as SKIPPED and fail the run loudly — a release never cuts on
# a partial battery.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/env.sh"
DONE="$LOG_DIR/gates.done"
: > "$DONE"

if [[ "$(gate_disk_free_gib)" -lt 25 ]]; then
  echo "REFUSING: only $(gate_disk_free_gib)G free on $GATE_ROOT — the model-download gates WILL hit the disk-full admission refusal; clean spent gate scratch first" >&2
  exit 2
fi

LOCAL_GATES=(g1-clean-install g2-online-9b g3-online-4b-card g4-offline g5-laya
             g6-codec g7a-packaged-icd g7b-system-install g8-kokoro
             g9-speak-queue g10-kokoro-primer g11-card-9b g12-kokoro-stream
             g14-routing)
ANE_GATES=(g7c-ane-worker-verify g7d-fresh-transcribe g13-gdn-maskless
            g15-trig)

FAILED=0
for g in "${LOCAL_GATES[@]}"; do
  echo "=== $g ==="
  if bash "$GATES_DIR/$g.sh"; then RC=0; else RC=$?; fi
  echo "${g}_RC=$RC" >> "$DONE"
  echo "$g RC=$RC"
  [[ $RC -eq 0 ]] || FAILED=1
done

if [[ "$RUN_ANE_GATES" != "1" ]]; then
  for g in "${ANE_GATES[@]}"; do
    echo "${g}_RC=SKIPPED" >> "$DONE"
    echo "SKIPPED $g (RUN_ANE_GATES!=1) — the release battery is INCOMPLETE"
  done
  echo "INCOMPLETE: ANE-host gates skipped; set RUN_ANE_GATES=1 ANE_GATE_HOST_SSH=<alias>" >&2
  exit 2
fi

# ---- ANE gate host: stage scripts + assets, announce, run in gpuwin windows
tar -C "$GATES_DIR/.." -czf "/tmp/${TAG}-release-gates.tgz" "$(basename "$GATES_DIR")"
ssh "$ANE_GATE_HOST_SSH" "rm -rf /tmp/omarchy-release-gates /tmp/${TAG}-gate-logs; mkdir -p /tmp/${TAG}-assets"
scp -q "/tmp/${TAG}-release-gates.tgz" "$ANE_GATE_HOST_SSH:/tmp/"
scp -q "$(gate_wheel)" "$(gate_vtar)" "$ANE_GATE_HOST_SSH:/tmp/${TAG}-assets/" || exit 2
[[ -n "${G7D_OVERLAY_CLI:-}" ]] && scp -q "$G7D_OVERLAY_CLI" "$ANE_GATE_HOST_SSH:/tmp/${TAG}-overlay-cli.py"
ssh "$ANE_GATE_HOST_SSH" "tar -xzf /tmp/${TAG}-release-gates.tgz -C /tmp && mv /tmp/$(basename "$GATES_DIR") /tmp/omarchy-release-gates"

# g13/g15's compile stages are CPU-only: build the doctest binaries over
# plain ssh BEFORE the windows so the gpuwin windows only run them.
echo "=== ANE gate host build stage: g13 g15 test binaries (no gpuwin) ==="
if ssh "$ANE_GATE_HOST_SSH" "TAG='$TAG' GATE_WORKTREE='$GATE_WORKTREE' LOG_DIR='/tmp/${TAG}-gate-logs' bash /tmp/omarchy-release-gates/g13-gdn-maskless.sh build && TAG='$TAG' GATE_WORKTREE='$GATE_WORKTREE' LOG_DIR='/tmp/${TAG}-gate-logs' bash /tmp/omarchy-release-gates/g15-trig.sh build"; then RC=0; else RC=$?; fi
echo "ane_build_stage_RC=$RC" >> "$DONE"
echo "ANE gate host build stage RC=$RC"
[[ $RC -eq 0 ]] || FAILED=1

if command -v herdr >/dev/null 2>&1; then
  herdr pane send-text w72:p1 "Release0715: starting ANE-host gpuwin windows for $TAG gates 7c/7d/13 (llm-inference pauses, auto-restore)" || true
  herdr pane send-keys w72:p1 Enter || true
fi

for g in "${ANE_GATES[@]}"; do
  echo "=== $g (ANE-host gpuwin) ==="
  ENV_PREFIX="TAG='$TAG' ASSETS_DIR='/tmp/${TAG}-assets' LOG_DIR='/tmp/${TAG}-gate-logs' PY_AARCH64='$PY_AARCH64' GATE_WORKTREE='$GATE_WORKTREE'"
  [[ -n "$SERVING_VENV" ]] && ENV_PREFIX+=" SERVING_VENV='$SERVING_VENV'"
  if [[ "$g" == g7d-* && -n "${G7D_OVERLAY_CLI:-}" ]]; then
    ENV_PREFIX+=" G7D_OVERLAY_CLI='/tmp/${TAG}-overlay-cli.py'"
  fi
  if ssh "$ANE_GATE_HOST_SSH" "/var/tmp/appbar/gpuwin.sh \"$ENV_PREFIX bash /tmp/omarchy-release-gates/$g.sh\""; then RC=0; else RC=$?; fi
  echo "${g}_RC=$RC" >> "$DONE"
  echo "$g RC=$RC"
  [[ $RC -eq 0 ]] || FAILED=1
done

scp -q "$ANE_GATE_HOST_SSH:/tmp/${TAG}-gate-logs/*" "$LOG_DIR/" 2>/dev/null || true
ssh "$ANE_GATE_HOST_SSH" "rm -rf /tmp/omarchy-release-gates '/tmp/${TAG}-release-gates.tgz'" || true

echo "=== summary ($DONE) ==="
cat "$DONE"
exit $FAILED
