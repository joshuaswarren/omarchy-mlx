#!/usr/bin/env bash
# jw16-gates — the jw16 window plan (w7K B2): g7c, g7d, g13, g15 together
# with their CPU build stages. Runs ON the jw16 host after the cluster.
#
# Stages (in order):
#   0. assets verified against SHA256SUMS; tag tree present (tarball extract)
#   1. CPU build stage: a wheel build tree at $GATE_WORKTREE/.work/mlx
#      (build-wheel.sh from the tag; g13/g15 reconfigure it with
#      MLX_BUILD_OMARCHY/TESTS=ON — CPU-only)
#   2. g13-gdn-maskless build   (CPU-only)
#   3. g15-trig build           (CPU-only)
#   4. g7c-ane-worker-verify    (GPU, ticket-or-lock)
#   5. g7d-fresh-transcribe     (GPU, ticket-or-lock)
#   6. g13-gdn-maskless run     (GPU, ticket-or-lock)
#   7. g15-trig run             (GPU, ticket-or-lock)
# Every RC is printed and teed; the summary line is JW16_GATES_PASS only when
# all seven steps are RC=0.
set -uo pipefail
: "${TAG:?set TAG=v0.7.30 (no defaults: a wrong tag silently tests the wrong cut)}"
: "${ASSETS_DIR:=$HOME/v0.7.29-assets}"
: "${GATE_ROOT:?set GATE_ROOT (no defaults)}"
: "${REPO:?set REPO (no defaults)}"
# The serving venv on this host: g7c REFUSES to touch it when named.
: "${SERVING_VENV:=$HOME/.local/share/mlx-omarchy/venv}"
export TAG ASSETS_DIR GATE_ROOT REPO SERVING_VENV
export GATE_WORKTREE="$REPO"          # g13/g15 use $GATE_WORKTREE/.work/mlx
GATES_DIR="$REPO/scripts/release-gates"
LOG_DIR="$GATE_ROOT/$TAG-gate-logs"
export LOG_DIR
RECEIPT="${JW16_RECEIPT:-$PWD/jw16-gates-receipt-$(date -u +%Y%m%dT%H%M%SZ).log}"
: > "$RECEIPT"
log() { tee -a "$RECEIPT"; }

# gpu-turn ticket when present; flock of the M1-family host lock otherwise.
if [[ -z "${GPU_LOCK:-}" ]]; then
  GPU_LOCK=/tmp/m1-gpu.lock
fi
runner() {
  if command -v gpu-turn >/dev/null 2>&1; then
    gpu-turn -m 15 timeout -k 60 900 "$@"
  else
    timeout -k 60 900 flock "$GPU_LOCK" "$@"
  fi
}

step() { # step <name> <cmd...>
  local name="$1"; shift
  echo "=== $name $(date -u +%T) ===" | log
  local s e rc
  s=$(date -u +%s)
  if "$@" >>"$RECEIPT" 2>&1; then rc=0; else rc=$?; fi
  e=$(date -u +%s)
  echo "STEP ${name}_RC=$rc wall=$((e-s))s" | log
  (( rc == 0 )) || { echo "FAIL at $name (rc=$rc)" | log; exit $rc; }
}

echo "== stage 0: assets + tag tree ==" | log
cd "$ASSETS_DIR"
while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  f="$(echo "$line" | awk '{print $2}')"
  [[ -f "$f" ]] || { echo "FAIL: staged asset missing: $f" | log; exit 1; }
done < SHA256SUMS
sha256sum -c --quiet SHA256SUMS | log && echo "ASSETS_OK" | log \
  || { echo "FAIL: assets do not match SHA256SUMS" | log; exit 1; }
if [[ ! -d "$REPO/scripts/release-gates" ]]; then
  mkdir -p "$REPO"
  tar -xf "$ASSETS_DIR/omarchy-mlx-0.7.29.tar.gz" -C "$REPO" --strip-components=1
fi
[[ -f "$GATES_DIR/g13-gdn-maskless.sh" ]] || { echo "FAIL: gate scripts missing under $REPO" | log; exit 1; }

echo "== stage 1: wheel build tree (CPU) ==" | log
if [[ ! -d "$REPO/.work/mlx" ]]; then
  BUNDLE="$ASSETS_DIR/bundle-extract"
  if [[ ! -d "$BUNDLE/mlx/share/mlx-omarchy/parakeet-1/bundles/parakeet-encoder-whole" ]]; then
    mkdir -p "$BUNDLE"
    python3 -m zipfile -e "$(echo "$ASSETS_DIR"/mlx_omarchy-*-cp314-cp314-linux_aarch64.whl)" "$BUNDLE"
  fi
  # The tag tarball has no .git, so build-wheel cannot stamp the commit by
  # itself: pin it to the tag sha (keeps the dev version segment identical to
  # the published wheel).
  step build-wheel env HOME="$HOME" DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=8 \
    MLX_OMARCHY_SOURCE_COMMIT="${MLX_OMARCHY_SOURCE_COMMIT:-d86daf9}" \
    MLX_OMARCHY_WHOLE_BUNDLE_DIR="$BUNDLE/mlx/share/mlx-omarchy/parakeet-1/bundles/parakeet-encoder-whole" \
    bash "$REPO/scripts/build-wheel.sh"
fi
[[ -d "$REPO/.work/mlx" ]] || { echo "FAIL: no build tree at $REPO/.work/mlx" | log; exit 1; }

step g13-build bash "$GATES_DIR/g13-gdn-maskless.sh" build
step g15-build bash "$GATES_DIR/g15-trig.sh" build
step g7c runner bash "$GATES_DIR/g7c-ane-worker-verify.sh"
step g7d runner bash "$GATES_DIR/g7d-fresh-transcribe.sh"
step g13-run runner bash "$GATES_DIR/g13-gdn-maskless.sh" run
step g15-run runner bash "$GATES_DIR/g15-trig.sh" run

echo "JW16_GATES_PASS receipt=$RECEIPT" | log
