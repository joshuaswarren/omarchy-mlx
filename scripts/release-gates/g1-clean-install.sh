#!/usr/bin/env bash
# Gate 1 — clean install into a throwaway HOME from the DRAFT assets.
# Runs on the M2 (aarch64): install.sh resolved for $TAG, assets from
# $ASSETS_DIR via MLX_OMARCHY_RELEASE_BASE. Post-fix wheels must also stage
# the mlx-omarchy-parakeet launcher (the g7d defect class).
set -uo pipefail
. "${GATES_DIR:-$(dirname "$(readlink -f "$0")")}/env.sh"
LOG="$LOG_DIR/g1-install.log"
: > "$LOG"
gate_begin "$LOG"

gate_refuse_existing "$GATE_HOME"
rm -rf "$GATE_HOME"; mkdir -p "$GATE_HOME"

if [[ -n "${GATE_INSTALL_SH:-}" ]]; then
  cp "$GATE_INSTALL_SH" "$GATE_ROOT/${TAG}-install.sh"
else
  curl -fsSL "https://raw.githubusercontent.com/joshuaswarren/omarchy-mlx/$TAG/install.sh" \
    -o "$GATE_ROOT/${TAG}-install.sh" 2>>"$LOG"
fi
gate_log "$LOG" "FETCH_EXIT $?"

flock "$GPU_LOCK" env -i PATH="$GATE_INSTALL_PATH" HOME="$GATE_HOME" \
  MLX_OMARCHY_VERSION="$TAG" MLX_OMARCHY_RELEASE_BASE="file://$ASSETS_DIR" TERM=dumb \
  bash "$GATE_ROOT/${TAG}-install.sh" >>"$LOG" 2>&1 </dev/null
rc=$?
gate_log "$LOG" "INSTALL_EXIT $rc $(date -u +%FT%TZ)"

# Wheel identity check: the install path picks the wheel from the SHA256SUMS
# the release base serves. If EXPECTED_WHEEL_SHA256 is set we assert the
# INSTALLED wheel version is the one the expected asset names. Without
# EXPECTED_WHEEL_SHA256 we just log whatever the venv has (a pass on a
# stale SHA256SUMS used to slip through here).
if [[ -n "${EXPECTED_WHEEL_SHA256:-}" ]]; then
  WHEEL_FILE="$(gate_wheel)"
  if [[ -z "$WHEEL_FILE" ]]; then
    gate_log "$LOG" "WHEEL_IDENTITY FAIL: no asset matches EXPECTED_WHEEL_SHA256=$EXPECTED_WHEEL_SHA256 in $ASSETS_DIR"
    rc=1
  else
    WANT_VER="$(basename "$WHEEL_FILE" | sed -E 's#^mlx_omarchy-(.+)-cp[0-9]+-cp[0-9]+-linux_aarch64\.whl$#\1#')"
    GOT_VER="$("$GATE_HOME/.local/share/mlx-omarchy/venv/bin/python" -c \
      'import importlib.metadata as md; print(md.version("mlx_omarchy"))' 2>/dev/null || echo absent)"
    if [[ "$WANT_VER" == "$GOT_VER" ]]; then
      gate_log "$LOG" "WHEEL_IDENTITY installed=$GOT_VER expected=$WANT_VER (sha=$EXPECTED_WHEEL_SHA256)"
    else
      gate_log "$LOG" "WHEEL_IDENTITY MISMATCH installed=$GOT_VER expected=$WANT_VER (sha=$EXPECTED_WHEEL_SHA256)"
      rc=1
    fi
  fi
else
  GOT_VER="$("$GATE_HOME/.local/share/mlx-omarchy/venv/bin/python" -c \
    'import importlib.metadata as md; print(md.version("mlx_omarchy"))' 2>/dev/null || echo absent)"
  gate_log "$LOG" "WHEEL_IDENTITY installed=$GOT_VER (no EXPECTED_WHEEL_SHA256)"
fi

env -i PATH="$GATE_INSTALL_PATH" HOME="$GATE_HOME" \
  "$GATE_HOME/.local/bin/mlx-omarchy-chat" --help >>"$LOG" 2>&1
gate_log "$LOG" "LAUNCHER_HELP_EXIT $?"

# Serve-entry startup check from the INSTALLED home (the release lane
# stages serve files from an explicit list; v0.7.15 shipped without
# perf_placement.py and every user install died at serve startup with
# ModuleNotFoundError). Replicates the launcher environment and executes
# the exact import pair the serve shim runs at startup — no server start,
# no model download. SERVE_ENTRY_OK is REQUIRED for a green gate.
P="$GATE_HOME/.local/share/mlx-omarchy"
env -i PATH="$GATE_INSTALL_PATH" HOME="$GATE_HOME" PYTHONPATH="$P" \
  "$P/venv/bin/python" -c '
import importlib
importlib.import_module("mlx_omarchy_serve._mlxlm_server")
importlib.import_module("mlx_omarchy_serve.catalog")
importlib.import_module("mlx_omarchy_serve.budget")
try:
    from mlx_omarchy_serve.perf_placement import apply_from_env
except ImportError:
    from perf_placement import apply_from_env
print("serve entry imports OK")
' >>"$LOG" 2>&1
SERVE_RC=$?
gate_log "$LOG" "SERVE_ENTRY_EXIT $SERVE_RC"
[[ $SERVE_RC -eq 0 ]] || rc=1

if [[ -x "$GATE_HOME/.local/bin/mlx-omarchy-parakeet" ]]; then
  gate_log "$LOG" "PARAKEET_LAUNCHER staged"
  env -i PATH="$GATE_INSTALL_PATH" HOME="$GATE_HOME" \
    "$GATE_HOME/.local/bin/mlx-omarchy-parakeet" --help >>"$LOG" 2>&1
  gate_log "$LOG" "PARAKEET_LAUNCHER_HELP_EXIT $?"
else
  gate_log "$LOG" "PARAKEET_LAUNCHER MISSING"
  rc=1
fi
ls "$GATE_HOME/.config/systemd/user/" >>"$LOG" 2>&1
gate_log "$LOG" "GATE1_DONE"
exit "$rc"
