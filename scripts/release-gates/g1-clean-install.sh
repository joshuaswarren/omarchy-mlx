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
