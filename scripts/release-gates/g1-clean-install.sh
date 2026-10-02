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
