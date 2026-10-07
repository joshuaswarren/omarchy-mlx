#!/usr/bin/env bash
# Gate 7b — offline system install from the uploaded vendor tar via the tag
# worktree's install.sh (--serve-src needs a repo tree), then the staged-tree
# readback including the post-fix mlx-omarchy-parakeet launcher and the
# launcher path-leak check.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/env.sh"
LOG="$LOG_DIR/g7b-system-install.log"
DEST="/tmp/${TAG}-sysinst"
VDIR="/tmp/${TAG}-vendor-sysinst"
: > "$LOG"
gate_begin "$LOG"

# Driver-staged tag tree (clone + HEAD assert). No manual worktree staging.
if ! gate_ensure_install_tree; then
  gate_log "$LOG" "INSTALL_TREE_STAGE FAIL (TAG_SHA=$TAG_SHA, INSTALL_TREE=$INSTALL_TREE)"
  gate_log "$LOG" "GATE7B_EXIT 1"
  exit 1
fi
gate_log "$LOG" "INSTALL_TREE $INSTALL_TREE HEAD=$(git -C "$INSTALL_TREE" rev-parse HEAD)"

VTAR="$(gate_vtar)"
gate_require_asset "$VTAR" "vendor tar"
rm -rf "$DEST" "$VDIR"
mkdir -p "$DEST" "$VDIR"
tar -xf "$VTAR" -C "$VDIR" --strip-components=1

unshare -n -r env -i PATH="$GATE_INSTALL_PATH" HOME="$GATE_HOME" MLX_OMARCHY_CONV_RING=0 TERM=dumb \
  bash "$INSTALL_TREE/install.sh" --system --dest-root "$DEST" \
  --vendor "$VDIR" --lock "$VDIR/requirements-lock.txt" --import-smoke >>"$LOG" 2>&1
RC=$?
gate_log "$LOG" "SYSINSTALL_EXIT $RC"

gate_log "$LOG" "== staged tree readback =="
SITE=$(echo "$DEST"/usr/lib/omarchy-mlx/venv/lib/python3.*/site-packages)
for f in usr/bin/mlx-omarchy usr/bin/mlx-omarchy-chat usr/bin/mlx-omarchy-parakeet \
         usr/lib/systemd/user/mlx-omarchy-chat.service \
         usr/share/applications/mlx-omarchy-chat.desktop usr/share/omarchy-mlx/paths.sh; do
  [[ -e "$DEST/$f" ]] && gate_log "$LOG" "STAGED_OK $f" || { gate_log "$LOG" "STAGED_MISSING $f"; RC=1; }
done
# Absolute site-packages entry (SITE already resolves under $DEST).
[[ -e "$SITE/mlx_omarchy_paths.py" ]] && gate_log "$LOG" "STAGED_OK site-packages/mlx_omarchy_paths.py" \
  || { gate_log "$LOG" "STAGED_MISSING site-packages/mlx_omarchy_paths.py"; RC=1; }
if grep -R -l -E "$DEST|$VDIR" "$DEST/usr/bin/" >/dev/null 2>&1; then
  gate_log "$LOG" "LAUNCHER_PATH_LEAK yes"; RC=1
else
  gate_log "$LOG" "LAUNCHER_PATH_LEAK no"
fi
gate_log "$LOG" "GATE7B_EXIT $RC"
exit "$RC"
