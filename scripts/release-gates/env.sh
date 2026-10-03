# env.sh — shared parameterisation for the release gate harness.
# Sourced by every gate script. Privacy contract: NO real hostnames,
# addresses, usernames, or personal paths are committed. Every host-specific
# value arrives through the environment (ssh ALIASES, never raw addresses);
# the defaults below are placeholders that fail loudly if used as-is.
#
#   TAG           DRAFT release tag the gates run against (required)
#   ASSETS_DIR    local dir with the draft assets (default /tmp/$TAG-assets):
#                 *cp314*linux_aarch64.whl, omarchy-mlx-vendor-wheels-*.tar,
#                 omarchy-mlx-vendor-wheels-*.tar.sha256, SHA256SUMS,
#                 *cp311*linux_x86_64.whl
#   GATE_ROOT     parent for all throwaway gate state (default $HOME)
#   M2_SSH        ssh ALIAS of the M2 gate host   (placeholder: M2_SSH_ALIAS)
#   JW16_SSH      ssh ALIAS of the jw16 ANE host  (placeholder: JW16_SSH_ALIAS)
#   GPU_LOCK      M2 serialization flock (default /tmp/m2-gpu.lock)
#   TTS_PACK_HOME pinned voice-pack home for gate 6 / gate 8
#   SERVING_VENV  jw16 serving venv path; g7c/g7d refuse to touch it when set
#   INSTALL_TREE  worktree holding the tag's install.sh for gate 7b

: "${TAG:?TAG is required: the DRAFT release tag, e.g. TAG=v0.7.15}"
: "${ASSETS_DIR:=/tmp/${TAG}-assets}"
: "${GATE_ROOT:=$HOME}"
: "${GATE_HOME:=$GATE_ROOT/${TAG}-gate-home}"
: "${GATE7D_HOME:=$GATE_ROOT/${TAG}-gate7d-home}"
: "${LOG_DIR:=$GATE_ROOT/${TAG}-gate-logs}"
: "${HF_CACHE:=$GATE_ROOT/${TAG}-fresh-hf}"
: "${ASSIST_9B:=$GATE_ROOT/${TAG}-assist-9b}"
: "${ASSIST_4B:=$GATE_ROOT/${TAG}-assist-4b}"
: "${ASSIST_OFF:=$GATE_ROOT/${TAG}-assist-off}"
: "${M2_SSH:=M2_SSH_ALIAS}"
: "${JW16_SSH:=JW16_SSH_ALIAS}"
: "${GPU_LOCK:=/tmp/m2-gpu.lock}"
: "${TTS_PACK_HOME:=${HOME}/mlx-tts-home}"
: "${PY_AARCH64:=python3.14}"
: "${INSTALL_TREE:=$GATE_ROOT/${TAG}-worktree}"
: "${GATE_WORKTREE:=$HOME/src/mlx-omarchy-v0715}"
: "${GATE_INSTALL_PATH:=/usr/local/bin:/usr/bin:/bin}"
: "${SERVING_VENV:=}"

GATES_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
mkdir -p "$LOG_DIR"

# Draft assets (gates install from these, never from a published release).
gate_wheel()   { ls "${G7D_WHEEL:-}" 2>/dev/null || ls "$ASSETS_DIR"/*cp314*linux_aarch64.whl 2>/dev/null | head -1; }
gate_vtar()    { ls "${G7D_VTAR:-}" 2>/dev/null || ls "$ASSETS_DIR"/omarchy-mlx-vendor-wheels-*.tar 2>/dev/null | head -1; }
gate_require_asset() { # gate_require_asset <glob-result-var> <description>
  [[ -s "$1" ]] || { echo "REFUSING: $2 not found in $ASSETS_DIR" >&2; return 1; }
}

gate_log() { # gate_log <logfile> <line...> — timestamped line to log AND stdout
  local f="$1"; shift
  printf '%s\n' "$*" | tee -a "$f"
}

gate_begin() { # gate_begin <logfile> — identity header every gate logs first
  gate_log "$1" "BEGIN $(date -u +%FT%TZ) tag=$TAG boot_id=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null) uname=$(uname -r) host_marker=${GATE_HOST_MARKER:-unset}"
}

# Refuse to run a gate when its target state already exists (freshness guard).
gate_refuse_existing() {
  for p in "$@"; do
    if [[ -e "$p" ]]; then echo "REFUSING non-fresh path $p" >&2; exit 2; fi
  done
}

# Free GiB on the filesystem holding $GATE_ROOT — the disk-full admission
# refusal pitfall: check BEFORE the model-download gates, not after.
gate_disk_free_gib() { df -BG "$GATE_ROOT" | tail -1 | awk '{print $4}' | tr -d G; }
