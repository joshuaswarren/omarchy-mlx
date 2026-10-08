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
#                 (auto-staged from $TAG by gate_ensure_install_tree when
#                  TAG_SHA is set)
#   TAG_SHA       full 40-char commit sha the tag points at; required by
#                 gate_ensure_install_tree to assert the staged tree
#   EXPECTED_WHEEL_SHA256
#                 when set, gate_wheel() picks the asset whose sha256
#                 matches and REFUSES otherwise
#   EXPECTED_VTAR_SHA256
#                 same contract for the vendor tar

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
: "${TAG_SHA:=}"
: "${EXPECTED_WHEEL_SHA256:=}"
: "${EXPECTED_VTAR_SHA256:=}"

GATES_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
mkdir -p "$LOG_DIR"

# Draft assets (gates install from these, never from a published release).
# When EXPECTED_*_SHA256 is set, the lookup is constrained to the asset whose
# sha256 matches; mismatch or missing = loud REFUSE. The unconstrained lookups
# still work for callers that intentionally point G7D_WHEEL/G7D_VTAR at a file.
gate_wheel() {
  if [[ -n "${G7D_WHEEL:-}" ]]; then ls "$G7D_WHEEL" 2>/dev/null; return; fi
  if [[ -n "$EXPECTED_WHEEL_SHA256" ]]; then
    local f
    f="$( cd "$ASSETS_DIR" && sha256sum -- *cp314*linux_aarch64.whl 2>/dev/null \
      | awk -v s="$EXPECTED_WHEEL_SHA256" '$1==s{print $2}' | head -1 )"
    [[ -n "$f" ]] && echo "$ASSETS_DIR/$f"
  else
    ls "$ASSETS_DIR"/*cp314*linux_aarch64.whl 2>/dev/null | head -1
  fi
}
gate_vtar() {
  if [[ -n "${G7D_VTAR:-}" ]]; then ls "$G7D_VTAR" 2>/dev/null; return; fi
  if [[ -n "$EXPECTED_VTAR_SHA256" ]]; then
    local f
    f="$( cd "$ASSETS_DIR" && sha256sum -- omarchy-mlx-vendor-wheels-*.tar 2>/dev/null \
      | awk -v s="$EXPECTED_VTAR_SHA256" '$1==s{print $2}' | head -1 )"
    [[ -n "$f" ]] && echo "$ASSETS_DIR/$f"
  else
    ls "$ASSETS_DIR"/omarchy-mlx-vendor-wheels-*.tar 2>/dev/null | head -1
  fi
}
gate_require_asset() { # gate_require_asset <path> <description>
  [[ -s "$1" ]] || { echo "REFUSING: $2 not found ($1)" >&2; return 1; }
}
# gate_ensure_install_tree — clone the tag into $INSTALL_TREE (default
# $GATE_ROOT/$TAG-worktree) and assert HEAD == $TAG_SHA. Idempotent: if the
# tree is already at the right sha, leave it. Otherwise re-clone: an
# extracted tarball (no .git/) or a stale tree is replaced by a fresh
# clone so the HEAD assert is meaningful every time. The "no manual
# staging" entry point called by g7b.
gate_ensure_install_tree() {
  : "${TAG_SHA:?TAG_SHA is required: full 40-char commit sha the tag points at}"
  : "${INSTALL_TREE:=$GATE_ROOT/${TAG}-worktree}"
  local want_head
  want_head="$TAG_SHA"
  local got
  got="$(git -C "$INSTALL_TREE" rev-parse HEAD 2>/dev/null || echo absent)"
  if [[ "$got" != "$want_head" ]]; then
    rm -rf "$INSTALL_TREE"
    git clone --depth 1 --branch "$TAG" \
      https://github.com/joshuaswarren/omarchy-mlx "$INSTALL_TREE" || return 1
    got="$(git -C "$INSTALL_TREE" rev-parse HEAD 2>/dev/null || echo absent)"
  fi
  [[ "$got" == "$want_head" ]] || {
    echo "REFUSING: INSTALL_TREE HEAD $got != TAG_SHA $want_head" >&2
    return 1
  }
  export INSTALL_TREE
}

gate_log() { # gate_log <logfile> <line...> — timestamped line to log AND stdout
  local f="$1"; shift
  printf '%s\n' "$*" | tee -a "$f"
}

# gate_log_wheel_identity <logfile> — print the INSTALLED wheel version and
# (when EXPECTED_WHEEL_SHA256 is set) the version implied by the expected
# asset, so every gate's log can be tied to a wheel identity. g1 calls this
# inline with its own MISMATCH/ok decision; other gates log a single line
# for the receipt.
gate_log_wheel_identity() {
  local log="$1"
  local got want
  got="$("$GATE_HOME/.local/share/mlx-omarchy/venv/bin/python" -c \
    'import importlib.metadata as md; print(md.version("mlx_omarchy"))' 2>/dev/null || echo absent)"
  if [[ -n "${EXPECTED_WHEEL_SHA256:-}" ]]; then
    local w
    w="$(gate_wheel 2>/dev/null)"
    want="$(basename "$w" 2>/dev/null | sed -E 's#^mlx_omarchy-(.+)-cp[0-9]+-cp[0-9]+-linux_aarch64\.whl$#\1#')"
    if [[ "$got" == "$want" && "$got" != "absent" ]]; then
      gate_log "$log" "WHEEL_IDENTITY installed=$got (sha=$EXPECTED_WHEEL_SHA256)"
    else
      gate_log "$log" "WHEEL_IDENTITY MISMATCH installed=$got expected=$want (sha=$EXPECTED_WHEEL_SHA256)"
    fi
  else
    gate_log "$log" "WHEEL_IDENTITY installed=$got"
  fi
}

gate_begin() { # gate_begin <logfile> — identity header every gate logs first
  gate_log "$1" "BEGIN $(date -u +%FT%TZ) tag=$TAG boot_id=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null) uname=$(uname -r) host_marker=${GATE_HOST_MARKER:-unset}"
}

# gate_lock — serialize GPU access. Two modes:
#   GPU_TURN_TICKET=1  inside a gpu-turn ticket: no-op. gpu-turn holds
#                      $GPU_LOCK for the whole ticket; the DRIVER (not
#                      gpu-turn itself, which exports nothing) must set
#                      GPU_TURN_TICKET=1 for the wrapped command.
#   unset (raw)        wraps the command in flock. The lock is probed
#                      first: if it is already held, gate_lock prints the
#                      named GPU_LOCK_BUSY error and returns 99 instead of
#                      silently hanging for the probe deadline.
gate_lock() {
  local -a opts=()
  while [[ "${1:-}" == -* ]]; do
    if [[ "$1" == "-w" ]]; then opts+=(-w "$2"); shift 2; else opts+=("$1"); shift; fi
  done
  if [[ -n "${GPU_TURN_TICKET:-}" ]]; then
    "$@"
    return
  fi
  if ! flock -n "$GPU_LOCK" true 2>/dev/null; then
    echo "GPU_LOCK_BUSY: $GPU_LOCK is held by another process and GPU_TURN_TICKET is not set." >&2
    echo "  Wrap this gate in a gpu-turn ticket whose driver exports GPU_TURN_TICKET=1," >&2
    echo "  or wait for the current lock holder to finish." >&2
    return 99
  fi
  flock "${opts[@]}" "$GPU_LOCK" "$@"
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
