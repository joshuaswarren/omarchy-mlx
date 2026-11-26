#!/usr/bin/env bash
# Decode-pair release gate for qualification kit tags (drill d-0075 p2).
#
# Before a qualification tag is cut, run the kit decode and the packaged-
# install decode on the same model on the reference M1 Max, inside the
# host's guard window. If the two rates differ by more than 5%, the tag is
# refused. Both rates, the delta, and the verdict are written into a JSON
# receipt that the release receipt embeds verbatim.
#
# Required env:
#   KIT_TAG            tag being gated, e.g. qualification-r8
#   KIT_GATE_HOST_SSH  ssh ALIAS of the reference host (never a raw address)
#   KIT_DIR            sealed kit directory on the host (holds qualify.py)
#   PACKAGED_PYTHON    packaged install's venv python on the host
#   MODEL_DIR          pinned model snapshot path on the host
# Optional env:
#   KIT_TOKENS         decode length (default 64, the kit's pinned length)
#   GUARD_WRAP         guard-window wrapper on the host
#                      (default /var/tmp/appbar/gpuwin.sh)
#   RECEIPT            local receipt path
#                      (default receipts/decode-pair-$KIT_TAG.json)
#   KIT_JSON / PACKAGED_JSON
#                      pre-collected result JSONs; skips the host run
#                      (the kit receipt's benchmark block, or one
#                      bench_decode result line)
#
# Exit: 0 gate passed (receipt written); 2 refused or unusable input.

set -euo pipefail
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPTS_DIR")"
GATE_PY="$REPO_DIR/qualification-kit/scripts/decode_pair.py"

: "${KIT_TAG:?KIT_TAG is required: the qualification tag being gated}"
: "${KIT_TOKENS:=64}"
: "${RECEIPT:=$REPO_DIR/receipts/decode-pair-$KIT_TAG.json}"

extract() {  # extract <json-or-log-file> -> prints "<tps> <ids_sha>"
  python3 - "$1" <<'PY'
import json, sys
d = None
try:
    d = json.load(open(sys.argv[1]))
except (json.JSONDecodeError, UnicodeDecodeError):
    pass
if d is None:
    for line in open(sys.argv[1]):
        line = line.strip()
        if line.startswith('{'):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                pass
if d is None:
    sys.exit(f'no JSON line in {sys.argv[1]}')
b = d.get('benchmark') if isinstance(d.get('benchmark'), dict) else d
tps, ids = b.get('decode_tps'), b.get('ids_sha256_16')
if not isinstance(tps, (int, float)) or tps <= 0:
    sys.exit(f'no usable decode_tps in {sys.argv[1]}')
print(tps, ids or '')
PY
}

read_pair() {  # read_pair <json-or-log-file> <arm-name> -> sets TPS IDS
  local pair
  pair="$(extract "$1")" || {
    echo "REFUSING: $2 arm produced no usable decode result ($1)" >&2
    exit 2
  }
  TPS="${pair%% *}"
  IDS="${pair##* }"
}

if [[ -n "${KIT_JSON:-}" && -n "${PACKAGED_JSON:-}" ]]; then
  read_pair "$KIT_JSON" kit;  KIT_TPS="$TPS";  KIT_IDS="$IDS"
  read_pair "$PACKAGED_JSON" packaged;  PKG_TPS="$TPS";  PKG_IDS="$IDS"
else
  : "${KIT_GATE_HOST_SSH:?KIT_GATE_HOST_SSH is required: ssh alias of the reference host}"
  : "${KIT_DIR:?KIT_DIR is required: sealed kit dir on the host}"
  : "${PACKAGED_PYTHON:?PACKAGED_PYTHON is required: packaged venv python on the host}"
  : "${MODEL_DIR:?MODEL_DIR is required: pinned model snapshot on the host}"
  : "${GUARD_WRAP:=/var/tmp/appbar/gpuwin.sh}"

  REMOTE=/tmp/kit-decode-pair-$KIT_TAG
  ssh "$KIT_GATE_HOST_SSH" "mkdir -p $REMOTE"
  # bench_decode imports caps_sim_guard, defer_tokens_guard and mlx_provenance
  # from its own directory; all are repo scripts, not part of the packaged install.
  scp -q "$SCRIPTS_DIR/bench_decode.py" "$SCRIPTS_DIR/caps_sim_guard.py" \
    "$SCRIPTS_DIR/defer_tokens_guard.py" "$SCRIPTS_DIR/mlx_provenance.py" \
    "$KIT_GATE_HOST_SSH:$REMOTE/"
  KIT_CMD="cd $KIT_DIR && python3 qualify.py --benchmark-only --machine-id dec0de00dec0de00 --model-dir $MODEL_DIR --out $REMOTE/kit.json && cat $REMOTE/kit.json"
  PKG_CMD="$PACKAGED_PYTHON $REMOTE/bench_decode.py --model $MODEL_DIR --tokens $KIT_TOKENS"
  kit_out="$(ssh "$KIT_GATE_HOST_SSH" "$GUARD_WRAP \"$KIT_CMD\"")"
  pkg_json="$(ssh "$KIT_GATE_HOST_SSH" "$GUARD_WRAP \"$PKG_CMD\"")"

  KIT_JSON="$REMOTE-dir-kit.json"
  PACKAGED_JSON="$REMOTE-dir-pkg.json"
  printf '%s' "$kit_out" > "$KIT_JSON"
  printf '%s' "$pkg_json" > "$PACKAGED_JSON"
  read_pair "$KIT_JSON" kit;  KIT_TPS="$TPS";  KIT_IDS="$IDS"
  read_pair "$PACKAGED_JSON" packaged;  PKG_TPS="$TPS";  PKG_IDS="$IDS"
fi

mkdir -p "$(dirname "$RECEIPT")"
python3 "$GATE_PY" \
  --kit-tps "$KIT_TPS" --packaged-tps "$PKG_TPS" \
  --kit-ids "$KIT_IDS" --packaged-ids "$PKG_IDS" \
  --model "${MODEL_DIR:-${KIT_JSON:-pre-collected}}" \
  --tokens "$KIT_TOKENS" --tag "$KIT_TAG" --out "$RECEIPT"
