#!/usr/bin/env bash
# Test for the stage-whole-bundle extract-from-wheel path (omarchy-mlx#708).
#
#   packaging/test-stage-whole-bundle-from-wheel.sh --wheel PUBLISHED.whl
#
# Builds a minimal work dir whose runtime pin declares
# parakeet-encoder-whole with the digests of the wheel's own embedded
# bundle, then asserts:
#   1. staging from the wheel succeeds and matches those digests;
#   2. staging from a tampered directory fails loudly;
#   3. no source at all refuses.
set -euo pipefail
WHEEL=""
WORK=""
while [ $# -gt 0 ]; do
  case "$1" in
    --wheel) WHEEL=$2; shift 2;;
    --work-dir) WORK=$2; shift 2;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
[ -n "$WHEEL" ] || { echo "usage: $0 --wheel PUBLISHED.whl [--work-dir DIR]" >&2; exit 2; }
[ -f "$WHEEL" ] || { echo "no wheel: $WHEEL" >&2; exit 2; }
WORK=${WORK:-$(mktemp -d)}
SHARE="$WORK/mlx/tools/mlx-omarchy-parakeet/share/mlx-omarchy/parakeet-1"
mkdir -p "$SHARE"
BUNDLE_MEMBERS=('mlx/share/mlx-omarchy/parakeet-1/bundles/parakeet-encoder-whole/manifest.json' \
                'mlx/share/mlx-omarchy/parakeet-1/bundles/parakeet-encoder-whole/program-0.anec')
MANIFEST_SHA=$(unzip -p "$WHEEL" "${BUNDLE_MEMBERS[0]}" | sha256sum | cut -d' ' -f1)
PROGRAM_SHA=$(unzip -p "$WHEEL" "${BUNDLE_MEMBERS[1]}" | sha256sum | cut -d' ' -f1)
python3 - "$MANIFEST_SHA" "$PROGRAM_SHA" "$SHARE/parakeet-runtime-pin.json" <<'EOF'
import json, sys
pin = {"assets": {"bundles": {"parakeet-encoder-whole": {
    "manifest.json": sys.argv[1], "program-0.anec": sys.argv[2]}}}}
with open(sys.argv[3], "w") as fh:
    json.dump(pin, fh)
EOF
SOURCE_THIS='source "$(dirname "$0")/stage-whole-bundle.sh"'
HERE=$(cd "$(dirname "$0")" && pwd)

# 1. extract-from-wheel staging succeeds and matches the wheel's digests
bash -c "set -eu; unset MLX_OMARCHY_WHOLE_BUNDLE_DIR; MLX_OMARCHY_WHOLE_BUNDLE_WHEEL='$WHEEL'; $SOURCE_THIS; stage_whole_bundle '$WORK'" >/dev/null
STAGED="$SHARE/bundles/parakeet-encoder-whole"
[ "$(sha256sum "$STAGED/manifest.json" | cut -d' ' -f1)" = "$MANIFEST_SHA" ] || { echo "FAIL manifest digest after wheel staging"; exit 1; }
[ "$(sha256sum "$STAGED/program-0.anec" | cut -d' ' -f1)" = "$PROGRAM_SHA" ] || { echo "FAIL program digest after wheel staging"; exit 1; }
echo "PASS wheel staging matches the wheel's embedded digests"

# 2. a tampered directory fails loudly
TAMPER=$(mktemp -d)
head -c 64 /dev/zero > "$TAMPER/manifest.json"
head -c 1024 /dev/zero > "$TAMPER/program-0.anec"
if bash -c "set -eu; MLX_OMARCHY_WHOLE_BUNDLE_DIR='$TAMPER'; $SOURCE_THIS; stage_whole_bundle '$WORK'" >/dev/null 2>&1; then
  echo "FAIL tampered dir was accepted"; exit 1
fi
echo "PASS tampered dir refused"

# 3. no source at all refuses
rm -rf "$STAGED"
if bash -c "set -eu; unset MLX_OMARCHY_WHOLE_BUNDLE_DIR MLX_OMARCHY_WHOLE_BUNDLE_WHEEL; $SOURCE_THIS; stage_whole_bundle '$WORK'" >/dev/null 2>&1; then
  echo "FAIL unset source was accepted"; exit 1
fi
echo "PASS unset source refused"

echo "ALL PASS"
