#!/usr/bin/env bash
# Focused host verification for the pinned ANE compiler (gate 46, host only).
#
# Proves, from a clean work area:
#   1. a wrong source archive is rejected before anything is built,
#   2. the pinned compiler builds and passes the compiler's host checks,
#   3. a known source-native H13 fixture compiles into a package with the
#      locked schema and passes device-free package inspection.
#
# This script never touches a device, never claims bundle adaptation
# (load_bundle), and never claims runtime qualification. The compiler's
# `emit` mode compiles H16G smoke fixtures and is NOT treated as H13
# evidence anywhere here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/ane-compiler.lock"

WORK_DIR="${MLX_OMARCHY_WORK_DIR:-$ROOT/.work}"
COMPILER_WORK="$WORK_DIR/ane-compiler"
ARCHIVE="$COMPILER_WORK/mil-hwx-compiler-$ANE_COMPILER_COMMIT.tar.gz"
SOURCE_DIR="$COMPILER_WORK/mil-hwx-compiler"
WRAPPER="$COMPILER_WORK/bin/mil-hwxc"

# Known H13 graph: the canonical conv+relu fixture from the locked source
# tree, with its in-tree model weights. The compiler's H16G `emit` smoke
# uses the same fixture; this qualification pins target H13 explicitly.
H13_FIXTURE="${H13_FIXTURE:-conv_relu}"
H13_MODEL_ROOT="${H13_MODEL_ROOT:-$SOURCE_DIR/tests/models/conv_relu}"

note() { printf '\n=== %s ===\n' "$1"; }

clean_area() {
  note "clean work area"
  rm -rf "$COMPILER_WORK"
}

prove_hash_rejection() {
  note "source hash rejection before build"
  mkdir -p "$COMPILER_WORK"
  # Valid gzip bytes that do not match the locked SHA-256: the archive
  # sanity sniff passes, the hash check must reject it before any build.
  head -c 65536 /dev/urandom | gzip > "$ARCHIVE"
  local log
  log="$(mktemp)"
  if "$ROOT/scripts/prepare-ane-compiler.sh" > "$log" 2>&1; then
    echo "prepare unexpectedly succeeded with a wrong archive" >&2
    cat "$log" >&2
    rm -f "$log"
    exit 1
  fi
  grep -q "archive SHA-256 mismatch" "$log" \
    || { echo "prepare failed for the wrong reason" >&2; cat "$log" >&2; rm -f "$log"; exit 1; }
  rm -f "$log"
  [[ ! -d "$SOURCE_DIR" ]] || { echo "source tree was built from rejected archive" >&2; exit 1; }
  [[ ! -x "$WRAPPER" ]] || { echo "wrapper was produced from rejected archive" >&2; exit 1; }
  rm -f "$ARCHIVE"
  echo "wrong archive rejected before extraction and build: PASS"
}

run_prepare() {
  note "prepare pinned compiler"
  "$ROOT/scripts/prepare-ane-compiler.sh"
}

qualify_known_h13() {
  note "known H13 graph host qualification"
  [[ -x "$WRAPPER" ]] || { echo "prepared compiler wrapper missing; run prepare first" >&2; exit 1; }
  local out="$COMPILER_WORK/h13-qualification/$H13_FIXTURE"
  rm -rf "$out"
  "$WRAPPER" \
    --target "$ANE_COMPILER_QUALIFIED_TARGET" \
    --mil "$SOURCE_DIR/tests/fixtures/$H13_FIXTURE.mil" \
    --model-root "$H13_MODEL_ROOT" \
    --output "$out"
  python3 - "$out" <<PY
import json
import pathlib
import sys

package = pathlib.Path(sys.argv[1])
manifest = json.loads((package / "manifest.json").read_text())
assert manifest["schema"] == "$ANE_COMPILER_PACKAGE_SCHEMA", manifest.get("schema")
assert manifest["target"] == "$ANE_COMPILER_QUALIFIED_TARGET", manifest.get("target")
assert manifest["artifactFormat"] == "anec", manifest.get("artifactFormat")
programs = manifest["programs"]
assert programs, "package has no programs"
for record in programs:
    assert (package / record["file"]).stat().st_size > 0, record["file"]
print(f"manifest: schema=$ANE_COMPILER_PACKAGE_SCHEMA target=$ANE_COMPILER_QUALIFIED_TARGET "
      f"format=anec programs={len(programs)}")
PY
  python3 "$SOURCE_DIR/research/inspect_anec.py" "$out"
  sha256sum "$out"/*
}

print_scope() {
  note "scope"
  cat <<EOF
provenance: commit=$ANE_COMPILER_COMMIT schema=$ANE_COMPILER_PACKAGE_SCHEMA qualified-target=$ANE_COMPILER_QUALIFIED_TARGET
device access: none (host compilation and device-free package inspection only)
not exercised: bundle adaptation (load_bundle), ANE submission, runtime qualification
EOF
}

case "${1:-all}" in
  clean) clean_area ;;
  reject) prove_hash_rejection ;;
  prepare) run_prepare ;;
  h13) qualify_known_h13 ;;
  all)
    clean_area
    prove_hash_rejection
    run_prepare
    qualify_known_h13
    print_scope
    echo "verify-ane-compiler: PASS"
    ;;
  *)
    echo "usage: $0 [all|clean|reject|prepare|h13]" >&2
    exit 64
    ;;
esac
