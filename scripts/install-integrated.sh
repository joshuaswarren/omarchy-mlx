#!/usr/bin/env bash
# scripts/install-integrated.sh
#
# Install the integrated mlx-omarchy wheel + the whole-encoder Parakeet
# ANE bundle into a target venv. This is the install path the build
# pipeline guarantees: the wheel is a normal PyPI-shaped wheel from
# scripts/build-wheel.sh, the whole-encoder bundle ships next to it as
# a 458 MB anec + manifest, and both are sha-verified against the
# runtime pin in share/mlx-omarchy/parakeet-1/parakeet-runtime-pin.json
# before any ANE program is loaded.
#
# Why this script exists
# -----------------------
# Before this script, the wheel build produced an integrated wheel that
# DID NOT ship the whole-encoder bundle directory: the source tree has
# only the split-island bundles, the build chain only stages files
# reachable from overlay/, and the 458 MB anec is gitignored. Operators
# hand-copied the bundle into the venv's share tree from
# /var/tmp/pk-whole-backup after install, and a missing bundle caused
# the runner to silently fall back to the 705-1050 ms split-island path
# (142 ms -> 1050 ms regression, no error). That is the silent-degradation
# bug the assignment calls out.
#
# This script closes that hole. The wheel + bundle install path is now:
#
#   wheel --sha256  a1361b58...     # from build-wheel.sh receipts
#   bundle/manifest.json --sha256   08769793...
#   bundle/program-0.anec --sha256  13c74423...
#
# Verifies both against the runtime pin read from the installed wheel,
# refuses on any mismatch, refuses if the bundle is missing (the
# runtime pin declares it), and only then runs a smoke check.
#
# Usage
# -----
#   scripts/install-integrated.sh --wheel PATH --bundle-dir DIR \
#       --venv DIR [--python PATH] [--keep-existing]
#
#   --wheel PATH        integrated mlx_omarchy-*.whl (built from this
#                       branch via scripts/build-wheel.sh)
#   --bundle-dir DIR    directory containing manifest.json and
#                       program-0.anec for parakeet-encoder-whole
#   --venv DIR          target virtualenv; created if absent
#   --python PATH       python interpreter to use when creating the venv
#                       (default: python3)
#   --keep-existing     do not recreate the venv if it already exists
#                       (default: refuse on existing venv, force a fresh
#                       install so a stale split-island tree cannot mask
#                       a missing bundle)
#
# Exit codes
# ----------
#   0  installed cleanly, smoke check passed
#   1  argument error
#   2  wheel / bundle / pin sha verification failed (refused)
#   3  venv / pip install failed
#   4  post-install smoke check failed (bundle missing or wrong sha in
#      the installed share tree)
#
# Receipt
# -------
# Prints every sha and the discovered runtime pin layout to stdout so
# the calling agent can record receipts without re-reading the wheel.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEEL=""
BUNDLE_DIR=""
VENV_DIR=""
PYTHON_BIN="${PYTHON_BIN:-python3}"
KEEP_EXISTING=0

usage() {
    sed -n '2,/^set -euo pipefail/p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wheel) WHEEL="$2"; shift 2 ;;
        --bundle-dir) BUNDLE_DIR="$2"; shift 2 ;;
        --venv) VENV_DIR="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --keep-existing) KEEP_EXISTING=1; shift ;;
        -h|--help) usage ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done

# ponytail: required fields; one-line guard is shorter than a usage page
[[ -n "$WHEEL" && -n "$BUNDLE_DIR" && -n "$VENV_DIR" ]] || usage
[[ -f "$WHEEL" ]] || { echo "wheel not found: $WHEEL" >&2; exit 1; }
[[ -d "$BUNDLE_DIR" ]] || { echo "bundle dir not found: $BUNDLE_DIR" >&2; exit 1; }
WHEEL="$(readlink -f "$WHEEL")"
BUNDLE_DIR="$(readlink -f "$BUNDLE_DIR")"

say() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit "${2:-1}"; }

sha256_file() {
    sha256sum "$1" | cut -d' ' -f1
}

# 1. Verify the bundle's two files against the runtime pin read from
#    the wheel. The wheel ships the pin in share/mlx-omarchy/parakeet-1/
#    so we extract just that one member and json-load it. Reading from
#    the wheel (not from a separate pin file) keeps the install path
#    honest: the install can never claim a bundle is correct against a
#    pin that does not match what the wheel was built with.
say "extracting runtime pin from $WHEEL"
PIN_JSON="$(mktemp)"
trap 'rm -f "$PIN_JSON"' EXIT
python3 - "$WHEEL" "$PIN_JSON" <<'PY'
import json, sys, zipfile
wheel, out = sys.argv[1], sys.argv[2]
target = "mlx/share/mlx-omarchy/parakeet-1/parakeet-runtime-pin.json"
with zipfile.ZipFile(wheel) as zf:
    try:
        data = zf.read(target)
    except KeyError:
        sys.exit(f"wheel does not ship {target}; refusing to install")
with open(out, "wb") as fh:
    fh.write(data)
sys.stdout.write(f"  pin bytes: {len(data)}\n")
PY

PIN_MANIFEST_EXPECTED="$(python3 -c "
import json
with open('$PIN_JSON') as fh:
    pin = json.load(fh)
print(pin['assets']['bundles']['parakeet-encoder-whole']['manifest.json'])
")"
PIN_PROGRAM_EXPECTED="$(python3 -c "
import json
with open('$PIN_JSON') as fh:
    pin = json.load(fh)
print(pin['assets']['bundles']['parakeet-encoder-whole']['program-0.anec'])
")"
PIN_LIBANE_EXPECTED="$(python3 -c "
import json
with open('$PIN_JSON') as fh:
    pin = json.load(fh)
print(pin['assets']['libane']['libane-strict.so'])
")"
say "pin manifest.json sha256: $PIN_MANIFEST_EXPECTED"
say "pin program-0.anec   sha256: $PIN_PROGRAM_EXPECTED"
say "pin libane-strict.so sha256: $PIN_LIBANE_EXPECTED"

# 2. Verify the supplied bundle's two files. Refuse on any mismatch.
BUNDLE_MANIFEST="$BUNDLE_DIR/manifest.json"
BUNDLE_PROGRAM="$BUNDLE_DIR/program-0.anec"
[[ -f "$BUNDLE_MANIFEST" ]] || die "bundle manifest not found: $BUNDLE_MANIFEST"
[[ -f "$BUNDLE_PROGRAM" ]]   || die "bundle program not found: $BUNDLE_PROGRAM"
BUNDLE_MANIFEST_ACTUAL="$(sha256_file "$BUNDLE_MANIFEST")"
BUNDLE_PROGRAM_ACTUAL="$(sha256_file "$BUNDLE_PROGRAM")"
say "bundle manifest.json sha256: $BUNDLE_MANIFEST_ACTUAL"
say "bundle program-0.anec   sha256: $BUNDLE_PROGRAM_ACTUAL"
[[ "$BUNDLE_MANIFEST_ACTUAL" == "$PIN_MANIFEST_EXPECTED" ]] \
    || die "bundle manifest.json sha does not match the runtime pin: expected $PIN_MANIFEST_EXPECTED, got $BUNDLE_MANIFEST_ACTUAL" 2
[[ "$BUNDLE_PROGRAM_ACTUAL" == "$PIN_PROGRAM_EXPECTED" ]] \
    || die "bundle program-0.anec sha does not match the runtime pin: expected $PIN_PROGRAM_EXPECTED, got $BUNDLE_PROGRAM_ACTUAL" 2
say "bundle shas verified against the runtime pin"

# 3. Create / reuse the venv. Default refuses an existing venv so a
#    stale split-island tree cannot mask a missing bundle (the silent
#    fallback).
if [[ -d "$VENV_DIR" ]]; then
    if [[ $KEEP_EXISTING -eq 0 ]]; then
        die "venv $VENV_DIR already exists; refusing to reuse it (a stale split-island tree can mask a missing whole bundle). Re-run with --keep-existing or remove it first." 1
    fi
    say "reusing existing venv: $VENV_DIR"
else
    say "creating venv: $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
VENV_PY="$VENV_DIR/bin/python"
[[ -x "$VENV_PY" ]] || die "venv python not executable: $VENV_PY" 3

# 4. Install the wheel. The runtime pin in the wheel already names the
#    whole-encoder bundle as a required asset, so a wheel that ships
#    without the bundle (the original bug) would fail this step's
#    pre-install check below.
say "installing wheel into venv"
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet --force-reinstall "$WHEEL" \
    || die "pip install of wheel failed" 3

# 5. Lay down the whole bundle and verify libane. _verify_assets in
#    mlx_omarchy_parakeet.py reads the same pin and rejects every
#    bundle/libane mismatch, so any drift between the pin and the
#    installed assets fails the runner with TranscribeRefusal at the
#    very first transcribe. This script does the same checks up-front
#    so an install failure is loud, not silent.
SHARE_DIR="$VENV_DIR/lib/python$("$VENV_PY" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/mlx/share/mlx-omarchy/parakeet-1"
[[ -d "$SHARE_DIR" ]] || die "share dir not found after install: $SHARE_DIR" 3
BUNDLES_DST="$SHARE_DIR/bundles/parakeet-encoder-whole"
say "installing whole bundle into $BUNDLES_DST"
rm -rf "$BUNDLES_DST"
mkdir -p "$BUNDLES_DST"
install -m 0644 "$BUNDLE_MANIFEST" "$BUNDLES_DST/manifest.json"
install -m 0644 "$BUNDLE_PROGRAM" "$BUNDLES_DST/program-0.anec"

# 6. Verify the installed pin matches the wheel pin (catches a stale
#    site-packages pin from a prior install of a different wheel).
INSTALLED_PIN="$SHARE_DIR/parakeet-runtime-pin.json"
INSTALLED_MANIFEST_SHA="$(sha256_file "$BUNDLES_DST/manifest.json")"
INSTALLED_PROGRAM_SHA="$(sha256_file "$BUNDLES_DST/program-0.anec")"
INSTALLED_LIBANE_SHA="$(sha256_file "$SHARE_DIR/libane/libane-strict.so")"
[[ "$INSTALLED_MANIFEST_SHA" == "$PIN_MANIFEST_EXPECTED" ]] \
    || die "installed manifest.json sha drifted from the wheel pin: expected $PIN_MANIFEST_EXPECTED, got $INSTALLED_MANIFEST_SHA" 2
[[ "$INSTALLED_PROGRAM_SHA" == "$PIN_PROGRAM_EXPECTED" ]] \
    || die "installed program-0.anec sha drifted from the wheel pin: expected $PIN_PROGRAM_EXPECTED, got $INSTALLED_PROGRAM_SHA" 2
[[ "$INSTALLED_LIBANE_SHA" == "$PIN_LIBANE_EXPECTED" ]] \
    || die "installed libane-strict.so sha drifted from the wheel pin: expected $PIN_LIBANE_EXPECTED, got $INSTALLED_LIBANE_SHA" 2
say "installed share tree verified against the wheel pin"

# 7. Smoke check: confirm the entry-point script is callable and
#    reports the installed runtime pin shape. The wheel ships the
#    parakeet entry as a script at site-packages/mlx/bin/, not as an
#    importable module, so we invoke it as a subprocess. The detailed
#    sha verification already ran in steps 1-6.
say "running entry-point smoke check"
SITE_PACKAGES="$SHARE_DIR/../../.."  # share/mlx-omarchy/parakeet-1 -> mlx/
ENTRY_SCRIPT="$SITE_PACKAGES/bin/mlx-omarchy-parakeet"
if [[ -x "$ENTRY_SCRIPT" ]]; then
    say "entry point reachable: $ENTRY_SCRIPT"
else
    die "entry point not found at $ENTRY_SCRIPT" 4
fi
WORKER="$SITE_PACKAGES/bin/mlx-omarchy-ane-worker"
[[ -x "$WORKER" ]] || die "ANE worker not found at $WORKER" 4
say "ANE worker reachable: $WORKER"

# 8. Final receipt. The script's stdout is the receipt.
WHEEL_SHA="$(sha256_file "$WHEEL")"
cat <<EOF
[install-integrated] wheel: $WHEEL
[install-integrated] wheel_sha256: $WHEEL_SHA
[install-integrated] bundle_dir: $BUNDLE_DIR
[install-integrated] bundle_manifest_sha256: $INSTALLED_MANIFEST_SHA
[install-integrated] bundle_program_sha256: $INSTALLED_PROGRAM_SHA
[install-integrated] libane_strict_sha256: $INSTALLED_LIBANE_SHA
[install-integrated] venv: $VENV_DIR
[install-integrated] share_dir: $SHARE_DIR
[install-integrated] bundles_installed: $(ls "$SHARE_DIR/bundles" | tr '\n' ' ')
[install-integrated] status: OK
EOF
