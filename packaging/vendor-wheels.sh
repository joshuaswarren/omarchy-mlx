#!/usr/bin/env bash
# Release-time: build the vendored wheel directory and hash-locked manifest.
#
#   packaging/vendor-wheels.sh --out DIR [--wheel mlx_omarchy-*.whl]
#
# Runs ONCE per release on a networked host with the pinned interpreter
# (python$PYTHON_VERSION from paths.sh); it resolves requirements-lock.in
# for that host's platform and writes DIR/requirements-lock.txt. A
# PKGBUILD never runs this: it consumes DIR + the lock offline through
# build-venv.sh. Voice extras are deliberately absent (system-package
# scope decision: voice stays install.sh-only for v1).
set -euo pipefail

self_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$self_dir/paths.sh"

out="" wheel=""
while (($#)); do
  case "$1" in
    --out) out="$2"; shift 2 ;;
    --wheel) wheel="$2"; shift 2 ;;
    *) echo "error: unknown option: $1" >&2; exit 1 ;;
  esac
done
if [[ -z $out ]]; then
  echo "error: --out DIR is required" >&2
  exit 1
fi
py="python$PYTHON_VERSION"
command -v "$py" >/dev/null 2>&1 ||
  { echo "error: $py not found; vendor builds use the pinned interpreter" >&2; exit 1; }

mkdir -p "$out"
echo "==> resolving requirements-lock.in for $(uname -m) $py"
"$py" -m pip download -r "$self_dir/requirements-lock.in" -d "$out"
if [[ -n $wheel ]]; then
  cp "$wheel" "$out/"
fi
"$self_dir/gen-vendor-lock.sh" "$out" >"$out/requirements-lock.txt"
"$self_dir/verify-vendor.sh" "$out" "$out/requirements-lock.txt"
echo "==> vendored wheels + lock written under $out"
