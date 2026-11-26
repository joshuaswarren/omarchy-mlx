#!/usr/bin/env bash
# Build the private system venv offline from vendored, hash-locked wheels.
#
#   packaging/build-venv.sh --vendor DIR --lock FILE --venv PATH [--python BIN]
#
# The interpreter comes from paths.sh (python$PYTHON_VERSION, the spec's
# pin) and every cp-tagged wheel in the vendor directory must match that
# interpreter exactly: a venv built against a removed or mismatched
# CPython minor is the failure this script must produce, never a silent
# install. The caller stages the tree; nothing here needs the network.
set -euo pipefail

self_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$self_dir/paths.sh"

vendor="" lock="" venv="" python_bin=""
while (($#)); do
  case "$1" in
    --vendor) vendor="$2"; shift 2 ;;
    --lock) lock="$2"; shift 2 ;;
    --venv) venv="$2"; shift 2 ;;
    --python) python_bin="$2"; shift 2 ;;
    *) echo "error: unknown option: $1" >&2; exit 1 ;;
  esac
done
for required in vendor lock venv; do
  if [[ -z ${!required} ]]; then
    echo "error: --$required is required" >&2
    exit 1
  fi
done
if [[ ! -d $vendor || ! -f $lock ]]; then
  echo "error: need --vendor DIR and --lock FILE" >&2
  exit 1
fi

py="${python_bin:-python$PYTHON_VERSION}"
command -v "$py" >/dev/null 2>&1 ||
  { echo "error: interpreter $py not found (pass --python for a staged build)" >&2; exit 1; }
pytag="$("$py" -c 'import sys; print("cp%d%d" % sys.version_info[:2])')"
tag_mismatch=0
for wheel in "$vendor"/*.whl; do
  [[ -e $wheel ]] || break
  case "$(basename "$wheel")" in
    *-cp*-cp*-*) ;;          # CPython-ABI wheel: tag must match exactly
    *) continue ;;           # pure-Python wheel: any interpreter
  esac
  wheel_tag="$(basename "$wheel" | sed -n 's/.*-\(cp[0-9]*\)-.*/\1/p')"
  if [[ $wheel_tag != "$pytag" ]]; then
    echo "error: $(basename "$wheel") targets $wheel_tag but $py provides $pytag" >&2
    tag_mismatch=1
  fi
done
if (( tag_mismatch )); then
  exit 1
fi

"$self_dir/verify-vendor.sh" "$vendor" "$lock"

echo "==> creating $venv with $py"
PYTHONNOUSERSITE=1 "$py" -m venv --clear "$venv"
if ! "$venv/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "error: $py -m venv produced no pip (ensurepip missing); cannot install offline" >&2
  exit 1
fi
"$venv/bin/python" -m pip install --no-index --find-links "$vendor" \
  --require-hashes -r "$lock"
echo "==> venv ready at $venv"
