#!/usr/bin/env bash
# Build the mlx-omarchy wheel end to end.
# This is the U5 packaging gate: scripts/build-wheel.sh
#
# Prepares the pinned upstream tree with the Omarchy-only backend, builds the
# python bindings, and writes exactly one wheel into dist/ at the repo root.
#
# Options:
#   --diagnostics  Build the dev diagnostics wheel instead of a release
#                  wheel. Compiles in the GPU profiling harness
#                  (-DMLX_OMARCHY_GPU_PROFILING=ON), stamps the version
#                  with a "+diag" local segment, and stages the
#                  mlx-omarchy-info binary next to the wheel. The harness
#                  runs slower and this artifact is not for production.
#                  Without the flag the build is unchanged: profiling
#                  harness compiled OUT, release wheel as before.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${MLX_OMARCHY_WORK_DIR:-$ROOT/.work}"
VENV_DIR="$WORK_DIR/venv-build"
DIST_DIR="$ROOT/dist"

DIAGNOSTICS=0
for arg in "$@"; do
  case "$arg" in
    --diagnostics) DIAGNOSTICS=1 ;;
    *) echo "unknown option: $arg (supported: --diagnostics)" >&2; exit 2 ;;
  esac
done

# Stamp the source commit into the version's local segment so every
# artifact is traceable to the exact commit that built it. v0.3.2
# recorded no commit anywhere in its wheel, and proving what it did or
# did not contain took a static-diff investigation
# (receipts/2026-09-03-wheel-delta-v0.3.2-vs-local.md). No
# byte-reproducibility requirement exists for these wheels - the dev
# segment already carries minute-level timestamps - so the stamp adds no
# new nondeterminism. DEV_RELEASE=1 (operator-exported for release
# builds) suppresses setup.py's own git-hash suffix, which is why the
# release path must stamp explicitly. Default dev builds keep setup.py's
# behavior untouched.
resolve_commit() {
  if [[ -n "${MLX_OMARCHY_SOURCE_COMMIT:-}" ]]; then
    printf '%s' "$MLX_OMARCHY_SOURCE_COMMIT"
    return
  fi
  git -C "$ROOT" rev-parse --short=7 HEAD 2>/dev/null || true
}

if [[ $DIAGNOSTICS -eq 1 ]]; then
  commit="$(resolve_commit)"
  if [[ -z "$commit" ]]; then
    echo "ERROR: cannot stamp the source commit: build from a git checkout or set MLX_OMARCHY_SOURCE_COMMIT" >&2
    exit 2
  fi
  export MLX_OMARCHY_LOCAL_VERSION="diag.$commit"
elif [[ "${DEV_RELEASE:-0}" == 1 && -z "${MLX_OMARCHY_LOCAL_VERSION:-}" ]]; then
  commit="$(resolve_commit)"
  if [[ -z "$commit" ]]; then
    echo "ERROR: cannot stamp the source commit: build from a git checkout or set MLX_OMARCHY_SOURCE_COMMIT" >&2
    exit 2
  fi
  export MLX_OMARCHY_LOCAL_VERSION="$commit"
fi

echo "== prepare upstream tree =="
"$ROOT/scripts/prepare-mlx.sh"

# Stage the whole-encoder bundle into the build source if the runtime
# pin declares it. The bundle is 458 MB and gitignored; it must be
# supplied via MLX_OMARCHY_WHOLE_BUNDLE_DIR, copied into the
# tools/mlx-omarchy-parakeet share tree after prepare-mlx.sh so the
# CMake install(DIRECTORY share/mlx-omarchy ...) rule picks it up.
#
# A wheel built without the whole bundle would still pass the rest of
# this script -- the build chain only stages files reachable from
# overlay/ -- and the runner would then silently fall back to the
# 705-1050 ms split-island path. Refusing here forces the bundle to
# travel with the build, which makes the silent fallback impossible.
#
# Set MLX_OMARCHY_WHOLE_BUNDLE_SKIP=1 to opt out of the whole-bundle
# pin requirement (for non-Parakeet-focused diagnostic builds).
echo "== stage whole-encoder bundle =="
PIN_PATH="$WORK_DIR/mlx/tools/mlx-omarchy-parakeet/share/mlx-omarchy/parakeet-1/parakeet-runtime-pin.json"
BUNDLE_STAGING="$WORK_DIR/mlx/tools/mlx-omarchy-parakeet/share/mlx-omarchy/parakeet-1/bundles/parakeet-encoder-whole"
if [[ -f "$PIN_PATH" ]] && python3 -c "
import json, sys
with open('$PIN_PATH') as fh:
    pin = json.load(fh)
sys.exit(0 if 'parakeet-encoder-whole' in pin.get('assets', {}).get('bundles', {}) else 1)
"; then
    if [[ "${MLX_OMARCHY_WHOLE_BUNDLE_SKIP:-0}" == "1" ]]; then
        echo "[bundle] MLX_OMARCHY_WHOLE_BUNDLE_SKIP=1; refusing to build without the whole bundle"
        exit 1
    fi
    if [[ -z "${MLX_OMARCHY_WHOLE_BUNDLE_DIR:-}" ]]; then
        echo "[bundle] runtime pin declares parakeet-encoder-whole but MLX_OMARCHY_WHOLE_BUNDLE_DIR is unset; refusing to build a wheel that would silently fall back" >&2
        echo "[bundle] supply the bundle dir (manifest.json + program-0.anec) via MLX_OMARCHY_WHOLE_BUNDLE_DIR=/path/to/dir" >&2
        exit 1
    fi
    if [[ ! -f "${MLX_OMARCHY_WHOLE_BUNDLE_DIR}/manifest.json" || ! -f "${MLX_OMARCHY_WHOLE_BUNDLE_DIR}/program-0.anec" ]]; then
        echo "[bundle] ${MLX_OMARCHY_WHOLE_BUNDLE_DIR} does not contain manifest.json + program-0.anec" >&2
        exit 1
    fi
    EXPECTED_MANIFEST="$(python3 -c "
import json
with open('$PIN_PATH') as fh:
    print(json.load(fh)['assets']['bundles']['parakeet-encoder-whole']['manifest.json'])
")"
    EXPECTED_PROGRAM="$(python3 -c "
import json
with open('$PIN_PATH') as fh:
    print(json.load(fh)['assets']['bundles']['parakeet-encoder-whole']['program-0.anec'])
")"
    ACTUAL_MANIFEST="$(sha256sum "${MLX_OMARCHY_WHOLE_BUNDLE_DIR}/manifest.json" | cut -d' ' -f1)"
    ACTUAL_PROGRAM="$(sha256sum "${MLX_OMARCHY_WHOLE_BUNDLE_DIR}/program-0.anec" | cut -d' ' -f1)"
    [[ "$ACTUAL_MANIFEST" == "$EXPECTED_MANIFEST" ]] || {
        echo "[bundle] manifest.json sha mismatch: pin=$EXPECTED_MANIFEST actual=$ACTUAL_MANIFEST" >&2; exit 1; }
    [[ "$ACTUAL_PROGRAM" == "$EXPECTED_PROGRAM" ]] || {
        echo "[bundle] program-0.anec sha mismatch: pin=$EXPECTED_PROGRAM actual=$ACTUAL_PROGRAM" >&2; exit 1; }
    rm -rf "$BUNDLE_STAGING"
    mkdir -p "$BUNDLE_STAGING"
    install -m 0644 "${MLX_OMARCHY_WHOLE_BUNDLE_DIR}/manifest.json" "$BUNDLE_STAGING/manifest.json"
    install -m 0644 "${MLX_OMARCHY_WHOLE_BUNDLE_DIR}/program-0.anec" "$BUNDLE_STAGING/program-0.anec"
    # Refresh mtime so the CMake install(DIRECTORY) rule treats the
    # bundle as newer than any prior build output (the same rule
    # prepare-mlx.sh applies to the overlay).
    touch "$BUNDLE_STAGING/manifest.json" "$BUNDLE_STAGING/program-0.anec"
    echo "[bundle] staged $BUNDLE_STAGING (manifest $ACTUAL_MANIFEST, program $ACTUAL_PROGRAM)"
else
    echo "[bundle] runtime pin does not declare parakeet-encoder-whole; skipping whole-bundle stage"
fi

echo "== build venv ($VENV_DIR) =="
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv --system-site-packages "$VENV_DIR"
fi
venv_python="$VENV_DIR/bin/python"

# Offline-first: keep system packages when they satisfy the pyproject build
# requirements (setuptools>=80, typing_extensions, cmake>=3.25); pip install
# only what is missing.
if ! "$venv_python" - <<'EOF'
import setuptools, typing_extensions
assert int(setuptools.__version__.split(".")[0]) >= 80
EOF
then
  "$venv_python" -m pip install 'setuptools>=80' typing_extensions
fi

cmake_version="$(cmake --version 2>/dev/null | sed -n '1s/^cmake version //p' || true)"
if [[ -z "$cmake_version" || "$(printf '%s\n' 3.25 "$cmake_version" | sort -V | head -n1)" != "3.25" ]]; then
  "$venv_python" -m pip install 'cmake>=3.25'
fi

echo "== wheel build =="
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

# setup.py appends CMAKE_ARGS to its cmake invocation; it splits the value on
# spaces, so keep each -D flag space separated.
export CMAKE_ARGS="-DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF"
if [[ -n "${MLX_OMARCHY_ANE_SOURCE_DIR:-}" ]]; then
  [[ -d "$MLX_OMARCHY_ANE_SOURCE_DIR" ]] || {
    echo "ERROR: MLX_OMARCHY_ANE_SOURCE_DIR is not a directory" >&2
    exit 2
  }
  [[ "$MLX_OMARCHY_ANE_SOURCE_DIR" != *" "* ]] || {
    echo "ERROR: MLX_OMARCHY_ANE_SOURCE_DIR cannot contain spaces" >&2
    exit 2
  }
  export CMAKE_ARGS="$CMAKE_ARGS -DMLX_OMARCHY_ANE_SOURCE_DIR=$MLX_OMARCHY_ANE_SOURCE_DIR"
fi
if [[ $DIAGNOSTICS -eq 1 ]]; then
  # Diagnostics wheel: compile in the env-gated GPU profiling harness.
  # DEV_RELEASE=1 keeps setup.py from appending its own git-hash local
  # segment; the MLX_OMARCHY_LOCAL_VERSION stamped above (diag.<commit>)
  # reaches the version through patches/mlx-version-time.patch instead.
  export CMAKE_ARGS="$CMAKE_ARGS -DMLX_OMARCHY_GPU_PROFILING=ON"
  export DEV_RELEASE=1
fi

# Honor an explicit override, else the machine's actual core count. A
# hardcoded 16 on the single-core M1 queues 96 compiler processes on one
# CPU (cmake+ninja+cc1plus fan-out) and thrashes 16 GB into the OOM zone.
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"
export PATH="$VENV_DIR/bin:$PATH"

# aarch64 wheels ship the Parakeet runtime surface: the standalone
# fd-protocol ANE worker (mlx/bin/mlx-omarchy-ane-worker) beside the
# pinned island bundles and strict libane under share/mlx-omarchy/. The
# worker only compiles under MLX_OMARCHY_ANE_DEVICE, and that flag
# builds nothing on other architectures (no libane surface).
case "$(uname -m)" in
  aarch64|arm64)
    export CMAKE_ARGS="$CMAKE_ARGS -DMLX_OMARCHY_ANE_DEVICE=ON"
    ;;
esac

"$venv_python" -m pip wheel --no-build-isolation --no-deps \
  --wheel-dir "$DIST_DIR" "$WORK_DIR/mlx"

shopt -s nullglob
wheels=("$DIST_DIR"/mlx_omarchy-*.whl)
shopt -u nullglob
if [[ ${#wheels[@]} -ne 1 ]]; then
  echo "expected exactly one mlx_omarchy wheel in $DIST_DIR, found ${#wheels[@]}" >&2
  exit 1
fi
wheel="${wheels[0]}"
if [[ $DIAGNOSTICS -eq 1 ]]; then
  echo "== diagnostics staging =="
  # The wheel must carry the profiling harness (the getenv literal only
  # exists when MLX_OMARCHY_GPU_PROFILING was ON) and the info tool. The
  # tool binary is staged from the wheel itself so the asset matches the
  # artifact byte for byte.
  python3 - "$wheel" "$DIST_DIR" <<'EOF'
import sys, zipfile
wheel, dist = sys.argv[1], sys.argv[2]
has_profile = False
tool = None
with zipfile.ZipFile(wheel) as zf:
    for name in zf.namelist():
        data = zf.read(name)
        if not has_profile and b"MLX_OMARCHY_GPU_PROFILE" in data:
            has_profile = True
        if name.endswith("bin/mlx-omarchy-info"):
            tool = data
if not has_profile:
    sys.exit("diagnostics wheel was built with the profiling harness compiled OUT")
if tool is None:
    sys.exit("diagnostics wheel does not ship mlx/bin/mlx-omarchy-info")
with open(f"{dist}/mlx-omarchy-info", "wb") as fh:
    fh.write(tool)
EOF
  chmod 755 "$DIST_DIR/mlx-omarchy-info"
  echo "[receipt] profiling harness: compiled IN (MLX_OMARCHY_GPU_PROFILE literal found in wheel)"
  echo "[receipt] staged tool: $DIST_DIR/mlx-omarchy-info"
fi


echo "== receipt =="
echo "[receipt] wheel: $wheel"
echo "[receipt] size: $(stat -c '%s' "$wheel") bytes"
echo "[receipt] sha256: $(sha256sum "$wheel" | cut -d' ' -f1)"
echo "[receipt] source commit: $(git -C "$ROOT" rev-parse --short=7 HEAD 2>/dev/null || echo unknown)"
cat <<'GATE'
[receipt] NEXT STEP - do not skip: after you upload this wheel to a
[receipt] release, verify the UPLOADED asset with
[receipt]   python3 scripts/verify-release-assets.py <tag>
[receipt] It must print VERIFIED before the release is announced as
[receipt] usable. See docs/release.md. A local build check does not
[receipt] verify what users download.
GATE
