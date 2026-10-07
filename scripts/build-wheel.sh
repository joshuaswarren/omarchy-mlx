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
# There is no opt-out: when the runtime pin declares
# parakeet-encoder-whole the build refuses unless
# MLX_OMARCHY_WHOLE_BUNDLE_DIR points at the exact pinned bytes
# (packaging/stage-whole-bundle.sh).
echo "== stage whole-encoder bundle =="
# shellcheck disable=SC1091
source "$ROOT/packaging/stage-whole-bundle.sh"
stage_whole_bundle "$WORK_DIR"

# The share tree ships verbatim into the wheel, so a dirty working copy
# (e.g. a locally rebuilt libane-strict.so that never moved the pin) would
# become a published wheel whose own worker seal refuses it. Fail here,
# at the build, with expected/actual digests.
echo "== verify pinned runtime assets =="
python3 "$ROOT/scripts/verify_runtime_assets.py" \
  "$WORK_DIR/mlx/tools/mlx-omarchy-parakeet/share/mlx-omarchy/parakeet-1"

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
# MLX's CPU backend needs lapacke.h and finds it only in /usr/include,
# /usr/local/include, and CMAKE_INCLUDE_PATH. Arch Linux ARM's openblas ships
# it in /usr/include/openblas (libopenblas also carries the LAPACKE symbols),
# so pass that directory in. Without any lapacke.h, configure fails with only
# LAPACK_INCLUDE_DIRS-NOTFOUND, so stop here and name the package instead.
lapacke_dirs=(/usr/include /usr/local/include /usr/include/*-linux-gnu)
IFS=: read -r -a cmake_include_dirs <<< "${CMAKE_INCLUDE_PATH:-}"
lapacke_found=0
for dir in "${lapacke_dirs[@]}" "${cmake_include_dirs[@]}"; do
  if [[ -n "$dir" && -f "$dir/lapacke.h" ]]; then lapacke_found=1; fi
done
if [[ "$lapacke_found" == 0 && -f /usr/include/openblas/lapacke.h ]]; then
  export CMAKE_ARGS="$CMAKE_ARGS -DLAPACK_INCLUDE_DIRS=/usr/include/openblas"
elif [[ "$lapacke_found" == 0 ]]; then
  echo "ERROR: lapacke.h not found. Install it: 'sudo pacman -S openblas' (or lapacke) on Arch/Omarchy," >&2
  echo "       'sudo apt install liblapacke-dev' on Debian-family distributions." >&2
  exit 2
fi
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
