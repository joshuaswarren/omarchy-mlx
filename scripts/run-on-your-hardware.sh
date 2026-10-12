#!/usr/bin/env bash
# One command for the non-Apple GPU path in docs/ANY-GPU.md: check the
# prerequisites (naming the Omarchy/Arch packages), stage the pinned MLX
# source, build and run the Vulkan runtime suite, print the quick
# capability report, and offer to publish it.
#
# Nothing uploads without --submit or the exact-word SUBMIT prompt. The
# full report is printed before any upload, and the test-suite log is
# never uploaded; it stays in the output directory.
set -euo pipefail

COMMUNITY_ENDPOINT="https://mlx-omarchy-community-data.joshua-s-warren.workers.dev"

usage() {
  cat <<'USAGE'
usage: scripts/run-on-your-hardware.sh [--submit] [--dry-run] [--out DIR]

Builds and runs the Vulkan runtime test suite, then prints the quick
capability report (docs/ANY-GPU.md is the manual version of this path).

  --submit     publish the quick report to the public community dataset
               after printing it in full
  --dry-run    check prerequisites and print the plan, build nothing
  --out DIR    keep the logs and the report in DIR
               (default: .work/hardware-results-<UTC timestamp>)
  -h, --help   this text

A plain run on a terminal offers the same upload behind the exact-word
SUBMIT prompt after the report prints; declining keeps everything
local. The suite log is not part of the published report; attach it to
an issue if you report a failure.

Multi-GPU or multi-driver hosts: export VK_DRIVER_FILES or
MLX_OMARCHY_DEVICE_INDEX before running. The exit status is the test
suite's exit status.
USAGE
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SUBMIT=0
DRY_RUN=0
OUT_DIR="$REPO_ROOT/.work/hardware-results-$(date -u +%Y%m%dT%H%M%SZ)"
while (($#)); do
  case "$1" in
    --submit) SUBMIT=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --out)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      OUT_DIR="$2"
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

MISSING_PKGS=()
need_cmd() {
  # need_cmd <command> <Arch/Omarchy package>
  if command -v "$1" >/dev/null 2>&1; then
    return 0
  fi
  echo "missing: $1 (package: $2)"
  MISSING_PKGS+=("$2")
}

echo "== prerequisites =="
need_cmd git git
need_cmd cmake cmake
need_cmd cc base-devel
need_cmd make base-devel
need_cmd python3 python
need_cmd vulkaninfo vulkan-tools
if ! command -v glslc >/dev/null 2>&1 &&
  ! command -v glslangValidator >/dev/null 2>&1; then
  echo "missing: glslc or glslangValidator (packages: shaderc or glslang;"
  echo "  the build needs one of them to compile the Vulkan shaders)"
  MISSING_PKGS+=(shaderc)
fi
if [[ ! -f /usr/include/vulkan/vulkan.h ]]; then
  echo "missing: Vulkan headers (package: vulkan-headers)"
  MISSING_PKGS+=(vulkan-headers)
fi
DRIVER_OK=1
if ! compgen -G '/usr/share/vulkan/icd.d/*.json' >/dev/null; then
  DRIVER_OK=0
  echo "missing: a Vulkan driver ICD under /usr/share/vulkan/icd.d/"
  echo "  AMD: vulkan-radeon   Intel: vulkan-intel   NVIDIA: nvidia-utils"
  echo "  no GPU at all: vulkan-swrast (the lavapipe software driver)"
fi

if ((${#MISSING_PKGS[@]} > 0)) || ((DRIVER_OK == 0)); then
  echo
  if ((${#MISSING_PKGS[@]} > 0)); then
    mapfile -t MISSING_PKGS < <(printf '%s\n' "${MISSING_PKGS[@]}" | sort -u)
    echo "install the missing packages on Omarchy or Arch with:"
    echo "  sudo pacman -S --needed ${MISSING_PKGS[*]}"
  fi
  if ((DRIVER_OK == 0)); then
    echo "install one Vulkan driver with:"
    echo "  sudo pacman -S --needed vulkan-radeon   # AMD"
    echo "  sudo pacman -S --needed vulkan-intel    # Intel"
    echo "  sudo pacman -S --needed nvidia-utils    # NVIDIA proprietary"
    echo "  sudo pacman -S --needed vulkan-swrast   # software fallback"
  fi
  exit 1
fi

GPU_NAME="$(vulkaninfo --summary 2>/dev/null |
  sed -n 's/^[[:space:]]*deviceName[[:space:]]*= *//p' | head -n 1 || true)"
if [[ -z "$GPU_NAME" ]]; then
  GPU_NAME="unknown"
  echo "warning: vulkaninfo reported no Vulkan device; the suite will"
  echo "  stop at device discovery. Check the driver packages above."
fi
if [[ -z "${MLX_OMARCHY_ALLOW_NON_APPLE:-}" && "$GPU_NAME" != *Apple* ]]; then
  export MLX_OMARCHY_ALLOW_NON_APPLE=1
  echo "non-Apple device ($GPU_NAME): MLX_OMARCHY_ALLOW_NON_APPLE=1 is set"
  echo "  for this run (docs/ANY-GPU.md)."
fi

echo
echo "== plan =="
echo "  1. scripts/prepare-mlx.sh"
echo "     stage the pinned MLX source into .work/mlx (network needed once)"
echo "  2. .work/mlx/tools/ci/run-omarchy-runtime.sh"
echo "     configure, build, and run the runtime test suite"
echo "  3. scripts/collect_quick.py"
echo "     print the quick capability report"
if ((SUBMIT)); then
  echo "  4. publish the report (--submit)"
elif [[ -t 0 ]]; then
  echo "  4. offer to publish the report (SUBMIT prompt)"
else
  echo "  (no publish step: pass --submit to opt in)"
fi
echo "  output directory: $OUT_DIR"
if ((DRY_RUN)); then
  echo "dry run: nothing was built or run."
  exit 0
fi

mkdir -p "$OUT_DIR"

echo
echo "== stage pinned MLX source =="
"$REPO_ROOT/scripts/prepare-mlx.sh" 2>&1 | tee "$OUT_DIR/prepare.log"

RUNNER="$REPO_ROOT/.work/mlx/tools/ci/run-omarchy-runtime.sh"
if [[ ! -f "$RUNNER" ]]; then
  echo "staged runner not found at $RUNNER; prepare-mlx.sh failed?" >&2
  exit 1
fi

echo
echo "== build and run the runtime suite =="
set +e
bash "$RUNNER" 2>&1 | tee "$OUT_DIR/runtime-tests.log"
SUITE_STATUS=${PIPESTATUS[0]}
set -e

echo
echo "== quick capability report =="
COLLECTOR_ARGS=(--out "$OUT_DIR/quick-report.json")
if ((SUBMIT)); then
  COLLECTOR_ARGS+=(--submit)
elif [[ -t 0 ]]; then
  export MLX_OMARCHY_SUBMIT_URL="$COMMUNITY_ENDPOINT"
  echo "the report below is exactly what would be published;"
  echo "type SUBMIT at the prompt to publish it."
fi
set +e
python3 "$REPO_ROOT/scripts/collect_quick.py" "${COLLECTOR_ARGS[@]}"
COLLECT_STATUS=$?
set -e

echo
echo "== summary =="
echo "checkout: $(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "suite exit status: $SUITE_STATUS  (full log: $OUT_DIR/runtime-tests.log)"
grep -E '\[doctest\] test cases:' "$OUT_DIR/runtime-tests.log" || true
echo "report: $OUT_DIR/quick-report.json (collector exit status: $COLLECT_STATUS)"
if ((COLLECT_STATUS == 0 && ! SUBMIT)); then
  echo "publish this report later with: python3 scripts/collect_quick.py --submit"
fi
echo "the suite log is not published; attach it to an issue if you report"
echo "a failure (docs/ANY-GPU.md lists what a report needs)."
exit "$SUITE_STATUS"
