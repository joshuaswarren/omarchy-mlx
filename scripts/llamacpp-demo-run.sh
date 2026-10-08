#!/usr/bin/env bash
# Run a llama.cpp binary on our Honeykrisp ICD, or refuse loudly.
# Usage from the repo root: bash scripts/llamacpp-demo-run.sh <relative-to-llamacpp-demo binary> [args...]
# Example: bash scripts/llamacpp-demo-run.sh llama-build/bin/llama-bench -m Qwen_Qwen3.5-9B-IQ2_M.gguf -p 512 -n 64 -ngl 99 -t 8
set -euo pipefail
DEMO_DIR="$PWD/llamacpp-demo"
[ "$1" = "--demo-dir" ] && { DEMO_DIR="$2"; shift 2; }
BIN="$1"; shift
ICD=$(ls "$DEMO_DIR"/mesa-install/share/vulkan/icd.d/*.json | head -1)
[ -f "$ICD" ] || { echo "ERROR: no ICD json found under $DEMO_DIR/mesa-install. Run scripts/llamacpp-demo.sh first."; exit 2; }
[ -x "$DEMO_DIR/$BIN" ] || { echo "ERROR: $DEMO_DIR/$BIN is not executable. Run scripts/llamacpp-demo.sh first."; exit 2; }
if ! command -v vulkaninfo >/dev/null 2>&1; then
  echo "ERROR: vulkaninfo not found. Install it: sudo pacman -S --needed vulkan-tools"
  exit 2
fi
if ! env VK_DRIVER_FILES="$ICD" vulkaninfo --summary 2>/dev/null | grep -q Honeykrisp; then
  echo "ERROR: the Honeykrisp driver did not load (vulkaninfo found no Honeykrisp device)."
  echo "llama.cpp would silently fall back to the CPU and produce meaningless numbers, so refusing to run."
  echo "Debug:  VK_LOADER_DEBUG=error,driver VK_DRIVER_FILES=$ICD vulkaninfo --summary"
  echo "Common cause: the ICD was built against older system libraries than the ones now installed."
  echo "Fix: delete the build dirs and rerun scripts/llamacpp-demo.sh so the driver is rebuilt against the current system:"
  echo "  cd $DEMO_DIR && rm -rf mesa-build mesa-install && cd .. && bash scripts/llamacpp-demo.sh"
  exit 2
fi
exec env VK_DRIVER_FILES="$ICD" "$DEMO_DIR/$BIN" "$@"
