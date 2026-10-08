#!/usr/bin/env bash
# Run a llama.cpp binary on our Honeykrisp ICD, or refuse loudly.
# Usage: bash scripts/llamacpp-demo-run.sh <binary relative to llamacpp-demo> [args...]   (from any directory)
# The binary and any file in the arguments (for example the model) are relative to llamacpp-demo: the wrapper runs there.
# Example: bash scripts/llamacpp-demo-run.sh llama-build/bin/llama-bench -m Qwen_Qwen3.5-9B-IQ2_M.gguf -p 512 -n 64 -ngl 99 -t 8
set -euo pipefail
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/llamacpp-demo"
[ "$1" = "--demo-dir" ] && { DEMO_DIR="$2"; shift 2; }
BIN="$1"; shift
ICD=$(ls "$DEMO_DIR"/mesa-install/share/vulkan/icd.d/*.json | head -1)
[ -f "$ICD" ] || { echo "ERROR: no ICD json found under $DEMO_DIR/mesa-install. Run scripts/llamacpp-demo.sh first."; exit 2; }
[ -x "$DEMO_DIR/$BIN" ] || { echo "ERROR: $DEMO_DIR/$BIN is not executable. Run scripts/llamacpp-demo.sh first."; exit 2; }

# --- Preflight: refuse before a model can push the machine into the OOM killer ----------------------------------
# The model and the GPU buffers live in unified memory. Refuse when MemAvailable is below the model size + 1 GiB; warn
# when it is below the model size + 3 GiB. (An 8 GB machine with about 4.5 GiB free runs the 4B model, not the 9B.)
# LLAMACPP_DEMO_MEMINFO replaces /proc/meminfo, for tests.
MEMINFO=${LLAMACPP_DEMO_MEMINFO:-/proc/meminfo}
GIB=$((1024 * 1024 * 1024))
model=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[i]}" in -m | --model) model="${args[i + 1]:-}" ;; esac
done
if [ -n "$model" ]; then
  case "$model" in /*) mpath="$model" ;; *) mpath="$DEMO_DIR/$model" ;; esac
  if [ -f "$mpath" ]; then
    size=$(stat -c%s "$mpath")
    avail=$(( $(awk '/^MemAvailable:/ {print $2}' "$MEMINFO") * 1024 ))
    if [ "$avail" -lt $((size + GIB)) ]; then
      echo "ERROR: not enough free memory: $((avail / 1048576)) MiB available, need at least $(((size + GIB) / 1048576)) MiB (model size + 1 GiB)."
      echo "Close other programs, or use a smaller model (the 4B model in this doc), then rerun."
      exit 2
    elif [ "$avail" -lt $((size + 3 * GIB)) ]; then
      echo "warning: only $((avail / 1048576)) MiB available; $(((size + 3 * GIB) / 1048576)) MiB (model size + 3 GiB) is comfortable. Close other programs if it stalls." >&2
    fi
  fi
fi
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
cd "$DEMO_DIR"
exec env VK_DRIVER_FILES="$ICD" "./$BIN" "$@"
