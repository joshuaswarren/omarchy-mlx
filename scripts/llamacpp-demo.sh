#!/usr/bin/env bash
# Build the demo stack from source in llamacpp-demo/ inside this repository (wherever you run it from): the
# Honeykrisp v3 Vulkan driver into a user prefix, llama.cpp with Vulkan, and the demo model.
# Safe to rerun; installs nothing system-wide.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p llamacpp-demo
cd llamacpp-demo

MESA_COMMIT=66cb84fd431d            # honeykrisp-omarchy-v3 head
LLAMA_COMMIT=65840ed                # receipt-pinned llama.cpp
MODEL_URL=https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF/resolve/main/Qwen_Qwen3.5-9B-IQ2_M.gguf
MODEL_FILE=Qwen_Qwen3.5-9B-IQ2_M.gguf

# The system must be fully updated first. Installing build packages on an out-of-date system is a partial
# upgrade (for example llvm 23 next to llvm-libs 22 breaks llvm-spirv and the driver build).
# Override, only if you know your system is current enough: LLAMACPP_DEMO_SKIP_UPDATE_CHECK=1 bash scripts/llamacpp-demo.sh
if [ "${LLAMACPP_DEMO_SKIP_UPDATE_CHECK:-0}" = 1 ]; then
  echo "skipping the pending-updates check (LLAMACPP_DEMO_SKIP_UPDATE_CHECK=1)"
else
  rc=0
  if command -v checkupdates >/dev/null 2>&1; then
    pending=$(checkupdates 2>/dev/null) || rc=$?   # 0 = updates pending, 2 = none, anything else = could not check
    case $rc in
      2) ;;
      0) echo "the system is not fully updated ($(printf '%s\n' "$pending" | wc -l) packages pending). Update first, then rerun this script:"
         echo "  omarchy update"
         exit 1 ;;
      *) echo "could not check for pending updates (network or mirror problem). Run this first, then rerun this script:"
         echo "  omarchy update"
         exit 1 ;;
    esac
  else
    pending=$(pacman -Qu 2>/dev/null || true)
    if [ -n "$pending" ]; then
      echo "the system is not fully updated ($(printf '%s\n' "$pending" | wc -l) packages pending). Update first, then rerun this script:"
      echo "  omarchy update"
      exit 1
    fi
  fi
fi

command -v meson ninja cmake make bison flex glslc git pkg-config >/dev/null 2>&1 && \
python3 -c 'import mako' >/dev/null 2>&1 && \
pacman -Qq expat libdrm libelf libunwind zstd zlib llvm spirv-tools spirv-llvm-translator libclc spirv-headers vulkan-tools >/dev/null 2>&1 || {
  echo "missing build dependencies. Install them now (only on a system that is up to date, see Step 0 in the doc):"
  echo "  sudo pacman -S --needed meson ninja cmake make bison flex shaderc glslang python-mako python-yaml pkgconf git expat libdrm libelf libunwind zstd zlib llvm spirv-tools spirv-llvm-translator libclc spirv-headers vulkan-tools"
  exit 1
}

# --- Honeykrisp v3 Vulkan driver, user prefix ---------------------------------
[ -d mesa-1 ] || git clone -b honeykrisp-omarchy-v3 --filter=blob:none https://github.com/joshuaswarren/mesa-1
git -C mesa-1 fetch origin honeykrisp-omarchy-v3
git -C mesa-1 checkout -f "$MESA_COMMIT"
T0=$SECONDS
[ -f mesa-build/build.ninja ] || meson setup mesa-build mesa-1 --prefix "$PWD/mesa-install" \
  -Dvulkan-drivers=asahi -Dgallium-drivers= -Dplatforms= -Dvulkan-layers=
ninja -C mesa-build
ninja -C mesa-build install
echo "MESA_BUILD_SECONDS=$((SECONDS-T0))"

# --- llama.cpp with Vulkan ----------------------------------------------------
[ -d llama.cpp ] || git clone --filter=blob:none https://github.com/ggml-org/llama.cpp
git -C llama.cpp fetch origin
git -C llama.cpp checkout -f "$LLAMA_COMMIT"
T0=$SECONDS
cmake -B llama-build -S llama.cpp -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build llama-build --target llama-bench llama-cli -j"$(nproc)"
echo "LLAMA_BUILD_SECONDS=$((SECONDS-T0))"

# --- demo model, 4.03 GB ------------------------------------------------------
EXPECTED_SIZE=4027985952   # bartowski Qwen3.5-9B-IQ2_M, verified 2026-10-08
until [ "$(stat -c%s "$MODEL_FILE" 2>/dev/null || echo 0)" -eq "$EXPECTED_SIZE" ]; do
  curl -L -C - --retry 8 --retry-all-errors -o "$MODEL_FILE" "$MODEL_URL"
done
sha256sum "$MODEL_FILE"

# On an 8 GB-class machine, also pull the 4B model: it is the one that fits
# a stage demo there, and the 9B needs -ub 256 -b 256 under the old driver.
MEMTOTAL_MB=$(awk '/MemTotal/{print int($2/1024)}' /proc/meminfo)
if [ "${MEMTOTAL_MB:-0}" -lt 12288 ]; then
  MODEL4_URL=https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF/resolve/main/Qwen_Qwen3.5-4B-IQ2_M.gguf
  MODEL4_FILE=Qwen_Qwen3.5-4B-IQ2_M.gguf
  EXPECTED4_SIZE=1953751008   # bartowski Qwen3.5-4B-IQ2_M, verified 2026-10-08
  until [ "$(stat -c%s "$MODEL4_FILE" 2>/dev/null || echo 0)" -eq "$EXPECTED4_SIZE" ]; do
    curl -L -C - --retry 8 --retry-all-errors -o "$MODEL4_FILE" "$MODEL4_URL"
  done
  echo 'fd6c12b5d4f76be174edbf4769d2a27b0384da020f5cf1241463679798a9f4ec  '"$MODEL4_FILE" | sha256sum -c -
  echo "4B model downloaded; benchmark and chat commands below use it in place of $MODEL_FILE on this machine."
fi

REPO=$(dirname "$(pwd)")
echo "everything is in: $(pwd)"
echo "benchmark commands, run from the repository folder ($REPO):"
echo "  (cd llamacpp-demo && llama-build/bin/llama-bench -m $MODEL_FILE -p 512 -n 64 -ngl 99 -t 8)"
echo "  bash scripts/llamacpp-demo-run.sh llama-build/bin/llama-bench -m $MODEL_FILE -p 512 -n 64 -ngl 99 -t 8"
