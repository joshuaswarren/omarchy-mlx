#!/usr/bin/env bash
# Build the demo stack from source in this directory: the Honeykrisp v3 Vulkan
# driver into a user prefix, llama.cpp with Vulkan, and the demo model.
# Safe to rerun; installs nothing system-wide.
set -euo pipefail
mkdir -p llamacpp-demo
cd llamacpp-demo

MESA_COMMIT=66cb84fd431d            # honeykrisp-omarchy-v3 head
LLAMA_COMMIT=65840ed                # receipt-pinned llama.cpp
MODEL_URL=https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF/resolve/main/Qwen_Qwen3.5-9B-IQ2_M.gguf
MODEL_FILE=Qwen_Qwen3.5-9B-IQ2_M.gguf

command -v meson ninja cmake glslc git >/dev/null || {
  echo "missing tools: sudo pacman -S --needed meson ninja cmake shaderc python-mako python-yaml pkgconf git"
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
cmake -B llama-build -S llama.cpp -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama-build --target llama-bench llama-cli -j"$(nproc)"
echo "LLAMA_BUILD_SECONDS=$((SECONDS-T0))"

# --- demo model, 4.03 GB ------------------------------------------------------
EXPECTED_SIZE=4027985952   # bartowski Qwen3.5-9B-IQ2_M, verified 2026-10-08
until [ "$(stat -c%s "$MODEL_FILE" 2>/dev/null || echo 0)" -eq "$EXPECTED_SIZE" ]; do
  curl -L -C - --retry 8 --retry-all-errors -o "$MODEL_FILE" "$MODEL_URL"
done
sha256sum "$MODEL_FILE"

ICD=$(ls mesa-install/share/vulkan/icd.d/*.json | head -1)
echo "everything is in: $(pwd)"
echo "ICD_JSON=$ICD"
echo "benchmark commands:"
echo "  llama-bench -m $MODEL_FILE -p 512 -n 64 -ngl 99 -t 8"
echo "  VK_DRIVER_FILES=$ICD llama-bench -m $MODEL_FILE -p 512 -n 64 -ngl 99 -t 8"
