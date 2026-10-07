#!/bin/bash
# MatmulGap H7 on jw16 (inside gpu-turn): shaderdb of the direct f16 kernel
# per arm, then the H6 GEMM cells and MLX-level cells with cand = the
# per-device gated build.
set -u
B=/var/tmp/w7q-mesa-build
S=/var/tmp/w7q-venv-stage
G=/var/tmp/w7q-gemm
echo "so base $(sha256sum $B/base/build/src/asahi/vulkan/libvulkan_asahi.so | cut -c1-16) cand $(sha256sum $B/cand/build/src/asahi/vulkan/libvulkan_asahi.so | cut -c1-16)"
for t in base cand; do
  echo "shaderdb $t $(MESA_SHADER_CACHE_DISABLE=1 AGX_MESA_DEBUG=shaderdb VK_ICD_FILENAMES=$B/$t.icd.json $G/gemm-bench --dtype f16 --mnk 128,128,128 --reps 1 --rounds 1 --side d=$G/sh/matmul_coopmat_direct.comp:64:64:0 2>&1 | grep 'CS shader' | cut -c1-160)"
done
$S/h6_run.sh
$S/h6b_run.sh
