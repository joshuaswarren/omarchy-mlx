#!/bin/bash
# MatmulGap H13 f16 a @ b cause probe (jw16, one gpu-turn ticket).
# 1) shaderdb of every pipeline the f16 a @ b run compiles, M vs C1 (bf16 ICD, cache off).
# 2) f16 a @ b 4096^3 timing, fresh process per sample, M C1 C1 M x3 (12 samples).
set -u
H=/var/tmp/w7q-h8
export VK_ICD_FILENAMES=$HOME/.local/share/coreglass/vulkan-4b-bf16cmat02/honeykrisp_icd.aarch64.json MESA_SHADER_CACHE_DISABLE=1
for v in venv-h13main venv-c1; do
  AGX_MESA_DEBUG=shaderdb "$H/$v/bin/python" "$H/nnprobe.py" 1 2>&1 | grep 'CS shader' | sort > "$H/nnprobe-shaderdb-$v.txt"
  echo "shaderdb $v: $(wc -l < "$H/nnprobe-shaderdb-$v.txt") pipelines"
done
cmp -s "$H/nnprobe-shaderdb-venv-h13main.txt" "$H/nnprobe-shaderdb-venv-c1.txt" && echo "shaderdb identical" || diff "$H/nnprobe-shaderdb-venv-h13main.txt" "$H/nnprobe-shaderdb-venv-c1.txt"
for r in 1 2 3; do
  for v in venv-h13main venv-c1 venv-c1 venv-h13main; do
    echo "$v $("$H/$v/bin/python" "$H/nnprobe.py" 30)"
  done
done
