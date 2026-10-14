#!/usr/bin/env bash
# Every qmm_vec / qmm_coopmat build that exists on the base commit compiles
# to the same SPIR-V from the base shader and from this branch's shader
# (glslc -O --target-env=vulkan1.3, the flags of embed_spirv.cmake).
# Usage: receipts/2026-10-13-qmm-formats/spirv-identity.sh [base-ref]
set -euo pipefail
base=${1:-origin/main}
root=$(git rev-parse --show-toplevel); cd "$root"
d=overlay/mlx/backend/omarchy
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
same=0; diff=0
for shader in qmm_vec qmm_coopmat; do
  git show "$base:$d/shaders/$shader.comp" > "$tmp/base.comp"
  while read -r name defs; do
    glslc -O --target-env=vulkan1.3 $defs "$tmp/base.comp" -o "$tmp/a.spv"
    glslc -O --target-env=vulkan1.3 $defs "$d/shaders/$shader.comp" -o "$tmp/b.spv"
    if cmp -s "$tmp/a.spv" "$tmp/b.spv"; then same=$((same + 1)); else
      diff=$((diff + 1)); echo "DIFF $name"; fi
  done < <(git show "$base:$d/CMakeLists.txt" |
    sed -nE "s/^omarchy_shader\(([a-z0-9_]+) shaders\/$shader\.comp ?(.*)\)$/\1 \2/p")
done
echo "identical: $same  different: $diff"
[ "$diff" = 0 ]
