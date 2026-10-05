#!/usr/bin/env bash
# H137b: generate the persistent two-stage GEMV shader from the bench copy of
# the production multi-row Q4 kernel. Usage: pair-gen.sh <out.comp>
# Line anchors are asserted; the script fails loudly if the base shader moved.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
src="$here/shaders/qmm_vec_base.comp"
out="$1"
chk() { [ "$(sed -n "$1p" "$src")" = "$2" ] || { echo "anchor $1 moved" >&2; exit 1; }; }
chk 574 'void main() {'
chk 585 '  uint workgroup = gl_WorkGroupID.x;'
chk 614 '  uint x_row = params.lhs_offset;'
chk 117 'layout(set = 0, binding = 0, std430) readonly buffer InputX {'
chk 151 'layout(set = 0, binding = B + 3, std430) writeonly buffer Output##I { \'
sed \
  -e '574s|.*|uint pair_xoff = 0u;\nvoid stage_main(uint pair_v) {|' \
  -e '585s|gl_WorkGroupID.x|pair_v|' \
  -e '614s|params.lhs_offset|params.lhs_offset + pair_xoff|' \
  -e '117s|readonly buffer|coherent readonly buffer|' \
  -e '121s|readonly buffer|coherent readonly buffer|' \
  -e '151s|writeonly buffer|coherent writeonly buffer|' \
  "$src" > "$out"
cat >> "$out" <<'EOF'
layout(set = 0, binding = 25, std430) coherent buffer PairSync {
  uint v[];
} pair_sync;
void pair_grid_barrier() {
  memoryBarrierBuffer();
  barrier();
  if (gl_LocalInvocationIndex == 0u) {
    uint g = gl_NumWorkGroups.x;
    uint gen = atomicAdd(pair_sync.v[1], 0u);
    uint old = atomicAdd(pair_sync.v[0], 1u);
    if (old == g - 1u) {
      atomicExchange(pair_sync.v[0], 0u);
      atomicAdd(pair_sync.v[1], 1u);
    } else {
      uint spins = 0u;
      while (atomicAdd(pair_sync.v[1], 0u) == gen) {
        if (++spins > 32768u) { atomicOr(pair_sync.v[2], 1u); break; }
      }
    }
  }
  barrier();
  memoryBarrierBuffer();
}
void main() {
  uint g0 = (params.shape[0] + COLUMNS_PER_GROUP - 1u) / COLUMNS_PER_GROUP;
  uint g1 = (params.shape[1] + COLUMNS_PER_GROUP - 1u) / COLUMNS_PER_GROUP;
  uint g = gl_NumWorkGroups.x;
  for (uint v = gl_WorkGroupID.x; v < g0; v += g) {
    pair_xoff = 0u;
    stage_main(v);
  }
  pair_grid_barrier();
  for (uint v = g0 + gl_WorkGroupID.x; v < g0 + g1; v += g) {
    pair_xoff = params.aux_offset;
    stage_main(v);
  }
}
EOF
