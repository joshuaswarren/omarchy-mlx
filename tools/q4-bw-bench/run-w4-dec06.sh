#!/usr/bin/env bash
# DEC-06 W4 screen on the 16-inch M1 Max (G13C): production W2-fast vs the
# QMM_W4 uvec4-load arm of the same shader, real GPU, bit-exactness compare
# plus GB/s. Two legs for repeatability. Runs under gpu-turn; every dispatch
# inside the bench is ms-scale.
set -euo pipefail
cd /tmp
SH=/tmp/qmm_vec.comp
B="-DUSE_BF16=1 -DUSE_SUBGROUP=1 -DQMM_VEC_Q4_WORD=1 -DQMM_VEC_MULTI=1 -DROWS_PER_SLOT=2 -DSLOTS_PER_GROUP=8 -DQMM_VEC_NO_SWIGLU=1 -DQMM_FAST_ONLY=1"
# Same compiler preference as omarchy_shader(): glslc where real (this box),
# glslangValidator fallback. A shader that only compiles under the validator
# would fail the real build.
compile() { # out defines...
  local out=$1; shift
  if glslc -fshader-stage=compute --target-env=vulkan1.3 -I/tmp "$@" -o "$out" "$SH" 2>/tmp/c.log; then return 0; fi
  glslangValidator -V --target-env vulkan1.3 -I/tmp "$@" -o "$out" "$SH"
}
compile /tmp/q4dec06_w2.spv $B -DQMM_W2=1
compile /tmp/q4dec06_w4.spv $B -DQMM_W2=1 -DQMM_W4=1
head -c 16 /tmp/q4dec06_w2.spv | od -An -tx1 | head -1
cmp -s /tmp/q4dec06_w2.spv /tmp/q4dec06_w4.spv && { echo "FATAL: variants identical"; exit 3; }
echo "variant blobs differ: $(stat -c %s /tmp/q4dec06_w2.spv) vs $(stat -c %s /tmp/q4dec06_w4.spv) bytes"
echo "== AGX register/occupancy dump (driver compile of both variants) =="
AGX_DEBUG=shaderdb,verbose /tmp/q4-bw-bench-w4 --gap --2b --quick --inc /tmp \
  --base-src "$SH" \
  --base-defs "$B -DQMM_W2=1" \
  --cand "$SH:8" --cand-def "$B -DQMM_W4=1" > /tmp/w4-dec06-agx.log 2>&1 || true
grep -aE "half|regs|loop|shader|SHADER" /tmp/w4-dec06-agx.log | head -40 || true
for leg in a b; do
  /tmp/q4-bw-bench-w4 --gap --2b --quick --inc /tmp \
    --base-src "$SH" \
    --base-defs "$B -DQMM_W2=1" \
    --cand "$SH:8" --cand-def "$B -DQMM_W4=1" 2>&1 \
    | tee "/tmp/w4-dec06-$leg.ndjson" | grep -aE '"k":"eq"|MM |widedep_cand|iso4_cand|widedep_base|iso4_base'
done
# k % 1024 != 0 shapes (kv512, qkv) never dispatch W4 in production (the
# host gate requires K % 1024 == 0); their W4-arm output is out of
# contract, so the verdict counts only the eligible shapes.
for leg in a b; do
  ok=$(grep -a '"k":"eq"' "/tmp/w4-dec06-$leg.ndjson" \
    | grep -avE '"shape":"(kv512|qkv)"' \
    | awk -F'bit_mismatches":' '{s += $2 + 0; n += 1} END {print s "/" n}')
  echo "leg-$leg eligible-shape bit-mismatches/total: $ok"
done
echo W4-DEC06-DONE
