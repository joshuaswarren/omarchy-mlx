#!/bin/bash
# MLXCOOPFIX: build the patched omarchy-mlx C++ test binaries NATIVELY on the
# J616s from our fork branch coopmat/qmm-prefill-shape-gate. CPU + omarchy
# (Honeykrisp Vulkan) backend, tests ON. The ICD (Apple GPU driver) is chosen
# at RUNTIME via VK_DRIVER_FILES, so the build does not need the overlay.
# Launched detached; writes build-omx.log and OMX-DONE (ok) / OMX-FAIL (error).
set -u
W=/var/tmp/mlxcoopfix; OMX=$W/omx; LOG=$W/build-omx.log
REPO=https://github.com/aurora-silicon/omarchy-mlx.git
BRANCH=coopmat/qmm-prefill-shape-gate
mkdir -p $W; : > $LOG; rm -f $W/OMX-DONE $W/OMX-FAIL
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
fail(){ log "FAIL: $*"; touch $W/OMX-FAIL; exit 1; }
command -v cmake >/dev/null || fail "no cmake"
command -v ninja >/dev/null || fail "no ninja"
command -v glslc >/dev/null || command -v glslangValidator >/dev/null || fail "no glsl compiler"

log "omx build on $(uname -n) $(uname -m); branch $BRANCH"
if [ ! -d $OMX/.git ]; then
  git clone --filter=blob:none $REPO $OMX >>$LOG 2>&1 || fail "clone"
fi
git -C $OMX fetch origin $BRANCH >>$LOG 2>&1 || fail "fetch"
git -C $OMX checkout -f $BRANCH >>$LOG 2>&1 || fail "checkout"
git -C $OMX reset --hard origin/$BRANCH >>$LOG 2>&1 || fail "reset"
log "omx HEAD=$(git -C $OMX rev-parse --short HEAD)"; git -C $OMX log --oneline -1 | tee -a $LOG

cd $OMX || fail "cd"
log "prepare-mlx (vendor pinned MLX + apply overlay/patches)"
bash scripts/prepare-mlx.sh >>$LOG 2>&1 || fail "prepare-mlx"
[ -f .work/mlx/mlx/backend/omarchy/coopmat_select.cpp ] || fail "coopmat_select.cpp not staged"

if [ ! -f .work/build/build.ninja ]; then
  log "cmake configure (CPU + omarchy, tests ON)"
  cmake -S .work/mlx -B .work/build -G Ninja \
    -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON \
    -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF \
    -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF \
    -DMLX_BUILD_BENCHMARKS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF \
    -DCMAKE_BUILD_TYPE=Release >>$LOG 2>&1 || fail "cmake configure"
fi
log "build omarchy_shaders"
cmake --build .work/build --target omarchy_shaders -j"$(nproc)" >>$LOG 2>&1 || fail "shaders"
log "build test targets (-j$(nproc))"
cmake --build .work/build \
  --target omarchy_coopmat_select_tests omarchy_matmul_family_tests \
           omarchy_runtime_tests omarchy_copy_offset_tests \
           omarchy_capability_sim_tests \
  -j"$(nproc)" >>$LOG 2>&1 || fail "test targets"
T=$OMX/.work/build/tests/omarchy
for b in omarchy_coopmat_select_tests omarchy_matmul_family_tests omarchy_runtime_tests omarchy_copy_offset_tests omarchy_capability_sim_tests; do
  [ -x "$T/$b" ] || fail "missing binary $b"
done
log "binaries in $T"
log "OMX-DONE"
touch $W/OMX-DONE
