#!/usr/bin/env bash
# Same-wheel A/B on the CPU-disabled head wheel: switch "0" (composition) vs unset (part default, fused on G15S). Budget unset.
set -uo pipefail
L=<lane>
R=<repo>
PN=$L/venv-nocpu/bin/python; WN=$(ls $L/dist-nocpu/*.whl)
M=<models>/qwen3-4b-2507-4bit
mkdir -p $L/runs/nocpu
G=$L/scripts/gpuleg.sh
$G N00-cpu-refused 120 $PN -I -c "import mlx.core as mx
a=mx.ones((4,)); mx.eval(a+a); print('gpu add ok', (a+a).tolist())
try:
    mx.eval(mx.add(a, a, stream=mx.cpu)); print('CPU ADD RAN')
except Exception as e: print('cpu add refused:', str(e)[:200])"
i=0
for arm in off on; do i=$((i+1)); sw="MLX_OMARCHY_SDPA_F16_NO_COOPMAT=0"; [ $arm = on ] && sw="-u MLX_OMARCHY_SDPA_F16_NO_COOPMAT"
  WAIT_LOAD=1 $G N1$i-percall-$arm 600 env -u MLX_OMARCHY_BATCH_WORK $sw $PN -I $L/scripts/sdpa_f16_ab.py $L/runs/nocpu/N1$i-$arm.json $L/runs/nocpu/N1$i-$arm.npz $WN $R N1$i-$arm; done
i=0
for arm in off on on off; do i=$((i+1)); sw="MLX_OMARCHY_SDPA_F16_NO_COOPMAT=0"; [ $arm = on ] && sw="-u MLX_OMARCHY_SDPA_F16_NO_COOPMAT"
  sleep 15
  WAIT_LOAD=1 $G E2$i-nocpu-$arm 1200 env -u MLX_OMARCHY_BATCH_WORK $sw MLX_OMARCHY_NO_MLXLM_WARNING=1 HF_HUB_OFFLINE=1 $PN -I $L/scripts/e2e_f16.py $M $L/runs/nocpu/E2$i-$arm.json $L/runs/nocpu/E2$i-$arm.npz $WN $R E2$i-nocpu-$arm $L/scripts/context-text.txt; done
echo "NOCPU_DONE $(date -u +%FT%TZ)" > $L/runs/nocpu/MARKER
