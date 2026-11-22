#!/usr/bin/env bash
# Round 2 GPU legs on code commit 95126c59f (G15S fused f16 decode below 1024 keys only). One lock hold per leg (gpuleg.sh).
set -uo pipefail
L=${LANE}
R=${REPO}
WS=${BASE_BUILD}
B="$R/.work/mlx/build-head"; T="$B/tests/omarchy"; INFO="$B/tools/mlx-omarchy-info/mlx-omarchy-info"
TB="$L/scratch/wt-before/.work/mlx/build-before/tests/omarchy"
M4B=${MODELS}/qwen3-4b-2507-4bit; TXT=$L/scripts/context-text.txt
GDB_CPU="$R/receipts/2026-09-30-speech-input-gpu/harness/count_cpu.gdb.py"
WB=$(ls $WS/dist-base/*.whl); PB=$WS/venv-base/bin/python
WH=$(ls $L/dist-head/*.whl); PH=$L/venv-head/bin/python
WN=$(ls $L/dist-nocpu/*.whl); PN=$L/venv-nocpu/bin/python
SUM=$L/logs/r2-summary.txt; : > "$SUM"; G=$L/scripts/gpuleg.sh
summ() { grep -E '^\[doctest\] (test cases|assertions|Status)' "$L/logs/$1.log" | tr '\n' ' '; }
leg() { local name=$1; shift; out=$("$G" "$name" "$@"); rc=$?; echo "$out | $(summ "$name")" | tee -a "$SUM"; return 0; }
O=$L/runs/r2; mkdir -p $O
NEWCASE="--test-case=G15S f16 decode stays on one-pass below 1024 keys and avoids two-pass overflow"
U="env -u MLX_OMARCHY_SDPA_F16_NO_COOPMAT -u MLX_OMARCHY_BATCH_WORK"

leg R2-T00-info-before 120 "$INFO" --json
leg R2-T01-newcase-before-8304e5dd3 300 "$TB/omarchy_sdpa_decode_fused_tests" "$NEWCASE"
leg R2-T02-newcase-head 300 "$T/omarchy_sdpa_decode_fused_tests" "$NEWCASE"
leg R2-T03-runtime-case-head 120 "$T/omarchy_runtime_tests" "--test-case=f16 decode sdpa without the cooperative matrix defaults on G16S and G15S one-pass"
leg R2-T04-sdpa-decode-fused-head 600 "$T/omarchy_sdpa_decode_fused_tests"
SUITES=(runtime primitive matmul_family fast_ops kv_ops indexing_ops reduce_ops shape_ops linalg_ops copy_offset distributed compiled_tape fft_ops fft_general eig_ops take_fill conv complex_ops select_layout fast_regression scatter_determinism eq_math fused_chain error_contract ane_bundle)
PROFILES=(m1-honeykrisp-fork m1-stock-no-coopmat subgroup-size-64 small-shared-memory no-cooperative-matrix m1-g13-legacy workgroup-256 workgroup-128)
n=0; for s in "${SUITES[@]}"; do n=$((n+1)); leg $(printf 'R2-S%02d-%s' $n $s) 1800 "$T/omarchy_${s}_tests"; done
n=0; for p in "${PROFILES[@]}"; do n=$((n+1)); leg $(printf 'R2-P%02d-%s' $n $p) 1800 env MLX_OMARCHY_ALLOW_NON_APPLE=1 "$T/omarchy_capability_sim_tests" "$p"; done

# Overflow input, defaults: base wheel (composition everywhere) and head wheel.
leg R2-X1-overflow-base-default 120 $U $PB -I $L/scripts/overflow_probe.py $WB $R
leg R2-X2-overflow-head-default 120 $U $PH -I $L/scripts/overflow_probe.py $WH $R

# Per-call A/B, base wheel default vs head wheel default, 3 interleaved pairs, shapes on both sides of 1024 keys.
i=0; for side in base head head base base head; do i=$((i+1)); P=$PB; W=$WB; [ $side = head ] && { P=$PH; W=$WH; }
  WAIT_LOAD=1 leg R2-B$i-percall-$side 600 $U SDPA_AB_SHAPES=r2 $P -I $L/scripts/sdpa_f16_ab.py $O/R2-B$i-$side.json $O/R2-B$i-$side.npz $W $R R2-B$i-$side; done
# End-to-end A/B, base vs head defaults, contexts 64 / 512 / 1100, budget unset then 15000.
i=0; for bw in unset 15000; do for side in base head head base base head; do i=$((i+1)); P=$PB; W=$WB; [ $side = head ] && { P=$PH; W=$WH; }
  benv="-u MLX_OMARCHY_BATCH_WORK"; [ $bw != unset ] && benv="MLX_OMARCHY_BATCH_WORK=$bw"
  sleep 15
  WAIT_LOAD=1 leg R2-E$(printf %02d $i)-bw$bw-$side 1200 env -u MLX_OMARCHY_SDPA_F16_NO_COOPMAT $benv E2E_CTX=64,512,1100 MLX_OMARCHY_NO_MLXLM_WARNING=1 HF_HUB_OFFLINE=1 $P -I $L/scripts/e2e_f16.py $M4B $O/R2-E$(printf %02d $i)-bw$bw-$side.json $O/R2-E$(printf %02d $i)-bw$bw-$side.npz $W $R R2-E$i-$side $TXT
done; done

# CPU-disabled head wheel: control, then same-wheel switch 0 vs unset.
leg R2-N00-cpu-refused 120 $PN -I $L/scripts/cpu_refused.py $WN $R
WAIT_LOAD=1 leg R2-N11-percall-off 600 env -u MLX_OMARCHY_BATCH_WORK MLX_OMARCHY_SDPA_F16_NO_COOPMAT=0 SDPA_AB_SHAPES=r2 $PN -I $L/scripts/sdpa_f16_ab.py $O/R2-N11-off.json $O/R2-N11-off.npz $WN $R R2-N11-off
WAIT_LOAD=1 leg R2-N12-percall-on 600 $U SDPA_AB_SHAPES=r2 $PN -I $L/scripts/sdpa_f16_ab.py $O/R2-N12-on.json $O/R2-N12-on.npz $WN $R R2-N12-on
i=0; for arm in off on on off; do i=$((i+1)); sw="MLX_OMARCHY_SDPA_F16_NO_COOPMAT=0"; [ $arm = on ] && sw="-u MLX_OMARCHY_SDPA_F16_NO_COOPMAT"; sleep 15
  WAIT_LOAD=1 leg R2-N2$i-e2e-$arm 1200 env -u MLX_OMARCHY_BATCH_WORK $sw E2E_CTX=64,512,1100 MLX_OMARCHY_NO_MLXLM_WARNING=1 HF_HUB_OFFLINE=1 $PN -I $L/scripts/e2e_f16.py $M4B $O/R2-N2$i-$arm.json $O/R2-N2$i-$arm.npz $WN $R R2-N2$i-$arm $TXT; done

# Zero CPU tensor dispatches, provenance printed in each process.
leg R2-H30-trace-cpu-control 600 $U gdb -batch -x "$GDB_CPU" --args $PH -I $L/scripts/trace_decode_f16.py $M4B 1 $WH $R --cpu-control
leg R2-H31-trace-f16-decode-head 900 $U gdb -batch -x "$GDB_CPU" --args $PH -I $L/scripts/trace_decode_f16.py $M4B 32 $WH $R
leg R2-H32-trace-f16-decode-base 900 $U gdb -batch -x "$GDB_CPU" --args $PB -I $L/scripts/trace_decode_f16.py $M4B 32 $WB $R
leg R2-H40-reopen-runtime 120 "$T/omarchy_runtime_tests" "--test-case=a fresh process reopens the Vulkan device"
leg R2-H41-reopen-smoke 120 "$INFO" --trace-smoke
leg R2-H42-info-after 120 "$INFO" --json
echo "R2_DONE $(date -u +%FT%TZ)" > $O/MARKER
