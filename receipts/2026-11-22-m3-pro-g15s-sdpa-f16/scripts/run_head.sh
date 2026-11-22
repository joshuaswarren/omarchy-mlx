#!/usr/bin/env bash
# Head verification on the M3 Pro: info, the new runtime case, the fused-decode
# suite, the standing battery (25 suites + 8 capability-sim profiles), the
# default-unset before/after (base wheel vs head wheel), the f16 decode CPU
# trace, and device reopen. Every leg through gpuleg.sh (lock, timeout, state).
set -uo pipefail
L=<lane>
R=<repo>
WS=<base-build>
B="$R/.work/mlx/build-head"; T="$B/tests/omarchy"; INFO="$B/tools/mlx-omarchy-info/mlx-omarchy-info"
M4B=<models>/qwen3-4b-2507-4bit
TXT=$L/scripts/context-text.txt
GDB_CPU="$R/receipts/2026-09-30-speech-input-gpu/harness/count_cpu.gdb.py"
WB=$(ls $WS/dist-base/*.whl); PB=$WS/venv-base/bin/python
WH=$(ls $L/dist-head/*.whl); PH=$L/venv-head/bin/python
SUM=$L/logs/head-summary.txt; : > "$SUM"
G=$L/scripts/gpuleg.sh
summ() { grep -E '^\[doctest\] (test cases|assertions|Status)' "$L/logs/$1.log" | tr '\n' ' '; }
leg() { local name=$1; shift; out=$("$G" "$name" "$@"); rc=$?; echo "$out | $(summ "$name")" | tee -a "$SUM"; return $rc; }
mkdir -p $L/runs/head
export -n MLX_OMARCHY_SDPA_F16_NO_COOPMAT MLX_OMARCHY_BATCH_WORK

leg H00-info-before 120 "$INFO" --json
leg H01-runtime-new-case 120 "$T/omarchy_runtime_tests" "--test-case=f16 decode sdpa without the cooperative matrix is a G16S and G15S default"
leg H02-sdpa-decode-fused 600 "$T/omarchy_sdpa_decode_fused_tests" -s || true

SUITES=(runtime primitive matmul_family fast_ops kv_ops indexing_ops reduce_ops shape_ops linalg_ops copy_offset distributed compiled_tape fft_ops fft_general eig_ops take_fill conv complex_ops select_layout fast_regression scatter_determinism eq_math fused_chain error_contract ane_bundle)
PROFILES=(m1-honeykrisp-fork m1-stock-no-coopmat subgroup-size-64 small-shared-memory no-cooperative-matrix m1-g13-legacy workgroup-256 workgroup-128)
n=0; for s in "${SUITES[@]}"; do n=$((n+1)); leg $(printf 'S%02d-%s' $n $s) 1800 "$T/omarchy_${s}_tests" || echo "NONZERO S$n" | tee -a "$SUM"; done
n=0; for p in "${PROFILES[@]}"; do n=$((n+1)); leg $(printf 'P%02d-%s' $n $p) 1800 env MLX_OMARCHY_ALLOW_NON_APPLE=1 "$T/omarchy_capability_sim_tests" "$p" || echo "NONZERO P$n" | tee -a "$SUM"; done

# Default (switch unset, budget unset) before/after: base wheel, then head wheel.
WAIT_LOAD=1 leg D01-percall-base-default 600 env -u MLX_OMARCHY_SDPA_F16_NO_COOPMAT -u MLX_OMARCHY_BATCH_WORK $PB -I $L/scripts/sdpa_f16_ab.py $L/runs/head/D01-base.json $L/runs/head/D01-base.npz $WB $R D01-base-default
WAIT_LOAD=1 leg D02-percall-head-default 600 env -u MLX_OMARCHY_SDPA_F16_NO_COOPMAT -u MLX_OMARCHY_BATCH_WORK $PH -I $L/scripts/sdpa_f16_ab.py $L/runs/head/D02-head.json $L/runs/head/D02-head.npz $WH $R D02-head-default
WAIT_LOAD=1 leg D03-e2e-head-default 1200 env -u MLX_OMARCHY_SDPA_F16_NO_COOPMAT -u MLX_OMARCHY_BATCH_WORK MLX_OMARCHY_NO_MLXLM_WARNING=1 HF_HUB_OFFLINE=1 $PH -I $L/scripts/e2e_f16.py $M4B $L/runs/head/D03-head.json $L/runs/head/D03-head.npz $WH $R D03-head-default $TXT

# Zero CPU tensor dispatches: control, then the f16 decode on the head wheel.
leg H30-trace-cpu-control 600 env -u MLX_OMARCHY_SDPA_F16_NO_COOPMAT gdb -batch -x "$GDB_CPU" --args $PH -I $L/scripts/trace_decode_f16.py $M4B 1 --cpu-control
leg H31-trace-f16-decode-head 900 env -u MLX_OMARCHY_SDPA_F16_NO_COOPMAT gdb -batch -x "$GDB_CPU" --args $PH -I $L/scripts/trace_decode_f16.py $M4B 32
leg H32-trace-f16-decode-base 900 env -u MLX_OMARCHY_SDPA_F16_NO_COOPMAT gdb -batch -x "$GDB_CPU" --args $PB -I $L/scripts/trace_decode_f16.py $M4B 32

leg H40-reopen-runtime 120 "$T/omarchy_runtime_tests" "--test-case=a fresh process reopens the Vulkan device"
leg H41-reopen-smoke 120 "$INFO" --trace-smoke
leg H42-info-after 120 "$INFO" --json
echo "HEAD_DONE $(date -u +%FT%TZ)" > $L/runs/head/MARKER
