#!/bin/bash
# gdn-s1.sh MODEL TAG SRC TESTBIN: STEP 1 check of the default of the f32 A_log GDN decode route on one machine, Qwen3.5-9B 4-bit, ONE wheel (commit 87bd20f9), ONE boot, env only.
# Part 0 (device test): the test binary omarchy_gdn_gqa_bf16a_tests of the same commit: the new case alone, then the whole binary, default env.
# Arms: D = the default (no GDN variable set), T = MLX_OMARCHY_GDN_DECODE_TILE=1 (the tiled kernel: the default before this change on G14C and G13C),
#       U = MLX_OMARCHY_GDN_DECODE_TILE=0 (the untiled composed kernel with the expansion copies: the default before this change on the 13-inch part).
# Part 1 (speed): bench_decode.py, 64 tokens, prompts P1 and P2, 5 rounds, order rotating D T U / T U D / U D T / D T U / T U D.
# Part 2 (numerics): the recorder gdn_logits.py, 128 greedy tokens, per-step log-probabilities (full vector for the first 32 steps): P1 with arms D, T, U, D again (D2); P2 with D and T.
# Part 3 (census): the recorder with MLX_OMARCHY_TRACE_DISPATCH=1, arms D, T and U (U = MLX_OMARCHY_GDN_DECODE_TILE=0, the untiled composed kernel with the expansion copies), 6 and 10 tokens of P1; per-token = difference over 4.
# Measure only. Writes only under ~/scratch/gdn/gdn-s1-TAG-1. Wheel and venv: the outputs of the build ticket step1 (the venv has only the wheel replaced).
set -uo pipefail
MODEL=${1:?model}
TAG=${2:?tag}
SRC=${3:?scripts source dir}
TB=${4:?test binary}
Q=$HOME/scratch/gdn/gdn2
B0=$HOME/scratch/gdn/step1-87bd20f9
VP=$B0/venv
O=$HOME/scratch/gdn/gdn-s1-$TAG-1
P1="Explain in plain words how a bicycle works."
P2="Write a short story about a lighthouse keeper."
WHL=$(find "$B0/wheel" -maxdepth 1 -name 'mlx_omarchy-*+87bd20f9-cp314-cp314-linux_aarch64.whl' | head -1)
[ -e "$O" ] && { echo "ABORT: $O exists"; exit 1; }
[ -n "$WHL" ] || { echo "ABORT no wheel"; exit 1; }
mkdir -p "$O"
L=$O/run.log
say() { echo "GDNS1 $*" | tee -a "$L"; }
say "start $(date -u +%FT%TZ) kernel=$(uname -r) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) governor=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor) load1=$(cut -d' ' -f1 /proc/loadavg) model=$MODEL"
say "env variables GDN/MLX/HK/AGX present: $(env | grep -cE '^(GDN|MLX_|HK_|AGX_)')"
say "wheel $(basename "$WHL") sha256 $(sha256sum "$WHL" | cut -d' ' -f1)"
say "libmlx $(sha256sum "$VP"/lib/python3.14/site-packages/mlx/lib/libmlx.so | cut -c1-16) vulkan driver library $(sha256sum /usr/lib/libvulkan_asahi.so | cut -c1-16)"
say "scripts bench_decode $(sha256sum "$SRC/scripts/bench_decode.py" | cut -c1-16) logits $(sha256sum "$Q/gdn_logits.py" | cut -c1-16)"
say "patches tree $(cd "$SRC/patches" && find . -type f | sort | xargs sha256sum | sha256sum | cut -c1-16) files $(find "$SRC/patches" -type f | wc -l)"
say "model snapshot $(ls "$HOME/.cache/huggingface/hub/models--${MODEL//\//--}/snapshots") MemAvailable_kB $(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
say "venv $("$VP/bin/python" -m pip list 2>/dev/null | grep -i '^mlx' | tr -s ' ' | tr '\n' ';')"
WANT_TB=dc1a37e6815cb314ad685663e23e96e5a3a89dc4cec065eb9457588c0d875dff
[ "$(sha256sum "$TB" | cut -c1-64)" = "$WANT_TB" ] || { say "ABORT test binary hash differs"; exit 1; }
say "device: $(vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverName|driverInfo|apiVersion' | tr -s ' ' | tr '\n' ';')"
say "test binary $TB sha256 $(sha256sum "$TB" | cut -c1-64) (equal to the build log of the commit)"
timeout 300 "$TB" --test-case="gdn raw decode f32 A_log: the default kernel by part*" > "$O/test-new-case.out" 2>&1
say "test new case rc=$? $(grep -a -E 'test cases:|assertions:' "$O/test-new-case.out" | tr -s ' ' | tr '\n' ';')"
say "test default line: $(grep -a 'gdn_pf_gqa_default' "$O/test-new-case.out" | grep -v provenance | head -1 | cut -c1-200)"
timeout 900 "$TB" > "$O/test-all.out" 2>&1
say "test whole binary rc=$? $(grep -a -E 'test cases:|assertions:' "$O/test-all.out" | tr -s ' ' | tr '\n' ';')"
export HF_HUB_OFFLINE=1
env_of() {
  case $1 in
    D|D2) echo "MLX_AB_UNUSED=1" ;;
    T) echo "MLX_OMARCHY_GDN_DECODE_TILE=1" ;;
    U) echo "MLX_OMARCHY_GDN_DECODE_TILE=0" ;;
  esac
}
prompt_of() { case $1 in P1) echo "$P1" ;; P2) echo "$P2" ;; esac; }
for i in 1 2 3 4 5; do
  for pn in P1 P2; do
    case $((i % 3)) in
      1) order="D T U" ;;
      2) order="T U D" ;;
      0) order="U D T" ;;
    esac
    for arm in $order; do
      # shellcheck disable=SC2046
      env $(env_of $arm) timeout 600 "$VP/bin/python" "$SRC/scripts/bench_decode.py" --model "$MODEL" --prompt "$(prompt_of $pn)" --tokens 64 --wheel "$WHL" > "$O/t-$pn-$arm-$i.out" 2> "$O/t-$pn-$arm-$i.err"
      say "time $pn round $i $arm rc=$? $(grep -a 'decode ' "$O/t-$pn-$arm-$i.out" | head -1 | cut -c1-60) $(grep -a 'generated_ids' "$O/t-$pn-$arm-$i.out" | head -1 | cut -c1-60) $(grep -a 'provenance' "$O/t-$pn-$arm-$i.out" | grep -o 'verified=[a-z]*')"
    done
  done
done
for run in "P1 D" "P1 T" "P1 U" "P1 D2" "P2 D" "P2 T"; do
  set -- $run
  pn=$1
  arm=$2
  # shellcheck disable=SC2046
  env $(env_of $arm) timeout 900 "$VP/bin/python" "$Q/gdn_logits.py" --model "$MODEL" --prompt "$(prompt_of $pn)" --tokens 128 --full-steps 32 --out "$O/num-$pn-$arm" --wheel "$WHL" --scripts "$SRC/scripts" > "$O/num-$pn-$arm.out" 2> "$O/num-$pn-$arm.err"
  say "numerics $pn $arm rc=$? $(grep -a 'logits-run' "$O/num-$pn-$arm.out" | head -1 | cut -c1-110) $(grep -a 'provenance' "$O/num-$pn-$arm.out" | grep -o 'verified=[a-z]*')"
done
for arm in D T U; do
  for n in 6 10; do
    # shellcheck disable=SC2046
    env $(env_of $arm) MLX_OMARCHY_TRACE_DISPATCH=1 timeout 600 "$VP/bin/python" "$Q/gdn_logits.py" --model "$MODEL" --prompt "$P1" --tokens $n --full-steps 32 --out "$O/cen-$arm-$n" --wheel "$WHL" --scripts "$SRC/scripts" 2>&1 >"$O/cen-$arm-$n.out" \
      | sed -n 's/.*DISPATCH kernel=\([0-9]*\) .*/\1/p' | sort -n | uniq -c | sort -k2,2n > "$O/hist-$arm-$n.txt"
    say "census $arm tokens=$n dispatches=$(awk '{s+=$1} END {print s+0}' "$O/hist-$arm-$n.txt") kernels=$(wc -l < "$O/hist-$arm-$n.txt") $(grep -a 'provenance' "$O/cen-$arm-$n.out" | grep -o 'verified=[a-z]*')"
  done
done
say "end $(date -u +%FT%TZ)"
