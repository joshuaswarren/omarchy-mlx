#!/bin/bash
# runprov.sh <state on|off> <tsv> <python> <key> <hf_repo> <rounds> <old_tree> <new_tree>
# ONE ticket = provenance + dispatch trace + timing + reopen, same boot, same wheel/model/driver, same ANE state.
#   P0 provrec.py JSON (kernel, wheel hashes vs RECORD, model snapshot + file sha256, quant, Vulkan device/driver, governor)
#   P1 dispatch trace per arm (V = staged route, P = default direct route): kernel-id histogram from
#      MLX_OMARCHY_TRACE_DISPATCH=1 (TRACE_ARMS, default V P) and a gdb count of CPU command-encoder calls (must be 0), 4 greedy tokens, untimed
#   P2 timed interleaved block (runwhl.sh, arms BLK_ARMS default V P, the stated rounds)
#   P3 device reopen: vulkaninfo and a fresh process doing one GPU op, rc recorded; GPU-timeout and OOM deltas
# The stated state must match the observed ANE state: "on" needs the filler running, "off" needs none. A mismatch
# aborts before any timing, so a row never claims a state it was not measured in. Quiet-OFF windows are booked and
# toggled by the caller (w7N filler-toggle.sh); this script never toggles the filler.
set -uo pipefail
S=$HOME/scratch/w71/score
STATE=$1 TSV=$2 PYB=$3 KEY=$4 REPO=$5 ROUNDS=$6 OLD=$7 NEW=$8
H=$HOME/.cache/huggingface/hub
cd "$S" || exit 1
T=prov-$KEY-$STATE-$(date -u +%H%M)
L=$S/sb-$T.log
say() { echo "PROV $T $*" | tee -a "$L"; }
say "command: runprov.sh $*"
d=$H/models--${REPO//\//--}
pre=0; [ -d "$d" ] && pre=1
if [ "$pre" = 0 ]; then
  gate=$(df -k /home | awk -v gb="${DL_GB:-20}" 'NR==2{used=$3; size=$2; avail=$4; add=gb*1048576; printf "%.1f %.1f", (used+add)*100/size, (avail-add)/1048576}')
  proj=${gate% *} freeafter=${gate#* }
  awk -v p="$proj" -v f="$freeafter" 'BEGIN{exit !(p<=82 && f>=50)}' || { say "ABORT: JIT download of ${DL_GB:-20} GB would leave /home at ${proj}% with ${freeafter} GB free (limits: 82% and 50 GiB)"; exit 1; }
  say "JIT gate ok: DL_GB=${DL_GB:-20} projected ${proj}% root use, ${freeafter} GB free after"
  export SCORE_TSV=$TSV
  PYTHONPATH=$S/mlxlm-latest-deps "$PYB" "$S/dl.py" "$KEY-blk" mlx "$REPO" >> "$L" 2>&1
  grep -q "^DL FAIL" "$L" && { say "ABORT: download failed"; exit 1; }
  say "JIT download done; the model is purged at the end"
fi
nane=$(pgrep -fc '[a]ne-run|[i]dle-guard-ane-job' || true)
case "$STATE" in
  on) [ "${nane:-0}" -ge 1 ] || { say "ABORT: state=on but no ANE filler is running"; exit 1; } ;;
  off) [ "${nane:-0}" -eq 0 ] || { say "ABORT: state=off but $nane ANE process(es) running"; exit 1; } ;;
  *) say "ABORT: state must be on or off"; exit 1 ;;
esac
say "state=$STATE ane_processes=$nane load1_pre=$(cut -d' ' -f1 /proc/loadavg) governor=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor)"
counts() {
  local j
  j=$(journalctl -k -b --no-pager 2> /dev/null)
  echo "gpu_timeouts=$(grep -c 'GPU timeout' <<< "$j") recovery_lines=$(grep -c 'Attempting recovery' <<< "$j") fault_info_lines=$(grep -c 'Fault info' <<< "$j") oom=$(grep -c 'Out of memory' <<< "$j")"
}
c0=$(counts); say "COUNTS before-trace $c0"
python3 "$S/provrec.py" "$S/$NEW" "$REPO" > "$S/$T.prov.json" 2>> "$L" || { say "ABORT: provrec failed"; exit 1; }
say "P0 provenance -> $T.prov.json sha256=$(sha256sum "$S/$T.prov.json" | cut -c1-16)"
export HF_HUB_OFFLINE=1 SB_SYSTEM=1
GDB=gdb
tfail=0
ran=""
for arm in ${TRACE_ARMS-V P}; do
  envv=(); [ "$arm" = V ] && envv=(MLX_OMARCHY_GATHER_QMM_DIRECT=0)
  timeout 300 env "${envv[@]}" PYTHONPATH="$S/$NEW:$S/mlxlm-latest-deps" MLX_OMARCHY_TRACE_DISPATCH=1 "$PYB" "$S/tracerun.py" "$REPO" 4 > "$S/$T.trace-$arm.out" 2> "$S/$T.trace-$arm.err"
  trc=$?
  say "P1 trace $arm rc=$trc $(grep -h TRACERUN "$S/$T.trace-$arm.out")"
  grep -h '\[rtmod\] DISPATCH' "$S/$T.trace-$arm.err" | sed 's/.*kernel=\([0-9]*\) .*/\1/' | sort -n | uniq -c | awk '{print $2":"$1}' | tr '\n' ' ' > "$S/$T.kernels-$arm.txt"
  say "P1 kernel-id:count $arm $(cat "$S/$T.kernels-$arm.txt")"
  timeout 360 env "${envv[@]}" PYTHONPATH="$S/$NEW:$S/mlxlm-latest-deps" "$GDB" -batch -x "$S/count_cpu.gdb.py" --args "$PYB" "$S/tracerun.py" "$REPO" 4 > "$S/$T.gdb-$arm.out" 2>&1
  inf=$(grep -o 'exited normally\|exited with code [0-9]*' "$S/$T.gdb-$arm.out" | tail -1)
  cc=$(sed -n 's/^CPU_COUNT //p' "$S/$T.gdb-$arm.out" | head -1)
  cpun=$(printf '%s' "$cc" | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print(d["cpu_command_encoder_calls"] if d["resolved_while_running"] and d["libmlx_loaded"] else "unresolved")' 2> /dev/null)
  say "P1 cpu-count $arm inferior=${inf:-none} calls=${cpun:-missing} raw=$(printf '%s' "$cc" | cut -c1-200)"
  ran="$ran$arm "
  pg=OK
  [ "$trc" = 0 ] || pg="trace rc=$trc"
  [ -s "$S/$T.kernels-$arm.txt" ] || pg="$pg; kernel-id file empty"
  grep -q 'tokens=4 ' "$S/$T.trace-$arm.out" || pg="$pg; no 4-token TRACERUN line"
  [ "$inf" = "exited normally" ] || pg="$pg; gdb inferior ${inf:-none}"
  grep -q '^TRACERUN .* tokens=4 ' "$S/$T.gdb-$arm.out" || pg="$pg; no 4-token TRACERUN in the gdb run"
  [ "$cpun" = 0 ] || pg="$pg; cpu calls ${cpun:-missing}"
  say "P1 gate $arm: $pg"
  [ "$pg" = OK ] || tfail=1
done
armsok=1
[ "${TRACE_ARMS-V P}" = "V P" ] && [ "${ran% }" = "V P" ] || armsok=0
if [ -z "${TRACE_ARMS-V P}" ]; then p1=SKIPPED; elif [ "$tfail" = 0 ] && [ "$armsok" = 1 ]; then p1=PASS; else p1=FAIL; fi
say "P1 RESULT: $p1 (arms ran: ${ran:-none})"
say "COUNTS after-trace $(counts)"
if [ -n "${PROV_SMOKE:-}" ]; then
  say "P2 skipped (PROV_SMOKE: harness check, no timing, nothing publishable)"
else
  BLK_ARMS=${BLK_ARMS:-V P} SB_GOV=performance bash "$S/runwhl.sh" "$TSV" "$PYB" "$KEY" "$ROUNDS" "$OLD" "$NEW" >> "$L" 2>&1
  say "P2 timed block rc=$?"
fi
c2=$(counts); say "COUNTS after-timed $c2"
vulkaninfo --summary > /dev/null 2>&1; vrc=$?; say "P3 reopen vulkaninfo rc=$vrc"
PYTHONPATH="$S/$NEW" "$PYB" -c "import mlx.core as mx; a=mx.ones(1024); mx.eval(a+a); print(mx.default_device())" > "$S/$T.reopen.out" 2>&1
grc=$?; say "P3 reopen gpu-op rc=$grc $(tail -1 "$S/$T.reopen.out")"
if [ "$pre" = 0 ]; then
  case "$d" in */hub/models--*--*) [ -d "$d" ] && rm -r "$d" && say "purged $d";; esac
  PYTHONPATH=$S/mlxlm-latest-deps "$PYB" "$S/orphans.py" "$H" --go --owner "models--${REPO//\//--}" 2> /dev/null | tail -1 >> "$L"
fi
say "DONE ane_end=$(pgrep -fc '[a]ne-run|[i]dle-guard-ane-job' || true)"
if [ -n "${PROV_SMOKE:-}" ]; then
  sf=""
  [ "$p1" = PASS ] || sf="$sf trace gate $p1;"
  [ "$vrc" = 0 ] || sf="$sf vulkaninfo rc=$vrc;"
  [ "$grc" = 0 ] || sf="$sf gpu-op rc=$grc;"
  [ "$c0" = "$c2" ] || sf="$sf COUNTS changed ($c0 -> $c2);"
  python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$S/$T.prov.json" 2> /dev/null || sf="$sf prov.json unreadable;"
  if [ -n "$sf" ]; then
    say "SMOKE RESULT: FAIL ($sf)"
    exit 1
  fi
  say "SMOKE RESULT: PASS"
fi
