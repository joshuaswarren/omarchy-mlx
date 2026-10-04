#!/usr/bin/env bash
# BarrierSched census inner: barrier/dispatch counters per decode cell,
# wave scheduler OFF then ON, same candidate wheel both sides. Runs INSIDE
# a gpuwin window. Env: BS_OUT, BS_PY, BS_MODEL, BS_PROMPTS, BS_BENCH,
# BS_CELLS ("label:depth:prefill:passes;...").
set -euo pipefail
OUT=${BS_OUT:?missing BS_OUT}
PY=${BS_PY:?missing BS_PY}
MODEL=${BS_MODEL:?missing BS_MODEL}
PROMPTS=${BS_PROMPTS:?missing BS_PROMPTS}
BENCH=${BS_BENCH:?missing BS_BENCH}
CELLS=${BS_CELLS:?missing BS_CELLS}
TOOLS=$(cd "$(dirname "$0")" && pwd)

test ! -e /var/tmp/JW16_MAINTENANCE
uptime_s=$(cut -d. -f1 /proc/uptime); test "$uptime_s" -ge 360
load1=$(cut -d' ' -f1 /proc/loadavg)
load_int=${load1%%.*}; load_frac=${load1#*.}00
(( $((10#$load_int * 100 + 10#${load_frac:0:2})) < 50 )) || { echo "load gate: $load1"; exit 1; }
psi=$(cat /proc/pressure/cpu)
case "$psi" in *"some avg10=0.00"*) ;; *) echo "PSI gate: $psi"; exit 1;; esac
printf 'measurement_host=%s boot_id=%s uptime_s=%s load1=%s psi=%s\n' \
  "$(hostname)" "$(cat /proc/sys/kernel/random/boot_id)" "$uptime_s" "$load1" "$psi" \
  | tee "$OUT/gates.txt"
"$PY" -m pip show mlx-omarchy | sed -n '1,2p' > "$OUT/package.txt"

run_cell() { # label depth prefill passes wave
  local label=$1 depth=$2 prefill=$3 passes=$4 wave=$5
  MLX_OMARCHY_WAVE_SCHED=$wave MLX_OMARCHY_NORM_APPLE=1 MLX_OMARCHY_GDN_BATCH=1 \
    MLX_OMARCHY_GDN_F16_STATE=0 \
    "$PY" "$TOOLS/bsched_run.py" "$OUT/$label.json" "$BENCH" \
      --model "$MODEL" --prompts "$PROMPTS" --limit 1 \
      --new-tokens "$depth" --prefill-tokens "$prefill" --warmup 1 \
      --passes "$passes" --label "$label" --out "$OUT/$label.json" \
      > "$OUT/$label.log" 2>&1
  "$PY" - "$OUT/$label.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
td = d.get("trace_delta") or {}
proto = d.get("protocol") or {}
print(f"[cell] label={d.get('label')} wave={d.get('wave')} "
      f"depth={proto.get('new_tokens')} prefill={proto.get('prefill_tokens')} "
      f"digest={d.get('ordered_records_sha256')} "
      f"tps={d.get('tokens_per_s')} dispatches={td.get('vk_compute_dispatches')} "
      f"barriers={td.get('barriers_emitted')} skipped={td.get('barriers_skipped')}",
      flush=True)
PYEOF
}

IFS=';' read -ra SPECS <<< "$CELLS"
for spec in "${SPECS[@]}"; do
  IFS=':' read -r label depth prefill passes wave <<< "$spec"
  run_cell "$label" "$depth" "$prefill" "$passes" "$wave"
done
echo "== census cells done =="
