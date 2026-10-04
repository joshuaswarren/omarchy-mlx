#!/usr/bin/env bash
# RingTurnaround W1 inner: whole-word HK_CDM_BARRIER_MASK probes on the SERVING ICD
# (1432df0196-transfer, dependency-tracked barrier lineage). No builds, no ICD changes.
# RT_PLAN spec: "label:depth:word[:passes]" — word = hex barrier word, or "ctl" (unset).
# Runs INSIDE a gpuwin window. Digests must equal the DrainFix pins (bit-exact numerics).
set -euo pipefail
OUT=${OUT:?missing OUT}
RT_PLAN=${RT_PLAN:?missing RT_PLAN}
PY=${RT_PY:-/var/tmp/v072-venv-fused/bin/python}
BENCH=${HOME}/bench-scripts/qwen38-mlx-bench.py
PROMPTS=${HOME}/bench-scripts/qwen38-2b-prompts.jsonl
MODEL=$(echo ${HOME}/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98b*)
TOOLS=/var/tmp/ringturn

test ! -e /var/tmp/JW16_MAINTENANCE
mkdir -p "$OUT"
uptime_s=$(cut -d. -f1 /proc/uptime); test "$uptime_s" -ge 360
# Bounded settle: llm-inference just stopped; 1-min load decays. Wait up to 240 s.
settle=0
while :; do
  load1=$(cut -d' ' -f1 /proc/loadavg)
  load_int=${load1%%.*}; load_frac=${load1#*.}00
  psi=$(cat /proc/pressure/cpu)
  if (( $((10#$load_int * 100 + 10#${load_frac:0:2})) < 50 )) && case "$psi" in *"some avg10=0.00"*) true;; *) false;; esac; then
    break
  fi
  settle=$((settle+15)); [ "$settle" -ge 240 ] && { echo "settle timeout: load1=$load1 psi=$psi"; exit 1; }
  echo "settle $settle s: load1=$load1 psi=$psi" >> "$OUT/settle.log"
  sleep 15
done
printf 'settle_wait_s=%s\n' "$settle" > "$OUT/settle.txt"
load1=$(cut -d' ' -f1 /proc/loadavg)
psi=$(cat /proc/pressure/cpu)
load_int=${load1%%.*}; load_frac=${load1#*.}00
(( $((10#$load_int * 100 + 10#${load_frac:0:2})) < 50 )) || { echo "load gate: $load1"; exit 1; }
case "$psi" in *"some avg10=0.00"*) ;; *) echo "PSI gate: $psi"; exit 1;; esac
printf 'measurement_host=%s boot_id=%s uptime_s=%s load1=%s psi=%s kernel=%s\n' \
  "$(hostname)" "$(cat /proc/sys/kernel/random/boot_id)" "$uptime_s" "$load1" "$psi" "$(uname -r)" | tee "$OUT/gates.txt"
printf 'RT_PLAN=%s\n' "$RT_PLAN" >> "$OUT/gates.txt"

"$PY" "$TOOLS/mlx_provenance.py" > "$OUT/provenance.json" 2>&1 || echo "provenance-run-failed"
head -c 400 "$OUT/provenance.json"; echo
"$PY" -m pip show mlx-omarchy | sed -n '1,2p' > "$OUT/package.txt" 2>/dev/null || true

run_one() { # label depth word passes
  local label=$1 depth=$2 word=$3 passes=${4:-1} extra=()
  if [ "$word" != ctl ]; then extra=(HK_CDM_BARRIER_MASK="$word"); fi
  local g1 p waited=0
  while :; do
    g1=$(cut -d' ' -f1 /proc/loadavg)
    case "$g1" in 0.*) p=$(cat /proc/pressure/cpu); case "$p" in *"some avg10=0.00"*) break;; esac;; esac
    waited=$((waited+10)); [ "$waited" -ge 120 ] && { echo "mid-plan gate timeout: load1=$g1"; exit 1; }
    sleep 10
  done
  [ "$waited" -gt 0 ] && echo "cell $label waited ${waited}s for gates" >> "$OUT/order.log"
  printf 'cell %s start %s word=%s load1=%s\n' "$label" "$(date -u +%FT%TZ)" "$word" "$g1" | tee -a "$OUT/order.log"
  env "${extra[@]}" MLX_OMARCHY_NORM_APPLE=1 MLX_OMARCHY_GDN_BATCH=1 MLX_OMARCHY_GDN_F16_STATE=0 \
    "$PY" "$BENCH" --model "$MODEL"/ --prompts "$PROMPTS" --limit 1 --new-tokens "$depth" \
    --prefill-tokens 0 --warmup 1 --passes "$passes" --label "$label" --out "$OUT/$label.json" \
    > "$OUT/$label.log" 2>&1
  "$PY" - "$OUT/$label.json" "$word" <<'PYEOF'
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("rt_pins", "/var/tmp/ringturn/pins.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
d = json.load(open(sys.argv[1]))
proto = d.get("protocol") or {}
key = (proto.get("new_tokens"), proto.get("passes", 1))
dig = d.get("ordered_records_sha256")
pin = mod.PINS.get(key)
rates = [r["decode_tok_rate"] for r in d.get("per_prompt", []) if r.get("decode_tok_rate")]
med = sorted(rates)[len(rates)//2] if rates else 0
status = "OK" if pin == dig else ("UNPINNED" if pin is None else "DIGEST-MISMATCH")
print(f"cell {d.get('label')} depth={key} word={sys.argv[2]} med={med:.2f} min={min(rates):.2f} max={max(rates):.2f} digest={status} {dig[:12] if dig else ''}")
if pin is not None and pin != dig:
    raise SystemExit(f"DIGEST-MISMATCH depth={key} word={sys.argv[2]} got={dig}")
PYEOF
  echo "ran $label depth=$depth word=$word passes=$passes" | tee -a "$OUT/order.log"
}

IFS=';' read -ra SPECS <<< "$RT_PLAN"
for spec in "${SPECS[@]}"; do
  IFS=':' read -r label depth word passes <<< "$spec"
  run_one "$label" "$depth" "$word" "${passes:-1}"
done
echo "== plan done =="
