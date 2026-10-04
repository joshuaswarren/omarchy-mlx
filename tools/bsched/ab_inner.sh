#!/usr/bin/env bash
# BarrierSched A/B inner: alternating ctl (MLX_OMARCHY_WAVE_SCHED=0) vs cand
# (=1) decode/prefill cells on ONE candidate wheel (same venv both arms; the
# gate is the only difference). Digest equality checked every round.
# Runs INSIDE a gpuwin window. Env: BS_OUT, BS_PY, BS_MODEL, BS_PROMPTS,
# BS_BENCH, BS_ROUNDS (default 5), BS_DEPTHS ("64 128 256 512"), BS_PREFILLS
# ("512" e.g.), BS_TAG (model tag for output naming).
set -euo pipefail
OUT=${BS_OUT:?missing BS_OUT}
PY=${BS_PY:?missing BS_PY}
MODEL=${BS_MODEL:?missing BS_MODEL}
PROMPTS=${BS_PROMPTS:?missing BS_PROMPTS}
BENCH=${BS_BENCH:?missing BS_BENCH}
TAG=${BS_TAG:-m}
ROUNDS=${BS_ROUNDS:-5}
DEPTHS=${BS_DEPTHS:-"64 512"}
PREFILLS=${BS_PREFILLS:-"512"}
TOOLS=$(cd "$(dirname "$0")" && pwd)

test ! -e /var/tmp/JW16_MAINTENANCE
uptime_s=$(cut -d. -f1 /proc/uptime); test "$uptime_s" -ge 360
load1=$(cut -d' ' -f1 /proc/loadavg)
load_int=${load1%%.*}; load_frac=${load1#*.}00
(( $((10#$load_int * 100 + 10#${load_frac:0:2})) < 50 )) || { echo "load gate: $load1"; exit 1; }
psi=$(cat /proc/pressure/cpu)
case "$psi" in *"some avg10=0.00"*) ;; *) echo "PSI gate: $psi"; exit 1;; esac
printf 'measurement_host=%s boot_id=%s uptime_s=%s load1=%s psi=%s tag=%s\n' \
  "$(hostname)" "$(cat /proc/sys/kernel/random/boot_id)" "$uptime_s" "$load1" "$psi" "$TAG" \
  | tee "$OUT/gates.txt"
"$PY" -m pip show mlx-omarchy | sed -n '1,2p' > "$OUT/package.txt"
# Prove the wave gate actually reaches the encoder in both arms.
for wv in 0 1; do
  MLX_OMARCHY_WAVE_SCHED=$wv "$PY" - <<'PYEOF'
import ctypes, os
class S(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in (
        "a","b","c","d","e","f","g","h","barriers_emitted","barriers_skipped")]
import mlx.core  # noqa
import mlx, os
so = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(mlx.__file__))), "mlx", "lib", "libmlx.so")
lib = ctypes.CDLL(so); lib.mlx_omarchy_trace_snapshot.argtypes = [ctypes.POINTER(S)]
s0, s1 = S(), S(); lib.mlx_omarchy_trace_snapshot(ctypes.byref(s0))
import mlx.core as mx
x = mx.ones(64); (x * 2).item() if False else mx.eval(x * 2)
lib.mlx_omarchy_trace_snapshot(ctypes.byref(s1))
print(f"gate-smoke wave={os.environ.get('MLX_OMARCHY_WAVE_SCHED')} dispatches={s1.a - s0.a} emitted={s1.barriers_emitted - s0.barriers_emitted} skipped={s1.barriers_skipped - s0.barriers_skipped}")
PYEOF
done | tee "$OUT/gate_smoke.txt"

run_arm() { # arm label depth prefill
  local arm=$1 label=$2 depth=$3 prefill=$4 wave
  case "$arm" in ctl) wave=0 ;; cand) wave=1 ;; *) echo "bad arm $arm"; exit 2 ;; esac
  MLX_OMARCHY_WAVE_SCHED=$wave MLX_OMARCHY_NORM_APPLE=1 MLX_OMARCHY_GDN_BATCH=1 \
    MLX_OMARCHY_GDN_F16_STATE=0 \
    "$PY" "$BENCH" --model "$MODEL" --prompts "$PROMPTS" --limit 1 \
      --new-tokens "$depth" --prefill-tokens "$prefill" --warmup 1 \
      --passes 1 --label "$label" --out "$OUT/$label.json" > "$OUT/$label.log" 2>&1
}

digest_of() {
  "$PY" - "$1" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get("ordered_records_sha256") or "NONE")
PYEOF
}

rate_of() { # file key(decode|prefill)
  "$PY" - "$1" "$2" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
key = sys.argv[2]
v = d.get(key)
if isinstance(v, dict):
    print(v.get("median", 0))
elif isinstance(v, list) and v:
    vals = sorted(x.get("pure_prefill_tok_rate", 0) for x in v)
    print(vals[len(vals) // 2])
else:
    print(v or 0)
PYEOF
}

: > "$OUT/pairs.jsonl"
round=1
while [ "$round" -le "$ROUNDS" ]; do
  for depth in $DEPTHS; do
    for prefill in $PREFILLS; do
      if (( round % 2 == 1 )); then order="ctl cand"; else order="cand ctl"; fi
      for arm in $order; do
        label="${TAG}_d${depth}_p${prefill}_${arm}_r${round}"
        run_arm "$arm" "$label" "$depth" "$prefill"
        echo "$label $(digest_of "$OUT/$label.json") $(tps_of "$OUT/$label.json")" >> "$OUT/pairs.jsonl"
      done
      ctl_label="${TAG}_d${depth}_p${prefill}_ctl_r${round}"
      cand_label="${TAG}_d${depth}_p${prefill}_cand_r${round}"
      cd=$(digest_of "$OUT/$ctl_label.json"); kd=$(digest_of "$OUT/$cand_label.json")
      if [ "$cd" != "$kd" ]; then echo "DIGEST-MISMATCH $ctl_label vs $cand_label ($cd vs $kd)"; exit 3; fi
      "$PY" - "$OUT/$ctl_label.json" "$OUT/$cand_label.json" "$depth" "$prefill" "$round" <<'PYEOF' >> "$OUT/pairs.jsonl"
import json, sys
c = json.load(open(sys.argv[1])); k = json.load(open(sys.argv[2]))
def med(v):
    if isinstance(v, dict):
        return v.get("median", 0)
    if isinstance(v, list) and v:
        vals = sorted(x.get("pure_prefill_tok_rate", 0) for x in v)
        return vals[len(vals) // 2]
    return v or 0
depth = int(sys.argv[3])
if depth <= 1:
    ct = med(c.get("pure_prefill")); kt = med(k.get("pure_prefill"))
else:
    ct = med(c.get("decode_tok_rate")); kt = med(k.get("decode_tok_rate"))
delta = (kt - ct) / ct * 100.0 if ct else 0.0
print(json.dumps({"depth": depth, "prefill": int(sys.argv[4]),
                  "round": int(sys.argv[5]), "ctl_tps": ct, "cand_tps": kt,
                  "ctl_digest": c.get("ordered_records_sha256"),
                  "cand_digest": k.get("ordered_records_sha256"),
                  "delta_pct": round(delta, 3),
                  "digest": c.get("ordered_records_sha256")}))
PYEOF
      echo "pair d$depth p$prefill r$round OK (digest $cd)"
    done
  done
  round=$((round + 1))
done
echo "== A/B rounds done =="
tail -n $((ROUNDS * ${#DEPTHS})) "$OUT/pairs.jsonl" | grep '"depth"' || true
