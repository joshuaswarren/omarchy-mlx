#!/bin/bash
# DecodeFuse4 COMBINED gate window (landing decision): F3+F4 flags both on
# vs serving, paired n=5 interleaved + ctl2 + kill-switch arm.
# INVOCATION: bash /var/tmp/appbar/gpuwin.sh 'bash /var/tmp/fuse4/combined.sh'
set -u
OUT=/var/tmp/fuse4/combined; mkdir -p "$OUT"
SERVE=/var/tmp/v072-venv-fused/bin/python3
CAND=/var/tmp/fuse4-venv/bin/python3
BENCH=$HOME/bench-scripts/qwen38-mlx-bench.py
MODEL=$(echo ~/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98b*)
{ date -u +%FT%TZ; cat /proc/sys/kernel/random/boot_id
  for py in "$SERVE" "$CAND"; do "$py" -c 'import mlx.core as mx; print(mx.__version__)'; done
} > "$OUT/identity.txt"

cell () { local N=$1 tag=$2 py=$3; shift 3
  env X=1 "$@" "$py" "$BENCH" --model "$MODEL"/ \
    --prompts "$HOME/bench-scripts/qwen38-2b-prompts.jsonl" \
    --limit 1 --new-tokens "$N" --warmup 1 --passes 5 --prefill-tokens 512 \
    --label "fuse4c-$N-$tag" --out "$OUT/d$N-$tag.json" > "$OUT/d$N-$tag.log" 2>&1
  "$SERVE" -c "
import json
try:
  d=json.load(open('$OUT/d$N-$tag.json'))
  print('d$N-$tag', round(d['decode_tok_rate']['median'],2), d['ordered_records_sha256'][:12])
except Exception as e:
  print('d$N-$tag', 'PARSE-FAIL', repr(e))"; }
for N in 64 128 256 512; do
  cell $N ctl "$SERVE"
  cell $N cand "$CAND" MLX_OMARCHY_ROPE_NORM_FUSE=1 MLX_OMARCHY_GDN_QKNORM_FUSE=1
done
cell 64 ctl2 "$SERVE"
cell 64 ksoff "$CAND"   # kill switch: no flags -> exact eager chain

# F6 adoption receipt: the planner trace shows whether the bf16 KV
# pairs classified (kinds 1,2 / 2,1 = keys_rope+values) or refused.
env X=1 MLX_OMARCHY_KV_TRACE=1 MLX_OMARCHY_ROPE_NORM_FUSE=1 \
  MLX_OMARCHY_GDN_QKNORM_FUSE=1 "$CAND" "$BENCH" --model "$MODEL"/ \
  --prompts "$HOME/bench-scripts/qwen38-2b-prompts.jsonl" \
  --limit 1 --new-tokens 8 --warmup 1 --passes 1 --prefill-tokens 512 \
  --label "fuse4c-kvtrace" --out "$OUT/kvtrace.json" \
  > "$OUT/kvtrace.log" 2>&1
grep -m 6 "kv-plan" "$OUT/kvtrace.log" > "$OUT/kvtrace-plan.txt" || true
echo "kv-trace: $(cat "$OUT/kvtrace-plan.txt" | head -2 | tr '\n' ' ')"
( cd "$OUT" && sha256sum ./* > SHA256SUMS 2>/dev/null )

python3 - "$OUT" > "$OUT/combined-final.txt" 2>&1 <<'PYEOF'
import json, sys
out = sys.argv[1]
PINS = {"64": "c84b3e7a", "128": "07c515e0", "256": "c6aabbf0", "512": "5c120987"}
ok = True
med = {}
for N, pin in PINS.items():
    for tag in ("ctl", "cand", "ctl2", "ksoff"):
        try:
            d = json.load(open(f"{out}/d{N}-{tag}.json"))
        except FileNotFoundError:
            continue
        dig = d["ordered_records_sha256"]
        r = round(d["decode_tok_rate"]["median"], 2)
        med[(N, tag)] = r
        if dig[:8] != pin:
            print(f"DIGEST-FAIL d{N}-{tag} {dig[:12]} != {pin}"); ok = False
        else:
            print(f"d{N}-{tag} {r} {dig[:12]} PIN-OK")
if all((n, "cand") in med for n in ("64", "128", "256", "512")):
    lo = min(med[("64","ctl")], med.get(("64","ctl2"), 1e9))
    hi = max(med[("64","ctl")], med.get(("64","ctl2"), 0))
    g64 = (med[("64","cand")] / med[("64","ctl")] - 1) * 100
    print(f"COMBINED d64 gain {g64:+.2f}% (ctl {med[('64','ctl')]} ctl2 {med.get(('64','ctl2'))} cand {med[('64','cand')]} ctl-range [{lo}, {hi}])")
    for n in ("128", "256", "512"):
        g = (med[(n,"cand")] / med[(n,"ctl")] - 1) * 100
        print(f"COMBINED d{n} gain {g:+.2f}%")
        if g <= 0:
            print(f"NEGATIVE-AT-d{n}"); ok = False
    if g64 < 2.0:
        print("LAND-BAR-FAIL: combined <+2% at d64"); ok = False
    if med[("64","cand")] <= hi:
        print("LAND-BAR-FAIL: cand inside ctl min-max at d64"); ok = False
else:
    print("LAND-BAR-FAIL: missing cells"); ok = False
print("COMBINED-FINAL", "PASS" if ok else "FAIL")
PYEOF
cat "$OUT/combined-final.txt"
