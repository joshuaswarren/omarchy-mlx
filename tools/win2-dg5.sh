#!/bin/bash
# DecodeGap5 win2: conv+silu gates + fused-stream census. INVOCATION:
#   bash /var/tmp/appbar/gpuwin.sh 'bash /var/tmp/dg5/w2.sh'
# Race battery is NOT run here (it launches its own gpuwin windows; nested
# gpuwin deadlocks on /tmp/gpuwin.mutex). It runs after this window returns.
set -u
OUT=/var/tmp/dg5/w2; mkdir -p "$OUT"
SERVE=/var/tmp/v072-venv-fused/bin/python3
CAND=/var/tmp/dg5-venv/bin/python3
DIAG=/var/tmp/dg5-diag-venv/bin/python3
BENCH=$HOME/bench-scripts/qwen38-mlx-bench.py
MODEL=$(echo ~/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98b*)
{ date -u +%FT%TZ; cat /proc/sys/kernel/random/boot_id
  for py in "$SERVE" "$CAND" "$DIAG"; do "$py" -c 'import mlx.core as mx; print(mx.__version__)'; done
} > "$OUT/identity.txt"

bitrun () { # tag py script outjson
  local tag=$1 py=$2 script=$3 outjson=$4
  env X=1 "$py" "$script" "$outjson" > "${outjson%.json}.log" 2>&1
  echo "bit $tag rc=$?"
}
bitrun conv-serve "$SERVE" /var/tmp/dg5/dg_bitcheck_conv.py "$OUT/bitconv-serve.json"
bitrun conv-cand  "$CAND"  /var/tmp/dg5/dg_bitcheck_conv.py "$OUT/bitconv-cand.json"
bitrun full-serve "$SERVE" /var/tmp/dg/dg_bitcheck.py "$OUT/bit-serve.json"
bitrun full-cand  "$CAND"  /var/tmp/dg/dg_bitcheck.py "$OUT/bit-cand.json"

# Gate 1: conv bitcheck. refnn/ref/state rows identical across venvs;
# fused == ref within candidate; fused ABSENT in serve.
python3 - "$OUT" > "$OUT/gate-conv.txt" 2>&1 <<'PYEOF'
import json, sys
out = sys.argv[1]
serve = json.load(open(f"{out}/bitconv-serve.json"))
cand = json.load(open(f"{out}/bitconv-cand.json"))
fails = []
for k, v in serve.items():
    if k.startswith(("convsilu.refnn.", "convsilu.ref.", "convsilu.state.")):
        if cand.get(k) != v:
            fails.append(f"cross-venv mismatch {k}: {v} vs {cand.get(k)}")
for c in ("rand", "special", "bigstate", "specialw"):
    if serve.get(f"convsilu.fused.{c}") != "ABSENT":
        fails.append(f"serve fused not ABSENT: {c}={serve.get(f'convsilu.fused.{c}')}")
    if cand.get(f"convsilu.fused.{c}") != cand.get(f"convsilu.ref.{c}"):
        fails.append(f"cand fused != ref: {c}")
print("GATE-CONV", "PASS" if not fails else "FAIL")
for f in fails:
    print(" ", f)
PYEOF
echo "gate-conv: $(cat "$OUT/gate-conv.txt" | head -1)"

# Gate 2: full bitcheck. Every row identical; only meta version may differ.
python3 - "$OUT" > "$OUT/gate-full.txt" 2>&1 <<'PYEOF'
import json, sys
out = sys.argv[1]
serve = json.load(open(f"{out}/bit-serve.json"))
cand = json.load(open(f"{out}/bit-cand.json"))
meta = serve.pop("meta", None); cand.pop("meta", None)
diff = {k for k in set(serve) | set(cand) if serve.get(k) != cand.get(k)}
print("GATE-FULL", "PASS" if not diff else "FAIL", f"rows={len(serve)}")
for k in sorted(diff)[:10]:
    print(" ", k, serve.get(k), "vs", cand.get(k))
if meta or cand.pop("meta", None):
    print("  (meta present, excluded)")
PYEOF
echo "gate-full: $(head -1 "$OUT/gate-full.txt")"

cell () { # N tag py
  local N=$1 tag=$2 py=$3
  env X=1 "$py" "$BENCH" --model "$MODEL"/ --prompts "$HOME/bench-scripts/qwen38-2b-prompts.jsonl" \
    --limit 1 --new-tokens "$N" --warmup 1 --passes 5 --prefill-tokens 512 \
    --label "dg5-$N-$tag" --out "$OUT/d$N-$tag.json" > "$OUT/d$N-$tag.log" 2>&1
  "$SERVE" -c "
import json
try:
  d=json.load(open('$OUT/d$N-$tag.json'))
  print('d$N-$tag', round(d['decode_tok_rate']['median'],2), d['ordered_records_sha256'])
except Exception as e:
  print('d$N-$tag', 'PARSE-FAIL', repr(e))"
}
for N in 64 128 256 512; do
  cell $N ctl "$SERVE"
  cell $N cand "$CAND"
done
cell 64 ctl2 "$SERVE"
cell 256 ctl2 "$SERVE"

# Census on the fused stream (diag wheel: profiling harness compiled in).
env X=1 MLX_OMARCHY_GPU_PROFILE="$OUT/census-fused.jsonl" \
  MLX_OMARCHY_GPU_PROFILE_LABEL=dg5-fused \
  "$DIAG" /var/tmp/dg/profile_decode_driver.py --model "$MODEL"/ \
  --warmup-tokens 16 --tokens 128 --out "$OUT/census-driver.jsonl" \
  > "$OUT/census-fused.log" 2>&1
echo "census rc=$?"
python3 /var/tmp/dg/profile_analyze.py "$OUT/census-fused.jsonl" > "$OUT/census-analyze.txt" 2>&1
echo "analyze rc=$?"
python3 /var/tmp/dg/analyze_decode_profile.py "$OUT/census-fused.jsonl" > "$OUT/census-edges.txt" 2>&1
echo "edges rc=$?"

( cd "$OUT" && sha256sum ./*.json ./*.txt ./*.jsonl > SHA256SUMS )
echo WIN2-DONE
