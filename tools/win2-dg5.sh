#!/bin/bash
# DecodeGap5 win2: conv+silu gates + fused-stream census.
# (a) dg_bitcheck_conv serve/cand  (b) dg_bitcheck.py full serve/cand
# (c) cells d64/128/256/512 n=5 interleaved serve vs cand, digests pinned
# (d) race battery gated_battery.sh 8 pairs on candidate venv
# (e) census: diag wheel venv, MLX_OMARCHY_GPU_PROFILE 128-token decode + profile_analyze
set -u
OUT=/var/tmp/dg5/w2; mkdir -p "$OUT"
SERVE=/var/tmp/v072-venv-fused/bin/python3
CAND=/var/tmp/dg5-venv/bin/python3
DIAG=/var/tmp/dg4b-venv/bin/python3
BENCH=$HOME/bench-scripts/qwen38-mlx-bench.py
MODEL=$(echo ~/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98b*)
{ date -u +%FT%TZ; cat /proc/sys/kernel/random/boot_id; } > "$OUT/identity.txt"

bitrun () { # tag py script out
  local tag=$1 py=$2 script=$3 outjson=$4
  env X=1 "$py" "$script" "$outjson" > "${outjson%.json}.log" 2>&1
  echo "bit $tag rc=$?"
}
bitrun conv-serve "$SERVE" /var/tmp/dg5/dg_bitcheck_conv.py "$OUT/bitconv-serve.json"
bitrun conv-cand "$CAND" /var/tmp/dg5/dg_bitcheck_conv.py "$OUT/bitconv-cand.json"
bitrun full-serve "$SERVE" /var/tmp/dg/dg_bitcheck.py "$OUT/bit-serve.json"
bitrun full-cand "$CAND" /var/tmp/dg/dg_bitcheck.py "$OUT/bit-cand.json"
# gate echoes: differences must be empty except the recorded version string
for pair in "bitconv-serve bitconv-cand" "bit-serve bit-cand"; do
  set -- $pair
  diff <(python3 -c "import json,sys;d=json.load(open('$OUT/$1.json'));d.pop('meta',None);[print(k,v) for k,v in sorted(d.items())]" 2>/dev/null) \
       <(python3 -c "import json,sys;d=json.load(open('$OUT/$2.json'));d.pop('meta',None);[print(k,v) for k,v in sorted(d.items())]" 2>/dev/null) \
       > "$OUT/diff-$1-$2.txt" 2>&1
  echo "diff $1 vs $2 lines=$(wc -l < "$OUT/diff-$1-$2.txt")"
done

cell () { # N tag py
  local N=$1 tag=$2 py=$3
  env X=1 "$py" "$BENCH" --model "$MODEL"/ --prompts "$HOME/bench-scripts/qwen38-2b-prompts.jsonl" \
    --limit 1 --new-tokens "$N" --warmup 1 --passes 5 --prefill-tokens 512 \
    --label "dg5-$N-$tag" --out "$OUT/d$N-$tag.json" > "$OUT/d$N-$tag.log" 2>&1
  "$SERVE" -c "
import json
try:
  d=json.load(open('$OUT/d$N-$tag.json'))
  print('d$N-$tag', round(d['decode_tok_rate']['median'],2), d['ordered_records_sha256'][:12])
except Exception as e:
  print('d$N-$tag', 'PARSE-FAIL', repr(e))"
}
for N in 64 128 256 512; do
  cell $N ctl "$SERVE"
  cell $N cand "$CAND"
done
cell 64 ctl2 "$SERVE"
cell 256 ctl2 "$SERVE"

bash /var/tmp/appbar/gated_battery.sh 8 "X=1" "$CAND" > "$OUT/battery.log" 2>&1
echo "battery rc=$?"

# census on the fused stream (diag wheel; profiled shares/counts are the signal)
env X=1 MLX_OMARCHY_GPU_PROFILE=1 "$DIAG" /var/tmp/dg/profile_decode_driver.py \
  --new-tokens 128 --warmup 16 --out "$OUT/census-fused.json" > "$OUT/census-fused.log" 2>&1
echo "census rc=$?"
python3 /var/tmp/dg/profile_analyze.py "$OUT/census-fused.json" > "$OUT/census-analyze.txt" 2>&1
echo "analyze rc=$?"

( cd "$OUT" && sha256sum ./*.json ./*.txt > SHA256SUMS )
echo WIN2-DONE
