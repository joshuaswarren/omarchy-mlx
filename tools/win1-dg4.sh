#!/bin/bash
# DecodeGap4 win1: chip-switch gates. bitcheck arms (tile/prefetch 0/1) + d64/d256 cells n=5
# arms: serve ctl, cand-default (G13C: both ON), cand-legacy (both OFF).
set -u
OUT=/var/tmp/dg4/w1; mkdir -p "$OUT"
SERVE=/var/tmp/v072-venv-fused/bin/python3
CAND=/var/tmp/dg4-venv/bin/python3
BENCH=$HOME/bench-scripts/qwen38-mlx-bench.py
MODEL=$(echo ~/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98b*)
{ date -u +%FT%TZ; cat /proc/sys/kernel/random/boot_id; } > "$OUT/identity.txt"
bitrun () { # name env...
  local name=$1; shift
  env "$@" "$CAND" /var/tmp/dg4/dg_bitcheck.py "$OUT/bit-$name.json" > "$OUT/bit-$name.log" 2>&1
  echo "bit $name rc=$?"
}
bitrun def X=1
bitrun legacy MLX_OMARCHY_GDN_DECODE_TILE=0 MLX_OMARCHY_LSE_ARGREDUCE_PREFETCH=0
bitrun tile1 MLX_OMARCHY_GDN_DECODE_TILE=1 MLX_OMARCHY_LSE_ARGREDUCE_PREFETCH=0
bitrun pf0 MLX_OMARCHY_GDN_DECODE_TILE=0 MLX_OMARCHY_LSE_ARGREDUCE_PREFETCH=1
env X=1 "$SERVE" /var/tmp/dg4/dg_bitcheck.py "$OUT/bit-serve.json" > "$OUT/bit-serve.log" 2>&1
echo "bit serve rc=$?"
cell () { # N tag py env...
  local N=$1 tag=$2 py=$3; shift 3
  env "$@" "$py" "$BENCH" --model "$MODEL"/ --prompts "$HOME/bench-scripts/qwen38-2b-prompts.jsonl" \
    --limit 1 --new-tokens "$N" --warmup 1 --passes 5 --prefill-tokens 512 \
    --label "dg4-$N-$tag" --out "$OUT/d$N-$tag.json" > "$OUT/d$N-$tag.log" 2>&1
  "$SERVE" -c "
import json
try:
  d=json.load(open('$OUT/d$N-$tag.json'))
  print('d$N-$tag', round(d['decode_tok_rate']['median'],2), d['ordered_records_sha256'][:12])
except Exception as e:
  print('d$N-$tag', 'PARSE-FAIL', repr(e))"
}
for N in 64 256; do
  cell $N ctl "$SERVE" X=1
  cell $N cand-def "$CAND" X=1
  cell $N cand-leg "$CAND" MLX_OMARCHY_GDN_DECODE_TILE=0 MLX_OMARCHY_LSE_ARGREDUCE_PREFETCH=0
  cell $N ctl2 "$SERVE" X=1
done
( cd "$OUT" && sha256sum ./*.json > SHA256SUMS )
echo WIN1-DONE
