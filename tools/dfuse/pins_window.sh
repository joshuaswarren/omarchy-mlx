#!/usr/bin/env bash
# Runs INSIDE one gpuwin window: 2B (bf16 A_log) bench digests on the v0.7.26
# reference venv vs a candidate venv, on every GDN decode kernel arm, plus 4B.
# usage: pins_window.sh <ref-python> <cand-python> <out-dir>
set -uo pipefail
REF="$1"; CAND="$2"; O="$3"
mkdir -p "$O"
B=$HOME/bench-scripts/qwen38-mlx-bench.py
P=$HOME/bench-scripts/qwen38-2b-prompts.jsonl
HF=$HOME/.cache/huggingface/hub
M2B=$HF/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e7c97b86b34b381
M4B=$(ls -d $HF/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/*/ | head -1)
printf "host=%s uptime_s=%s load1=%s psi=%s\n" "$(hostname)" "$(cut -d. -f1 /proc/uptime)" \
  "$(cut -d' ' -f1 /proc/loadavg)" "$(head -1 /proc/pressure/cpu)" | tee "$O/gates.txt"
for SIDE in ref cand; do
  PY=$REF; [ "$SIDE" = cand ] && PY=$CAND
  "$PY" /var/tmp/dfuse-build/scripts/mlx_provenance.py > "$O/$SIDE-provenance.txt" 2>&1 || echo "$SIDE provenance refused"
  echo "== $SIDE $("$PY" -m pip show mlx-omarchy | sed -n 2p)"
done
cell() { # side arm model tag depth env...
  local side=$1 arm=$2 model=$3 tag=$4 depth=$5; shift 5
  local py=$REF; [ "$side" = cand ] && py=$CAND
  local lbl="$tag-$arm-d$depth-$side"
  env "$@" timeout 300 "$py" "$B" --model "$model" --prompts "$P" --limit 1 --warmup 1 \
    --passes 1 --new-tokens "$depth" --prefill-tokens 512 --label "$lbl" \
    --out "$O/$lbl.json" > "$O/$lbl.log" 2> "$O/$lbl.err"
  echo "$lbl rc=$? $(grep -o '"ordered_records_sha256": *"[0-9a-f]\{16\}' "$O/$lbl.json" | tail -c 17) decode=$(python3 -c "import json;print(round(json.load(open('$O/$lbl.json'))['decode_tok_rate']['median'],2))" 2>/dev/null)"
}
for depth in 64 128; do
  for side in ref cand; do
    cell $side tiled "$M2B" 2b $depth MLX_OMARCHY_GDN_DECODE_TILE=1
    cell $side pf "$M2B" 2b $depth MLX_OMARCHY_GDN_DECODE_TILE=0
    cell $side perrow "$M2B" 2b $depth MLX_OMARCHY_GDN_DECODE_TILE=0 MLX_OMARCHY_GDN_PF=0
  done
done
for side in ref cand; do cell $side default "$M4B" 4b 64 X_UNUSED=1; done
