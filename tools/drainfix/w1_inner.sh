#!/usr/bin/env bash
# DrainFix W1 inner: decode timeline census on the serving stack.
# Runs INSIDE a gpuwin window. Gates: uptime>=360s, load1<0.5, PSI cpu some avg10=0.00.
set -euo pipefail
OUT=${OUT:?missing OUT}
PY=/var/tmp/v072-venv-fused/bin/python
BENCH=${HOME}/bench-scripts/qwen38-mlx-bench.py
PROMPTS=${HOME}/bench-scripts/qwen38-2b-prompts.jsonl
MODEL=${HOME}/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e7c97b86b34b381
BENCHENV=(MLX_OMARCHY_NORM_APPLE=1 MLX_OMARCHY_GDN_BATCH=1 MLX_OMARCHY_GDN_F16_STATE=0)

test ! -e /var/tmp/JW16_MAINTENANCE
uptime_s=$(cut -d. -f1 /proc/uptime); test "$uptime_s" -ge 360
load1=$(cut -d' ' -f1 /proc/loadavg)
load_int=${load1%%.*}; load_frac=${load1#*.}00
(( $((10#$load_int * 100 + 10#${load_frac:0:2})) < 50 )) || { echo "load gate: $load1"; exit 1; }
psi=$(cat /proc/pressure/cpu)
case "$psi" in *"some avg10=0.00"*) ;; *) echo "PSI gate: $psi"; exit 1;; esac
printf 'measurement_host=%s uptime_s=%s load1=%s psi=%s\n' "$(hostname)" "$uptime_s" "$load1" "$psi" | tee "$OUT/gates.txt"

bench() { env "${BENCHENV[@]}" "$PY" "$BENCH" --model "$MODEL" --prompts "$PROMPTS" \
  --limit 1 --new-tokens "$1" --prefill-tokens 0 --warmup 1 --passes "$2" \
  --label "$3" --out "$OUT/$3.json"; }

echo "== w1 ctl d64 unstraced =="; bench 64 1 w1-ctl-d64
echo "== w1 straced d64 =="
strace -f -y -tt -T -e trace=ioctl -o "$OUT/strace-d64.log" \
  env "${BENCHENV[@]}" "$PY" "$BENCH" --model "$MODEL" --prompts "$PROMPTS" \
  --limit 1 --new-tokens 64 --prefill-tokens 0 --warmup 1 --passes 1 \
  --label w1-strace-d64 --out "$OUT/w1-strace-d64.json"
echo "== w1 straced d512 =="
strace -f -y -tt -T -e trace=ioctl -o "$OUT/strace-d512.log" \
  env "${BENCHENV[@]}" "$PY" "$BENCH" --model "$MODEL" --prompts "$PROMPTS" \
  --limit 1 --new-tokens 512 --prefill-tokens 0 --warmup 1 --passes 1 \
  --label w1-strace-d512 --out "$OUT/w1-strace-d512.json"
echo "== w1 done =="
