#!/usr/bin/env bash
# DispatchFuse re-gate: final rebased wheel, default-ON fold, digest pins.
set -euo pipefail
C=/var/tmp/dfuse-cand/bin/python
B=${HOME}/bench-scripts/qwen38-mlx-bench.py
P=${HOME}/bench-scripts/qwen38-2b-prompts.jsonl
HF=${HOME}/.cache/huggingface/hub
M4B=$(ls -d $HF/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/*/ | head -1)
M2B=$HF/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e7c97b86b34b381
O=/var/tmp/dfuse/regate
mkdir -p "$O"

test ! -e /var/tmp/JW16_MAINTENANCE
uptime_s=$(cut -d. -f1 /proc/uptime); test "$uptime_s" -ge 360
load1=$(cut -d" " -f1 /proc/loadavg)
li=${load1%%.*}; lf=${load1#*.}00
(( $((10#$li * 100 + 10#${lf:0:2})) < 50 )) || { echo "load gate: $load1"; exit 1; }
case $(cat /proc/pressure/cpu) in *"some avg10=0.00"*) ;; *) echo "PSI gate"; exit 1;; esac

run() { # py model depth out
  timeout 300 "$1" "$B" --model "$2" --prompts "$P" --limit 1 --warmup 1 \
    --passes 1 --new-tokens "$3" --prefill-tokens 512 --label "$4" --out "$O/$4.json" \
    > "$O/$4.log" 2> "$O/$4.err"
  grep -o '"ordered_records_sha256[^,}]*' "$O/$4.json"
}

sudo journalctl --flush; sync
bash /var/tmp/appbar/gpuwin.sh "timeout 700 env DF_REGATE=1 bash /var/tmp/dfuse/regate_inner2.sh"
svc=$(systemctl is-active llm-inference || true)
printf "post-window llm-inference: %s\n" "$svc"
