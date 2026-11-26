#!/usr/bin/env bash
# Runs INSIDE a gpuwin window: perrow order-match verification on the FORCED
# legacy path (MLX_OMARCHY_GDN_DECODE_TILE=0), jw16.
# 1) captured-operand composed-vs-fused bit identity through perrow
# 2) free-run identity 5 prompts x 512 through perrow
# 3) default-path spot pins (tiled untouched): 9B d64 + 2B d64
set -euo pipefail
C=/var/tmp/dfuse-venv/bin/python
OLD=/var/tmp/v072-venv-fused.pre-20261004T000642/bin/python
B=${HOME}/bench-scripts/qwen38-mlx-bench.py
P=${HOME}/bench-scripts/qwen38-2b-prompts.jsonl
M9B=$(ls -d ${HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/*/ | head -1)
M2B=${HOME}/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e7c97b86b34b381
O=/var/tmp/dfuse/perrow
mkdir -p "$O"

test ! -e /var/tmp/JW16_MAINTENANCE
uptime_s=$(cut -d. -f1 /proc/uptime); test "$uptime_s" -ge 360
load1=$(cut -d" " -f1 /proc/loadavg)
li=${load1%%.*}; lf=${load1#*.}00
(( $((10#$li * 100 + 10#${lf:0:2})) < 50 )) || { echo "load gate: $load1"; exit 1; }
case $(cat /proc/pressure/cpu) in *"some avg10=0.00"*) ;; *) echo "PSI gate"; exit 1;; esac
printf "host=%s boot=%s\n" "$(hostname)" "$(cat /proc/sys/kernel/random/boot_id)" > "$O/gates.txt"

echo "== 1) captured-operand bit identity via PERROW (forced legacy path)"
env -u MLX_OMARCHY_GDN_RAW_REPEAT MLX_OMARCHY_GDN_DECODE_TILE=0 "$C" /var/tmp/dfuse/gdu_fp64_probe.py --capture "$O/gdu-operands.npz" > "$O/capture.log" 2>&1 || echo "capture FAILED"
env MLX_OMARCHY_GDN_RAW_REPEAT=1 MLX_OMARCHY_GDN_DECODE_TILE=0 "$C" /var/tmp/dfuse/gdu_fp64_probe.py --compare "$O/gdu-operands.npz" "$O/fp64-perrow.json" > "$O/fp64.log" 2>&1 || echo "fp64 FAILED"
python3 - <<'PYEOF'
import json
r = json.load(open("/var/tmp/dfuse/perrow/fp64-perrow.json"))
c, f = r["composed_vs_fp64"]["out"]["max_abs"], r["fused_vs_fp64"]["out"]["max_abs"]
cs, fs = r["composed_vs_fp64"]["state"]["max_abs"], r["fused_vs_fp64"]["state"]["max_abs"]
print(f"out: composed {c} fused {f} | state: composed {cs} fused {fs}")
print("PERROW BIT-IDENTICAL:", c == f and cs == fs)
PYEOF

echo "== 2) free-run identity via PERROW (5 prompts x 512)"
env MLX_OMARCHY_GDN_RAW_REPEAT=1 MLX_OMARCHY_GDN_DECODE_TILE=0 "$C" /var/tmp/dfuse/free_run_identity.py "$M9B" "$O/identity-perrow.json" 5 > "$O/identity.log" 2>&1 || echo "identity FAILED"
tail -2 "$O/identity.log"

echo "== 3) default-path (tiled) spot pins"
env -u MLX_OMARCHY_GDN_RAW_REPEAT "$C" "$B" --model "$M9B" --prompts "$P" --limit 1 --warmup 1 --passes 1 --new-tokens 64 --prefill-tokens 512 --label t9b-d64 --out "$O/t9b-d64.json" > "$O/t1.log" 2>&1
echo "9B d64 tiled: $(grep -o '"ordered_records_sha256[^,}]*' "$O/t9b-d64.json")"
env -u MLX_OMARCHY_GDN_RAW_REPEAT "$C" "$B" --model "$M2B" --prompts "$P" --limit 1 --warmup 1 --passes 1 --new-tokens 64 --prefill-tokens 512 --label t2b-d64 --out "$O/t2b-d64.json" > "$O/t2.log" 2>&1
echo "2B d64 tiled: $(grep -o '"ordered_records_sha256[^,}]*' "$O/t2b-d64.json")"

svc=$(systemctl is-active llm-inference || true)
printf "post-window llm-inference: %s\n" "$svc"
