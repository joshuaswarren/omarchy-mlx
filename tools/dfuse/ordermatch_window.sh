#!/usr/bin/env bash
# Runs INSIDE a gpuwin window: order-match verification.
# 1) captured-operand composed-vs-fused bit identity; 2) free-run identity (5 prompts, d512);
# 3) perf cost: 3 paired d64 cells + 2 paired d512 cells (ctl rollback = composed, on = diag fused).
set -euo pipefail
C=/var/tmp/dfuse-venv/bin/python
OLD=/var/tmp/v072-venv-fused.pre-20261004T000642/bin/python
B=${HOME}/bench-scripts/qwen38-mlx-bench.py
P=${HOME}/bench-scripts/qwen38-2b-prompts.jsonl
M9B=$(ls -d ${HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/*/ | head -1)
O=/var/tmp/dfuse/ordermatch
mkdir -p "$O"

test ! -e /var/tmp/JW16_MAINTENANCE
uptime_s=$(cut -d. -f1 /proc/uptime); test "$uptime_s" -ge 360
load1=$(cut -d" " -f1 /proc/loadavg)
li=${load1%%.*}; lf=${load1#*.}00
(( $((10#$li * 100 + 10#${lf:0:2})) < 50 )) || { echo "load gate: $load1"; exit 1; }
case $(cat /proc/pressure/cpu) in *"some avg10=0.00"*) ;; *) echo "PSI gate"; exit 1;; esac
printf "host=%s boot=%s\n" "$(hostname)" "$(cat /proc/sys/kernel/random/boot_id)" > "$O/gates.txt"

echo "== 1) captured-operand bit identity"
env -u MLX_OMARCHY_GDN_RAW_REPEAT "$C" /var/tmp/dfuse/gdu_fp64_probe.py --compare \
  /var/tmp/dfuse/tf9/gdu-operands.npz "$O/fp64-ordermatch.json" > "$O/fp64.log" 2>&1 || echo "fp64 probe FAILED"
python3 - <<'PYEOF'
import json
r = json.load(open("/var/tmp/dfuse/ordermatch/fp64-ordermatch.json"))
c, f = r["composed_vs_fp64"]["out"]["max_abs"], r["fused_vs_fp64"]["out"]["max_abs"]
cs, fs = r["composed_vs_fp64"]["state"]["max_abs"], r["fused_vs_fp64"]["state"]["max_abs"]
print(f"out: composed {c} fused {f} | state: composed {cs} fused {fs}")
print("BIT-IDENTICAL:", c == f and cs == fs)
PYEOF

echo "== 2) free-running greedy identity (5 prompts x 512)"
env MLX_OMARCHY_GDN_RAW_REPEAT=1 "$C" /var/tmp/dfuse/free_run_identity.py "$M9B" "$O/identity.json" 5 \
  > "$O/identity.log" 2>&1 || echo "identity FAILED"
tail -2 "$O/identity.log"

echo "== 3) perf cost pairs"
pair() { # tag depth
  env -u MLX_OMARCHY_GDN_RAW_REPEAT "$OLD" "$B" --model "$M9B" --prompts "$P" --limit 1 --warmup 1 \
    --passes 1 --new-tokens "$2" --prefill-tokens 512 --label "om-$1-ctl" --out "$O/om-$1-ctl.json" \
    > "$O/om-$1-ctl.log" 2>&1
  env MLX_OMARCHY_GDN_RAW_REPEAT=1 "$C" "$B" --model "$M9B" --prompts "$P" --limit 1 --warmup 1 \
    --passes 1 --new-tokens "$2" --prefill-tokens 512 --label "om-$1-on" --out "$O/om-$1-on.json" \
    > "$O/om-$1-on.log" 2>&1
  python3 - "$O/om-$1-ctl.json" "$O/om-$1-on.json" "$1" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1]))["decode_tok_rate"]["median"]
o = json.load(open(sys.argv[2]))["decode_tok_rate"]["median"]
print(f"pair {sys.argv[3]}: ctl={c:.2f} on={o:.2f} delta={100*(o-c)/c:+.2f}%")
PYEOF
}
pair d64-a 64
pair d64-b 64
pair d64-c 64
pair d512-a 512
pair d512-b 512

svc=$(systemctl is-active llm-inference || true)
printf "post-window llm-inference: %s\n" "$svc"
