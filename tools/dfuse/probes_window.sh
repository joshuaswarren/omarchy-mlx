#!/usr/bin/env bash
# Runs INSIDE a gpuwin window: token-316 near-tie + fp64 per-op accuracy.
set -euo pipefail
C=/var/tmp/dfuse-cand/bin/python
O=/var/tmp/dfuse/tf9
mkdir -p "$O"
CTL=/var/tmp/dfuse/ab-w9D-20261004T053232Z/9-d512-r1-ctl.json
test -e "$CTL" || { echo "missing ctl json"; exit 1; }
env -u MLX_OMARCHY_GDN_RAW_REPEAT "$C" /var/tmp/dfuse/tf316.py "$CTL" "$O/tf316.json" > "$O/tf316.log" 2>&1 || echo "tf316 FAILED"
cat "$O/tf316.json"
env -u MLX_OMARCHY_GDN_RAW_REPEAT "$C" /var/tmp/dfuse/gdu_fp64_probe.py --capture "$O/gdu-operands.npz" > "$O/capture.log" 2>&1 || { echo "capture FAILED"; tail -3 "$O/capture.log"; exit 1; }
env -u MLX_OMARCHY_GDN_RAW_REPEAT "$C" /var/tmp/dfuse/gdu_fp64_probe.py --compare "$O/gdu-operands.npz" "$O/fp64-compare.json" > "$O/fp64.log" 2>&1 || { echo "fp64 FAILED"; tail -3 "$O/fp64.log"; exit 1; }
cat "$O/fp64-compare.json"
svc=$(systemctl is-active llm-inference || true)
printf "post-window llm-inference: %s\n" "$svc"
