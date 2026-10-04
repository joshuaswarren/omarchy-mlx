#!/usr/bin/env bash
# Runs INSIDE a gpuwin window: H257 owner-bar measurements on jw16
# FORCED LEGACY (MLX_OMARCHY_GDN_DECODE_TILE=0, PF default = the jwm1 path).
# (1) fp64 state/output error, both paths, captured operands
# (2) free-run 10x512 with per-divergence composed top-2 gap
# (3) S=1 decode-route-sensitive PPL
set -euo pipefail
C=/var/tmp/dfuse-venv/bin/python
M9B=$(ls -d ${HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/*/ | head -1)
O=/var/tmp/dfuse/h257bar
mkdir -p "$O"

test ! -e /var/tmp/JW16_MAINTENANCE
uptime_s=$(cut -d. -f1 /proc/uptime); test "$uptime_s" -ge 360
case $(cat /proc/pressure/cpu) in *"some avg10=0.00"*) ;; *) echo "PSI gate"; exit 1;; esac
printf "host=%s boot=%s\n" "$(hostname)" "$(cat /proc/sys/kernel/random/boot_id)" > "$O/gates.txt"

echo "== (1) fp64 state/output error (forced legacy, both paths)"
env -u MLX_OMARCHY_GDN_RAW_REPEAT MLX_OMARCHY_GDN_DECODE_TILE=0 "$C" /var/tmp/dfuse/gdu_fp64_probe.py --capture "$O/gdu-operands.npz" > "$O/capture.log" 2>&1 || echo "capture FAILED"
env MLX_OMARCHY_GDN_RAW_REPEAT=1 MLX_OMARCHY_GDN_DECODE_TILE=0 "$C" /var/tmp/dfuse/gdu_fp64_probe.py --compare "$O/gdu-operands.npz" "$O/fp64.json" > "$O/fp64.log" 2>&1 || echo "fp64 FAILED"
cat "$O/fp64.json" 2>/dev/null || tail -5 "$O/fp64.log"

echo "== (2) free-run 10x512 with composed top-2 gaps at divergences"
env MLX_OMARCHY_GDN_RAW_REPEAT=1 MLX_OMARCHY_GDN_DECODE_TILE=0 "$C" /var/tmp/dfuse/free_run_gaps.py \
  "$M9B" "$O/free-run-gaps.json" 10 512 > "$O/gaps.log" 2>&1 || echo "gaps FAILED"
cat "$O/free-run-gaps.json" 2>/dev/null | tail -20

echo "== (3) S=1 decode-route-sensitive PPL"
env MLX_OMARCHY_GDN_RAW_REPEAT=1 MLX_OMARCHY_GDN_DECODE_TILE=0 "$C" /var/tmp/dfuse/ppl_s1.py \
  "$M9B" "$O/ppl-s1.json" > "$O/ppl.log" 2>&1 || echo "ppl FAILED"
cat "$O/ppl-s1.json" 2>/dev/null || tail -5 "$O/ppl.log"

sha256sum "$O"/*.json > "$O/SHA256SUMS" 2>/dev/null || true
svc=$(systemctl is-active llm-inference || true)
printf "post-window llm-inference: %s\n" "$svc"
