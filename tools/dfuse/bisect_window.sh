#!/usr/bin/env bash
# Runs INSIDE a gpuwin window: perrow bisect (forced legacy path).
set -euo pipefail
C=/var/tmp/dfuse-venv/bin/python
M9B=$(ls -d ${HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/*/ | head -1)
O=/var/tmp/dfuse/perrow
mkdir -p "$O"

test ! -e /var/tmp/JW16_MAINTENANCE
uptime_s=$(cut -d. -f1 /proc/uptime); test "$uptime_s" -ge 360
case $(cat /proc/pressure/cpu) in *"some avg10=0.00"*) ;; *) echo "PSI gate"; exit 1;; esac
printf "host=%s boot=%s\n" "$(hostname)" "$(cat /proc/sys/kernel/random/boot_id)" > "$O/gates-bisect.txt"

env -u MLX_OMARCHY_GDN_RAW_REPEAT MLX_OMARCHY_GDN_DECODE_TILE=0 MLX_OMARCHY_GDN_PF=0 \
  "$C" /var/tmp/dfuse/perrow_bisect.py "$M9B" "$O" 30 1 > "$O/bisect.log" 2>&1 || { echo "BISECT FAILED"; tail -8 "$O/bisect.log"; exit 1; }
cat "$O/bisect.json"
svc=$(systemctl is-active llm-inference || true)
printf "post-window llm-inference: %s\n" "$svc"
