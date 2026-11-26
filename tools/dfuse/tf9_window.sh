#!/usr/bin/env bash
# 9B numerics gate, NormApple methodology. Runs INSIDE a gpuwin window.
# Phase 1: teacher corpus (route OFF). Phase 2: TF baseline (OFF) + candidate (ON) + compare.
set -euo pipefail
C=/var/tmp/dfuse-cand/bin/python
HF=${HOME}/.cache/huggingface/hub
M9B=$(ls -d $HF/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/*/ | head -1)
O=/var/tmp/dfuse/tf9
mkdir -p "$O"

test ! -e /var/tmp/JW16_MAINTENANCE
uptime_s=$(cut -d. -f1 /proc/uptime); test "$uptime_s" -ge 360
load1=$(cut -d" " -f1 /proc/loadavg)
li=${load1%%.*}; lf=${load1#*.}00
(( $((10#$li * 100 + 10#${lf:0:2})) < 50 )) || { echo "load gate: $load1"; exit 1; }
case $(cat /proc/pressure/cpu) in *"some avg10=0.00"*) ;; *) echo "PSI gate"; exit 1;; esac
printf "host=%s boot=%s uptime_s=%s load1=%s\n" "$(hostname)" \
  "$(cat /proc/sys/kernel/random/boot_id)" "$uptime_s" "$load1" | tee "$O/gates.txt"

if [ ! -e "$O/teacher.json" ]; then
  env -u MLX_OMARCHY_GDN_RAW_REPEAT "$C" /var/tmp/dfuse/mk_teacher_9b.py "$O/teacher.json" \
    > "$O/teacher.log" 2>&1
  echo "teacher: $(tail -1 "$O/teacher.log")"
fi

TF() { # envflag out
  env MLX_OMARCHY_GDN_RAW_REPEAT="$1" "$C" /var/tmp/dfuse/tf9.py "$M9B" \
    /var/tmp/Jw16NumericsGpu/qwen38-2b-prompts-10.jsonl "$O/teacher.json" "$O/$2.json" \
    > "$O/$2.log" 2>&1
  tail -1 "$O/$2.log"
}

TF 0 baseline
TF 1 candidate
/usr/bin/python3 /var/tmp/dfuse/compare_tf9.py "$O/baseline.json" "$O/candidate.json" \
  > "$O/summary.json" || echo "COMPARE FAILED (gate not met)"
cat "$O/summary.json"
sha256sum "$O"/baseline.json "$O"/candidate.json "$O"/teacher.json > "$O/SHA256SUMS"
svc=$(systemctl is-active llm-inference || true)
printf "post-window llm-inference: %s\n" "$svc"
