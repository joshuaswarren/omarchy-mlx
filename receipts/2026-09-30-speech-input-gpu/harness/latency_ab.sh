#!/bin/bash
# GPU turn: direct Recognition latency, final worker (main) vs the worker before the empty-transcript
# retry, interleaved final/pre/final/pre/final. A monitor records loadavg, CPU frequency, top
# processes and render-node holders every second.
set -u
A=<home>/agents/SpeechInputGpu
R=$A/runs/latency-ab-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$R"
PY=<home>/.local/share/mlx-omarchy/venv/bin/python
export HF_HUB_OFFLINE=1
{ date -u +%FT%TZ; cat /proc/sys/kernel/random/boot_id; cat /proc/loadavg
  ps -eo pid,etime,cmd | grep -E "python|mlx" | grep -v grep
  fuser -v /dev/dri/renderD128 2>&1
  PYTHONPATH=$A/repo/serve $PY $A/repo/scripts/mlx_provenance.py; } > "$R/env.txt" 2>&1
( while true; do { date -u +%T.%N; cat /proc/loadavg
    cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq | sort | uniq -c | tr '\n' ' '; echo
    fuser /dev/dri/renderD128 2>&1; echo
    ps -eo pid,pcpu,rss,etime,cmd --sort=-pcpu | head -5 | cut -c1-200; } >> "$R/monitor.txt"; sleep 1; done ) &
MON=$!
trap 'kill $MON 2>/dev/null; sync' EXIT
for label in final pre final pre final; do
  if [ $label = final ]; then SP=$A/repo/serve; else SP=$A/repo-preretry/serve; fi
  echo "== $label $(date -u +%T.%N)" >> "$R/monitor.txt"
  PYTHONPATH=$SP:<home>/voice-site timeout 150 $PY $A/latency_only.py "$R/$label-$(date -u +%H%M%S).json" $label >> "$R/stdout.txt" 2>> "$R/stderr.txt"
  echo "exit=$?" >> "$R/stderr.txt"
done
echo "== end $(date -u +%T.%N)" >> "$R/monitor.txt"
echo "$R"
