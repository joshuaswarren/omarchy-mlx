#!/bin/bash
# GPU turn: app + Everyday pair with voice resident; HTTP transcribe latency (2 warm-ups + 30),
# then browser stop-to-transcript latency (2 warm-ups + 30, 5.0 s recordings). A monitor samples
# processes and load every 2 s so foreign GPU work during the window is visible.
set -u
A=<home>/agents/SpeechInputGpu
R=$A/runs/e2e-latency-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$R"
PY=<home>/.local/share/mlx-omarchy/venv/bin/python
export PYTHONPATH=$A/repo/serve:<home>/voice-site
PORT=47811
{ date -u +%FT%TZ; cat /proc/sys/kernel/random/boot_id; cat /proc/loadavg
  ps -eo pid,etime,cmd | grep -E "python|mlx" | grep -v grep; } > "$R/env.txt" 2>&1
$PY $A/make_wavs.py "$R" > "$R/wavs.json" 2>&1
rm -f <app-home>/assistant/application.json
cd "$A/repo"
setsid $PY -m mlx_omarchy_assistant --home $A/home --no-browser --port $PORT > "$R/app.log" 2>&1 &
APP=$!
( while true; do { date -u +%T; cat /proc/loadavg; cat /sys/class/thermal/thermal_zone0/temp
    ps -eo pid,pcpu,rss,etime,cmd --sort=-pcpu | head -8; } >> "$R/monitor.txt"; sleep 2; done ) &
MON=$!
trap 'pkill -KILL -f -- "--user-data-dir=$R/profile"; kill $MON 2>/dev/null; kill -TERM -$APP 2>/dev/null; sleep 3; kill -KILL -$APP 2>/dev/null; sync' EXIT
for i in $(seq 60); do [ -e <app-home>/assistant/application.json ] && break; sleep 0.5; done
timeout 240 $PY $A/app_client.py setup $A/home > "$R/setup.json" 2>&1
echo "== http $(date -u +%T)" >> "$R/monitor.txt"
timeout 150 $PY $A/app_client.py latency $A/home "$R/five_s.wav" 30 > "$R/http_latency.json" 2>&1
echo "== browser $(date -u +%T)" >> "$R/monitor.txt"
URL=$($PY $A/app_client.py launch $A/home)
timeout 400 node $A/e2e.mjs "$URL" "$R/fake_mic_long.wav" "$R" latency > "$R/e2e.json" 2> "$R/e2e.stderr"
echo "exit=$?" >> "$R/e2e.stderr"
echo "== end $(date -u +%T)" >> "$R/monitor.txt"
echo "$R"
