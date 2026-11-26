#!/bin/bash
# GPU turn: app + Everyday pair with voice; 30 timed GET /api/status, then HTTP transcribe latency
# (2 warm-ups + 30). Tests whether the per-request manager.status() in the transcribe route is the
# gap between direct (p50 ~425 ms) and HTTP (p50 ~1 s) transcription.
set -u
A=<home>/agents/SpeechInputGpu
R=$A/runs/route-probe-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$R"
PY=<home>/.local/share/mlx-omarchy/venv/bin/python
export PYTHONPATH=$A/repo/serve:<home>/voice-site
PORT=47811
{ date -u +%FT%TZ; cat /proc/sys/kernel/random/boot_id; cat /proc/loadavg
  ps -eo pid,etime,cmd | grep -E "python|mlx" | grep -v grep; } > "$R/env.txt" 2>&1
$PY $A/make_wavs.py "$R" > "$R/wavs.json" 2>&1
rm -f <app-home>/assistant/application.json
cd "$A/repo"
git -C "$A/repo" log -1 --format=%H > "$R/source.txt" 2>/dev/null
setsid $PY -m mlx_omarchy_assistant --home $A/home --no-browser --port $PORT > "$R/app.log" 2>&1 &
APP=$!
trap 'kill -TERM -$APP 2>/dev/null; sleep 3; kill -KILL -$APP 2>/dev/null; sync' EXIT
for i in $(seq 60); do [ -e <app-home>/assistant/application.json ] && break; sleep 0.5; done
timeout 240 $PY $A/app_client.py setup $A/home > "$R/setup.json" 2>&1
timeout 60 $PY $A/app_client.py status_latency $A/home 30 > "$R/status_latency.json" 2>&1
timeout 150 $PY $A/app_client.py latency $A/home "$R/five_s.wav" 30 > "$R/http_latency.json" 2>&1
ps -eo pid,etime,rss,cmd | grep -E "python|mlx" | grep -v grep >> "$R/ps_after.txt"
echo "$R"
