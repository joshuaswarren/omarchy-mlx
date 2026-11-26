#!/bin/bash
# GPU turn: app + Everyday pair with voice; Chromium WITHOUT the fake-UI flag, microphone denied.
# The server always stops before this script exits.
set -u
A=<home>/agents/SpeechInputGpu
R=$A/runs/e2e-denied-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$R"
PY=<home>/.local/share/mlx-omarchy/venv/bin/python
export PYTHONPATH=$A/repo/serve:<home>/voice-site
PORT=47811
{ date -u +%FT%TZ; cat /proc/sys/kernel/random/boot_id; cat /proc/loadavg
  ps -eo pid,etime,cmd | grep -E "python|mlx" | grep -v grep; } > "$R/env.txt" 2>&1
cp $A/runs/e2e-20260930T035118Z/fake_mic.wav "$R/"
rm -f <app-home>/assistant/application.json
cd "$A/repo"
setsid $PY -m mlx_omarchy_assistant --home $A/home --no-browser --port $PORT > "$R/app.log" 2>&1 &
APP=$!
trap 'pkill -KILL -f -- "--user-data-dir=$R/profile"; kill -TERM -$APP 2>/dev/null; sleep 3; kill -KILL -$APP 2>/dev/null; sync' EXIT
for i in $(seq 60); do [ -e <app-home>/assistant/application.json ] && break; sleep 0.5; done

timeout 240 $PY $A/app_client.py setup $A/home > "$R/setup.json" 2>&1
URL=$($PY $A/app_client.py launch $A/home)
timeout 200 node $A/e2e.mjs "$URL" "$R/fake_mic.wav" "$R" denied > "$R/e2e.json" 2> "$R/e2e.stderr"
echo "exit=$?" >> "$R/e2e.stderr"
echo "$R"
