#!/bin/bash
# GPU turn: system MemAvailable while (A) the recognizer runs alone through Recognition, then
# (B) the app starts the Everyday pair with voice and the recognizer loads beside it. A sampler
# writes "<epoch> <MemAvailable kB> <foreign mlx procs>" every 0.25 s; phase markers go to marks.txt.
set -u
A=<home>/agents/SpeechInputGpu
R=$A/runs/mem-probe-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$R"
PY=<home>/.local/share/mlx-omarchy/venv/bin/python
export PYTHONPATH=$A/repo/serve:<home>/voice-site
export HF_HUB_OFFLINE=1
PORT=47811
mark() { echo "$(date +%s.%N) $1" >> "$R/marks.txt"; }
{ date -u +%FT%TZ; cat /proc/sys/kernel/random/boot_id; cat /proc/loadavg
  ps -eo pid,etime,cmd | grep -E "python|mlx" | grep -v grep; } > "$R/env.txt" 2>&1
$PY $A/make_wavs.py "$R" > "$R/wavs.json" 2>&1
( while true; do
    echo "$(date +%s.%N) $(awk '/MemAvailable/{print $2}' /proc/meminfo) $(ps -eo cmd | grep -E 'mlx_omarchy|mlx_lm|laya' | grep -vc SpeechInputGpu)"
    sleep 0.25; done >> "$R/memavail.txt" ) &
MON=$!
APP=
trap 'kill $MON 2>/dev/null; [ -n "$APP" ] && { kill -TERM -$APP 2>/dev/null; sleep 3; kill -KILL -$APP 2>/dev/null; }; sync' EXIT
mark baseline_start; sleep 5; mark baseline_end
mark alone_start
timeout 150 $PY $A/mem_alone.py "$R/five_s.wav" > "$R/alone.json" 2> "$R/alone.stderr"
mark alone_end; sleep 5; mark settle_end
rm -f <app-home>/assistant/application.json
cd "$A/repo"
setsid $PY -m mlx_omarchy_assistant --home $A/home --no-browser --port $PORT > "$R/app.log" 2>&1 &
APP=$!
for i in $(seq 60); do [ -e <app-home>/assistant/application.json ] && break; sleep 0.5; done
mark setup_start
timeout 240 $PY $A/app_client.py setup $A/home > "$R/setup.json" 2>&1
mark setup_end; sleep 10; mark pair_settled
timeout 150 $PY $A/app_client.py latency $A/home "$R/five_s.wav" 30 > "$R/http_latency.json" 2>&1
mark transcribes_end; sleep 5; mark recognizer_settled
kill -TERM -$APP 2>/dev/null; sleep 3; kill -KILL -$APP 2>/dev/null; APP=
sleep 5; mark after_stop
echo "$R"
