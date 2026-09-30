#!/bin/bash
# GPU turn: app + Everyday pair with voice, Chromium fake-mic E2E, HTTP upload latency.
# The server always stops before this script exits (no server left under the lock).
set -u
A=<home>/agents/SpeechInputGpu
R=$A/runs/e2e-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$R"
PY=<home>/.local/share/mlx-omarchy/venv/bin/python
export PYTHONPATH=$A/repo/serve:<home>/voice-site
PORT=47811
{ date -u +%FT%TZ; cat /proc/sys/kernel/random/boot_id; cat /proc/loadavg
  ps -eo pid,etime,cmd | grep -E "python|mlx" | grep -v grep; } > "$R/env.txt" 2>&1
$PY $A/make_wavs.py "$R" > "$R/wavs.json" 2>&1
T2=$A/runs/turn6
[ -e "$T2/eval.done" ] && [ -e "$T2/overlimit.done" ] && [ -e "$T2/worker_trace.done" ] || { echo "turn2 incomplete; not running E2E" > "$R/qualify.json"; echo "$R"; exit 0; }
$PY $A/score.py $A/corpus/manifest.json "$T2/eval/clips.jsonl+$T2/eval/overlimit.jsonl" > "$T2/scores.json" 2> "$R/score.stderr"
if ! $PY $A/qualify.py $A/home $T2/scores.json $T2/eval/latency.json $T2/worker.gdb.txt runs/turn6 > "$R/qualify.json" 2>&1; then
  echo "$R"; exit 0
fi
cd "$A/repo"
setsid $PY -m mlx_omarchy_assistant --home $A/home --no-browser --port $PORT > "$R/app.log" 2>&1 &
APP=$!
trap 'pkill -KILL -f -- "--user-data-dir=$R/profile"; kill -TERM -$APP 2>/dev/null; sleep 3; kill -KILL -$APP 2>/dev/null; sync' EXIT
for i in $(seq 60); do [ -e <app-home>/assistant/application.json ] && break; sleep 0.5; done
timeout 240 $PY $A/app_client.py setup $A/home > "$R/setup.json" 2>&1
URL=$($PY $A/app_client.py launch $A/home)
timeout 300 node $A/e2e.mjs "$URL" "$R/fake_mic.wav" "$R" > "$R/e2e.json" 2> "$R/e2e.stderr"
echo "exit=$?" >> "$R/e2e.stderr"
ps -eo pid,etime,rss,cmd | grep -E "python|mlx" | grep -v grep >> "$R/ps_before_latency.txt"; cat /proc/loadavg >> "$R/ps_before_latency.txt"
timeout 150 $PY $A/app_client.py latency $A/home "$R/five_s.wav" 30 > "$R/http_latency.json" 2>&1
echo "$R"
