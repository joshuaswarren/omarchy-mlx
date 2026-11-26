#!/bin/bash
# GPU turn 6 (empty-transcript retry) (worker without the models-package import): smoke (new cancel path), full-corpus eval + latency, worker under gdb with dispatch trace.
set -u
A=<home>/agents/SpeechInputGpu
R=$A/runs/turn6
mkdir -p "$R"
PY=<home>/.local/share/mlx-omarchy/venv/bin/python
export PYTHONPATH=$A/repo/serve:<home>/voice-site
export HF_HUB_OFFLINE=1
cd "$A/repo"
step() { [ -e "$R/$1.done" ] && return 1; echo "== $1 $(date -u +%FT%TZ)"; return 0; }
done_() { touch "$R/$1.done"; sync; }

if step env; then
  { date -u +%FT%TZ; cat /proc/sys/kernel/random/boot_id; cat /proc/loadavg
    ps -eo pid,etime,cmd | grep -E "python|mlx" | grep -v grep
    $PY scripts/mlx_provenance.py; } > "$R/env.txt" 2>&1; done_ env
fi
if step smoke; then
  timeout 240 $PY $A/smoke_path.py > "$R/smoke.json" 2> "$R/smoke.stderr"; echo "exit=$?" >> "$R/smoke.stderr"; done_ smoke
fi
if step eval; then
  timeout 420 $PY $A/eval_run.py "$R/eval" > "$R/eval.stdout" 2> "$R/eval.stderr"
  rc=$?; echo "exit=$rc" >> "$R/eval.stderr"; [ $rc -eq 0 ] && done_ eval
fi
if step overlimit && [ -e "$R/eval.done" ]; then
  timeout 120 $PY $A/overlimit.py "$R/eval/overlimit.jsonl" > "$R/overlimit.stdout" 2> "$R/overlimit.stderr"
  rc=$?; echo "exit=$rc" >> "$R/overlimit.stderr"; [ $rc -eq 0 ] && done_ overlimit
fi
if step frames; then
  $PY $A/build_frames.py $A/corpus/manifest.json "$R/frames.bin" > "$R/frames.txt" 2>&1 && done_ frames
fi
if step worker_trace && [ -e "$R/frames.done" ]; then
  CPU_COUNT_REDIRECT="< $R/frames.bin > $R/worker.out 2> $R/worker.trace" MLX_OMARCHY_TRACE_DISPATCH=1 \
    timeout 300 gdb -batch -x $A/count_cpu.gdb.py --args $PY -m mlx_omarchy_assistant.gpu_stt_worker \
    --model-dir <app-home>/voice/parakeet-tdt-0.6b-v3 > "$R/worker.gdb.txt" 2>&1
  echo "exit=$?" >> "$R/worker.gdb.txt"; done_ worker_trace
fi
echo "== turn6 end $(date -u +%FT%TZ)"; sync
