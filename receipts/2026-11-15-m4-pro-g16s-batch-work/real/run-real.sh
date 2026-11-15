#!/bin/bash
# Submission GPU time on a release build from the DRM scheduler tracepoints
# (gpu_scheduler:drm_sched_job_{queue,run,done}), with vkcube presenting under
# the Wayland compositor during the run. Needs root for tracefs.
# PY = python of a venv with the release build; MODEL = local model snapshot dir.
T=/sys/kernel/tracing; OUT=${OUT:-.}
tr() { sudo -n sh -c "$1"; }
setup() { tr "echo 0 > $T/tracing_on; echo mono > $T/trace_clock; echo 65536 > $T/buffer_size_kb; for e in queue run done; do echo 1 > $T/events/gpu_scheduler/drm_sched_job_\$e/enable; done; echo > $T/trace"; }
restore() { tr "echo 0 > $T/tracing_on; for e in queue run done; do echo 0 > $T/events/gpu_scheduler/drm_sched_job_\$e/enable; done; echo local > $T/trace_clock; echo > $T/trace; echo 1 > $T/tracing_on"; }
trap restore EXIT
one() { # label budget
  setup; vkcube --wsi wayland --width 256 --height 256 > /dev/null 2>&1 & VK=$!
  sleep 3; tr "echo 1 > $T/tracing_on"
  MLX_OMARCHY_BATCH_WORK=$2 "$PY" -I reald.py "$MODEL" > $OUT/$1.drv 2>&1
  tr "echo 0 > $T/tracing_on; cat $T/trace" | xz -9 > $OUT/$1.trace.xz
  kill $VK; wait $VK 2>/dev/null
}
for bw in 5000 10000 15000 20000 30000; do one "$(basename "$MODEL")-$bw" $bw; done
