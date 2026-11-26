#!/bin/bash
# Positive (mx.cpu) and negative (mx.gpu) controls for the gdb CPU-dispatch counter.
A=<home>/agents/SpeechInputGpu
R=$A/runs/controls-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$R"
PY=<home>/.local/share/mlx-omarchy/venv/bin/python
for dev in gpu cpu; do
  CPU_COUNT_REDIRECT="> $R/control_$dev.out 2>&1" timeout 90 gdb -batch -x $A/count_cpu.gdb.py \
    --args $PY $A/control.py $dev > "$R/control_$dev.gdb.txt" 2>&1
done
echo "$R"; sync
