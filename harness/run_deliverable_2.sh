#!/usr/bin/env bash
# PairGates deliverable 2 orchestrator. Runs three workers under
# gpu-turn (one ticket each) with the count_cpu gdb breakpoint:
#   1. chat worker (qwen3.8-2b-4bit) -> artifacts/PairGates/<ts>-chat-cpu
#   2. decision worker (laya-mlx) -> artifacts/PairGates/<ts>-decision-cpu
#   3. TTS worker (qwen3-tts-0.6b-customvoice-4bit) -> artifacts/PairGates/<ts>-tts-cpu
#
# Each runs the gdb control (mx.gpu = 0, mx.cpu = 3) then captures
# counts + native chains + python stacks if nonzero.
#
# Each worker is launched via gpu-turn -m 30 -- bash run_one_worker_gdb.sh.
set -uo pipefail
RUN_ID=${1:-$(date -u +%Y-%m-%dT%H-%MZ)}
ART_BASE=${ARTIFACT_BASE:-<home>/.local/share/apple-silicon-lab/artifacts/PairGates}
mkdir -p "$ART_BASE"

HARNESS=<home>/agents/PairGates/worktree/harness

for label in chat decision tts; do
  OUT="$ART_BASE/${RUN_ID}-${label}-cpu"
  mkdir -p "$OUT"
  echo "=== $label -> $OUT ==="
  nohup bash -c "<home>/bin/gpu-turn -m 30 -- $HARNESS/run_one_worker_gdb.sh $label $OUT" \
    >> "$OUT/wrapper.log" 2>&1
  echo "done $label"
done

echo "all 3 done"