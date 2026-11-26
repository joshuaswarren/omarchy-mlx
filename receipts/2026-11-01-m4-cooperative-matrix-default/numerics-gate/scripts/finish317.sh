#!/bin/bash
# After the 20:00Z start guard stops runner_all.sh: when the Llama reference is
# merged (and only until 20:12Z), run the short CPU-only #317 analysis.
O=$HOME/gpu/upr/gate/out
while [ ! -s $O/ref-Llama-3.2-1B-Instruct-bf16.json ]; do
  [ "$(date -u +%H%M)" -ge 2012 ] && { echo "finish317: reference not ready by 20:12Z; not started"; exit 0; }
  sleep 15
done
sleep 40  # let runner_all.sh claim the analysis first if its 20:00Z guard allows
pgrep -f "[r]un_gate.sh analyze317" > /dev/null && { echo "finish317: runner is analyzing"; exit 0; }
grep -q "analyze317 start" $HOME/gpu/upr/logs/gate-runner-all.log && { echo "finish317: runner took it"; exit 0; }
[ -s $O/gate-317-Llama-3.2-1B-Instruct-bf16.json ] && { echo "finish317: already analyzed"; exit 0; }
[ "$(date -u +%H%M)" -ge 2012 ] && { echo "finish317: too late to start"; exit 0; }
echo "== analyze317 start $(date -u +%FT%TZ)"
bash $HOME/gpu/upr/gate/run_gate.sh analyze317 >> $HOME/gpu/upr/logs/gate-analyze317.log 2>&1
echo "== analyze317 rc=$? $(date -u +%FT%TZ)"
