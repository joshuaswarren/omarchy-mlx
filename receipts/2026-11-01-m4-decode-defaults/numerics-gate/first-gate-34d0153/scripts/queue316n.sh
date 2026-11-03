#!/bin/bash
# Evening queue, in order: wait for the narrowed wheels and suites (setup316n.sh), the #321 gate
# (runner321.sh) and diag317; build the profiling wheels dp2 (#317) and dp6 (#321); the per-op
# kernel lists for #317 and #321; then the #316n speed A/B on a quiet host; then the #316n gate.
set -uo pipefail
U=$HOME/gpu/upr; L=$U/logs; G=$U/gate
while pgrep -f "[s]etup316n.sh|[r]unner321.sh|[d]iag317.sh" > /dev/null; do sleep 20; done
grep -q SETUP316N_DONE $L/setup316n.log || { echo "QUEUE_STOP: setup316n not done"; exit 2; }
for spec in dp2:91d9f8a dp6:89b54a0; do
  v=${spec%%:*}; c=${spec##*:}
  grep -q "^$v " $U/commits.txt || echo "$v $c" >> $U/commits.txt
  if ! grep -q BUILD_OK $L/build-$v.log 2>/dev/null; then bash $U/probes/build.sh $v $c 1 || { echo "QUEUE_STOP: build $v"; exit 5; }; fi
  [ -x $U/venv-$v/bin/python ] || bash $U/probes/mkvenv.sh $v >> $L/mkvenv-$v.log 2>&1 || { echo "QUEUE_STOP: venv $v"; exit 5; }
  echo "== $v ready $(date -u +%FT%TZ)"
done
for pr in 317 321; do
  echo "== ${pr}k start $(date -u +%FT%TZ)"
  bash $G/run_gatek.sh $pr >> $L/gate-${pr}k.log 2>&1 || echo "== ${pr}k FAILED rc=$? (continuing; traces are not a gate check)"
done
echo "== speed316n start $(date -u +%FT%TZ)"
bash $G/speed316n.sh >> $L/speed316n.log 2>&1 || { echo "QUEUE_STOP: speed316n rc=$?"; exit 3; }
for ph in arms perop analyze; do
  echo "== 316n $ph start $(date -u +%FT%TZ)"
  bash $G/run_gate316n.sh $ph >> $L/gate-316n-$ph.log 2>&1 || { echo "QUEUE_STOP: 316n $ph rc=$?"; exit 4; }
done
echo "QUEUE316N_DONE $(date -u +%FT%TZ)"
