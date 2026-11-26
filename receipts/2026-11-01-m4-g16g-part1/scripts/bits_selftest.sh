#!/bin/bash
# Compare exit status on a matching and a mismatching baseline, per probe.
# usage: bits_selftest.sh <arm>   (writes logs/bits-selftest-<arm>.log)
U=$HOME/gpu/upr; A=$1; cd $U/probes; mkdir -p $U/bits/selftest; O=$U/logs/bits-selftest-$A.log; : > $O
export AB_NOWAIT=1
for p in qmv rope sdpa qmm_fma; do
  env=""; [ $p = qmm_fma ] && env="HK_APPLE9_COOPMAT=0"
  f=$U/bits/selftest/$p.npz; t=$U/bits/selftest/$p-tampered.npz
  AB_ENV="$env" python3 once.py $U/logs/bits-selftest.jsonl $A -- ${p}_bits.py dump $f > /dev/null 2>&1
  $U/venv-$A/bin/python tamper.py $f $t >> $O 2>&1
  AB_ENV="$env" python3 once.py $U/logs/bits-selftest.jsonl $A -- ${p}_bits.py compare $f > /dev/null 2>&1; r1=$?
  AB_ENV="$env" python3 once.py $U/logs/bits-selftest.jsonl $A -- ${p}_bits.py compare $t > /dev/null 2>&1; r2=$?
  echo "${p}_bits.py compare: matching baseline rc=$r1 (want 0), mismatching baseline rc=$r2 (want 1)" >> $O
done
cat $O
