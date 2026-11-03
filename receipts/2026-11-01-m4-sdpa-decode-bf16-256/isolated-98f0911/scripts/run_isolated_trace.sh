#!/bin/bash
# Joshua liaison, #320 (13:59Z review): isolated M4 dispatch trace of this commit alone.
# A = dmain (profiling build of main a69d9e6, the PR's base), B = dp320 (profiling build of 98f0911, the PR's code).
# The receipt's own trace.sh / trace_gen.py / profile_analyze.py; granite-4.0-1b-4bit, 32 greedy tokens; order A B B A.
set -uo pipefail
U=$HOME/gpu/upr; O=$U/logs/liaison-trace320; mkdir -p $O
{ date -u +%FT%TZ; uname -r; grep -o 'nr_cpus=[0-9]*' /proc/cmdline; echo "nproc $(nproc)"; } > $O/host.txt
grep -q "^dp320 " $U/commits.txt || echo "dp320 98f0911" >> $U/commits.txt
grep -q BUILD_OK $U/logs/build-dp320.log 2>/dev/null || bash $U/probes/build.sh dp320 98f0911 1 || { echo "build dp320 failed" >> $O/done.txt; exit 3; }
[ -x $U/venv-dp320/bin/python ] || bash $U/probes/mkvenv.sh dp320 || { echo "mkvenv failed" >> $O/done.txt; exit 4; }
[ -d $U/wt-dmain ] && [ -x $U/venv-dmain/bin/python ] || { echo "dmain build missing" >> $O/done.txt; exit 5; }
RP=receipts/2026-11-01-m4-sdpa-decode-bf16-256/scripts
for f in trace.sh trace_gen.py; do
  git -C $U/repo.git show liaison/320-head:$RP/$f > $O/$f || { echo "cannot read $f from the #320 head" >> $O/done.txt; exit 6; }
  [ -e $U/probes/$f ] || cp $O/$f $U/probes/$f
  cmp -s $O/$f $U/probes/$f || { echo "probes/$f differs from the #320 receipt copy" >> $O/done.txt; exit 6; }
done
for w in dmain dp320; do [ -f $U/wt-$w/scripts/profile_analyze.py ] || { echo "wt-$w has no profile_analyze.py" >> $O/done.txt; exit 7; }; done
REPO=mlx-community/granite-4.0-1b-4bit
ls -d $HOME/.cache/huggingface/hub/models--mlx-community--granite-4.0-1b-4bit/snapshots/* >> $O/host.txt 2>&1
echo "repo=$REPO" >> $O/host.txt
i=0
for v in dmain dp320 dp320 dmain; do
  i=$((i+1)); bash $U/probes/trace.sh $v granite-iso$i "$REPO" 32 > $O/trace-$i-$v.out 2>&1
  cp $U/traces/$v-granite-iso$i/analysis.txt $O/analysis-$i-$v.txt 2>/dev/null; cp $U/traces/$v-granite-iso$i/run.out $O/run-$i-$v.out 2>/dev/null
  grep -E "SdpaDecodeNativeBF16Hd128|GPU busy per step" $O/analysis-$i-$v.txt | sed "s/^/$i $v: /" >> $O/summary.txt
done
echo "TRACE320_DONE $(date -u +%FT%TZ)" >> $O/done.txt
