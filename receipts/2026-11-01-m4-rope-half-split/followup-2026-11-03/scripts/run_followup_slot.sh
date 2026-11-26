#!/bin/bash
# Joshua liaison, #319 follow-ups (02:06Z). Full-CPU Gravity boot required.
# 1) 6 interleaved rope_bits.py timing pairs: A = p2j (main 999afe65 tree), B = p3m (3c2160b1, merged #319).
# 2) p5 = 1b1b71b1 (half-split default keyed on G16G): build + omarchy_fast_ops_tests unset / =1 / =0,
#    then the four bit compares against the p2j dump.
set -uo pipefail
U=$HOME/gpu/upr; O=$U/logs/liaison-slot319; mkdir -p $O
{ date -u +%FT%TZ; uname -a; cat /proc/cmdline; echo "nproc $(nproc)"; lsmod | grep asahi_m4; } > $O/host.txt
[ "$(nproc)" -ge 4 ] || { echo "nproc $(nproc): not a full-CPU boot; stop" | tee -a $O/done.txt; exit 2; }
[ -s $U/bits/rope-p2j.npz ] || { echo "no rope-p2j.npz" | tee -a $O/done.txt; exit 2; }
cd $U/probes
python3 ab.py $O/perf-3c2160b1-rope.jsonl 6 p2j p3m -- rope_bits.py compare $U/bits/rope-p2j.npz > $O/ab-rope.log 2>&1
echo "ab_rc=$?" >> $O/done.txt
python3 $U/liaison/rope_summary.py $O/perf-3c2160b1-rope.jsonl > $O/perf-3c2160b1-rope-summary.txt 2>&1
echo "summary_rc=$?" >> $O/done.txt
echo "TIMING_DONE $(date -u +%FT%TZ)" >> $O/done.txt
grep -q "^p5 " $U/commits.txt || echo "p5 1b1b71b" >> $U/commits.txt
grep -q BUILD_OK $U/logs/build-p5.log 2>/dev/null || bash $U/probes/build.sh p5 1b1b71b1 0 omarchy_fast_ops_tests || { echo "build p5 failed" >> $O/done.txt; exit 3; }
[ -x $U/venv-p5/bin/python ] || bash $U/probes/mkvenv.sh p5 || { echo "mkvenv p5 failed" >> $O/done.txt; exit 4; }
BATTERY_TAG=-unset bash $U/probes/battery.sh p5 omarchy_fast_ops_tests
BATTERY_TAG=-no1 BATTERY_ENV="MLX_OMARCHY_NO_ROPE_HALF_SPLIT=1" bash $U/probes/battery.sh p5 omarchy_fast_ops_tests
BATTERY_TAG=-no0 BATTERY_ENV="MLX_OMARCHY_NO_ROPE_HALF_SPLIT=0" bash $U/probes/battery.sh p5 omarchy_fast_ops_tests
bash $U/probes/bits.sh p5 compare p2j > $O/bits-p5-vs-main.log 2>&1; echo "bits_p5_rc=$?" >> $O/done.txt
echo "SLOT319_DONE $(date -u +%FT%TZ)" >> $O/done.txt
