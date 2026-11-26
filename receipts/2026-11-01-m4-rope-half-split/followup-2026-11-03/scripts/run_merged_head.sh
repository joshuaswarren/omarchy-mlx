#!/bin/bash
# Joshua liaison: #319 after merging main (head 3c2160b1) against main 999afe65.
# p2j = 2e5d7ea4 wheel, whose tree equals main 999afe65 (the #317 squash); p3m = 3c2160b1.
# Host booted with nr_cpus=1 (affects build time only).
# omarchy_fast_ops_tests on p3m (GPU lock), then the four bit probes: dump on p2j, compare on p3m.
set -uo pipefail
U=$HOME/gpu/upr
echo "start $(date -u +%FT%TZ) nproc=$(nproc) $(grep -o 'nr_cpus=[0-9]*' /proc/cmdline)"
grep -q "^p3m " $U/commits.txt || echo "p3m 3c2160b" >> $U/commits.txt
grep -q BUILD_OK $U/logs/build-p3m.log 2>/dev/null || bash $U/probes/build.sh p3m 3c2160b1 0 omarchy_fast_ops_tests || { echo "build p3m failed rc=$?"; exit 3; }
[ -x $U/venv-p3m/bin/python ] || bash $U/probes/mkvenv.sh p3m || { echo "mkvenv p3m failed"; exit 4; }
echo "BUILT319 $(date -u +%FT%TZ)"
BATTERY_TAG=-fastops bash $U/probes/battery.sh p3m omarchy_fast_ops_tests
bash $U/probes/bits.sh p2j dump > $U/logs/liaison-bits-p2j.log 2>&1; echo "bits dump rc=$?"
bash $U/probes/bits.sh p3m compare p2j > $U/logs/liaison-bits-p3m.log 2>&1; echo "bits compare rc=$?"
echo "RUN319M_DONE $(date -u +%FT%TZ)"
