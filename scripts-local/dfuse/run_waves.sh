#!/usr/bin/env bash
# DispatchFuse wave runner: sequential A/B waves with retry on reboot/gate
# refusal. usage: run_waves.sh "wA wB wC wD" [max_tries_per_wave]
set -euo pipefail
WAVES=${1:?wave list}
TRIES=${2:-4}
for w in $WAVES; do
  ok=0
  for t in $(seq 1 "$TRIES"); do
    echo "=== wave $w try $t $(date -u +%FT%TZ)"
    if ssh -o ConnectTimeout=8 jw16mbp1-linux \
        'while [ $(cut -d. -f1 /proc/uptime) -lt 420 ]; do sleep 15; done; for i in $(seq 1 40); do case $(cat /proc/pressure/cpu) in *"some avg10=0.00"*) break;; esac; sleep 6; done; wm=0; while pgrep -f "[a]ppbar/gpuwin" >/dev/null; do wm=$((wm+10)); [ "$wm" -ge 1200 ] && { echo "WAIT-TIMEOUT gpuwin still busy after 20 min"; exit 1; }; sleep 10; done; systemctl is-active llm-inference >/dev/null || /var/tmp/appbar/gpuwin.sh true; sleep 20; sudo journalctl --flush; sync; bash /var/tmp/dfuse/run_ab.sh '"$w"' 2>&1 | tail -14; rc=${PIPESTATUS[0]}; systemctl is-active llm-inference; exit $rc'; then
      ok=1
      break
    fi
    echo "wave $w try $t failed; backing off"
    sleep 60
  done
  [ "$ok" = 1 ] || { echo "WAVE $w EXHAUSTED"; exit 1; }
done
echo "ALL WAVES DONE"
