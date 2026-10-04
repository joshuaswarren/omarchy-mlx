#!/usr/bin/env bash
# RingTurnaround W2 inner: chain slopes per barrier word + (optional) landing decode cells.
# RT_CHAIN_WORDS="ctl:0x178:0x177" -> chain_slopes.py per word, each isolated (failures logged).
# RT_PLAN is the optional decode-cell plan (same spec language as w1_inner.sh).
set -euo pipefail
OUT=${OUT:?missing OUT}
TOOLS=/var/tmp/ringturn
PY=${RT_PY:-/var/tmp/v072-venv-fused/bin/python}
RT_CHAIN_WORDS=${RT_CHAIN_WORDS:-ctl}

test ! -e /var/tmp/JW16_MAINTENANCE
mkdir -p "$OUT"
uptime_s=$(cut -d. -f1 /proc/uptime); test "$uptime_s" -ge 360
settle=0
while :; do
  load1=$(cut -d' ' -f1 /proc/loadavg)
  load_int=${load1%%.*}; load_frac=${load1#*.}00
  psi=$(cat /proc/pressure/cpu)
  if (( $((10#$load_int * 100 + 10#${load_frac:0:2})) < 50 )) && case "$psi" in *"some avg10=0.00"*) true;; *) false;; esac; then
    break
  fi
  settle=$((settle+15)); [ "$settle" -ge 240 ] && { echo "settle timeout: load1=$load1 psi=$psi"; exit 1; }
  echo "settle $settle s: load1=$load1 psi=$psi" >> "$OUT/settle.log"
  sleep 15
done
printf 'measurement_host=%s boot_id=%s uptime_s=%s settle_wait_s=%s kernel=%s\n' \
  "$(hostname)" "$(cat /proc/sys/kernel/random/boot_id)" "$uptime_s" "$settle" "$(uname -r)" | tee "$OUT/gates.txt"
"$PY" "$TOOLS/mlx_provenance.py" > "$OUT/provenance.json" 2>&1 || echo "provenance-run-failed"

IFS=':' read -ra WORDS <<< "$RT_CHAIN_WORDS"
for w in "${WORDS[@]}"; do
  echo "chain $w start $(date -u +%FT%TZ)" | tee -a "$OUT/order.log"
  if [ "$w" = ctl ]; then
    set +e; "$PY" "$TOOLS/chain_slopes.py" > "$OUT/chain-ctl.json" 2> "$OUT/chain-ctl.err"
  else
    set +e; HK_CDM_BARRIER_MASK="$w" "$PY" "$TOOLS/chain_slopes.py" > "$OUT/chain-$w.json" 2> "$OUT/chain-$w.err"
  fi
  rc=$?
  set -e
  echo "chain $w rc=$rc $(head -c 200 "$OUT/chain-$w.json" 2>/dev/null || echo no-json)" | tee -a "$OUT/order.log"
done

echo "== chains done =="
