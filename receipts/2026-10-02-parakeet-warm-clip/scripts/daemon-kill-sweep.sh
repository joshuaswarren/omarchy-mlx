#!/usr/bin/env bash
# kill -9 the daemon at several offsets into a daemon-attached call so at
# least one lands inside the ANE submit; record the client outcome and the
# device state after each.
set -u
V=/var/tmp/pkwarm-venv/lib/python3.14/site-packages/mlx
S=$V/share/mlx-omarchy/parakeet-1
PY=/var/tmp/pkwarm-venv/bin/python3.14
CLI=$V/bin/mlx-omarchy-parakeet
SOCK=/tmp/pkwarm-h000/ane.sock
OUT=/tmp/pkwarm-h000/res
mkdir -p "$OUT"
for delay in 0.45 0.50 0.55 0.60; do
  rm -f "$SOCK" "$SOCK.lock" "$SOCK.pid"
  setsid "$V/bin/mlx-omarchy-ane-worker" --daemon --socket "$SOCK" \
    --idle-time-ms 600000 \
    --bundle parakeet-encoder-whole="$S/bundles/parakeet-encoder-whole" \
    --libane "$S/libane/libane-strict.so" --deadline-ms 20000 \
    </dev/null >>"$OUT/daemon.log" 2>&1 &
  for _ in $(seq 1 50); do [ -S "$SOCK" ] && break; sleep 0.1; done
  dpid=$(cat "$SOCK.pid")
  d="$OUT/vd-$delay"
  mkdir -p "$d"; rm -f "$d/transcribe-report.json"
  MLX_OMARCHY_PK_KEEP_WORKER=1 MLX_OMARCHY_ANE_SOCK="$SOCK" \
    "$PY" "$CLI" transcribe -o "$d" >"$d.log" 2>&1 &
  victim=$!
  sleep "$delay"; kill -9 "$dpid"; wait "$victim"; rc=$?
  sleep 1
  report=no; [ -f "$d/transcribe-report.json" ] && report=yes
  err=$(grep -oE 'ResidentWorkerError: .{0,90}' "$d.log" | tail -1)
  echo "delay=$delay client_rc=$rc report=$report err=$err"
  echo "  workers=$(ps -eo cmd | grep -c '[m]lx-omarchy-ane-worker')" \
       "accel0=[$(fuser /dev/accel/accel0 2>&1 | tr -s ' ')]" \
       "dmesg_ane_dart=$(sudo -n dmesg | grep -ciE 'ane|dart')"
done
sudo -n dmesg | grep -iE 'ane|dart' | tail -3
