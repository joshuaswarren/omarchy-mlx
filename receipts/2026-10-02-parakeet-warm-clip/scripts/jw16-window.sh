#!/usr/bin/env bash
# Body of one jw16 gpuwin window: provenance, daemon A/B, resilience.
# Run as: gpuwin.sh 'bash <this file>'. Never takes /tmp/gpuwin.mutex.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
V=/var/tmp/pkwarm-venv/lib/python3.14/site-packages/mlx
S=$V/share/mlx-omarchy/parakeet-1
SOCK=/tmp/pkwarm-h000/ane.sock
mkdir -p /tmp/pkwarm-h000
stop_daemon() { [ -f "$SOCK.pid" ] && kill "$(cat "$SOCK.pid")" 2>/dev/null; sleep 1; }
trap stop_daemon EXIT
echo "boot_id $(cat /proc/sys/kernel/random/boot_id) uptime $(cut -d' ' -f1 /proc/uptime)"
echo "loadavg $(cat /proc/loadavg) | psi $(head -1 /proc/pressure/cpu)"
/var/tmp/pkwarm-venv/bin/python3.14 "$HERE/../../../scripts/mlx_provenance.py" | head -12
stop_daemon
rm -f "$SOCK" "$SOCK.lock" "$SOCK.pid"
setsid "$V/bin/mlx-omarchy-ane-worker" --daemon --socket "$SOCK" \
  --idle-time-ms 600000 \
  --bundle parakeet-encoder-whole="$S/bundles/parakeet-encoder-whole" \
  --libane "$S/libane/libane-strict.so" --deadline-ms 20000 \
  </dev/null >/tmp/pkwarm-h000/daemon-jw16.log 2>&1 &
for _ in $(seq 1 50); do [ -S "$SOCK" ] && break; sleep 0.1; done
cat /tmp/pkwarm-h000/daemon-jw16.log
/var/tmp/pkwarm-venv/bin/python3.14 "$HERE/ab-wall-bench.py" 10 | tail -2
stop_daemon
bash "$HERE/daemon-resilience.sh"
