#!/usr/bin/env bash
# Interleaved daemon vs private A/B with MLX_OMARCHY_PK_TRUST_CACHE=1 in
# both arms (L1 + L3 combined). One fresh daemon; results land in
# /tmp/pkwarm-h000/ab/ab-summary.json.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
V=/var/tmp/pkwarm-venv/lib/python3.14/site-packages/mlx
S=$V/share/mlx-omarchy/parakeet-1
SOCK=/tmp/pkwarm-h000/ane.sock
mkdir -p /tmp/pkwarm-h000
[ -f "$SOCK.pid" ] && kill "$(cat "$SOCK.pid")" 2>/dev/null
sleep 1
rm -f "$SOCK" "$SOCK.lock" "$SOCK.pid"
setsid "$V/bin/mlx-omarchy-ane-worker" --daemon --socket "$SOCK" \
  --idle-time-ms 600000 \
  --bundle parakeet-encoder-whole="$S/bundles/parakeet-encoder-whole" \
  --libane "$S/libane/libane-strict.so" --deadline-ms 20000 \
  </dev/null >/tmp/pkwarm-h000/daemon-trust.log 2>&1 &
for _ in $(seq 1 50); do [ -S "$SOCK" ] && break; sleep 0.1; done
export MLX_OMARCHY_PK_TRUST_CACHE=1
# One warm-up call refreshes .verified-hashes if the sidecar is stale.
/var/tmp/pkwarm-venv/bin/python3.14 "$V/bin/mlx-omarchy-parakeet" transcribe \
  -o /tmp/pkwarm-h000/trust-warmup >/dev/null 2>&1
echo "loadavg $(cat /proc/loadavg) | psi $(head -1 /proc/pressure/cpu)"
/var/tmp/pkwarm-venv/bin/python3.14 "$HERE/ab-wall-bench.py" "${1:-10}" | tail -1 >/dev/null
kill "$(cat "$SOCK.pid")" 2>/dev/null
