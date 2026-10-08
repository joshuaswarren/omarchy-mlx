#!/usr/bin/env bash
# SushiHang: wait for the zig fetch to land, then build sushi and arm capture.
set -uo pipefail
cd /var/tmp/sushi-rr
nohup bash -c '
  for i in $(seq 1 240); do
    Z=$(ls -d /var/tmp/sushi-rr/zig-aarch64-linux-*/zig 2>/dev/null | head -1)
    [ -n "$Z" ] && [ -x "$Z" ] && break
    # re-run the fetch if it died without producing the binary
    pgrep -f fetch-zig.sh >/dev/null || bash scripts/fetch-zig.sh >> zigfetch.log 2>&1
    sleep 30
  done
  Z=$(ls -d /var/tmp/sushi-rr/zig-aarch64-linux-*/zig 2>/dev/null | head -1)
  echo "ZIG=$Z" >> sushibuild.log
  [ -n "$Z" ] && [ -x "$Z" ] && "$Z" build -Doptimize=ReleaseFast >> sushibuild.log 2>&1
  ls -la zig-out/bin/ >> sushibuild.log 2>&1
  echo SUSHI-BUILD-DONE >> sushibuild.log
' >> /var/tmp/sushi-rr/finishbuild.log 2>&1 &
echo poller-armed
