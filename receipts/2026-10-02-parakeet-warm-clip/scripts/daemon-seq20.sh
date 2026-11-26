#!/usr/bin/env bash
# 20 sequential daemon-attached calls against one fresh daemon.
set -u
V=/var/tmp/pkwarm-venv/lib/python3.14/site-packages/mlx
S=$V/share/mlx-omarchy/parakeet-1
SOCK=/tmp/pkwarm-h000/ane.sock
[ -f "$SOCK.pid" ] && kill "$(cat "$SOCK.pid")" 2>/dev/null
sleep 1
rm -f "$SOCK" "$SOCK.lock" "$SOCK.pid"
setsid "$V/bin/mlx-omarchy-ane-worker" --daemon --socket "$SOCK" \
  --idle-time-ms 600000 \
  --bundle parakeet-encoder-whole="$S/bundles/parakeet-encoder-whole" \
  --libane "$S/libane/libane-strict.so" --deadline-ms 20000 \
  </dev/null >/tmp/pkwarm-h000/daemon-20.log 2>&1 &
for _ in $(seq 1 50); do [ -S "$SOCK" ] && break; sleep 0.1; done
echo "loadavg: $(cat /proc/loadavg)"; echo "psi: $(head -1 /proc/pressure/cpu)"
/var/tmp/pkwarm-venv/bin/python3.14 $(dirname "$0")/wall-bench.py 20 daemon >/dev/null 2>&1
/var/tmp/pkwarm-venv/bin/python3.14 - <<'PYEOF'
import json, statistics
s = json.load(open("/tmp/pkwarm-h000/wall-daemon/summary.json"))
ok = [r for r in s if r.get("status") == "match"]
print("n", len(s), "match", len(ok),
      "transports", sorted({r["transport"] for r in ok}),
      "tsha", sorted({r["tsha"] for r in ok}),
      "failed", sorted({c for r in ok for c in r["checks_failed"]}),
      "cpu", sorted({r["cpu_events"] for r in ok}),
      "emissions_slots", sorted({r["emissions"] for r in ok}))
w = [r["wall_ms"] for r in ok]
print("wall median", round(statistics.median(w), 1), "max", round(max(w), 1),
      "open median", statistics.median(r["session_open_ms"] for r in ok))
PYEOF
