#!/usr/bin/env bash
# kill -9 of a daemon client mid-run, kill -9 of the daemon mid-run,
# idle exit; device state (dmesg ANE/DART, accel0 holders) around each.
set -u
V=/var/tmp/pkwarm-venv/lib/python3.14/site-packages/mlx
S=$V/share/mlx-omarchy/parakeet-1
PY=/var/tmp/pkwarm-venv/bin/python3.14
CLI=$V/bin/mlx-omarchy-parakeet
SOCK=/tmp/pkwarm-h000/ane.sock
OUT=/tmp/pkwarm-h000/res
mkdir -p "$OUT"

dev_state() {
  echo "== device state: $1"
  echo "workers: $(ps -eo pid,stat,cmd | grep -c '[m]lx-omarchy-ane-worker')"
  echo "accel0 holders: [$(fuser /dev/accel/accel0 2>&1 | tr -s ' ')]"
  echo "dmesg ane/dart lines: $(sudo -n dmesg 2>/dev/null | grep -ciE 'ane|dart' || echo unknown)"
  sudo -n dmesg 2>/dev/null | grep -iE 'ane .*(fault|error|timeout)|dart.*(fault|error)' | tail -3
}
start_daemon() {
  rm -f "$SOCK" "$SOCK.lock" "$SOCK.pid"
  setsid "$V/bin/mlx-omarchy-ane-worker" --daemon --socket "$SOCK" \
    --idle-time-ms "$1" \
    --bundle parakeet-encoder-whole="$S/bundles/parakeet-encoder-whole" \
    --libane "$S/libane/libane-strict.so" --deadline-ms 20000 \
    </dev/null >>"$OUT/daemon.log" 2>&1 &
  for _ in $(seq 1 50); do [ -S "$SOCK" ] && break; sleep 0.1; done
}
transcribe() {  # $1=label
  mkdir -p "$OUT/$1"
  rm -f "$OUT/$1/transcribe-report.json"
  MLX_OMARCHY_PK_KEEP_WORKER=1 MLX_OMARCHY_ANE_SOCK="$SOCK" \
    timeout 60 "$PY" "$CLI" transcribe -o "$OUT/$1" >"$OUT/$1.log" 2>&1
  local rc=$?
  "$PY" - "$OUT/$1" "$rc" <<'PYEOF'
import hashlib, json, sys
d, rc = sys.argv[1], sys.argv[2]
try:
    r = json.load(open(d + "/transcribe-report.json"))
except OSError:
    print(f"{d.rsplit('/',1)[1]}: rc={rc} NO REPORT"); sys.exit()
print(f"{d.rsplit('/',1)[1]}: rc={rc} status={r['status']} "
      f"sha={hashlib.sha256(r['transcript'].encode()).hexdigest()[:16]} "
      f"transport={r['ane']['session'].get('transport')} "
      f"open_ms={r['ane']['session']['open_ms']} "
      f"cpu={r['execution']['cpu_tensor_events']} "
      f"failed={[c['check'] for c in r['verification']['checks'] if not c['pass']]}")
PYEOF
}

pkill -f '[m]lx-omarchy-ane-worker --daemon'; sleep 1
dev_state "baseline"

echo "### 1. kill -9 client mid-run"
start_daemon 600000
transcribe warm0
for delay in 0.55 0.65 0.75; do
  MLX_OMARCHY_PK_KEEP_WORKER=1 MLX_OMARCHY_ANE_SOCK="$SOCK" \
    "$PY" "$CLI" transcribe -o "$OUT/victim-$delay" >"$OUT/victim-$delay.log" 2>&1 &
  victim=$!
  sleep "$delay"; kill -9 "$victim"; wait "$victim" 2>/dev/null
  echo "killed client at ${delay}s"
  transcribe "after-client-kill-$delay"
done
tail -3 "$OUT/daemon.log"
dev_state "after client kills"

echo "### 2. kill -9 daemon mid-run"
dpid=$(cat "$SOCK.pid")
MLX_OMARCHY_PK_KEEP_WORKER=1 MLX_OMARCHY_ANE_SOCK="$SOCK" \
  "$PY" "$CLI" transcribe -o "$OUT/victim-daemon" >"$OUT/victim-daemon.log" 2>&1 &
victim=$!
sleep 0.65; kill -9 "$dpid"; wait "$victim"; echo "client rc after daemon kill: $?"
sleep 2
dev_state "after daemon kill -9"
echo "stale socket present: $([ -S "$SOCK" ] && echo yes || echo no)"
transcribe "fallback-after-daemon-kill"
start_daemon 600000
transcribe "restart-1"
transcribe "restart-2"

echo "### 3. idle exit"
pkill -f '[m]lx-omarchy-ane-worker --daemon'; sleep 1
start_daemon 3000
transcribe "idle-pre"
sleep 5
echo "daemon alive after idle window: $(ps -eo cmd | grep -c '[m]lx-omarchy-ane-worker --daemon')"
dev_state "after idle exit"
tail -4 "$OUT/daemon.log"
