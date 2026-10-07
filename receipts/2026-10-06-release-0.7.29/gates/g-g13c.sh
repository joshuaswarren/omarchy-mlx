#!/usr/bin/env bash
# g-g13c — on the G13C (M1 Max) host, post-cluster window.
# Verifies the CPU PD hold GATE on this chip: OFF by default (even with full
# write access — the probe runs as root so permissions cannot mask the gate),
# and forced-on engages and releases. No udev rule is installed by this gate
# (the permission path was verified separately; see the phase-A receipt).
# GPU work runs under the host's GPU lock.
#
# Usage (on the host):
#   G13C_WHEEL=/tmp/v0.7.29-assets/mlx_omarchy-*-cp314-cp314-linux_aarch64.whl \
#     bash g-g13c.sh
# Expected PASS output:
#   G13C_DEVICE Apple M1 Max (G13C C0)
#   G13C_GATE default during=0 after=0
#   G13C_GATE forced during>=1 after=0
#   G13C_PASS
# Any FAIL line: stop and report (the tag never moves).
set -euo pipefail
WHEEL="${G13C_WHEEL:?set G13C_WHEEL to the release wheel path on this host}"
# Host-detected GPU lock: M2-class hosts use /tmp/m2-gpu.lock, the M1 family
# /tmp/m1-gpu.lock. An explicit GPU_LOCK always wins.
if [[ -z "${GPU_LOCK:-}" ]]; then
  if grep -aq "t6021\|t6011\|t6010" /proc/device-tree/compatible 2>/dev/null; then
    GPU_LOCK=/tmp/m2-gpu.lock
  else
    GPU_LOCK=/tmp/m1-gpu.lock
  fi
fi
command -v python3.14 >/dev/null || { echo "FAIL python3.14 missing"; exit 1; }

VENV="$(mktemp -d "${TMPDIR:-/tmp}/g13c-gate.XXXX")"
PROBE="$(mktemp "${TMPDIR:-/tmp}/g13c-probe.XXXX.py")"
cleanup() { rm -rf "$VENV" "$PROBE"; }
trap cleanup EXIT

python3.14 -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet "$WHEEL"

echo "== device =="
"$VENV/bin/python" - <<'EOF' | sed 's/^/G13C_DEVICE /'
import mlx.core as mx
info = mx.device_info()
print(info.get("marketing_name") or info)
EOF

cat >"$PROBE" <<'EOF'
import os, threading, time
import mlx.core as mx

def holds():
    n = 0
    for fd in os.listdir("/proc/self/fd"):
        try:
            if os.readlink("/proc/self/fd/" + fd) == "/dev/cpu_dma_latency":
                n += 1
        except OSError:
            pass
    return n

a = mx.random.normal((4096, 4096)).astype(mx.float16)
b = mx.random.normal((4096, 4096)).astype(mx.float16)
mx.eval(a, b)
seen = []
stop = False
def sampler():
    while not stop:
        seen.append(holds())
        time.sleep(0.02)
th = threading.Thread(target=sampler)
th.start()
t0 = time.time()
while time.time() - t0 < 1.5:
    mx.eval(a @ b)
during = max(seen) if seen else -1
time.sleep(1.0)
stop = True
th.join()
print("RESULT during=%d after=%d" % (during, holds()))
EOF

run_probe() { # $1: env value ("1") or empty for the default arm; runs as root
  if [[ -n "${1:-}" ]]; then
    sudo env MLX_OMARCHY_CPU_PD_HOLD="$1" "$VENV/bin/python" "$PROBE"
  else
    sudo "$VENV/bin/python" "$PROBE"
  fi
}

echo "== GPU probes under lock $GPU_LOCK (root: isolates the gate decision from permissions) =="

# flock runs a COMMAND, not a shell function: wrap the probe in bash -c.
out_default="$(flock "$GPU_LOCK" bash -c "sudo '$VENV/bin/python' '$PROBE'")"
echo "$out_default" | grep -q "RESULT during=0 after=0" \
  && echo "G13C_GATE default during=0 after=0" \
  || { echo "FAIL default arm: $out_default"; exit 1; }

out_forced="$(flock "$GPU_LOCK" bash -c "sudo env MLX_OMARCHY_CPU_PD_HOLD=1 '$VENV/bin/python' '$PROBE'")"
echo "$out_forced" | grep -qE "RESULT during=[1-9][0-9]* after=0" \
  && echo "G13C_GATE forced during>=1 after=0" \
  || { echo "FAIL forced arm: $out_forced"; exit 1; }

echo "G13C_PASS"
