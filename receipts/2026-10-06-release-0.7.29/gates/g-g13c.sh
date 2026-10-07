#!/usr/bin/env bash
# g-g13c — on the G13C (M1 Max) host, post-cluster window.
# Verifies the CPU PD hold GATE on this chip: OFF by default (even with full
# write access — the probe runs as root so permissions cannot mask the gate),
# and forced-on engages and releases. No udev rule is installed by this gate
# (the permission path was verified separately; see the phase-A receipt).
#
# w7K review fixes: G1 probes run as direct commands (no function exec), G2
# the chip is asserted from the backend's own device info (must contain
# G13C), G3 gpu-turn tickets when present with no inner flock of the same
# lock, G4 every line is teed to a receipt file, probes wait 2 s before the
# release check and run under `timeout -k 30 120`.
#
# Usage (on the host):
#   G13C_WHEEL=/tmp/v0.7.29-assets/mlx_omarchy-*-cp314-cp314-linux_aarch64.whl \
#     bash g-g13c.sh
# Expected PASS output (and the same lines in the receipt file):
#   G13C_DEVICE device:            Apple M1 Max (G13C C0)
#   G13C_CHIP_OK G13C
#   G13C_GATE default during=0 after=0
#   G13C_GATE forced during>=1 after=0
#   G13C_PASS
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
RECEIPT="${G13C_RECEIPT:-$PWD/g-g13c-receipt-$(date -u +%Y%m%dT%H%M%SZ).log}"
: > "$RECEIPT"
log() { tee -a "$RECEIPT"; }
command -v python3.14 >/dev/null || { echo "FAIL python3.14 missing" | log; exit 1; }

VENV="$(mktemp -d "${TMPDIR:-/tmp}/g13c-gate.XXXX")"
PROBE="$(mktemp "${TMPDIR:-/tmp}/g13c-probe.XXXX.py")"
cleanup() { rm -rf "$VENV" "$PROBE"; }
trap cleanup EXIT

python3.14 -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet "$WHEEL"

echo "== device (from the wheel's own mlx-omarchy-info) ==" | log
INFO="$("$VENV/bin/python" - <<'EOF'
import os, mlx
print(next(p for root in mlx.__path__
           if os.access(p := os.path.join(root, "bin", "mlx-omarchy-info"), os.X_OK)))
EOF
)"
DEVLINE="$("$INFO" | grep '^  device:')" | log
echo "$DEVLINE" | log
echo "$DEVLINE" | grep -q "G13C" \
  && echo "G13C_CHIP_OK G13C" | log \
  || { echo "FAIL chip assertion: '$DEVLINE' lacks G13C" | log; exit 1; }

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
time.sleep(2.0)          # 2 s: strictly outside the documented 1 s release bound
stop = True
th.join()
print("RESULT during=%d after=%d" % (during, holds()))
EOF

# gpu-turn ticket when the host hands out GPU time that way; plain flock of
# the host lock only where gpu-turn is absent. Never both.
runner() {
  if command -v gpu-turn >/dev/null 2>&1; then
    gpu-turn -m 2 timeout -k 30 120 "$@"
  else
    timeout -k 30 120 flock "$GPU_LOCK" "$@"
  fi
}

echo "== GPU probes (root: isolates the gate decision from permissions) ==" | log
out_default="$(runner sudo "$VENV/bin/python" "$PROBE" 2>&1 | tee -a "$RECEIPT")"
echo "$out_default" | grep -q "RESULT during=0 after=0" \
  && echo "G13C_GATE default during=0 after=0" | log \
  || { echo "FAIL default arm: $out_default" | log; exit 1; }

out_forced="$(runner sudo env MLX_OMARCHY_CPU_PD_HOLD=1 "$VENV/bin/python" "$PROBE" 2>&1 | tee -a "$RECEIPT")"
echo "$out_forced" | grep -qE "RESULT during=[1-9][0-9]* after=0" \
  && echo "G13C_GATE forced during>=1 after=0" | log \
  || { echo "FAIL forced arm: $out_forced" | log; exit 1; }

echo "G13C_PASS receipt=$RECEIPT" | log
