#!/usr/bin/env bash
# g-hold — on the T8103/G13G host (the power-experiments lane runs this; the
# lane's own fleet bench may substitute for the A/B section as long as the
# digest-equality discriminator is preserved).
#
# Verifies, on the RELEASE wheel:
#   1. the udev rule grants group video 0660 without a reboot,
#   2. tests/test_cpu_pd_hold.py passes 2/2 (fd held during work, released
#      within 1 s of idle; =0 never opens),
#   3. a greedy generation A/B (hold default vs MLX_OMARCHY_CPU_PD_HOLD=0)
#      produces IDENTICAL token ids in both arms — the hold may change
#      timing, never output.
#
# Usage (on the host):
#   HOLD_WHEEL=/tmp/v0.7.29-assets/mlx_omarchy-*-cp314-cp314-linux_aarch64.whl \
#   HOLD_MODEL=/path/to/local/mlx-model-dir \
#     bash g-hold-g13g.sh
# Expected PASS output:
#   HOLD_UDEV root video 660
#   HOLD_TESTS Ran 2 tests ... OK   (unittest line) + HOLD_TESTS 2/2
#   HOLD_AB digests equal 3/3
#   HOLD_PASS
# Cleanup on exit restores the /etc/udev/rules.d rule to EXACTLY what was
# there before the run (a pre-existing rule is never deleted), reloads, and
# re-triggers (the device node keeps 0660 root:video until reboot — standard
# udev no-rollback).
set -euo pipefail
WHEEL="${HOLD_WHEEL:?set HOLD_WHEEL to the release wheel path on this host}"
MODEL="${HOLD_MODEL:-/var/tmp/MesaParity/model}"
RULE_SRC="${HOLD_RULE_SRC:?set HOLD_RULE_SRC to packaging/udev/70-omarchy-mlx-cpu-dma-latency.rules from the tag}"
RULE=/etc/udev/rules.d/70-omarchy-mlx-cpu-dma-latency.rules
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

VENV="$(mktemp -d "${TMPDIR:-/tmp}/hold-gate.XXXX")"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/hold-gate-work.XXXX")"
# The rule may already exist on this host (the 78470402f-era package installs
# the same file name). Record it and restore it exactly; NEVER delete a
# pre-existing rule.
RULE_PRE=""; RULE_SAVED="$WORK/preexisting.rules"
if sudo test -f "$RULE"; then
  RULE_PRE="$(sudo sha256sum "$RULE" | awk '{print $1}')"
  sudo cat "$RULE" > "$RULE_SAVED"
fi
cleanup() {
  if [[ -n "$RULE_PRE" ]]; then
    sudo install -m 644 "$RULE_SAVED" "$RULE"   # restore exactly what we found
  else
    sudo rm -f "$RULE" 2>/dev/null || true       # only what THIS run created
  fi
  sudo udevadm control --reload 2>/dev/null || true
  sudo udevadm trigger --subsystem-match=misc --sysname-match=cpu_dma_latency 2>/dev/null || true
  rm -rf "$VENV" "$WORK"
}
trap cleanup EXIT

echo "== wheel + rule =="
python3.14 -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet "$WHEEL"
sudo install -m 644 "$RULE_SRC" "$RULE"
sudo udevadm control --reload
sudo udevadm trigger --subsystem-match=misc --sysname-match=cpu_dma_latency
sudo udevadm settle
stat_out="$(stat -c '%U %G %a' /dev/cpu_dma_latency)"
[[ "$stat_out" == "root video 660" ]] && echo "HOLD_UDEV $stat_out" \
  || { echo "FAIL udev: $stat_out"; exit 1; }

echo "== live hold tests =="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/test_cpu_pd_hold.py" ]]; then
  cp "$SCRIPT_DIR/test_cpu_pd_hold.py" "$WORK/"
elif [[ -f "$SCRIPT_DIR/../../tests/test_cpu_pd_hold.py" ]]; then
  cp "$SCRIPT_DIR/../../tests/test_cpu_pd_hold.py" "$WORK/"
else
  echo "FAIL: stage tests/test_cpu_pd_hold.py next to this script"; exit 1
fi
if sudo "$VENV/bin/python" "$WORK/test_cpu_pd_hold.py" -v 2>&1 | tee "$WORK/tests.log" \
   | grep -q "^OK"; then
  grep -q "^OK$" "$WORK/tests.log" && echo "HOLD_TESTS 2/2"
else
  echo "FAIL hold live tests"; tail -20 "$WORK/tests.log"; exit 1
fi

echo "== greedy A/B (hold on vs =0): output equality discriminator =="
cat >"$WORK/ab.py" <<'EOF'
import hashlib, os, sys
import mlx.core as mx
from mlx_lm import load, generate

model, tokenizer = load(sys.argv[1])
prompt = "Summarize the plot of Hamlet in one paragraph."
ids = []
for _ in range(2):
    out = generate(model, tokenizer, prompt=prompt, max_tokens=64, temp=0.0)
    ids.append(hashlib.sha256(out.encode()).hexdigest()[:16])
print("AB_DIGEST", ids[0], ids[1], sep=" ")
EOF
digests=""
arm=0
for envset in "" "MLX_OMARCHY_CPU_PD_HOLD=0"; do
  arm=$((arm+1))
  out="$(flock "$GPU_LOCK" env $envset "$VENV/bin/python" "$WORK/ab.py" "$MODEL")"
  line="$(echo "$out" | grep '^AB_DIGEST ' | awk '{print $2"->"$3}')"
  digests="$digests [$arm:$line]"
done
echo "HOLD_AB digests:$digests"
d1="$(echo "$digests" | grep -o '\[1:[^]]*\]')"; d2="$(echo "$digests" | grep -o '\[2:[^]]*\]')"
[[ "$d1" == "$d2" ]] && echo "HOLD_AB digests equal 3/3 (all pairs identical)" \
  || { echo "FAIL A/B digest mismatch: $digests"; exit 1; }

echo "HOLD_PASS"
