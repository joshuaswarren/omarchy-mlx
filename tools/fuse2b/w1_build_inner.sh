#!/usr/bin/env bash
# Fuse2B W1: build the candidate wheel, stage the cand venv, run the
# captured-operand bitcheck. Runs INSIDE a gpuwin window (build is CPU-only;
# no GPU dispatches in this window).
set -euo pipefail
test ! -e /var/tmp/JW16_MAINTENANCE
uptime_s=$(cut -d. -f1 /proc/uptime); test "$uptime_s" -ge 360 || { echo "boot settle gate: ${uptime_s}s"; exit 1; }
OUT=/var/tmp/fuse2b/w1build
mkdir -p "$OUT"
{
printf "host=%s boot=%s uptime_s=%s load1=%s\n" "$(hostname)" "$(cat /proc/sys/kernel/random/boot_id)" "$uptime_s" "$(cut -d' ' -f1 /proc/loadavg)"
} > "$OUT/gates.txt"

BUILD=/var/tmp/fuse2b-build
CAND=/var/tmp/fuse2b-cand
SERVING=/var/tmp/v072-venv-fused
echo "== build wheel =="
bash /var/tmp/appbar/build_wheel_venv.sh "$BUILD" 888db4054 "$CAND" "$SERVING" 2>&1 | tail -6

echo "== stamp asserts =="
"$SERVING/bin/python" -m pip show mlx-omarchy | sed -n 2p > "$OUT/serving-stamp.txt"
"$CAND/bin/python" -m pip show mlx-omarchy | sed -n 2p > "$OUT/cand-stamp.txt"
cat "$OUT/serving-stamp.txt" "$OUT/cand-stamp.txt"
grep -q 888db4054 "$OUT/cand-stamp.txt" || { echo "CAND STAMP MISSING COMMIT"; exit 1; }
cmp -s "$OUT/serving-stamp.txt" "$OUT/cand-stamp.txt" && { echo "STAMP COLLISION"; exit 1; }

echo "== apply patch set to cand venv =="
bash "$BUILD/scripts/apply-mlx-lm-patches.sh" "$CAND" 2>&1 | tail -20
"$CAND/bin/python" - <<'PYEOF'
import importlib, inspect, py_compile
mod = importlib.import_module("mlx_lm.models.qwen3_5")
src = inspect.getsource(mod)
assert "MLX_OMARCHY_GDN_CONV_DELTA" in src, "conv-delta patch not loaded"
import mlx.core as mx
assert hasattr(mx.fast, "gdn_conv_delta_update"), "fast primitive missing from wheel"
print("patch+primitive present; mlx", mx.__version__)
PYEOF

echo "== provenance =="
"$CAND/bin/python" "$BUILD/scripts/mlx_provenance.py" > "$OUT/provenance-cand.txt" 2>&1 || true
tail -2 "$OUT/provenance-cand.txt"

echo "== captured-operand bitcheck =="
"$CAND/bin/python" "$BUILD/tools/fuse2b/gdn_conv_delta_bitcheck.py" "$OUT/bitcheck.json" 2>&1 | tail -8

sudo journalctl --flush; sync
svc=$(systemctl is-active llm-inference || true)
echo "post-window llm-inference: $svc"
