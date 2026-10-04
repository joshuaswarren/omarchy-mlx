#!/usr/bin/env bash
# Fuse2B W1c: rebuild the wheel with the pipeline registration (7fa356dd8),
# restage the cand venv, apply the conv-delta patcher, run the bitcheck.
set -euo pipefail
OUT=/var/tmp/fuse2b/w1c
mkdir -p "$OUT"
test ! -e /var/tmp/JW16_MAINTENANCE
uptime_s=$(cut -d. -f1 /proc/uptime); test "$uptime_s" -ge 360 || { echo "boot settle gate: ${uptime_s}s"; exit 1; }
load1=$(cut -d" " -f1 /proc/loadavg)
li=${load1%%.*}; lf=${load1#*.}00
psi=$(cat /proc/pressure/cpu)
case "$psi" in *"some avg10=0.00"*) ;; *) echo "PSI gate: $psi"; exit 1;; esac
# Build + bit-identity window: timing comes only from the A/B windows, so the
# strict load<0.5 timing gate does not apply here; PSI must still be flat and
# the load must be under 5 (the 40G btrfs-cleaner after Jw16Cleaner keeps the
# 1-min load above 0.5 for long stretches).
(( $((10#$li * 100 + 10#${lf:0:2})) < 500 )) || { echo "load gate: $load1"; exit 1; }
printf "host=%s boot=%s uptime_s=%s load1=%s psi=%s\n" "$(hostname)" \
  "$(cat /proc/sys/kernel/random/boot_id)" "$uptime_s" "$load1" "$psi" > "$OUT/gates.txt"

BUILD=/var/tmp/fuse2b-build
CAND=/var/tmp/fuse2b-cand
SERVING=/var/tmp/v072-venv-fused

echo "== rebuild wheel =="
rm -rf "$CAND"
cp -a "$SERVING" "$CAND"
cd "$BUILD"
DEV_RELEASE=1 MLX_OMARCHY_SOURCE_COMMIT=7fa356dd8 \
  MLX_OMARCHY_WHOLE_BUNDLE_DIR=/var/tmp/encoder-whole/bundle \
  bash scripts/build-wheel.sh > "$OUT/build-full.log" 2>&1 || {
  echo "BUILD FAILED - full log in $OUT/build-full.log"
  grep -B3 -A15 -i -E "error|failed" "$OUT/build-full.log" | head -80
  exit 1
}
tail -3 "$OUT/build-full.log"
"$CAND/bin/python" -m pip install --force-reinstall --no-deps -q "$BUILD"/dist/*.whl
"$CAND/bin/python" -m pip list 2>/dev/null | grep -i mlx-omarchy

echo "== stamps =="
"$SERVING/bin/python" -m pip show mlx-omarchy | sed -n 2p > "$OUT/serving-stamp.txt"
"$CAND/bin/python" -m pip show mlx-omarchy | sed -n 2p > "$OUT/cand-stamp.txt"
cat "$OUT/serving-stamp.txt" "$OUT/cand-stamp.txt"
grep -q 7fa356dd8 "$OUT/cand-stamp.txt" || { echo "CAND STAMP MISSING COMMIT"; exit 1; }
cmp -s "$OUT/serving-stamp.txt" "$OUT/cand-stamp.txt" && { echo "STAMP COLLISION"; exit 1; }

echo "== apply conv-delta patcher =="
python3 "$BUILD/scripts/patch-mlx-lm-gdn-conv-delta.py" "$CAND"
"$CAND/bin/python" - <<'PYEOF'
import importlib, inspect
mod = importlib.import_module("mlx_lm.models.qwen3_5")
assert "MLX_OMARCHY_GDN_CONV_DELTA" in inspect.getsource(mod), "patch not loaded"
import mlx.core as mx
assert hasattr(mx.fast, "gdn_conv_delta_update"), "fast primitive missing"
print("patch+primitive present; mlx", mx.__version__)
PYEOF

echo "== provenance =="
"$CAND/bin/python" "$BUILD/scripts/mlx_provenance.py" > "$OUT/provenance-cand.txt" 2>&1 || true
tail -2 "$OUT/provenance-cand.txt"

echo "== captured-operand bitcheck =="
"$CAND/bin/python" "$BUILD/tools/fuse2b/gdn_conv_delta_bitcheck.py" "$OUT/bitcheck.json" 2>&1 | tail -8

sudo journalctl --flush; sync
echo "post-window llm-inference: $(systemctl is-active llm-inference || true)"
