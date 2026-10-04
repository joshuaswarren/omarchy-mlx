#!/usr/bin/env bash
# On-box release-wheel build + serving deploy loop (survives route flaps).
set -uo pipefail
LOG=/var/tmp/dfuse/rel-build.log
for attempt in 1 2 3 4 5 6; do
  echo "=== rel build attempt $attempt $(date -u +%FT%TZ)" > "$LOG"
  if cd /var/tmp/dfuse-build && bash /var/tmp/appbar/build_wheel_venv.sh \
      /var/tmp/dfuse-build dfuse.d86ea8815 \
      /var/tmp/dfuse-cand /var/tmp/v072-venv-fused >> "$LOG" 2>&1; then
    WHEEL=$(ls /var/tmp/dfuse-build/dist/*dfuse.d86ea8815*.whl | head -1)
    sha256sum "$WHEEL" >> "$LOG"
    bash /var/tmp/appbar/deploy_wheel.sh "$WHEEL" /var/tmp/v072-venv-fused >> "$LOG" 2>&1
    python3 /var/tmp/dfuse/patch-mlx-lm-qwen3-rope-norm.py /var/tmp/v072-venv-fused >> "$LOG" 2>&1
    python3 /var/tmp/dfuse/patch-mlx-lm-gdn-raw-repeat.py /var/tmp/v072-venv-fused >> "$LOG" 2>&1
    python3 - <<'PYEOF' >> "$LOG" 2>&1
f = "/var/tmp/v072-venv-fused/lib/python3.14/site-packages/mlx_lm/models/gated_delta.py"
s = open(f).read()
old = 'os.environ.get("MLX_OMARCHY_GDN_RAW_REPEAT", "0") == "1"'
new = 'os.environ.get("MLX_OMARCHY_GDN_RAW_REPEAT", "1") == "1"'
if old in s:
    open(f, "w").write(s.replace(old, new, 1))
    print("gdn default flipped ON")
else:
    print("gdn default already ON" if new in s else "GDN ANCHOR MISSING")
PYEOF
    /var/tmp/v072-venv-fused/bin/python -c "import mlx.core as mx; print('serving:', mx.__version__)" >> "$LOG" 2>&1
    echo "REL-COMPLETE" >> "$LOG"
    exit 0
  fi
  echo "attempt $attempt failed" >> "$LOG"
  sleep 60
done
echo "REL-EXHAUSTED" >> "$LOG"
exit 1
