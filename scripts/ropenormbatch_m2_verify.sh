#!/usr/bin/env bash
# RopeNormBatch post-reopen verification on the M2.
# ONE build at a time; run via gpu-turn ticket after Main reopens the box.
# Artifacts under /var/tmp/ropenormbatch-* (survives the reboots).
set -euo pipefail

cd /var/tmp/ropenormbatch-wheel

if pgrep -f '[c]c1plus|[c]make|[n]inja' > /dev/null; then
  echo "BUILD BUSY: another build is running; aborting without starting one." >&2
  exit 3
fi

echo "== prepare"
./scripts/prepare-mlx.sh

echo "== build (DEV_RELEASE=1)"
DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=3 \
  nice -n 19 ionice -c3 ./scripts/build-wheel.sh \
  > /var/tmp/ropenormbatch-build.log 2>&1
tail -4 /var/tmp/ropenormbatch-build.log

WHEEL=$(ls -t dist/mlx_omarchy-*.whl | head -1)
echo "== wheel: $WHEEL"

echo "== private venv f1 (base has mlx_lm 0.31.3 for the behavior pytest)"
BASE_VENV="${RNB_BASE_VENV:-$HOME/bench-qwen38-venv}"
if [ ! -d /var/tmp/ropenormbatch-venv-f1 ]; then
  cp -a "$BASE_VENV" /var/tmp/ropenormbatch-venv-f1
  /var/tmp/ropenormbatch-venv-f1/bin/python -m venv --upgrade /var/tmp/ropenormbatch-venv-f1
fi
/var/tmp/ropenormbatch-venv-f1/bin/python -m pip install --no-deps --force-reinstall -q "$WHEEL"
/var/tmp/ropenormbatch-venv-f1/bin/python -c 'import mlx.core as mx; print("f1 stamp", mx.__version__)'

echo "== op-level matrix: OLD wheel (b0) with hashes"
/var/tmp/ropenormbatch-venv-b0/bin/python /var/tmp/ropenormbatch-opcheck.py \
  > /var/tmp/ropenormbatch-opcheck-b0-v2.log 2>&1
grep -E "SUMMARY|PROVENANCE" /var/tmp/ropenormbatch-opcheck-b0-v2.log

echo "== op-level matrix: NEW wheel (f1)"
/var/tmp/ropenormbatch-venv-f1/bin/python /var/tmp/ropenormbatch-opcheck.py \
  > /var/tmp/ropenormbatch-opcheck-f1.log 2>&1
grep -E "SUMMARY|PROVENANCE" /var/tmp/ropenormbatch-opcheck-f1.log

echo "== B=1 fused-bit stability across wheels (q_hash old vs new, int rows)"
python3 - <<'PY'
import json
def rows(p):
    out = {}
    for line in open(p):
        line = line.strip()
        if line.startswith("{"):
            r = json.loads(line)
            out[(r["B"], r["L"], r["off"])] = r
    return out
b0 = rows("/var/tmp/ropenormbatch-opcheck-b0-v2.log")
f1 = rows("/var/tmp/ropenormbatch-opcheck-f1.log")
stable = eq = True
for k in sorted(b0):
    if "error" in b0[k]:
        continue
    if k not in f1 or "error" in f1[k]:
        stable = False
        print("MISSING", k)
        continue
    if b0[k].get("q_hash") != f1[k].get("q_hash"):
        stable = False
        print("HASH-CHANGE", k, b0[k].get("q_hash"), "->", f1[k].get("q_hash"))
for k, r in f1.items():
    if not (r.get("q_eq") and r.get("k_eq")):
        eq = False
        print("NOT-BIT-EXACT", k, r)
print("B1_FUSED_STABLE" if stable else "B1_FUSED_CHANGED")
print("ALL_BIT_EXACT" if eq else "BIT_EXACT_FAILURES")
PY

echo "== behavior pytest (end to end, real patcher)"
/var/tmp/ropenormbatch-venv-f1/bin/python -m pip install -q pytest 2>&1 | tail -1
cd /var/tmp/ropenormbatch-wheel
/var/tmp/ropenormbatch-venv-f1/bin/python -m pytest tests/test_rope_norm_batch_behavior.py -q \
  > /var/tmp/ropenormbatch-pytest-f1.log 2>&1
tail -4 /var/tmp/ropenormbatch-pytest-f1.log

echo "== C++ doctest: runs on the dev box (scripts/ropenormbatch_devbox_doctest.sh);"
echo "   wheel builds set MLX_BUILD_TESTS=OFF so it is not built here."

echo "== verify script done"
