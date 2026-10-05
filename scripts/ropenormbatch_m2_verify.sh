#!/usr/bin/env bash
# RopeNormBatch golden-flow verification on the M2.
# Clone the golden tree, transplant the fix delta at the pushed wip sha,
# full rebuild (the patch change forces prepare -> no warm objects),
# golden venv clone + fix wheel, then the op matrix and the behavior pytest.
set -euo pipefail

MYSHA=885509f
WHEELDIR=/var/tmp/rnb-golden-tree
VENV=/var/tmp/rnb-venv

if pgrep -f '[c]c1plus|[c]make|[n]inja' > /dev/null; then
  echo "BUILD BUSY: another build is running; aborting without starting one." >&2
  exit 3
fi

echo "== clone golden tree"
rm -rf "$WHEELDIR" 2>/dev/null || true
bash /var/tmp/golden-clone-tree.sh "$WHEELDIR"

echo "== checkout the wip sha (carries the whole delta)"
cd "$WHEELDIR"
git fetch origin "refs/heads/agent/rope-norm-batch-wip:refs/heads/wip" 2>&1 | tail -1
git checkout -q -f wip && git status --short | head -3 || true
git rev-parse --short=7 HEAD
test "$(git rev-parse --short=7 HEAD)" = "$MYSHA" || { echo "FATAL: sha mismatch" >&2; exit 4; }
grep -q "norm_weight ? default_inv_freqs" patches/mlx-rope-rms-norm.patch
grep -q "serves per-batch array offsets" overlay/tests/omarchy/test_fast_ops.cpp
echo "delta ok"

echo "== prepare + build (DEV_RELEASE=1, nice 10, -j4)"
./scripts/prepare-mlx.sh
export MLX_OMARCHY_WHOLE_BUNDLE_DIR="${MLX_OMARCHY_WHOLE_BUNDLE_DIR:-$(cat /var/tmp/.golden-status/bundledir 2>/dev/null || true)}"
DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 \
  nice -n 10 ./scripts/build-wheel.sh \
  > /var/tmp/rnb-build.log 2>&1
tail -3 /var/tmp/rnb-build.log
WHEEL=$(ls -t dist/mlx_omarchy-*.whl | head -1)
echo "wheel: $WHEEL"
sha256sum "$WHEEL"

echo "== golden venv clone + fix wheel"
rm -rf "$VENV" 2>/dev/null || true
bash /var/tmp/golden-clone-venv.sh "$VENV"
"$VENV/bin/python" -m pip install --no-deps --force-reinstall -q "$WHEEL"
"$VENV/bin/python" -c 'import mlx.core as mx; print("f1 stamp", mx.__version__)'

echo "== op-level matrix: OLD wheel (b0, hash-pinned)"
/var/tmp/ropenormbatch-venv-b0/bin/python scripts/rope_norm_batch_opcheck.py \
  > /var/tmp/rnb-opcheck-b0.log 2>&1
grep -E "SUMMARY|PROVENANCE" /var/tmp/rnb-opcheck-b0.log

echo "== op-level matrix: NEW wheel (fix)"
"$VENV/bin/python" /var/tmp/rnb-delta/scripts/rope_norm_batch_opcheck.py \
  > /var/tmp/rnb-opcheck-f1.log 2>&1
grep -E "SUMMARY|PROVENANCE" /var/tmp/rnb-opcheck-f1.log

echo "== cross-wheel B=1 fused-bit stability + bit-exactness"
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
b0 = rows("/var/tmp/rnb-opcheck-b0.log")
f1 = rows("/var/tmp/rnb-opcheck-f1.log")
stable = eq = True
for k in sorted(b0):
    if "error" in b0[k]:
        continue
    if k not in f1 or "error" in f1[k]:
        stable = False; print("MISSING", k); continue
    if b0[k].get("q_hash") != f1[k].get("q_hash"):
        stable = False; print("HASH-CHANGE", k, b0[k].get("q_hash"), "->", f1[k].get("q_hash"))
for k, r in f1.items():
    if "error" in r:
        eq = False; print("ERROR-ROW", k, r["error"]); continue
    if not (r.get("q_eq") and r.get("k_eq")):
        eq = False; print("NOT-BIT-EXACT", k, r)
print("B1_FUSED_STABLE" if stable else "B1_FUSED_CHANGED")
print("ALL_BIT_EXACT" if eq else "BIT_EXACT_FAILURES")
PY

echo "== behavior pytest (end to end, real patcher, array offsets)"
"$VENV/bin/python" -m pip install -q pytest 2>&1 | tail -1
"$VENV/bin/python" -m pytest tests/test_rope_norm_batch_behavior.py -q \
  > /var/tmp/rnb-pytest-f1.log 2>&1
tail -4 /var/tmp/rnb-pytest-f1.log

echo "== verify script done"
