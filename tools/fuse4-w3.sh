#!/bin/bash
# DecodeFuse4 W3 window: F4+F6 wheel (1248d88fe) + F3 flag — isos, whole-model
# bitcheck, combined paired cells, F6 adoption trace.
# INVOCATION: bash /var/tmp/appbar/gpuwin.sh 'bash /var/tmp/fuse4/w3.sh'
set -u
OUT=/var/tmp/fuse4/combined; mkdir -p "$OUT"
SERVE=/var/tmp/v072-venv-fused/bin/python3
CAND=/var/tmp/fuse4-venv/bin/python3

echo "=== qknorm iso (candidate) ==="
env X=1 "$CAND" /var/tmp/fuse4/dg_bitcheck_qknorm.py "$OUT/qknorm-iso.json" \
  > "$OUT/qknorm-iso.log" 2>&1
echo "qknorm-iso rc=$? $(head -1 "$OUT/qknorm-iso.log")"

echo "=== rope-norm iso v3 (candidate; F3 path on the combined wheel) ==="
env X=1 "$CAND" /var/tmp/fuse3/rope_norm_bitcheck3.py "$OUT/rope-norm-iso.json" \
  > "$OUT/rope-norm-iso.log" 2>&1
echo "rope-iso rc=$? $(head -1 "$OUT/rope-norm-iso.log")"

echo "=== whole-model bitcheck serve vs cand ==="
env X=1 "$SERVE" /var/tmp/dg/dg_bitcheck.py "$OUT/bit-serve.json" \
  > "${OUT}/bit-serve.log" 2>&1; echo "bit-serve rc=$?"
env X=1 "$CAND" /var/tmp/dg/dg_bitcheck.py "$OUT/bit-cand.json" \
  > "${OUT}/bit-cand.log" 2>&1; echo "bit-cand rc=$?"
python3 - "$OUT" <<'PYEOF'
import json, sys
out = sys.argv[1]
def leaves(d):
    rows = json.load(open(d)).get("rows", {})
    return {k: v for k, v in rows.items() if isinstance(v, str)}
s, c = leaves(f"{out}/bit-serve.json"), leaves(f"{out}/bit-cand.json")
diff = {k for k in set(s) | set(c) if s.get(k) != c.get(k)}
print("GATE-FULL", "PASS" if not diff else "FAIL", f"rows={len(s)}")
for k in sorted(diff)[:10]:
    print(" ", k, s.get(k), "vs", c.get(k))
PYEOF

echo "=== combined paired cells (both flags) ==="
bash /var/tmp/fuse4/combined.sh
