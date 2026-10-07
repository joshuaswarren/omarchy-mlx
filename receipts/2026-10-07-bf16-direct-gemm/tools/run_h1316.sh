#!/bin/bash
# MatmulGap H13 (b) / H14 / H15 / H16 three-arm run on one chip.
# Usage: run_h1316.sh <icd.json> <round 1|2|3|final>
#   round 1: M C1 C2   round 2: C2 C1 M   round 3: M C1 C2   (each slot: metal_baseline + rowcheck)
#   final: identity per arm, mlxcheck2 per arm, family tests at C2.
# jw16: one ~/bin/gpu-turn -m 12 ticket per round; jwm1: H8_FLOCK=1 (whole call under the lock).
set -u
H=/var/tmp/w7q-h8
ICD=$1
ROUND=$2
declare -A VENV=([M]=venv-h13main [C1]=venv-c1 [C2]=venv-c2)
if [ "${H8_FLOCK:-0}" = 1 ]; then
  exec 9>/tmp/m1-gpu.lock
  flock -w 1800 9 || { echo LOCK-TIMEOUT; exit 2; }
fi
export VK_ICD_FILENAMES=$ICD MESA_SHADER_CACHE_DISABLE=1
echo "round $ROUND start $(date -u +%FT%TZ) so $(sha256sum "$(grep -o '"library_path": *"[^"]*"' "$ICD" | cut -d'"' -f4)" | cut -c1-16)"
slot() { # arm, rep
  local py=$H/${VENV[$1]}/bin/python
  echo "== $1 r$2 $(date -u +%T)"
  "$py" "$H/metal_baseline.py" > "$H/h1316-mb-$1-r$2.jsonl"
  "$py" "$H/rowcheck.py" > "$H/h1316-rc-$1-r$2.jsonl"
}
case $ROUND in
  1) slot M 1; slot C1 1; slot C2 1 ;;
  2) slot C2 2; slot C1 2; slot M 2 ;;
  3) slot M 3; slot C1 3; slot C2 3 ;;
  final)
    for a in M C1 C2; do
      "$H/${VENV[$a]}/bin/python" -c 'import mlx.core as mx; i = mx.device_info(); print(mx.__version__, i.get("device_name"), "bf16_8", i.get("cooperative_matrix_bf16_8"))'
      "$H/${VENV[$a]}/bin/python" "$H/mlxcheck2.py" > "$H/h1316-mlxcheck2-$a.json" 2>&1
    done
    python3 - "$H" <<'EOF'
import json, sys
h = sys.argv[1]
m = json.load(open(f"{h}/h1316-mlxcheck2-M.json"))["cases"]
for a in ("C1", "C2"):
    c = json.load(open(f"{h}/h1316-mlxcheck2-{a}.json"))["cases"]
    print(f"mlxcheck2 {a} vs M diffs: {[k for k in m if m[k]['sha'] != c[k]['sha']]} of {len(m)}")
EOF
    bin=$(find "$H/tbuild-c2" -name omarchy_matmul_family_tests -type f -perm -u+x | head -1)
    "$bin" > "$H/h1316-family-c2.log" 2>&1
    echo "family (C2) rc=$? $(grep -E '^\[doctest\] (test cases|assertions)' "$H/h1316-family-c2.log" | tr '\n' ' ')"
    ;;
  *) echo "usage: run_h1316.sh <icd> 1|2|3|final"; exit 2 ;;
esac
echo "round $ROUND end $(date -u +%FT%TZ)"
