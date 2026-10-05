#!/usr/bin/env bash
# GdnPrefill2 pre-push build check (dev box; no GPU needed).
# 1. Every omarchy_shader -DFOO for a stub/variant shader must appear in the
#    shader's preprocessor conditionals (#if/#ifdef/#elif defined(FOO)).
# 2. C++ syntax check of the omarchy backend translation units that the GDN
#    variants touch (primitives.cpp) against the prepared tree, with minimal
#    stub headers generated for the build-time embedded shader headers.
# Run from the repo root of the branch under test. Exits nonzero on any gap.
set -euo pipefail
ROOT="${1:-.}"
SHADERS="$ROOT/overlay/mlx/backend/omarchy/shaders"
OM="$ROOT/overlay/mlx/backend/omarchy"
WORK="$ROOT/.work/mlx"
fail=0

echo "== 1. CMake -D names vs shader conditionals (GDN variant shaders) =="
# Collect omarchy_shader lines with their -D flags (multi-line aware).
python3 - "$OM/CMakeLists.txt" "$SHADERS" <<'PY'
import re
import sys

cmake, sdir = sys.argv[1], sys.argv[2]
text = open(cmake).read()
pairs = re.findall(
    r"omarchy_shader\(\s*([\w]+)\s+shaders/([\w.]+)([^)]*)\)", text)
watch = ("gated_delta", "gdn_stub")
bad = 0
for name, fname, args in pairs:
    if not fname.startswith(watch):
        continue
    defines = re.findall(r"-D(\w+)=", args)
    if not defines:
        continue
    src = open(f"{sdir}/{fname}").read()
    known = {a or b for a, b in re.findall(r"defined\((\w+)\)|(?:#if|#elif|ifdef)\s+(\w+)", src)}
    for d in defines:
        if d not in known:
            print(f"GAP: {name} ({fname}) defines -D{d} but {fname} never tests it")
            bad += 1
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] || fail=1

echo "== 2. C++ syntax check (primitives.cpp) =="
if [ ! -d "$WORK" ]; then
  echo "SKIP: no prepared tree at $WORK (run scripts/prepare-mlx.sh on a scratch clone first)"
  exit $fail
fi
GEN=$(mktemp -d)
# Minimal declarations for every embedded-shader header the TUs include.
for h in $(grep -rhoE '#include "([a-z0-9_]+)\.h"' "$OM/compute.cpp" | sed 's/#include "//; s/\.h"//' | sort -u); do
  if [ ! -f "$WORK/mlx/backend/omarchy/shaders/$h.h" ] && [ ! -f "$GEN/$h.h" ]; then
    n="$h"
    printf 'namespace {\nextern const unsigned char %s[];\nextern const long unsigned int %s_size;\n}\n' "$n" "$n" > "$GEN/$h.h"
  fi
done
g++ -fsyntax-only -std=c++20 \
  -I"$ROOT/overlay" -I"$WORK" -I"$GEN" \
  "$OM/primitives.cpp" 2>&1 | head -30
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  echo "SYNTAX FAIL (rc=$rc) — see above"
  fail=1
else
  echo "primitives.cpp: syntax OK"
fi
rm -rf "$GEN"
exit $fail
