#!/usr/bin/env bash
# GdnPrefill2 pre-push build check (dev box; no GPU needed).
# 1. Every omarchy_shader -DFOO for a GDN variant shader must appear in the
#    shader's preprocessor conditionals.
# 2. REAL shader compile: if glslc exists, every GDN variant shader compiles
#    with its exact CMake -D set (--target-env=vulkan1.3, -O). If glslc is
#    ABSENT, this falls back to brace/paren balance + preprocess-only and
#    PRINTS "NOT A COMPILE" — preprocess is not a compile.
# 3. C++ syntax check of the omarchy backend translation units the GDN
#    variants touch (primitives.cpp) against the prepared tree, with
#    minimal stub headers for the build-time embedded shader headers.
# Run from the repo root of the branch under test. Exits nonzero on any gap.
set -euo pipefail
ROOT="${1:-.}"
SHADERS="$ROOT/overlay/mlx/backend/omarchy/shaders"
OM="$ROOT/overlay/mlx/backend/omarchy"
WORK="$ROOT/.work/mlx"
fail=0

echo "== 1. CMake -D names vs shader conditionals (GDN variant shaders) =="
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

echo "== 2. Shader compile (glslc) per GDN variant + define combo =="
if command -v glslc >/dev/null 2>&1; then
  python3 - "$OM/CMakeLists.txt" "$SHADERS" <<'PY'
import re
import subprocess
import sys
import tempfile

cmake, sdir = sys.argv[1], sys.argv[2]
text = open(cmake).read()
pairs = re.findall(
    r"omarchy_shader\(\s*([\w]+)\s+shaders/([\w.]+)([^)]*)\)", text)
bad = 0
for name, fname, args in pairs:
    if not fname.startswith(("gated_delta", "gdn_stub")):
        continue
    flags = re.findall(r"-D\w+=\d+", args)
    with tempfile.NamedTemporaryFile(suffix=".spv") as out:
        cmd = ["glslc", "-O", "--target-env=vulkan1.3"] + flags + \
            [f"{sdir}/{fname}", "-o", out.name]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"COMPILE FAIL: {name} ({' '.join(flags) or 'no defines'})")
            print(r.stderr[:2000])
            bad += 1
        else:
            print(f"compile OK: {name} ({' '.join(flags) or 'no defines'})")
sys.exit(1 if bad else 0)
PY
  [ $? -eq 0 ] || fail=1
else
  echo "WARNING: glslc NOT FOUND — falling back to balance + preprocess. NOT A COMPILE."
  fail=1
  while IFS= read -r f; do
    o=$(grep -o '{' "$f" | wc -l); c=$(grep -o '}' "$f" | wc -l)
    op=$(grep -o '(' "$f" | wc -l); cp=$(grep -o ')' "$f" | wc -l)
    if [ "$o" -ne "$c" ] || [ "$op" -ne "$cp" ]; then
      echo "BALANCE FAIL: $f braces $o/$c parens $op/$cp"
      fail=1
    fi
    glslangValidator -E "$f" >/dev/null 2>&1 || { echo "PREPROCESS FAIL: $f"; fail=1; }
  done < <(ls "$SHADERS"/gated_delta_*.comp "$SHADERS"/gdn_stub_*.comp 2>/dev/null)
fi

echo "== 3. C++ syntax check (primitives.cpp) =="
if [ ! -d "$WORK" ]; then
  echo "SKIP: no prepared tree at $WORK (run scripts/prepare-mlx.sh on a scratch clone first)"
  exit $fail
fi
GEN=$(mktemp -d)
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
