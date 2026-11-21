#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
python3 -m venv "$tmp/venv"
py="$tmp/venv/bin/python"
if ! command -v patchelf >/dev/null 2>&1; then
  "$py" -m pip -q install patchelf
fi
if command -v patchelf >/dev/null 2>&1; then
  patchelf_bin=$(command -v patchelf)
else
  patchelf_bin="$tmp/venv/bin/patchelf"
fi
work="$tmp/chain"
mkdir -p "$work/licences"
printf 'synthetic test fixture: GPL-3.0 text\n' > "$work/licences/COPYING3"
printf 'synthetic test fixture: GCC Runtime Library Exception\n' > "$work/licences/COPYING.RUNTIME"
printf 'synthetic test fixture: LGPL-2.1 text\n' > "$work/licences/COPYING.LIB"
printf 'void fixture(void) {}\n' > "$work/f.c"
build_lib() { gcc -shared -fPIC -Wl,-soname,"$1" -o "$work/$1" "$work/f.c" "${@:2}"; }
build_lib libquadmath-e2e.so.0
build_lib libgfortran-e2e.so.5 -Wl,--no-as-needed -L"$work" -l:libquadmath-e2e.so.0
build_lib libgomp-e2e.so.1
build_lib libopenblas.so.0 -Wl,--no-as-needed -L"$work" -l:libgfortran-e2e.so.5 -l:libgomp-e2e.so.1
build_lib libmlx.so -Wl,--no-as-needed -L"$work" -l:libopenblas.so.0
wheel="$work/mlx-0.1-cp312-cp312-linux_x86_64.whl"
python3 - "$wheel" "$work/libmlx.so" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1], 'w') as z:
    z.write(sys.argv[2], 'mlx/lib/libmlx.so')
    z.writestr('mlx-0.1.dist-info/RECORD', 'mlx-0.1.dist-info/RECORD,,\n')
PY
"$py" "$root/scripts/openblas_isolation.py" vendor "$wheel" --openblas "$work/libopenblas.so.0" --patchelf "$patchelf_bin"
"$py" "$root/scripts/openblas_isolation.py" check "$wheel"
mkdir -p "$work/install/mlx/lib"
python3 - "$wheel" "$work/install/mlx/lib" <<'PY'
import sys, zipfile, pathlib
with zipfile.ZipFile(sys.argv[1]) as z:
  for n in z.namelist():
    if n.startswith('mlx/lib/') and not n.endswith('/'):
      (pathlib.Path(sys.argv[2])/pathlib.Path(n).name).write_bytes(z.read(n))
PY
blas=$(find "$work/install/mlx/lib" -name 'libopenblas-*.so.0' -print -quit)
"$patchelf_bin" --remove-rpath "$blas"
if "$py" - "$work/install/mlx/lib/libmlx.so" <<'PY'
import ctypes, sys
ctypes.CDLL(sys.argv[1])
PY
then
  echo 'FAIL: import unexpectedly succeeded without OpenBLAS RUNPATH' >&2
  exit 1
fi
"$patchelf_bin" --add-rpath '$ORIGIN' "$blas"
"$py" - "$work/install/mlx/lib/libmlx.so" <<'PY'
import ctypes, sys
ctypes.CDLL(sys.argv[1])
PY
echo 'PASS: missing RUNPATH reproduced; vendor/check and repaired transitive chain load successfully'
