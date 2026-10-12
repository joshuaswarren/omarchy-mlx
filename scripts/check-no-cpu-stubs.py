#!/usr/bin/env python3
"""check-no-cpu-stubs: every eval_cpu implementation the tree compiles must have a NO_CPU stub for CPU-off builds.

A primitive with an eval_cpu body in mlx/backend/cpu but no NO_CPU stub in mlx/backend/no_cpu/primitives.cpp links
with MLX_BUILD_CPU=ON and fails with MLX_BUILD_CPU=OFF (undefined vtable/typeinfo). The FloorDivide backport
(omarchy-mlx#156) added eval_cpu + NO_GPU(FloorDivide) and no NO_CPU(FloorDivide).

The check scans a prepared MLX source tree (scripts/prepare-mlx.sh output: patches applied, overlay not yet copied):

  check-no-cpu-stubs.py <mlx-source-dir>

It reports every class with a `ClassName::eval_cpu` definition in backend/cpu that lacks a matching
`NO_CPU(ClassName)` / `NO_CPU_MULTI(ClassName)` stub, ignoring fast:: helpers (ConvertFP8, Quantize: separate
translation units that link). Exit 0 when clean, 1 with the missing names.

Usage:
  check-no-cpu-stubs.py /path/to/.work/mlx
Exit: 0 clean, 1 missing stubs / unreadable inputs (fail-closed).
"""
import re
import sys
from pathlib import Path

EVAL_RE = re.compile(r"^void ([A-Za-z0-9_]+)::eval_cpu", re.M)
STUB_RE = re.compile(r"NO_CPU(?:_MULTI)?\(([A-Za-z0-9_]+)\)")
# fast:: helpers live in their own translation units and link without stubs.
# Compiled::eval_cpu takes (inputs, outputs) and is defined in compiled.cpp
# itself, so it never needs a stub either.
IGNORE = {"ConvertFP8", "Quantize", "Compiled", "func"}


def die(msg):
    sys.stderr.write("no-cpu-stubs: %s\n" % msg)
    sys.exit(1)


def main():
    if len(sys.argv) != 2:
        die("usage: check-no-cpu-stubs.py <mlx-source-dir>")
    root = Path(sys.argv[1])
    cpu_dir = root / "mlx" / "backend" / "cpu"
    stub_file = root / "mlx" / "backend" / "no_cpu" / "primitives.cpp"
    if not cpu_dir.is_dir():
        die("no such directory: %s" % cpu_dir)
    if not stub_file.is_file():
        die("no such file: %s" % stub_file)
    impl = set()
    for path in sorted(cpu_dir.rglob("*.cpp")):
        try:
            text = path.read_text(errors="replace")
        except OSError as e:
            die("cannot read %s: %s" % (path, e))
        impl.update(EVAL_RE.findall(text))
    stubs = set(STUB_RE.findall(stub_file.read_text(errors="replace")))
    missing = sorted(n for n in impl - stubs if n not in IGNORE)
    if missing:
        die("eval_cpu without NO_CPU stub: %s" % ", ".join(missing))
    print("no-cpu-stubs: ok (%d impls, %d stubs)" % (len(impl), len(stubs)))


if __name__ == "__main__":
    main()
