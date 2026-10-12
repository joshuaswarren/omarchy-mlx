"""Issue #156 regression: every class that gains an eval_cpu body in the patch
series must also gain a cpu-off stub in the same patch.

The 2026-10-12 report: patches/mlx-floor-divide.patch added NO_GPU(FloorDivide)
but no NO_CPU(FloorDivide), so MLX_BUILD_CPU=OFF failed to link with an
undefined vtable. The runtime CI gate builds with the CPU backend on, so it
never sees that configuration.

This test parses every patches/*.patch for a C++ member definition matching
`X::eval_cpu(` and requires the same patch to add a `NO_CPU[_MULTI](X)` line.
Classes whose bodies live outside the series (pristine upstream files,
fast:: helpers with CPU-kept bodies, Compiled with its own no_cpu file) are
not the series' responsibility and are not checked here.
"""

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PATCHES = REPO / "patches"

EVAL_DEF = re.compile(r"^\+(?:void\s+)?([A-Za-z_][\w:]*)\s*::\s*eval_cpu\s*\(")
STUB_ADD = re.compile(r"^\+NO_CPU(?:_MULTI)?\(([^)]+)\)")


def patch_coverage(path: Path):
    target = None
    defined, stubbed = set(), set()
    for line in path.read_text().splitlines():
        if line.startswith("+++ b/"):
            target = line[6:].split("\t")[0]
            continue
        m = EVAL_DEF.match(line)
        if m and target and not target.startswith("tests/"):
            defined.add(m.group(1))
        m = STUB_ADD.match(line)
        if m:
            stubbed.add(m.group(1))
    return defined - stubbed


class StubParityTests(unittest.TestCase):
    def test_every_eval_cpu_body_has_a_stub_in_the_same_patch(self):
        uncovered = {}
        for patch in sorted(PATCHES.glob("*.patch")):
            missing = patch_coverage(patch)
            if missing:
                uncovered[patch.name] = sorted(missing)
        self.assertEqual(uncovered, {}, f"eval_cpu bodies without a NO_CPU stub: {uncovered}")

    def test_floor_divide_stub_is_present(self):
        text = (PATCHES / "mlx-floor-divide.patch").read_text()
        self.assertIn("+NO_CPU(FloorDivide)", text.splitlines())


if __name__ == "__main__":
    unittest.main()
