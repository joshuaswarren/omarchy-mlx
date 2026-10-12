"""Regression tests for scripts/check-no-cpu-stubs.py.

omarchy-mlx#156: the FloorDivide backport added an eval_cpu body and a
NO_GPU stub but no NO_CPU stub, so MLX_BUILD_CPU=OFF failed to link
(undefined vtable/typeinfo) while CI (CPU=ON) stayed green. The gate
re-derives the eval_cpu set from backend/cpu and the stub set from
backend/no_cpu and fails on drift. Synthetic fixture trees only; no
compiler, no network.
"""

import pathlib
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
GATE = REPO / "scripts" / "check-no-cpu-stubs.py"

CPU_SRC = """#include "mlx/primitives.h"
namespace mlx::core {
void Divide::eval_cpu(const std::vector<array>& inputs, array& out) {}
void FloorDivide::eval_cpu(const std::vector<array>& inputs, array& out) {}
}
"""

STUBS = """#include "mlx/primitives.h"
namespace mlx::core {
NO_CPU(Divide)
%s
}
"""


def make_tree(root, with_stub):
    cpu = pathlib.Path(root) / "mlx" / "backend" / "cpu"
    nocpu = pathlib.Path(root) / "mlx" / "backend" / "no_cpu"
    cpu.mkdir(parents=True)
    nocpu.mkdir(parents=True)
    (cpu / "binary.cpp").write_text(CPU_SRC)
    (nocpu / "primitives.cpp").write_text(STUBS % ("NO_CPU(FloorDivide)" if with_stub else ""))
    return root


def run_gate(root):
    return subprocess.run(
        ["python3", str(GATE), str(root)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)


class NoCpuStubsGate(unittest.TestCase):
    def test_missing_stub_fails_and_names_it(self):
        # The #156 class: impl without a stub.
        with tempfile.TemporaryDirectory() as d:
            make_tree(d, with_stub=False)
            p = run_gate(d)
        self.assertEqual(p.returncode, 1)
        self.assertIn("FloorDivide", p.stderr.decode())

    def test_present_stub_passes(self):
        with tempfile.TemporaryDirectory() as d:
            make_tree(d, with_stub=True)
            p = run_gate(d)
        self.assertEqual(p.returncode, 0, p.stderr.decode())
        self.assertIn("no-cpu-stubs: ok", p.stdout.decode())


if __name__ == "__main__":
    unittest.main()
