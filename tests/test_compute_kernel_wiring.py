"""Regression tests for scripts/check_compute_kernel_wiring.py.

Commit 5f05b1ee2 landed Bonsai shader dispatch rows that referenced 17
ComputeKernel values compute.h never declared, so the feature never
compiled and nothing static caught it. The gate re-derives the enum,
dispatch and shader sets and fails on drift. These tests pin the gate's
contract:

- the real tree is wired (would have caught 5f05b1ee2: verified against
  `git show 5f05b1ee2` in the landing receipt, not pinned here to keep
  the suite independent of history);
- each failure class exits 1 and names the offender: undeclared
  reference, row without a shader target, shader target without a row,
  referenced value without a dispatch row, _r<N> target without the
  matching ROWS_PER_SLOT spec constant;
- dead declared entries (profile-id placeholders) pass as INFO;
- missing inputs fail closed.

Synthetic fixture trees only; no compiler, no network, no GPU.
"""

import pathlib
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
GATE = REPO / "scripts" / "check_compute_kernel_wiring.py"

HEADER = """#pragma once
#include <cstdint>
namespace mlx::core::omarchy {
enum class ComputeKernel : uint16_t {
  @VALUES@,
};
} // namespace mlx::core::omarchy
"""

CPP = """#include "mlx/backend/omarchy/compute.h"
namespace mlx::core::omarchy {{
namespace {{
using ShaderBytes = std::pair<const unsigned char*, size_t>;
ShaderBytes shader_bytes(ComputeKernel kernel) {{
  using namespace shaders;
  switch (kernel) {{
@ROWS@
    case ComputeKernel::Custom:
    case ComputeKernel::Count:
      break;
  }}
  throw std::invalid_argument("[omarchy] invalid compute kernel.");
}}
}} // namespace
}} // namespace mlx::core::omarchy
"""

CMAKE = """@TARGETS@
"""


def row(value, symbol):
    return "    case ComputeKernel::%s:\n      return {%s, %s_size};" % (
        value, symbol, symbol)


def make_tree(root, values, rows_, targets, extra_src=""):
    values = list(dict.fromkeys(list(values) + ["Custom", "Count"]))
    omarchy = pathlib.Path(root) / "overlay" / "mlx" / "backend" / "omarchy"
    omarchy.mkdir(parents=True)
    (omarchy / "compute.h").write_text(HEADER.replace("@VALUES@", ",\n  ".join(values)))
    (omarchy / "compute.cpp").write_text(CPP.replace("@ROWS@", "\n".join(rows_)))
    (omarchy / "CMakeLists.txt").write_text(CMAKE.replace("@TARGETS@", "\n".join(targets)))
    if extra_src:
        (omarchy / "user.cpp").write_text(extra_src)
    return root


def run_gate(root):
    return subprocess.run(
        ["python3", str(GATE), str(root)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)


class ComputeKernelWiringGate(unittest.TestCase):
    def test_real_tree_is_wired(self):
        p = run_gate(REPO)
        self.assertEqual(p.returncode, 0, p.stderr.decode())
        self.assertIn("compute-kernel-wiring: ok", p.stdout.decode())

    def test_undeclared_reference_fails(self):
        # The 5f05b1ee2 class: dispatch rows name enum values nobody declared.
        with tempfile.TemporaryDirectory() as d:
            make_tree(d, ["ThingF32", "Custom", "Count"],
                      [row("ThingF32", "thing_f32")],
                      ["omarchy_shader(thing_f32 shaders/thing.comp)"],
                      extra_src="int x = sizeof(ComputeKernel::GhostF32);\n")
            p = run_gate(d)
        self.assertEqual(p.returncode, 1)
        err = p.stderr.decode()
        self.assertIn("not declared in compute.h", err)
        self.assertIn("GhostF32", err)

    def test_row_without_shader_target_fails(self):
        with tempfile.TemporaryDirectory() as d:
            make_tree(d, ["ThingF32", "Custom", "Count"],
                      [row("ThingF32", "thing_f32")], [])
            p = run_gate(d)
        self.assertEqual(p.returncode, 1)
        self.assertIn("no omarchy_shader target", p.stderr.decode())
        self.assertIn("thing_f32", p.stderr.decode())

    def test_target_without_row_fails(self):
        with tempfile.TemporaryDirectory() as d:
            make_tree(d, ["ThingF32", "Custom", "Count"],
                      [row("ThingF32", "thing_f32")],
                      ["omarchy_shader(thing_f32 shaders/thing.comp)",
                       "omarchy_shader(orphan_f32 shaders/orphan.comp)"])
            p = run_gate(d)
        self.assertEqual(p.returncode, 1)
        self.assertIn("never returned by a dispatch row", p.stderr.decode())
        self.assertIn("orphan_f32", p.stderr.decode())

    def test_referenced_value_without_row_fails(self):
        # Declared and dispatched from source, but no shader_bytes row:
        # create_pipeline would throw invalid_argument at runtime.
        with tempfile.TemporaryDirectory() as d:
            make_tree(d, ["ThingF32", "OtherF32", "Custom", "Count"],
                      [row("ThingF32", "thing_f32")],
                      ["omarchy_shader(thing_f32 shaders/thing.comp)"],
                      extra_src="int y = sizeof(ComputeKernel::OtherF32);\n")
            p = run_gate(d)
        self.assertEqual(p.returncode, 1)
        self.assertIn("no shader_bytes dispatch row", p.stderr.decode())
        self.assertIn("OtherF32", p.stderr.decode())

    def test_dead_entry_passes_with_info(self):
        # Declared, never referenced, never wired: the append-only
        # GPU-profile id convention. Must pass, must be named.
        with tempfile.TemporaryDirectory() as d:
            make_tree(d, ["ThingF32", "PlaceholderF32", "Custom", "Count"],
                      [row("ThingF32", "thing_f32")],
                      ["omarchy_shader(thing_f32 shaders/thing.comp)"])
            p = run_gate(d)
        self.assertEqual(p.returncode, 0, p.stderr.decode())
        out = p.stdout.decode()
        self.assertIn("1 dead entries", out)
        self.assertIn("PlaceholderF32", out)

    def test_rn_target_requires_rows_per_slot(self):
        with tempfile.TemporaryDirectory() as d:
            make_tree(d, ["ThingF32", "Custom", "Count"],
                      [row("ThingF32", "thing_f32")],
                      ["omarchy_shader(thing_f32 shaders/thing.comp)",
                       "omarchy_shader(wide_r3 shaders/wide.comp -DUSE_SUBGROUP=1)"])
            p = run_gate(d)
        self.assertEqual(p.returncode, 1)
        self.assertIn("ROWS_PER_SLOT=3", p.stderr.decode())

    def test_missing_inputs_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p = run_gate(d)
        self.assertEqual(p.returncode, 1)
        self.assertIn("missing", p.stderr.decode())


if __name__ == "__main__":
    unittest.main()
