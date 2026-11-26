#!/usr/bin/env python3
"""Issue #58 part 3: unit tests for harness/kernel_battery.py.

Stdlib unittest only, no network, no mlx import: extract_kernels and the
MSL device-pointer classifier run against real .py sources written to a
temp directory.
"""

import ast
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "harness"))

import kernel_battery as kb  # noqa: E402

SYNTHETIC_KERNELS = '''
import mlx.core as mx

BLOCK = 64

ADD_SRC = """
kernel void add_one(device const float* x [[buffer(0)]],
                    device float* y [[buffer(1)]],
                    uint gid [[thread_position_in_grid]]) {
  y[gid] = x[gid] + 1.0f;
}
"""

def direct():
    return mx.fast.metal_kernel(
        name="direct_add",
        input_names=["x"], output_names=["y"],
        source="""
kernel void scaled_copy(device const float2* x [[buffer(0)]],
                        device float2* y [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
  y[gid] = x[gid];
}
""",
        template={"BLOCK": 256},
    )

def via_var():
    return mx.fast.metal_kernel(
        name=f"add_{BLOCK}",
        input_names=["x"], output_names=["y"],
        source=ADD_SRC,
        template={"BLOCK": BLOCK},
    )

def texture_user():
    return mx.fast.metal_kernel(
        name="tex", input_names=["t"], output_names=["o"],
        source="kernel void tex(texture2d<float> t [[texture(0)]]) {}",
    )

def decoy():
    return mx.fast.some_other_kernel(name="nope", source="kernel void z(){}")
'''


class ExtractKernelsTests(unittest.TestCase):
    def _extract(self, text):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "kernels.py"
            path.write_text(text)
            return list(kb.extract_kernels(path, "test"))

    def test_extracts_direct_variable_and_fstring_kernels(self):
        kernels = self._extract(SYNTHETIC_KERNELS)
        by_name = {k.name: k for k in kernels}
        self.assertEqual(sorted(by_name), ["<fstring>", "direct_add", "tex"])

        direct = by_name["direct_add"]
        self.assertEqual(direct.source, "test")
        self.assertEqual(direct.input_names, ["x"])
        self.assertEqual(direct.output_names, ["y"])
        self.assertEqual(direct.template_args, {"BLOCK": 256})
        self.assertEqual(direct.gmsl_features, [])
        self.assertGreater(direct.line, 0)
        self.assertTrue(direct.file.endswith("kernels.py"))
        self.assertIn("scaled_copy", direct.msllib)

        fstring = by_name["<fstring>"]
        self.assertIn("add_one", fstring.msllib)
        # name= was an f-string and template used a Name: both best-effort.
        self.assertEqual(fstring.template_args, {"BLOCK": "<expr>"})
        self.assertEqual(fstring.gmsl_features, [])

    def test_refused_keyword_scan_flags_texture(self):
        kernels = self._extract(SYNTHETIC_KERNELS)
        tex = [k for k in kernels if k.name == "tex"][0]
        self.assertEqual(tex.gmsl_features, ["texture"])

    def test_decoy_non_metal_kernel_call_is_ignored(self):
        kernels = self._extract(SYNTHETIC_KERNELS)
        self.assertNotIn("nope", [k.name for k in kernels])

    def test_syntax_error_file_yields_nothing(self):
        self.assertEqual(self._extract("def broken(:\n"), [])

    def test_empty_source_call_is_skipped(self):
        # source resolved from an unknown name -> no msllib -> skipped
        text = (
            "import mlx.core as mx\n"
            "def f():\n"
            "    return mx.fast.metal_kernel(name='n', source=UNKNOWN,\n"
            "        input_names=['x'], output_names=['y'])\n"
        )
        self.assertEqual(self._extract(text), [])


class PtrAliasClassTests(unittest.TestCase):
    def test_scalar_alias(self):
        self.assertEqual(
            kb.ptr_alias_class("device const float* xi = x + off;"), "scalar")

    def test_vector_alias(self):
        self.assertEqual(
            kb.ptr_alias_class("device float2* xi = x;"), "vector")

    def test_pointer_array(self):
        self.assertEqual(
            kb.ptr_alias_class("device float* xs[3] = {a, b, c};"),
            "ptr-array")

    def test_no_alias(self):
        self.assertEqual(kb.ptr_alias_class("kernel void z() {}"), "none")


class CallStrTests(unittest.TestCase):
    def test_dotted_call(self):
        tree = ast.parse("mx.fast.metal_kernel(x)")
        self.assertEqual(kb.call_str(tree.body[0].value),
                         "mx.fast.metal_kernel(...)")

    def test_plain_call(self):
        tree = ast.parse("run(a, b)")
        self.assertEqual(kb.call_str(tree.body[0].value), "run(...)")


class ExtractFromTreeTests(unittest.TestCase):
    def test_walks_tree_and_skips_pycache(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "pkg").mkdir()
            (root / "pkg" / "__pycache__").mkdir()
            (root / "pkg" / "real.py").write_text(SYNTHETIC_KERNELS)
            (root / "pkg" / "__pycache__" / "junk.py").write_text(
                SYNTHETIC_KERNELS)
            names = {k.name for k in kb.extract_from_tree(root, "t")}
        self.assertEqual(names, {"<fstring>", "direct_add", "tex"})


if __name__ == "__main__":
    unittest.main()
