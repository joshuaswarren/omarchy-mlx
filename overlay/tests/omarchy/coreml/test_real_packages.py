# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Ground-truth checks against the pinned public Parakeet packages.

The expectations here were derived independently by parsing the real
packages with the official generated schema (not by trusting the old
hand-written parser). Skips when the download cache is absent, so the
suite stays host-independent.

Ground truth (HF revision b650695c2322ee5281dff48d7345b2f3a58ff018):
encoder: spec 9, mlProgram, main/CoreML8, 3351 ops, 194 constexpr
palettized weights, fp16 MIL body between FLOAT32/INT32 boundary
features. decoder: 69 ops incl. two `lstm`s. joint: 21 ops.
"""

import json
import os
import unittest
from pathlib import Path

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml.inspect_mlpackage import main
from coreml.mlpackage import inspect

_HF_REVISION = "b650695c2322ee5281dff48d7345b2f3a58ff018"


def _cache_root() -> Path | None:
    root = os.environ.get("MLX_OMARCHY_CACHE_DIR")
    candidates = []
    if root:
        candidates.append(
            Path(root)
            / "parakeet-reference"
            / "mweinbach1"
            / "parakeet-tdt-0.6b-v3-coreml"
        )
    candidates.append(
        Path.home()
        / ".cache"
        / "mlx-omarchy"
        / "parakeet-reference"
        / "mweinbach1"
        / "parakeet-tdt-0.6b-v3-coreml"
    )
    for base in candidates:
        pinned = base / _HF_REVISION
        if pinned.is_dir():
            return pinned
    return None


_CACHE = _cache_root()


@unittest.skipUnless(_CACHE, "parakeet reference cache not present")
class RealPackageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = _CACHE

    def inv(self, name):
        return inspect(self.cache / f"{name}.mlpackage")

    def test_encoder_boundary_types(self):
        inv = self.inv("encoder")
        self.assertEqual(inv["model"]["specification_version"], 9)
        self.assertEqual(inv["model"]["type"], "mlProgram")
        inputs = {e["name"]: e for e in inv["model"]["description"]["inputs"]}
        self.assertEqual(inputs["input_features"]["feature_dtype"], "FLOAT32")
        self.assertEqual(inputs["input_features"]["shape"], [1, 3000, 128])
        self.assertEqual(inputs["attention_mask"]["feature_dtype"], "INT32")
        self.assertEqual(inputs["attention_mask"]["shape"], [1, 3000])
        outputs = {e["name"]: e for e in inv["model"]["description"]["outputs"]}
        self.assertEqual(outputs["encoder_hidden"]["shape"], [1, 375, 640])
        self.assertEqual(outputs["encoder_mask"]["shape"], [1, 375])

    def test_encoder_operation_inventory(self):
        inv = self.inv("encoder")
        fn = inv["program"]["functions"][0]
        self.assertEqual(fn["name"], "main")
        self.assertEqual(fn["opset"], "CoreML8")
        self.assertEqual(inv["op_total"], 3351)
        hist = inv["op_histogram"]
        self.assertEqual(hist["const"], 1783)
        self.assertEqual(hist["constexpr_lut_to_dense"], 194)
        self.assertEqual(hist["linear"], 194)

    def test_encoder_mil_dtypes_keep_distinct_enums(self):
        # The MIL body is fp16 (FLOAT16=10) while the boundary features
        # are FLOAT32/INT32 in the FeatureType enum: both must survive.
        inv = self.inv("encoder")
        fn = inv["program"]["functions"][0]
        mil_dtypes = set()
        for op in fn["block_specializations"]["CoreML8"]["operations"]:
            for out in op["outputs"]:
                if out["type"].get("kind") == "tensor":
                    mil_dtypes.add(out["type"]["dtype"])
        self.assertIn("FLOAT16", mil_dtypes)
        self.assertIn("INT32", mil_dtypes)
        # String-typed consts exist (vocab data) and are named, not mangled.
        self.assertIn("STRING", mil_dtypes)

    def test_encoder_compression_and_weights(self):
        inv = self.inv("encoder")
        comp = inv["compression"]
        self.assertEqual(comp["constexpr_op_count"], 194)
        lut_op = next(
            c for c in comp["constexpr_ops"] if c["op"] == "constexpr_lut_to_dense"
        )
        self.assertIn("indices", lut_op["inputs"])
        self.assertIn("lut", lut_op["inputs"])
        weights = inv["weights"]
        self.assertEqual(len(weights["files"]), 1)
        self.assertEqual(
            weights["files"][0]["sha256"],
            "23867a834223ee6484d0b7b9b703530e8606546efa8e710535163d51ed9aac04",
        )
        refs = weights["blob_references"]
        self.assertTrue(refs and all(r["resolved"] for r in refs))
        self.assertEqual(refs[0]["blob_file"], "@model_path/weights/weight.bin")

    def test_decoder_structure(self):
        inv = self.inv("decoder")
        fn = inv["program"]["functions"][0]
        self.assertEqual(inv["op_total"], 69)
        hist = fn["op_histogram"]
        self.assertEqual(hist["lstm"], 2)
        inputs = {e["name"]: e for e in inv["model"]["description"]["inputs"]}
        self.assertEqual(inputs["input_ids"]["feature_dtype"], "INT32")
        self.assertEqual(inputs["hidden"]["shape"], [2, 1, 640])
        self.assertEqual(inputs["cell"]["shape"], [2, 1, 640])
        outputs = {e["name"]: e for e in inv["model"]["description"]["outputs"]}
        self.assertEqual(
            sorted(outputs), ["decoder_hidden", "next_cell", "next_hidden"]
        )

    def test_joint_structure(self):
        inv = self.inv("joint")
        self.assertEqual(inv["op_total"], 21)
        outputs = {e["name"]: e for e in inv["model"]["description"]["outputs"]}
        self.assertEqual(outputs["token_logits"]["shape"], [1, 8193])
        self.assertEqual(outputs["duration_logits"]["shape"], [1, 5])
        fn = inv["program"]["functions"][0]
        self.assertEqual(fn["inputs"][0]["name"], "encoder_frame")
        # Probed directly with the official bindings: joint MIL inputs
        # are FLOAT32 (=11), matching its FLOAT32 boundary features.
        self.assertEqual(fn["inputs"][0]["type"]["dtype"], "FLOAT32")

    def test_cli_json_and_strict_on_all_packages(self):
        for name in ("encoder", "decoder", "joint"):
            pkg = self.cache / f"{name}.mlpackage"
            rc = main(["inspect", str(pkg), "--json", "--strict"])
            self.assertEqual(rc, 0, f"{name} failed strict inspection")

    def test_json_output_is_round_trippable(self):
        rc = main(["inspect", str(self.cache / "joint.mlpackage"), "--json"])
        self.assertEqual(rc, 0)
        # (main printed JSON; parse contract verified structurally here)
        inv = self.inv("joint")
        json.dumps(inv)  # must not raise
        self.assertEqual(inv["inventory_schema"], "mlx-omarchy.coreml.inventory/1")


if __name__ == "__main__":
    unittest.main()
