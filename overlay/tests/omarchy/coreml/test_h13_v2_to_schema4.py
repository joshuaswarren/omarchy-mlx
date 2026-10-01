#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host-only tests for H13 v2 → schema-4 wrapping."""

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

try:
    from _bootstrap import _TOOLS  # noqa: F401
except ImportError:
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml.h13_v2_to_schema4 import (
    AdapterError,
    convert,
    drop_unknown_tensor_fields,
)

REPO = Path(__file__).resolve().parents[4]
FIXTURE = (
    Path(__file__).resolve().parents[1] / "ane" / "fixtures" / "h13-explicit-chain-add-mul"
)
ISLAND = REPO / "receipts/2026-09-13-attn-select-island/attn-select-island"
IDENTITY = {
    "name": "h13-first-run-chain-add-mul",
    "graph_hash": "5584d0fd8d40027229890408e924e6f7930cd5f02516466a442193408482ce76",
    "compiler_host_build": "Linux x86_64 explicit ANEC fixture",
    "compiler_toolchain": "mil-hwxc aa688df66cbc2110e0df94f0d50fb72c7fa30a18",
    "source_repo": "joshuaswarren/omarchy-mlx",
    "source_commit": "309cd745d40117b689e8936d3ef62fcea262b232",
    "exported_at": "2026-09-12",
    "model": "h13-first-run-chain-add-mul",
}
ISLAND_IDENTITY = {
    "name": "schema4-attn-select-island",
    "graph_hash": "e0daeb7fd056f79f97a1552a1e11b0a92146dd480dcec216fac750613ea1b745",
    "compiler_host_build": "Linux 6.17.2-1-pve x86_64",
    "compiler_toolchain": (
        "mil-hwxc c2cf32e4aa0200d72cecbce203bf6b47f50f729e "
        "sha256:82f1d4ce44dc8f2cb5eefd843eea41c47527af3fbff2def0ed1fc154ccd3ce71"
    ),
    "source_repo": "joshuaswarren/omarchy-mlx",
    "source_commit": "947ece40ec49671f9cb423aff7401b0e88f94680",
    "exported_at": "2026-09-13",
    "model": "attn-select-island",
}


class DropFieldsTest(unittest.TestCase):
    def test_drops_unknown_tensor_fields_and_keeps_bindings(self):
        source = {
            "tensors": {
                "cond": {
                    "dtype": "bool",
                    "logicalBytes": 1,
                    "role": "input",
                    "shape": [1],
                    "extra": True,
                },
                "x": {
                    "logicalBytes": 2,
                    "role": "input",
                    "shape": [1],
                    "accumulation": "sum",
                },
            },
            "programs": [{"inputs": [{"dtype": "bool", "name": "cond"}]}],
        }
        cleaned, dropped = drop_unknown_tensor_fields(source)
        self.assertEqual(sorted(dropped), [("cond", "dtype"), ("cond", "extra")])
        self.assertEqual(
            cleaned["tensors"]["cond"],
            {"logicalBytes": 1, "role": "input", "shape": [1]},
        )
        self.assertEqual(cleaned["tensors"]["x"]["accumulation"], "sum")
        self.assertEqual(cleaned["programs"][0]["inputs"][0]["dtype"], "bool")
        self.assertEqual(source["tensors"]["cond"]["dtype"], "bool")


class ConvertTest(unittest.TestCase):
    def test_drops_tensor_dtype_keeps_original_and_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "package"
            shutil.copytree(FIXTURE, package)
            original = (package / "manifest.json").read_bytes()
            source = json.loads(original)
            source["tensors"]["a"]["dtype"] = "float16"
            source["tensors"]["a"]["bogus"] = 1
            (package / "manifest.json").write_text(json.dumps(source))
            before = (package / "manifest.json").read_bytes()
            manifest, dropped = convert(package, Path(directory) / "bundle", IDENTITY)
            self.assertEqual(sorted(dropped), [("a", "bogus"), ("a", "dtype")])
            self.assertEqual(manifest["manifest_version"], 4)
            self.assertEqual(
                [binding["dtype"] for binding in manifest["programs"][0]["inputs"]],
                ["float16", "float16"],
            )
            self.assertEqual((package / "manifest.json").read_bytes(), before)
            self.assertIn("dtype", json.loads(before)["tensors"]["a"])

    def test_unknown_binding_field_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "package"
            shutil.copytree(FIXTURE, package)
            source = json.loads((package / "manifest.json").read_text())
            source["programs"][0]["inputs"][0]["bogus"] = True
            (package / "manifest.json").write_text(json.dumps(source))
            with self.assertRaisesRegex(AdapterError, "unknown field"):
                convert(package, Path(directory) / "bundle", IDENTITY)

    def test_island_keeps_bool_binding_and_copies_anecs(self):
        if not (ISLAND / "manifest.json").is_file():
            self.skipTest("attn-select island package not present")
        original = json.loads((ISLAND / "manifest.json").read_text())
        self.assertEqual(original["tensors"]["cond"]["dtype"], "bool")
        self.assertEqual(
            original["programs"][1]["inputs"][2]["dtype"], "bool"
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest, dropped = convert(
                ISLAND, Path(directory) / "bundle", ISLAND_IDENTITY
            )
            self.assertEqual(dropped, [("cond", "dtype")])
            self.assertEqual(manifest["manifest_version"], 4)
            self.assertEqual(
                [program["operation"] for program in manifest["programs"]],
                ["matmul", "select"],
            )
            cond = next(item for item in manifest["inputs"] if item["name"] == "cond")
            self.assertEqual(cond["dtype"], "bool")
            binding = manifest["programs"][1]["inputs"][2]
            self.assertEqual(binding["tensor"], "cond")
            self.assertEqual(binding["dtype"], "bool")
            self.assertNotIn("dtype", original["tensors"]["ninf_rt"])
            self.assertEqual(
                original["tensors"]["cond"]["dtype"], "bool"
            )
            for name, digest in (
                (
                    "program-0.anec",
                    "b801f6212f65da3f5ffdab3d6e21cd49e5ae0c16b709ff945de9aaee93fcb629",
                ),
                (
                    "program-1.anec",
                    "0879c6277cf3d89888735ea69ef96d29fb492d43eacfe42a0f154f8f0f91c8eb",
                ),
            ):
                payload = (Path(directory) / "bundle" / name).read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)
                self.assertEqual(payload, (ISLAND / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
