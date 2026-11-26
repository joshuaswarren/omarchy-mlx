# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Per-binding dtype ABI acceptance (mlx-omarchy side, 2026-09-13).

Compiler yield: bool surfaces use the existing per-binding ``dtype``
field (no separate elementDtype field exists). ``dtype: "bool"`` maps
to a 1-byte surface with logical_bytes == element count; absent change
means fp16 and adapts unchanged. Everything is host-only.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

import sys

_TOOLS_DIR = Path(_TOOLS).resolve()
for _entry in (
    str(_TOOLS_DIR),
    str(_TOOLS_DIR / "coreml"),
    str(_TOOLS_DIR / "ane-export"),
):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import h13_package_to_bundle as adapter  # noqa: E402

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "ane"
    / "fixtures"
    / "h13-explicit-chain-add-mul"
)


def _copy_fixture(root: Path) -> Path:
    target = root / "pkg"
    shutil.copytree(_FIXTURE, target)
    return target


def _mutate_binding_dtype(package: Path, dtype) -> None:
    manifest = json.loads((package / "manifest.json").read_text())
    binding = manifest["programs"][0]["inputs"][0]
    if dtype is not None:
        binding["dtype"] = dtype
    (package / "manifest.json").write_text(json.dumps(manifest))


class ElementDtypeTest(unittest.TestCase):
    def test_fp16_package_adapts_unchanged_back_compat(self):
        with tempfile.TemporaryDirectory() as directory:
            package = _copy_fixture(Path(directory))
            manifest = json.loads((package / "manifest.json").read_text())
            self.assertEqual(
                manifest["programs"][0]["inputs"][0]["dtype"], "float16"
            )
            result = adapter.adapt(
                package,
                Path(directory) / "bundle",
                {
                    "name": "compat",
                    "graph_hash": "a" * 64,
                    "compiler_host_build": "h",
                    "compiler_toolchain": "t",
                    "source_repo": "r",
                    "source_commit": "c" * 40,
                    "exported_at": "2026-09-13",
                    "model": "m",
                },
            )
            dtypes = {
                b["dtype"]
                for p in result["programs"]
                for b in p["inputs"] + p["outputs"]
            }
            self.assertEqual(dtypes, {"float16"})

    def test_bool_surface_maps_to_one_byte_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            package = _copy_fixture(Path(directory))
            _mutate_binding_dtype(package, "bool")
            # Fix geometry: bool elements are 1 byte, so logicalBytes
            # must equal the element count for that binding and tensor.
            manifest = json.loads((package / "manifest.json").read_text())
            binding = manifest["programs"][0]["inputs"][0]
            count = 1
            for dim in binding["shape"]:
                count *= dim
            binding["logicalBytes"] = count
            tensor = manifest["tensors"][binding["name"]]
            tensor["logicalBytes"] = count
            (package / "manifest.json").write_text(json.dumps(manifest))
            result = adapter.adapt(
                package,
                Path(directory) / "bundle",
                {
                    "name": "bool-surface",
                    "graph_hash": "a" * 64,
                    "compiler_host_build": "h",
                    "compiler_toolchain": "t",
                    "source_repo": "r",
                    "source_commit": "c" * 40,
                    "exported_at": "2026-09-13",
                    "model": "m",
                },
            )
            first = result["programs"][0]["inputs"][0]
            self.assertEqual(first["dtype"], "bool")
            self.assertEqual(first["logical_bytes"], count)

    def test_unknown_element_dtype_is_named(self):
        with tempfile.TemporaryDirectory() as directory:
            package = _copy_fixture(Path(directory))
            _mutate_binding_dtype(package, "int4")
            with self.assertRaises(adapter.AdapterError) as caught:
                adapter.adapt(
                    package,
                    Path(directory) / "bundle",
                    {
                        "name": "bad",
                        "graph_hash": "a" * 64,
                        "compiler_host_build": "h",
                        "compiler_toolchain": "t",
                        "source_repo": "r",
                        "source_commit": "c" * 40,
                        "exported_at": "2026-09-13",
                        "model": "m",
                    },
                )
            self.assertIn(".dtype 'int4' is unsupported", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
