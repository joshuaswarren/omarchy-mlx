#!/usr/bin/env python3
"""Host-only tests for the explicit H13 ANEC package adapter."""

import hashlib
import importlib.util
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
ADAPTER_PATH = REPO / "overlay/tools/ane-export/h13_package_to_bundle.py"
sys.path.insert(0, str(ADAPTER_PATH.parent))
FIXTURE = Path(__file__).with_name("fixtures") / "h13-explicit-chain-add-mul"
SPEC = importlib.util.spec_from_file_location("h13_package_to_bundle", ADAPTER_PATH)
ADAPTER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ADAPTER)

IDENTITY = {
    "name": "h13-first-run-chain-add-mul",
    "graph_hash": "5584d0fd8d40027229890408e924e6f7930cd5f02516466a442193408482ce76",
    "compiler_host_build": "Linux x86_64 explicit ANEC fixture",
    "compiler_toolchain": "mil-hwxc aa688df66cbc2110e0df94f0d50fb72c7fa30a18",
    "source_repo": "joshuaswarren/mlx-omarchy",
    "source_commit": "309cd745d40117b689e8936d3ef62fcea262b232",
    "exported_at": "2026-09-12",
    "model": "h13-first-run-chain-add-mul",
}


class AdapterTest(unittest.TestCase):
    def test_real_two_program_package_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            manifest = ADAPTER.adapt(FIXTURE, output, IDENTITY)
            self.assertEqual(manifest["manifest_version"], 4)
            self.assertEqual(manifest["driver_abi_major"], 1)
            self.assertEqual(manifest["dispatch_plan"], [0, 1])
            self.assertEqual([p["operation"] for p in manifest["programs"]], ["add", "mul"])
            self.assertEqual(manifest["programs"][0]["inputs"][0]["channel"], 5)
            self.assertEqual(manifest["programs"][1]["outputs"][0]["allocation_bytes"], 16384)
            self.assertEqual(manifest["outputs"], [{
                "name": "y", "index": 0, "dtype": "float16",
                "shape": [1, 64, 1, 1], "byte_size": 128, "stride": 16384,
            }])
            self.assertEqual(manifest["logical_results"], [{
                "name": "y", "dtype": "float16", "shape": [1, 64, 1, 1],
                "tensor": "y", "element_offset": 0, "element_count": 64,
                "conversion": "identity",
            }])
            self.assertEqual(
                manifest["payloads"][1]["sha256"],
                "62595e4a61db24066b83a4f7c077d459c69a6a740109098ee0122b4b46dcd3cc",
            )
            self.assertEqual((output / "program-0.anec").read_bytes(),
                             (FIXTURE / "program-0.anec").read_bytes())
            encoded = json.dumps(
                sorted(manifest["payloads"], key=lambda payload: payload["path"]),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
            self.assertEqual(
                manifest["release_asset"]["model_sha256"],
                hashlib.sha256(encoded).hexdigest(),
            )

    def test_malformed_allocation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "package"
            shutil.copytree(FIXTURE, package)
            source = json.loads((package / "manifest.json").read_text())
            source["programs"][0]["inputs"][0]["allocationBytes"] = 128
            (package / "manifest.json").write_text(json.dumps(source))
            with self.assertRaisesRegex(ADAPTER.AdapterError, "allocationBytes"):
                ADAPTER.adapt(package, Path(directory) / "bundle", IDENTITY)

    def test_malformed_channel_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "package"
            shutil.copytree(FIXTURE, package)
            source = json.loads((package / "manifest.json").read_text())
            source["programs"][0]["inputs"][0]["index"] = 6
            source["programs"][0]["inputs"][1]["index"] = 5
            (package / "manifest.json").write_text(json.dumps(source))
            with self.assertRaisesRegex(
                    ADAPTER.AdapterError,
                    r"declares input channels \[6, 5\] but the task stream "
                    r"binds inputs"):
                ADAPTER.adapt(package, Path(directory) / "bundle", IDENTITY)

    def _build_reverse_fixture(
            self, directory, *,
            manifest_input_index,
            manifest_output_index=4):
        """Build a self-contained package whose ANEC declares selectors
        0x25864 with header src=1 dst=1. Derived map: dst=5, src=4. The
        package manifest binds its single input at `manifest_input_index`
        (5 for the formula layout, 4 for the derived layout) and its output
        at 4 (formula) or 5 (derived). Mirrors fb4dfa86 test_bundle.cpp
        """
        package = Path(directory) / "package"
        package.mkdir()
        nchw = [1, 64, 1, 1, 64, 64]
        anec = bytearray(ADAPTER._ANEC_PAYLOAD_OFFSET + 0x4000)
        struct.pack_into("<Q", anec, 0, 0x4000)
        struct.pack_into("<I", anec, 8, 64)
        struct.pack_into("<I", anec, 12, 1)
        struct.pack_into("<Q", anec, 16, 0x1f8)
        struct.pack_into("<Q", anec, 24, 0x400)
        struct.pack_into("<I", anec, 32, 1)
        struct.pack_into("<I", anec, 36, 1)
        struct.pack_into("<I", anec, 40, 1)
        for ch in (4, 5):
            struct.pack_into("<I", anec, 40 + ch * 4, 1)
            nchw_off = 40 + 32 * 4 + ch * len(nchw) * 8
            for i, v in enumerate(nchw):
                struct.pack_into("<Q", anec, nchw_off + i * 8, v)
        struct.pack_into("<I", anec, ADAPTER._ANEC_PAYLOAD_OFFSET + 4,
                         (16 - 1) << 16)
        struct.pack_into("<I", anec, ADAPTER._ANEC_PAYLOAD_OFFSET + 32, 0x25864)
        at = ADAPTER._ANEC_PAYLOAD_OFFSET + 40
        for reg, cfg in ((0x13800, 0x00033881), (0x13804, 0x00033881),
                         (0x17800, 0x040000c1)):
            struct.pack_into("<I", anec, at, reg)
            struct.pack_into("<I", anec, at + 4, cfg)
            at += 8
        assert at == ADAPTER._ANEC_PAYLOAD_OFFSET + 64
        (package / "model.anec").write_bytes(bytes(anec))
        manifest = {
            "artifactFormat": "anec",
            "dispatchPlan": [0],
            "intermediates": [],
            "logicalResults": [{
                "name": "y", "dtype": "float16", "shape": [1, 64, 1, 1],
                "physical": {"tensor": "y", "elementOffset": 0, "elementCount": 64},
                "conversion": "identity",
            }],
            "physicalOutputs": [{
                "tensor": "y", "dtype": "float16",
                "shape": [1, 64, 1, 1], "logicalBytes": 128,
            }],
            "programs": [{
                "bytes": len(anec),
                "constantBytes": 0,
                "constantInputs": [],
                "constantOffset": 0,
                "encoder": "h13-test",
                "file": "model.anec",
                "inputs": [{
                    "allocationBytes": 0x4000, "dtype": "float16",
                    "index": manifest_input_index,
                    "logicalBytes": 128, "name": "a", "nchw": nchw,
                    "shape": [1, 64, 1, 1],
                }],
                "operation": "add",
                "outputs": [{
                    "allocationBytes": 0x4000, "dtype": "float16",
                    "index": manifest_output_index,
                    "logicalBytes": 128, "name": "y", "nchw": nchw,
                    "shape": [1, 64, 1, 1],
                }],
                "scratchBytes": 0,
                "taskDescriptors": 1,
            }],
            "schema": "mil-hwxc.h13-anec-package.v2",
            "target": "H13",
            "tensors": {
                "a": {"role": "input", "logicalBytes": 128,
                       "shape": [1, 64, 1, 1]},
                "y": {"role": "output", "logicalBytes": 128,
                       "shape": [1, 64, 1, 1]},
            },
        }
        (package / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return package

    def test_derived_reverse_map_rejects_a_positional_declaration(self):
        with tempfile.TemporaryDirectory() as directory:
            package = self._build_reverse_fixture(
                directory, manifest_input_index=5)
            with self.assertRaisesRegex(
                    ADAPTER.AdapterError,
                    r"declares output channels \[4\] but the task stream "
                    r"binds outputs \[5\]"):
                ADAPTER.adapt(package, Path(directory) / "bundle", IDENTITY)

    def test_derived_reverse_map_accepts_the_task_stream_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            package = self._build_reverse_fixture(
                directory, manifest_input_index=4, manifest_output_index=5)
            bundle = Path(directory) / "bundle"
            ADAPTER.adapt(package, bundle, IDENTITY)
            emitted = json.loads((bundle / "manifest.json").read_text())
            self.assertEqual(emitted["programs"][0]["outputs"][0]["channel"], 5)
            self.assertEqual(emitted["programs"][0]["inputs"][0]["channel"], 4)

    def test_non_anec_package_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "package"
            shutil.copytree(FIXTURE, package)
            source = json.loads((package / "manifest.json").read_text())
            source["artifactFormat"] = "hwx"
            (package / "manifest.json").write_text(json.dumps(source))
            with self.assertRaisesRegex(ADAPTER.AdapterError, "artifactFormat"):
                ADAPTER.adapt(package, Path(directory) / "bundle", IDENTITY)

    def test_graph_tensor_dtype_is_accepted_and_checked(self):
        # The compiler spells `dtype` on a graph tensor for a bool surface,
        # so an island with a bool select cond adapts without hand-editing
        # the compiler manifest. A dtype that contradicts the program
        # binding is still a hard refusal.
        for declared, expect in (("float16", None), ("bool", "dtype")):
            with self.subTest(dtype=declared), tempfile.TemporaryDirectory() as directory:
                package = Path(directory) / "package"
                output = Path(directory) / "bundle"
                shutil.copytree(FIXTURE, package)
                source = json.loads((package / "manifest.json").read_text())
                source["tensors"]["a"]["dtype"] = declared
                (package / "manifest.json").write_text(json.dumps(source))
                if expect is None:
                    manifest = ADAPTER.adapt(package, output, IDENTITY)
                    self.assertEqual(manifest["inputs"][0]["dtype"], "float16")
                else:
                    with self.assertRaisesRegex(ADAPTER.AdapterError, expect):
                        ADAPTER.adapt(package, output, IDENTITY)
                    self.assertFalse(output.exists())

    def test_logical_result_order_duplicates_and_overlapping_views_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "package"
            shutil.copytree(FIXTURE, package)
            source = json.loads((package / "manifest.json").read_text())
            source["logicalResults"] = [
                {"name": "tail", "dtype": "float16", "shape": [4, 8],
                 "physical": {"tensor": "y", "elementOffset": 32,
                              "elementCount": 32}, "conversion": "identity"},
                {"name": "whole", "dtype": "float16", "shape": [8, 8],
                 "physical": {"tensor": "y", "elementOffset": 0,
                              "elementCount": 64}, "conversion": "identity"},
                {"name": "tail", "dtype": "float16", "shape": [4, 8],
                 "physical": {"tensor": "y", "elementOffset": 32,
                              "elementCount": 32}, "conversion": "identity"},
            ]
            (package / "manifest.json").write_text(json.dumps(source))
            manifest = ADAPTER.adapt(package, Path(directory) / "bundle", IDENTITY)
            self.assertEqual(
                [(item["name"], item["shape"], item["element_offset"])
                 for item in manifest["logical_results"]],
                [("tail", [4, 8], 32), ("whole", [8, 8], 0),
                 ("tail", [4, 8], 32)],
            )

    def test_invalid_logical_result_contract_is_rejected_without_output(self):
        cases = {
            "unsupported conversion": ("conversion", "cast", "conversion"),
            "identity dtype mismatch": ("dtype", "float32", "dtype"),
        }
        for name, (field, value, error) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                package = Path(directory) / "package"
                output = Path(directory) / "bundle"
                shutil.copytree(FIXTURE, package)
                source = json.loads((package / "manifest.json").read_text())
                source["logicalResults"][0][field] = value
                (package / "manifest.json").write_text(json.dumps(source))
                with self.assertRaisesRegex(ADAPTER.AdapterError, error):
                    ADAPTER.adapt(package, output, IDENTITY)
                self.assertFalse(output.exists())

    def test_unsupported_physical_output_dtype_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "package"
            shutil.copytree(FIXTURE, package)
            source = json.loads((package / "manifest.json").read_text())
            source["programs"][1]["outputs"][0]["dtype"] = "float32"
            source["programs"][1]["outputs"][0]["logicalBytes"] = 256
            source["tensors"]["y"]["logicalBytes"] = 256
            source["physicalOutputs"][0]["dtype"] = "float32"
            source["physicalOutputs"][0]["logicalBytes"] = 256
            source["logicalResults"][0]["dtype"] = "float32"
            (package / "manifest.json").write_text(json.dumps(source))
            with self.assertRaisesRegex(ADAPTER.AdapterError, "hardware dtype"):
                ADAPTER.adapt(package, Path(directory) / "bundle", IDENTITY)

    def test_logical_result_geometry_and_unknown_fields_are_rejected(self):
        mutations = (
            (lambda source: source["logicalResults"][0]["physical"].update(
                elementCount=63), "elementCount"),
            (lambda source: source["logicalResults"][0]["physical"].update(
                elementOffset=1), "storage"),
            (lambda source: source["logicalResults"][0]["physical"].update(
                physicalElements=64), "unknown field"),
            (lambda source: source["logicalResults"][0].update(extra=True),
             "unknown field"),
            (lambda source: source["physicalOutputs"][0].update(extra=True),
             "unknown field"),
        )
        for mutate, error in mutations:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                package = Path(directory) / "package"
                shutil.copytree(FIXTURE, package)
                source = json.loads((package / "manifest.json").read_text())
                mutate(source)
                (package / "manifest.json").write_text(json.dumps(source))
                with self.assertRaisesRegex(ADAPTER.AdapterError, error):
                    ADAPTER.adapt(package, Path(directory) / "bundle", IDENTITY)

    def test_old_producer_schema_is_rejected_without_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "package"
            output = Path(directory) / "bundle"
            shutil.copytree(FIXTURE, package)
            source = json.loads((package / "manifest.json").read_text())
            source["schema"] = "mil-hwxc.h13-anec-package.v1"
            del source["physicalOutputs"]
            del source["logicalResults"]
            (package / "manifest.json").write_text(json.dumps(source))
            with self.assertRaisesRegex(ADAPTER.AdapterError, "schema"):
                ADAPTER.adapt(package, output, IDENTITY)
            self.assertFalse(output.exists())

    def test_top_level_offset_is_independent_of_physical_slice_span(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "package"
            shutil.copytree(FIXTURE, package)
            source = json.loads((package / "manifest.json").read_text())
            source["tensors"]["a"]["shape"] = [1, 1024, 1, 1]
            source["tensors"]["a"]["logicalBytes"] = 2048
            binding = source["programs"][0]["inputs"][0]
            binding["shape"] = [1, 384, 1, 1]
            binding["nchw"] = [1, 512, 1, 1, 32, 32]
            binding["logicalBytes"] = 768
            binding["slice"] = {
                "tensor": "a",
                "elementOffset": 512,
                "elementCount": 384,
                "physicalElements": 512,
            }
            (package / "manifest.json").write_text(json.dumps(source))
            manifest = ADAPTER.adapt(package, Path(directory) / "bundle", IDENTITY)
            self.assertEqual(
                manifest["programs"][0]["inputs"][0]["element_offset"], 512)

    def test_output_ranges_require_complete_union_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "package"
            shutil.copytree(FIXTURE, package)
            source = json.loads((package / "manifest.json").read_text())
            source["tensors"]["y"]["shape"] = [1, 128, 1, 1]
            source["tensors"]["y"]["logicalBytes"] = 256
            source["physicalOutputs"][0]["shape"] = [1, 128, 1, 1]
            source["physicalOutputs"][0]["logicalBytes"] = 256
            source["logicalResults"][0]["shape"] = [1, 128, 1, 1]
            source["logicalResults"][0]["physical"]["elementCount"] = 128
            output = source["programs"][1]["outputs"][0]
            output["nchw"][1] = 128
            output["slice"] = {
                "tensor": "y",
                "elementOffset": 0,
                "elementCount": 64,
                "physicalElements": 128,
            }
            (package / "manifest.json").write_text(json.dumps(source))
            with self.assertRaisesRegex(ADAPTER.AdapterError, "not fully written"):
                ADAPTER.adapt(package, Path(directory) / "gap", IDENTITY)

            source["intermediates"] = []
            del source["tensors"]["sum"]
            for index, program in enumerate(source["programs"]):
                program["inputs"][0]["name"] = "a"
                program["inputs"][0].pop("role", None)
                output = program["outputs"][0]
                output["name"] = "y"
                output.pop("role", None)
                output["nchw"][1] = 128
                output["slice"] = {
                    "tensor": "y",
                    "elementOffset": index * 64,
                    "elementCount": 64,
                    "physicalElements": 128,
                }
            (package / "manifest.json").write_text(json.dumps(source))
            manifest = ADAPTER.adapt(package, Path(directory) / "complete", IDENTITY)
            self.assertEqual(
                [program["outputs"][0]["element_offset"]
                 for program in manifest["programs"]],
                [0, 64],
            )

    def test_generation_receipt_binds_compiler_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "package"
            output = Path(directory) / "bundle"
            shutil.copytree(FIXTURE, package)
            source = json.loads((package / "manifest.json").read_text())
            source["programs"][1]["inputs"][1]["name"] = "a"
            (package / "manifest.json").write_text(json.dumps(source))
            result = subprocess.run(
                [
                    sys.executable,
                    str(ADAPTER_PATH),
                    str(package),
                    "--out-dir", str(output),
                    "--graph-source", str(package / "model.mil"),
                    "--compiler-source", str(REPO),
                    "--name", "mutated-package",
                    "--source-repo", "mil-hwxc",
                    "--source-commit", "aa688df66cbc2110e0df94f0d50fb72c7fa30a18",
                    "--model", "mutated-package",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn(
                "compiler receipt compiler_manifest_sha256 does not match compiler manifest",
                result.stderr,
            )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
