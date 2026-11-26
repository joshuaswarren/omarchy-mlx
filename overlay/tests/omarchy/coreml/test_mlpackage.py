# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Package reader tests: synthetic packages built with the official
schema (validation, type semantics, flexibility, failures) plus the
consumer contract."""

import atexit
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml import proto
from coreml.inspect_mlpackage import main
from coreml.mlpackage import INVENTORY_SCHEMA, MlPackageError, inspect
from coreml.schema import MIL_pb2, Model_pb2

# ---------------------------------------------------------------------------
# Synthetic package builder (official bindings only — no wire handiwork)
# ---------------------------------------------------------------------------


def build_model() -> Model_pb2.Model:
    """A minimal but semantically rich ML Program:

    * FeatureType input: FLOAT32 with a shapeRange (flexible batch dim)
    * FeatureType output: FLOAT16 static shape
    * MIL body: fp16 const + fp16 linear (bound by name), one symbolic
      (unknown) intermediate dimension, a state feature, and a blob
      reference into weights/weight.bin for the lut payload.
    """
    model = Model_pb2.Model()
    model.specificationVersion = 9

    inp = model.description.input.add()
    inp.name = "audio"
    arr = inp.type.multiArrayType
    arr.dataType = Model_pb2.ArrayFeatureType.FLOAT32
    arr.shape.extend([1, 100, 80])
    rng = arr.shapeRange.sizeRanges.add()
    rng.lowerBound, rng.upperBound = 1, -1  # unbounded batch dim
    rng = arr.shapeRange.sizeRanges.add()
    rng.lowerBound, rng.upperBound = 100, 100
    rng = arr.shapeRange.sizeRanges.add()
    rng.lowerBound, rng.upperBound = 80, 80

    out = model.description.output.add()
    out.name = "hidden"
    arr = out.type.multiArrayType
    arr.dataType = Model_pb2.ArrayFeatureType.FLOAT16
    arr.shape.extend([1, 50, 64])

    state = model.description.state.add()
    state.name = "kv_cache"
    arr = state.type.stateType.arrayType
    arr.dataType = Model_pb2.ArrayFeatureType.FLOAT16
    arr.shape.extend([2, 4, 64])

    program = model.mlProgram
    program.version = 3
    fn = program.functions["main"]
    fn.opset = "CoreML8"

    # Function-level named input mirrors the model input as fp16 MIL.
    named = fn.inputs.add()
    named.name = "audio"
    tt = named.type.tensorType
    tt.dataType = MIL_pb2.FLOAT16
    tt.rank = 3
    tt.dimensions.add().constant.size = 1
    tt.dimensions.add().constant.size = 100
    tt.dimensions.add().constant.size = 80

    # const op holding an inline fp16 weight (contents never reported).
    const = fn.block_specializations[fn.opset].operations.add()
    const.type = "const"
    cout = const.outputs.add()
    cout.name = "weight_fp16"
    tt = cout.type.tensorType
    tt.dataType = MIL_pb2.FLOAT16
    tt.rank = 2
    tt.dimensions.add().constant.size = 64
    tt.dimensions.add().constant.size = 64

    # linear op referencing the const by name; its lut argument points
    # at the external weight blob.
    linear = fn.block_specializations[fn.opset].operations.add()
    linear.type = "linear"
    x = linear.inputs["x"].arguments.add()
    x.name = "audio"
    w = linear.inputs["weight"].arguments.add()
    w.name = "weight_fp16"
    lut = linear.inputs["lut"].arguments.add()
    lut.value.type.tensorType.dataType = MIL_pb2.UINT4
    lut.value.type.tensorType.rank = 4
    for _ in range(4):
        lut.value.type.tensorType.dimensions.add().constant.size = 2
    lut.value.blobFileValue.fileName = "@model_path/weights/weight.bin"
    lut.value.blobFileValue.offset = 64
    lout = linear.outputs.add()
    lout.name = "hidden_fp16"
    tt = lout.type.tensorType
    tt.dataType = MIL_pb2.FLOAT16
    tt.rank = 3
    tt.dimensions.add().constant.size = 1
    tt.dimensions.add().unknown.variadic = False  # symbolic dimension
    tt.dimensions.add().constant.size = 64
    return model


def write_package(root: Path, model: Model_pb2.Model, *, weights: bool = True) -> Path:
    (root / "Data" / "com.apple.CoreML" / "weights").mkdir(parents=True)
    (root / "Data" / "com.apple.CoreML" / "model.mlmodel").write_bytes(
        model.SerializeToString()
    )
    if weights:
        (root / "Data" / "com.apple.CoreML" / "weights" / "weight.bin").write_bytes(
            b"\x00" * 128
        )
    manifest = {
        "fileFormatVersion": "1.0.0",
        "itemInfoEntries": {
            "model-id": {
                "author": "com.apple.CoreML",
                "name": "model.mlmodel",
                "path": "com.apple.CoreML/model.mlmodel",
            },
            "weights-id": {
                "author": "com.apple.CoreML",
                "name": "weights",
                "path": "com.apple.CoreML/weights",
            },
        },
        "rootModelIdentifier": "model-id",
    }
    (root / "Manifest.json").write_text(json.dumps(manifest))
    return root


def make_package(model: Model_pb2.Model, **kwargs) -> Path:
    tmp = tempfile.TemporaryDirectory()
    atexit.register(tmp.cleanup)
    return write_package(Path(tmp.name) / "model.mlpackage", model, **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class CoverageEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.package = make_package(build_model())
        self.model = self.package / "Data/com.apple.CoreML/model.mlmodel"
        self.weight = self.package / "Data/com.apple.CoreML/weights/weight.bin"
        self.report = {
            "model_sha256": hashlib.sha256(self.model.read_bytes()).hexdigest(),
            "weights": {
                self.weight.relative_to(self.package).as_posix(): hashlib.sha256(
                    self.weight.read_bytes()
                ).hexdigest()
            },
            "compiler": {
                "repository": "https://example.com/compiler",
                "commit": "a" * 40,
                "target": "H13",
            },
            "source": "https://example.com/coverage",
            "counts": {
                "direct": 0,
                "direct-alias": 0,
                "direct-const": 1,
                "normalization-needed": 0,
                "missing-envelope": 0,
                "missing": 1,
            },
        }
        self.report_path = self.package.parent / "coverage.json"

    def run_report(self):
        self.report_path.write_text(json.dumps(self.report))
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = main(
                [
                    "inspect",
                    str(self.package),
                    "--json",
                    "--compiler-coverage",
                    str(self.report_path),
                ]
            )
        return code, output.getvalue(), errors.getvalue()

    def test_source_matched_report_does_not_claim_compilation(self):
        code, output, _ = self.run_report()
        self.assertEqual(code, 0)
        eligibility = json.loads(output)["eligibility"]
        self.assertEqual(eligibility["counts"]["missing"], 1)
        self.assertIsNone(eligibility["compilable"])
        self.assertEqual(eligibility["compiler"]["commit"], "a" * 40)

    def test_changed_source_or_weights_rejects_stale_report(self):
        self.weight.write_bytes(b"changed weights")
        self.assertEqual(self.run_report()[0], 1)
        self.weight.write_bytes(b"\x00" * 128)
        model = build_model()
        model.description.metadata.author = "different model"
        self.model.write_bytes(model.SerializeToString())
        self.assertEqual(self.run_report()[0], 1)

    def test_incomplete_or_invalid_counts_reject_report(self):
        for count in (0, True, -1):
            with self.subTest(count=count):
                self.report["counts"]["missing"] = count
                self.assertEqual(self.run_report()[0], 1)


class InventorySemanticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pkg_path = make_package(build_model())
        cls.inv = inspect(cls.pkg_path)

    def test_consumer_contract_keys(self):
        # What a phase-3 adapter relies on must exist and be consistent.
        inv = self.inv
        self.assertEqual(inv["inventory_schema"], INVENTORY_SCHEMA)
        self.assertEqual(sum(inv["op_histogram"].values()), inv["op_total"])
        fn = inv["program"]["functions"][0]
        self.assertEqual(fn["opset"], "CoreML8")
        self.assertEqual(
            {op["type"] for op in fn["block_specializations"]["CoreML8"]["operations"]},
            {"const", "linear"},
        )

    def test_feature_vs_mil_dtype_enums_preserved(self):
        inv = self.inv
        # Model description speaks the FeatureType enum (0x10000|bits).
        self.assertEqual(
            inv["model"]["description"]["inputs"][0]["feature_dtype"], "FLOAT32"
        )
        self.assertEqual(
            inv["model"]["description"]["outputs"][0]["feature_dtype"], "FLOAT16"
        )
        # MIL block values speak the MIL enum, a different numbering.
        ops = inv["program"]["functions"][0]["block_specializations"]["CoreML8"][
            "operations"
        ]
        lin = next(op for op in ops if op["type"] == "linear")
        self.assertEqual(lin["outputs"][0]["type"]["dtype"], "FLOAT16")
        # Input is f32 at the model boundary and fp16 inside MIL: both
        # readings coexist; nothing was coerced to one dtype.
        named = inv["program"]["functions"][0]["inputs"][0]
        self.assertEqual(named["type"]["dtype"], "FLOAT16")

    def test_shape_range_flexibility_reported_not_forced_static(self):
        inp = self.inv["model"]["description"]["inputs"][0]
        flex = inp["shape_flexibility"]
        self.assertEqual(flex["kind"], "shapeRange")
        self.assertTrue(flex["size_ranges"][0]["unbounded"])

    def test_symbolic_mil_dimension_explicit(self):
        ops = self.inv["program"]["functions"][0]["block_specializations"]["CoreML8"][
            "operations"
        ]
        lin = next(op for op in ops if op["type"] == "linear")
        dims = lin["outputs"][0]["type"]["dimensions"]
        self.assertEqual(dims[1], {"unknown": True, "variadic": False})

    def test_state_reported(self):
        state = self.inv["model"]["description"]["state"]
        self.assertEqual([s["name"] for s in state], ["kv_cache"])
        self.assertEqual(state[0]["feature_dtype"], "FLOAT16")

    def test_blob_reference_resolved(self):
        ref = self.inv["weights"]["blob_references"][0]
        self.assertEqual(ref["blob_file"], "@model_path/weights/weight.bin")
        self.assertTrue(ref["resolved"])
        self.assertEqual(ref["distinct_offsets"], 1)

    def test_no_tensor_data_in_inventory(self):
        # The inventory carries types and references, not tensor bytes.
        raw = json.dumps(self.inv)
        self.assertNotIn("immediate_data", raw)

    def test_ssa_bindings_are_validated(self):
        self.assertEqual(self.inv["validity"]["ssa_bindings"], "ok")


class ConsumerContractTest(unittest.TestCase):
    """The contract a downstream compiler adapter builds on: the
    inventory must expose every binding with enough detail that a
    broken package fails with a precise, actionable error — never a
    silent blank."""

    def test_dangling_binding_fails_with_op_and_value_named(self):
        model = build_model()
        fn = model.mlProgram.functions["main"]
        block = fn.block_specializations[fn.opset]
        # Point the linear's weight at a value nobody produces.
        block.operations[1].inputs["weight"].arguments[0].name = "ghost_value"
        pkg = make_package(model)
        inv = inspect(pkg)
        self.assertEqual(inv["validity"]["ssa_bindings"], "invalid")
        message = "\n".join(inv["validity"]["notes"])
        self.assertIn("ghost_value", message)
        self.assertIn("linear", message)
        self.assertIn("weight", message)
        self.assertEqual(main(["inspect", str(pkg), "--strict"]), 2)

    def test_corrupt_package_fails_with_named_cause(self):
        pkg = make_package(build_model())
        model_file = pkg / "Data" / "com.apple.CoreML" / "model.mlmodel"
        model_file.write_bytes(b"\x0a\xff" + b"\x00" * 40)  # invalid length
        try:
            inspect(pkg)
            self.fail("expected ModelSpecError")
        except proto.ModelSpecError as exc:
            self.assertIn("model.mlmodel", str(exc))
            self.assertIn("parseable", str(exc))


class PackageValidationTest(unittest.TestCase):
    def test_path_escape_refused(self):
        pkg = make_package(build_model())
        manifest = json.loads((pkg / "Manifest.json").read_text())
        manifest["itemInfoEntries"]["model-id"]["path"] = "../escapee/model.mlmodel"
        (pkg / "Manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(MlPackageError) as ctx:
            inspect(pkg)
        self.assertIn("escapes the package", str(ctx.exception))

    def test_absolute_manifest_path_refused(self):
        pkg = make_package(build_model())
        manifest = json.loads((pkg / "Manifest.json").read_text())
        manifest["itemInfoEntries"]["model-id"]["path"] = "/etc/model.mlmodel"
        (pkg / "Manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(MlPackageError) as ctx:
            inspect(pkg)
        self.assertIn("absolute", str(ctx.exception))

    def test_unparsable_manifest_refused(self):
        pkg = make_package(build_model())
        (pkg / "Manifest.json").write_text("{not json")
        with self.assertRaises(MlPackageError) as ctx:
            inspect(pkg)
        self.assertIn("unparsable manifest", str(ctx.exception))

    def test_missing_model_file_refused(self):
        pkg = make_package(build_model())
        (pkg / "Data" / "com.apple.CoreML" / "model.mlmodel").unlink()
        with self.assertRaises(MlPackageError) as ctx:
            inspect(pkg)
        self.assertIn("model file named by manifest is missing", str(ctx.exception))

    def test_truncated_model_spec_is_parse_error_not_crash(self):
        pkg = make_package(build_model())
        model_file = pkg / "Data" / "com.apple.CoreML" / "model.mlmodel"
        model_file.write_bytes(model_file.read_bytes()[:7])
        with self.assertRaises(proto.ModelSpecError) as ctx:
            inspect(pkg)
        self.assertIn("not a parseable Core ML Model", str(ctx.exception))

    def test_not_a_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "somefile"
            target.write_text("x")
            with self.assertRaises(MlPackageError) as ctx:
                inspect(target)
        self.assertIn("not a directory", str(ctx.exception))


class MissingWeightsTest(unittest.TestCase):
    def test_unresolved_blob_reference_is_explicit_and_strict_fails(self):
        pkg = make_package(build_model(), weights=False)
        inv = inspect(pkg)  # non-strict inspection succeeds with a finding
        ref = inv["weights"]["blob_references"][0]
        self.assertFalse(ref["resolved"])
        self.assertIsNone(ref["resolved_file"])
        # Strict mode must turn the same finding into a failure.
        rc = main(["inspect", str(pkg), "--json", "--strict"])
        self.assertEqual(rc, 2)

    def test_weight_file_hash_matches_contents(self):
        pkg = make_package(build_model(), weights=True)
        inv = inspect(pkg)
        expected = hashlib.sha256(b"\x00" * 128).hexdigest()
        self.assertEqual(inv["weights"]["files"][0]["sha256"], expected)


class CompleteInventoryTest(unittest.TestCase):
    def test_nested_attribute_weights_and_operations_are_inventoried(self):
        model = build_model()
        block = model.mlProgram.functions["main"].block_specializations["CoreML8"]
        outer = block.operations.add()
        outer.type = "while_loop"
        inner = outer.blocks.add().operations.add()
        inner.type = "cond"
        inner.inputs["x"].arguments.add().name = "audio"
        const = inner.blocks.add().operations.add()
        const.type = "const"
        value = const.attributes["val"]
        value.blobFileValue.fileName = "@model_path/weights/weight.bin"
        value.blobFileValue.offset = 96
        with tempfile.TemporaryDirectory() as tmp:
            inv = inspect(write_package(Path(tmp), model))
        self.assertEqual(inv["op_total"], 5)
        self.assertEqual(inv["op_histogram"]["const"], 2)
        self.assertEqual(inv["control_flow"]["ops"], ["cond", "while_loop"])
        self.assertEqual(inv["weights"]["blob_references"][0]["offsets"], [64, 96])
        self.assertEqual(inv["validity"]["ssa_bindings"], "ok")

    def test_manifest_root_selects_model_independent_of_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = write_package(Path(tmp), build_model())
            old = pkg / "Data/com.apple.CoreML/model.mlmodel"
            old.rename(old.with_name("encoder.mlmodel"))
            manifest = json.loads((pkg / "Manifest.json").read_text())
            manifest["itemInfoEntries"]["model-id"].update(
                path="com.apple.CoreML/encoder.mlmodel", name="encoder.mlmodel"
            )
            (pkg / "Manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(inspect(pkg)["op_total"], 2)

    def test_unknown_root_identifier_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = write_package(Path(tmp), build_model())
            manifest = json.loads((pkg / "Manifest.json").read_text())
            manifest["rootModelIdentifier"] = "missing-id"
            (pkg / "Manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(MlPackageError):
                inspect(pkg)

    def test_blob_resolution_requires_exact_path_and_valid_offset(self):
        model = build_model()
        value = (
            model.mlProgram.functions["main"]
            .block_specializations["CoreML8"]
            .operations[1]
            .inputs["lut"]
            .arguments[0]
            .value
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = write_package(Path(tmp), model)
            weight = pkg / "Data/com.apple.CoreML/weights/weight.bin"
            weight.rename(weight.with_name("wrongweight.bin"))
            self.assertFalse(inspect(pkg)["weights"]["blob_references"][0]["resolved"])
            weight.with_name("wrongweight.bin").rename(weight)
            value.blobFileValue.offset = 128
            (pkg / "Data/com.apple.CoreML/model.mlmodel").write_bytes(
                model.SerializeToString()
            )
            self.assertFalse(inspect(pkg)["weights"]["blob_references"][0]["resolved"])


if __name__ == "__main__":
    unittest.main()
