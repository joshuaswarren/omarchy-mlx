# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Vendored official schema: provenance, enums, and parse errors."""

import hashlib
import json
import sys
import unittest
from pathlib import Path

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401
from coreml import proto
from coreml.schema import MIL_pb2, Model_pb2


class VendoredSchemaTest(unittest.TestCase):
    def test_manifest_pins_every_vendored_file(self):
        schema = Path(proto.__file__).parent / "schema"
        manifest = json.loads((schema / "VENDORED.json").read_text())["vendored_schema"]
        self.assertEqual(manifest["upstream_tag"], "9.0")
        pinned = manifest["files"]
        on_disk = sorted(
            p.relative_to(schema).as_posix()
            for p in list(schema.glob("*_pb2.py"))
            + list(schema.glob("proto/*.proto"))
            + [schema / "__init__.py", schema / "LICENSE.txt"]
        )
        self.assertEqual(sorted(pinned), on_disk)
        for rel, digest in pinned.items():
            actual = hashlib.sha256((schema / rel).read_bytes()).hexdigest()
            self.assertEqual(actual, digest, f"{rel} drifted from the pinned hash")

    def test_no_coremltools_or_mlx_import(self):
        self.assertNotIn("coremltools", sys.modules)
        self.assertNotIn("mlx", sys.modules)
        self.assertNotIn("tensorflow", sys.modules)


class OfficialEnumSemanticsTest(unittest.TestCase):
    """The dtype enums, straight from the generated descriptors.

    These exact mappings are what the deleted hand-written tables got
    wrong (it labeled MIL BOOL=1 as float16). If the schema changes,
    these fail loudly instead of silently misreporting dtypes.
    """

    def test_mil_dtype_enum(self):
        self.assertEqual(MIL_pb2.BOOL, 1)
        self.assertEqual(MIL_pb2.STRING, 2)
        self.assertEqual(MIL_pb2.FLOAT16, 10)
        self.assertEqual(MIL_pb2.FLOAT32, 11)
        self.assertEqual(MIL_pb2.INT32, 23)
        self.assertEqual(MIL_pb2.UINT4, 35)

    def test_feature_dtype_enum_is_a_different_numbering(self):
        # Model-description ArrayFeatureType values are 0x10000|bits etc.
        self.assertEqual(Model_pb2.ArrayFeatureType.FLOAT32, 65568)
        self.assertEqual(Model_pb2.ArrayFeatureType.FLOAT16, 65552)
        self.assertEqual(Model_pb2.ArrayFeatureType.INT32, 131104)
        # The two enums must never be conflated: 65568 is not a MIL dtype.
        with self.assertRaises(ValueError):
            MIL_pb2.DataType.Name(65568)

    def test_naming_helpers_use_official_enums(self):
        self.assertEqual(proto.mil_dtype_name(10), "FLOAT16")
        self.assertEqual(proto.mil_dtype_name(1), "BOOL")
        self.assertEqual(proto.feature_dtype_name(65568), "FLOAT32")
        self.assertEqual(proto.feature_dtype_name(131104), "INT32")

    def test_unknown_enum_values_are_explicit_never_blank(self):
        self.assertEqual(proto.mil_dtype_name(999), "UNRECOGNIZED(999)")
        self.assertEqual(proto.feature_dtype_name(999), "UNRECOGNIZED(999)")


class ModelParseTest(unittest.TestCase):
    def test_truncated_spec_raises_model_spec_error(self):
        model = Model_pb2.Model()
        model.specificationVersion = 9
        entry = model.description.input.add()
        entry.name = "x" * 64  # long enough that a mid-field cut truncates
        raw = model.SerializeToString()
        self.assertGreater(len(raw), 20)
        with self.assertRaises(proto.ModelSpecError) as ctx:
            proto.load_model(raw[:10])
        self.assertIn("not a parseable Core ML Model", str(ctx.exception))

    def test_random_bytes_raise_model_spec_error(self):
        with self.assertRaises(proto.ModelSpecError):
            proto.load_model(b"\xff\x27\x01this is not protobuf" * 8)

    def test_unknown_schema_fields_do_not_break_parse(self):
        # Forward compatibility: an unknown field (992, length-delimited)
        # appended after a valid spec must still parse.
        model = Model_pb2.Model()
        model.specificationVersion = 9
        raw = model.SerializeToString() + bytes([0x82, 0x3E, 0x03]) + b"abc"
        parsed = proto.load_model(raw)
        self.assertEqual(parsed.specificationVersion, 9)

    def test_unknown_fields_are_counted_not_hidden(self):
        # A package written by a schema newer than the vendored one:
        # parse succeeds, and the unknown field shows up in the scan
        # instead of silently disappearing.
        model = Model_pb2.Model()
        model.specificationVersion = 9
        raw = model.SerializeToString() + bytes([0x82, 0x3E, 0x03]) + b"abc"
        parsed = proto.load_model(raw)
        self.assertEqual(proto.count_unknown_fields(parsed), 1)
        self.assertEqual(proto.count_unknown_fields(model), 0)


if __name__ == "__main__":
    unittest.main()
