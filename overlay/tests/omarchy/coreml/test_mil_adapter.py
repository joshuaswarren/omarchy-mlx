# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Strict textual MIL adapter contracts and pinned encoder verification."""

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml.mil_adapter import (
    AdapterError,
    Emission,
    EmittedRecord,
    _Adapter,
    emit_mlpackage,
)
from coreml.mlpackage import open_mlpackage
from coreml.proto import load_model, mil_dtype_name
from coreml.schema import MIL_pb2, Model_pb2


def _tensor_type(container, dtype, shape):
    tensor = container.tensorType
    tensor.dataType = dtype
    tensor.rank = len(shape)
    for dimension in shape:
        tensor.dimensions.add().constant.size = dimension



def _array_feature(container, name, shape):
    feature = container.add()
    feature.name = name
    feature.type.multiArrayType.dataType = Model_pb2.ArrayFeatureType.FLOAT16
    feature.type.multiArrayType.shape.extend(shape)


def _string_value(value, text):
    _tensor_type(value.type, MIL_pb2.STRING, ())
    value.immediateValue.tensor.strings.values.append(text)


def _minimal_model() -> Model_pb2.Model:
    model = Model_pb2.Model()
    model.specificationVersion = 9
    program = model.mlProgram
    program.version = 1
    function = program.functions["main"]
    function.opset = "CoreML8"
    named = function.inputs.add()
    named.name = "x"
    _tensor_type(named.type, MIL_pb2.FLOAT16, (1,))
    operation = function.block_specializations["CoreML8"].operations.add()
    operation.type = "relu"
    operation.inputs["x"].arguments.add().name = "x"
    output = operation.outputs.add()
    output.name = "y"
    _tensor_type(output.type, MIL_pb2.FLOAT16, (1,))
    function.block_specializations["CoreML8"].outputs.append("y")
    _array_feature(model.description.input, "x", (1,))
    _array_feature(model.description.output, "y", (1,))
    return model


def _write_package(root: Path, model: Model_pb2.Model, weight: bytes | None = None) -> Path:
    package = root / "encoder.mlpackage"
    model_dir = package / "Data" / "com.apple.CoreML"
    model_dir.mkdir(parents=True)
    (model_dir / "model.mlmodel").write_bytes(model.SerializeToString())
    entries = {
        "model": {
            "author": "com.apple.CoreML",
            "name": "model.mlmodel",
            "path": "com.apple.CoreML/model.mlmodel",
        }
    }
    if weight is not None:
        weights = model_dir / "weights"
        weights.mkdir()
        (weights / "weight.bin").write_bytes(weight)
        entries["weights"] = {
            "author": "com.apple.CoreML",
            "name": "weights",
            "path": "com.apple.CoreML/weights",
        }
    manifest = {
        "fileFormatVersion": "1.0.0",
        "itemInfoEntries": entries,
        "rootModelIdentifier": "model",
    }
    (package / "Manifest.json").write_text(json.dumps(manifest))
    return package


def _blob_file(records: list[tuple[int, bytes]]) -> bytes:
    data = bytearray(struct.pack("<II", len(records), 2) + b"\0" * 56)
    for code, payload in records:
        data.extend(b"\0" * ((-len(data)) % 64))
        offset = len(data)
        data.extend(
            struct.pack("<IIQQ", 0xDEADBEEF, code, len(payload), offset + 64)
            + b"\0" * 40
        )
        data.extend(payload)
    return bytes(data)


def _blob_value(value, dtype, shape, offset):
    _tensor_type(value.type, dtype, shape)
    value.blobFileValue.fileName = "@model_path/weights/weight.bin"
    value.blobFileValue.offset = offset


def _shape(value_type):
    return tuple(int(item.constant.size) for item in value_type.tensorType.dimensions)


def _type_text(value_type):
    names = {
        "BOOL": "bool",
        "FLOAT16": "fp16",
        "FLOAT32": "fp32",
        "INT32": "int32",
        "STRING": "string",
        "UINT4": "uint4",
    }
    shape = ", ".join(map(str, _shape(value_type)))
    return f"tensor<{names[mil_dtype_name(value_type.tensorType.dataType)]}, [{shape}]>"


class AdapterUnitTest(unittest.TestCase):
    def test_cli_rejects_package_and_lock_errors_without_tracebacks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = _write_package(root, _minimal_model())
            lock = root / "invalid.lock"
            lock.write_text("not json")
            output = root / "output"
            for source in (root / "missing.mlpackage", package):
                with self.subTest(package=source):
                    result = subprocess.run(
                        [sys.executable, str(Path(_TOOLS) / "coreml" / "mil_adapter.py"),
                         "emit", str(source), str(output), "--reference-lock", str(lock)],
                        capture_output=True, text=True, check=False, timeout=15,
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertTrue(result.stderr.startswith("mil_adapter: "), result.stderr)
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertEqual(result.stdout, "")
                    self.assertFalse(output.exists())

    def _validate(self, model):
        with tempfile.TemporaryDirectory() as directory:
            package = open_mlpackage(_write_package(Path(directory), model))
            _Adapter(package, load_model(package.model_path.read_bytes())).validate()

    def test_refuses_unknown_nested_state_dynamic_and_multi_binding(self):
        model = _minimal_model()
        raw = model.SerializeToString() + bytes([0x82, 0x3E, 0x03]) + b"abc"
        with tempfile.TemporaryDirectory() as directory:
            package = open_mlpackage(
                _write_package(Path(directory), load_model(raw))
            )
            with self.assertRaisesRegex(AdapterError, "unknown protobuf"):
                _Adapter(package, load_model(package.model_path.read_bytes())).validate()

        model = _minimal_model()
        model.mlProgram.functions["main"].block_specializations["CoreML8"].operations[0].blocks.add()
        with self.assertRaisesRegex(AdapterError, "nested blocks"):
            self._validate(model)

        model = _minimal_model()
        model.description.state.add().name = "cache"
        with self.assertRaisesRegex(AdapterError, "state features"):
            self._validate(model)

        model = _minimal_model()
        tensor = model.mlProgram.functions["main"].inputs[0].type.tensorType
        tensor.dimensions[0].ClearField("constant")
        tensor.dimensions[0].unknown.variadic = False
        with self.assertRaisesRegex(AdapterError, "dynamic tensor"):
            self._validate(model)

        model = _minimal_model()
        array = model.description.input[0].type.multiArrayType
        array.shapeRange.sizeRanges.add(lowerBound=1, upperBound=2)
        with self.assertRaisesRegex(AdapterError, "dynamic model input"):
            self._validate(model)

        model = _minimal_model()
        argument = model.mlProgram.functions["main"].block_specializations["CoreML8"].operations[0].inputs["x"]
        argument.arguments.add().name = "x"
        with self.assertRaisesRegex(AdapterError, "2 bindings"):
            self._validate(model)

    def test_lsb_first_uint4_and_group_block_indexing(self):
        model = Model_pb2.Model()
        model.specificationVersion = 9
        program = model.mlProgram
        program.version = 1
        function = program.functions["main"]
        function.opset = "CoreML8"
        operation = function.block_specializations["CoreML8"].operations.add()
        operation.type = "constexpr_lut_to_dense"
        output = operation.outputs.add()
        output.name = "dense"
        _tensor_type(output.type, MIL_pb2.FLOAT16, (4, 4))
        _string_value(operation.attributes["name"], "dense")
        indices = bytes([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE])
        first_palette = b"".join(struct.pack("<H", 0x1000 + i) for i in range(16))
        second_palette = b"".join(struct.pack("<H", 0x2000 + i) for i in range(16))
        weight = _blob_file([(11, indices), (1, first_palette + second_palette)])
        second_offset = 64 + 64 + len(indices)
        second_offset = (second_offset + 63) & ~63
        _blob_value(
            operation.inputs["indices"].arguments.add().value,
            MIL_pb2.UINT4,
            (4, 4),
            64,
        )
        _blob_value(
            operation.inputs["lut"].arguments.add().value,
            MIL_pb2.FLOAT16,
            (2, 1, 16, 1),
            second_offset,
        )
        function.block_specializations["CoreML8"].outputs.append("dense")
        _array_feature(model.description.output, "dense", (4, 4))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = open_mlpackage(_write_package(root, model, weight))
            output_root = root / "out"
            emission = _Adapter(
                package, load_model(package.model_path.read_bytes())
            ).emit(output_root)
            record = emission.normalized_records[0]
            blob = Path(emission.normalized_blob_path).read_bytes()
            start = record.header_offset + 64
            actual = blob[start : start + record.payload_size]
            unpacked = [n for byte in indices for n in (byte & 0x0F, byte >> 4)]
            expected = b"".join(
                struct.pack("<H", (0x1000 if row < 2 else 0x2000) + unpacked[row * 4 + column])
                for row in range(4)
                for column in range(4)
            )
            self.assertEqual(actual, expected)
            self.assertEqual(struct.unpack("<II", blob[:8]), (1, 2))
            self.assertEqual(
                struct.unpack("<IIQQ", blob[64:88]),
                (0xDEADBEEF, 1, 32, 128),
            )

    def test_refuses_nonstandard_blob_header(self):
        model = _minimal_model()
        operation = model.mlProgram.functions["main"].block_specializations["CoreML8"].operations[0]
        value = operation.attributes["bad"]
        _blob_value(value, MIL_pb2.FLOAT16, (1,), 64)
        weight = _blob_file([(1, b"\0\0")])
        weight = weight[:64] + b"\0\0\0\0" + weight[68:]
        with tempfile.TemporaryDirectory() as directory:
            package = open_mlpackage(_write_package(Path(directory), model, weight))
            with self.assertRaisesRegex(AdapterError, "invalid record"):
                _Adapter(package, load_model(package.model_path.read_bytes())).validate()


class PinnedEncoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package_env = os.environ.get("MLX_OMARCHY_PINNED_ENCODER")
        if not package_env:
            raise unittest.SkipTest("MLX_OMARCHY_PINNED_ENCODER is not set")
        cls.package_path = Path(package_env)
        cls.package = open_mlpackage(cls.package_path)
        cls.model = load_model(cls.package.model_path.read_bytes())
        emission_env = os.environ.get("MLX_OMARCHY_PINNED_EMISSION")
        summary_env = os.environ.get("MLX_OMARCHY_PINNED_SUMMARY")
        cls.temporary = None
        if emission_env and summary_env:
            payload = json.loads(Path(summary_env).read_text())
            payload["normalized_records"] = [
                EmittedRecord(**item) for item in payload["normalized_records"]
            ]
            payload["immediate_records"] = [
                EmittedRecord(**item) for item in payload["immediate_records"]
            ]
            cls.emission = Emission(**payload)
            cls.output = Path(emission_env)
        else:
            cls.temporary = tempfile.TemporaryDirectory()
            cls.output = Path(cls.temporary.name) / "emission"
            lock = Path(_TOOLS) / "coreml" / "parakeet-reference.lock"
            cls.emission = emit_mlpackage(cls.package_path, cls.output, lock)

    @classmethod
    def tearDownClass(cls):
        if cls.temporary is not None:
            cls.temporary.cleanup()

    def test_all_results_constants_and_apple_lut_reference(self):
        program = self.model.mlProgram
        function = program.functions["main"]
        block = function.block_specializations["CoreML8"]
        emission = self.emission
        self.assertEqual(program.version, emission.program_version, 1)
        self.assertEqual(function.opset, emission.function_opset, "CoreML8")
        self.assertEqual(emission.operation_count, 3351)
        self.assertEqual(emission.operation_result_count, 3375)
        self.assertEqual(emission.split_count, 24)
        self.assertEqual(emission.split_result_count, 48)
        self.assertEqual(emission.normalized_constexpr_count, 194)
        self.assertEqual(emission.dense_fp16_payload_bytes, 1016332288)
        self.assertEqual(emission.fp16_immediate_count, 120)
        self.assertEqual(emission.fp16_immediate_payload_bytes, 240)

        text = Path(emission.mil_path).read_text()
        self.assertTrue(text.startswith("program(1)\n{\n  func main<CoreML8>("))
        self.assertNotIn("constexpr_lut_to_dense", text)
        operation_lines = [line for line in text.splitlines() if line.startswith("    ")]
        self.assertEqual(len(operation_lines), len(block.operations))
        for source, line in zip(block.operations, operation_lines):
            results = [f"{_type_text(item.type)} {item.name}" for item in source.outputs]
            lhs = results[0] if len(results) == 1 else f"({', '.join(results)})"
            emitted_type = "const" if source.type == "constexpr_lut_to_dense" else source.type
            self.assertTrue(line.startswith(f"    {lhs} = {emitted_type}("), line)
            for key, argument in source.inputs.items():
                binding = argument.arguments[0]
                if binding.WhichOneof("binding") == "name":
                    self.assertIn(f"{key} = {binding.name}", line)
        used_names = [
            binding.name
            for operation in block.operations
            for argument in operation.inputs.values()
            for binding in argument.arguments
            if binding.WhichOneof("binding") == "name"
        ]
        for operation in block.operations:
            if operation.type == "split":
                for output in operation.outputs:
                    self.assertEqual(used_names.count(output.name), 1)
        self.assertIn(f"  }} -> ({', '.join(block.outputs)});", text)

        immediate_source = [
            bytes(value.immediateValue.tensor.bytes.values)
            for operation in block.operations
            for value in _Adapter._operation_values(operation)
            if value.WhichOneof("value") == "immediateValue"
            and value.immediateValue.WhichOneof("value") == "tensor"
            and value.immediateValue.tensor.WhichOneof("value") == "bytes"
        ]
        self.assertEqual(len(immediate_source), 120)
        self.assertIn(struct.pack("<H", 0xFC00), immediate_source)
        normalized_blob = Path(emission.normalized_blob_path)
        with normalized_blob.open("rb") as stream:
            for source, record in zip(immediate_source, emission.immediate_records):
                stream.seek(record.header_offset + 64)
                self.assertEqual(stream.read(record.payload_size), source)

        import numpy as np
        from coremltools.converters.mil.mil.ops.defs.iOS18.compression import constexpr_lut_to_dense
        from coremltools.libmilstoragepython import _BlobStorageReader

        source_reader = _BlobStorageReader(str(self.package.weight_files[0].path))
        output_reader = _BlobStorageReader(str(normalized_blob))
        lut_ops = [op for op in block.operations if op.type == "constexpr_lut_to_dense"]
        self.assertEqual(len(lut_ops), len(emission.normalized_records))
        checked_bytes = 0
        for operation, record in zip(lut_ops, emission.normalized_records):
            indices_value = operation.inputs["indices"].arguments[0].value
            lut_value = operation.inputs["lut"].arguments[0].value
            indices = np.asarray(
                source_reader.read_uint4_data(indices_value.blobFileValue.offset),
                dtype=np.uint8,
            ).reshape(_shape(indices_value.type))
            lut_bits = np.asarray(
                source_reader.read_fp16_data(lut_value.blobFileValue.offset),
                dtype=np.uint16,
            )
            lut = lut_bits.view(np.float16).reshape(_shape(lut_value.type))
            expected = constexpr_lut_to_dense.decompress(indices, lut, None)
            actual = np.asarray(
                output_reader.read_fp16_data(record.header_offset), dtype=np.uint16
            ).reshape(_shape(operation.outputs[0].type))
            np.testing.assert_array_equal(actual, expected.view(np.uint16))
            checked_bytes += actual.nbytes
        self.assertEqual(checked_bytes, 1016332288)


if __name__ == "__main__":
    unittest.main()
