# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host tests for constexpr_lut_to_dense pre-expansion.

A small synthetic palettized package (uint4 indices, fp16 grouped lut,
blob storage with Apple's metadata layout) proves the expansion is
byte-exact and that the transformed package's op histogram contains no
``constexpr_lut_to_dense``.
"""

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml.depalettize import (
    ALIGNMENT,
    BLOB_METADATA,
    BLOB_RESERVED,
    STORAGE_HEADER,
    DepalettizeError,
    expand_lut_constants,
    read_blob_metadata,
    unpack_packed_uints,
)
from coreml.mlpackage import inspect
from coreml.proto import load_model
from coreml.schema import Model_pb2

FLOAT16 = 10
UINT4 = 35

# Distinct fp16 bit patterns per palette entry, including +0, -0, inf and
# NaN payloads: the gather must copy raw bytes, not recompute floats.
_LUT_PATTERNS = [
    0x0000, 0x8000, 0x3C00, 0xBC00, 0x7C00, 0xFC00, 0x7E01, 0x7E42,
    0x3555, 0xB555, 0x4248, 0xC248, 0x1E70, 0x9E70, 0x5A1B, 0xDA1B,
]

_WEIGHTS_REL = "com.apple.CoreML/weights"
_MODEL_REL = "com.apple.CoreML/model.mlmodel"


def _pack_uint4(values):
    """Apple PackSubByteVecImpl order: element 2k low nibble, 2k+1 high."""
    out = bytearray(len(values) // 2)
    for i, v in enumerate(values):
        out[i // 2] |= (v & 0xF) << (4 * (i % 2))
    return bytes(out)


def _blob_entry(handle, payload, dtype_code):
    handle.seek(0, 2)
    end = handle.tell()
    if end % ALIGNMENT:
        handle.write(b"\x00" * (ALIGNMENT - end % ALIGNMENT))
    metadata_offset = handle.tell()
    handle.write(
        BLOB_METADATA.pack(0xDEADBEEF, dtype_code, len(payload),
                           metadata_offset + 64, 0)
        + BLOB_RESERVED
    )
    remainder = len(payload) % ALIGNMENT
    handle.write(payload if remainder == 0 else
                 payload + b"\x00" * (ALIGNMENT - remainder))
    return metadata_offset


def _build_weight_file(path, indices, lut16):
    with path.open("wb") as handle:
        handle.write(b"\x00" * ALIGNMENT)
        handle.seek(0)
        handle.write(STORAGE_HEADER.pack(2, 2))
        indices_offset = _blob_entry(handle, _pack_uint4(indices), 11)
        lut_offset = _blob_entry(
            handle, np.asarray(lut16, dtype="<u2").tobytes(), 1
        )
    return indices_offset, lut_offset


def _tensor_binding(binding, dtype, shape, file_name, offset):
    value = binding.value
    tensor_type = value.type.tensorType
    tensor_type.dataType = dtype
    tensor_type.rank = len(shape)
    for dim in shape:
        tensor_type.dimensions.add().constant.size = dim
    value.blobFileValue.fileName = file_name
    value.blobFileValue.offset = offset


def _build_package(root, indices, lut16):
    weights_dir = root / "Data" / _WEIGHTS_REL
    weights_dir.mkdir(parents=True)
    weight_file = weights_dir / "weight.bin"
    indices_offset, lut_offset = _build_weight_file(weight_file, indices, lut16)

    spec = Model_pb2.Model()
    spec.specificationVersion = 9
    block = spec.mlProgram.functions["main"].block_specializations["CoreML8"]
    op = block.operations.add()
    op.type = "constexpr_lut_to_dense"
    _tensor_binding(
        op.inputs["indices"].arguments.add(),
        UINT4, [8, 12], "@model_path/weights/weight.bin", indices_offset,
    )
    _tensor_binding(
        op.inputs["lut"].arguments.add(),
        FLOAT16, [2, 1, 16, 1], "@model_path/weights/weight.bin", lut_offset,
    )
    output = op.outputs.add()
    output.name = "dense_weight"
    output.type.tensorType.dataType = FLOAT16
    output.type.tensorType.rank = 2
    output.type.tensorType.dimensions.add().constant.size = 8
    output.type.tensorType.dimensions.add().constant.size = 12

    model_dir = root / "Data" / "com.apple.CoreML"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.mlmodel").write_bytes(spec.SerializeToString())
    (root / "Manifest.json").write_text(
        json.dumps(
            {
                "fileFormatVersion": "1.0.0",
                "itemInfoEntries": {
                    "MODEL": {
                        "author": "test",
                        "description": "model",
                        "name": "model.mlmodel",
                        "path": _MODEL_REL,
                    },
                    "WEIGHTS": {
                        "author": "test",
                        "description": "weights",
                        "name": "weights",
                        "path": _WEIGHTS_REL,
                    },
                },
                "rootModelIdentifier": "MODEL",
            }
        )
    )
    return root


def _model_path(package):
    return package / "Data" / "com.apple.CoreML" / "model.mlmodel"


def _weights_dir(package):
    return package / "Data" / "com.apple.CoreML" / "weights"


class DepalettizeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_uint4_unpack_matches_apple_lsb_first_order(self):
        packed = bytes([0x21, 0xF0, 0x84])
        values = unpack_packed_uints(packed, 4, 6)
        self.assertEqual(values.tolist(), [1, 2, 0, 15, 4, 8])

    def test_expanded_const_is_byte_exact_and_histogram_is_clean(self):
        rng = np.random.default_rng(20260913)
        indices = rng.integers(0, 16, size=96).astype(np.uint8).tolist()
        lut16 = []
        for palette in range(2):
            lut16.extend(
                (_LUT_PATTERNS[entry] + palette * 0x0101) & 0xFFFF
                for entry in range(16)
            )
        package = _build_package(self.root / "pkg", indices, lut16)

        before = inspect(package)
        self.assertEqual(
            before["op_histogram"], {"constexpr_lut_to_dense": 1}
        )

        spec = load_model(_model_path(package).read_bytes())
        expansion = _weights_dir(package) / "weight.bin"
        expanded = expand_lut_constants(spec, _weights_dir(package), expansion)
        _model_path(package).write_bytes(spec.SerializeToString())

        self.assertEqual(len(expanded), 1)
        record = expanded[0]
        self.assertEqual(record.output_name, "dense_weight")
        self.assertEqual(record.shape, (8, 12))
        self.assertEqual(record.dtype_name, "fp16")
        self.assertEqual(record.element_count, 96)
        self.assertEqual(record.byte_count, 192)

        # Independent pure-Python expected gather: palette = row // 4
        # (block 8 / 2), one column palette (block 12 / 1), entry = the
        # uint4 index; the fp16 payload is copied as raw bytes.
        expected = bytearray()
        for p in range(96):
            row, _col = divmod(p, 12)
            palette = row // 4
            expected += struct.pack(
                "<H", lut16[palette * 16 + indices[p]]
            )
        with expansion.open("rb") as handle:
            metadata = read_blob_metadata(handle, record.blob_offset)
            self.assertEqual(metadata.dtype_code, 1)
            self.assertEqual(metadata.size_bytes, 192)
            handle.seek(metadata.data_offset)
            actual = handle.read(metadata.size_bytes)
        self.assertEqual(bytes(expected), actual)

        after = inspect(package)
        self.assertEqual(after["op_histogram"], {"const": 1})
        self.assertNotIn("constexpr_lut_to_dense", after["op_histogram"])
        fn = after["program"]["functions"][0]
        (op,) = fn["block_specializations"]["CoreML8"]["operations"]
        self.assertEqual(op["type"], "const")
        self.assertIn("val", op["attributes"])

    def test_named_errors_for_unsupported_palettization(self):
        rng = np.random.default_rng(7)
        indices = rng.integers(0, 16, size=96).astype(np.uint8).tolist()
        package = _build_package(self.root / "pkg", indices, list(range(32)))
        weights = _weights_dir(package)
        spec = load_model(_model_path(package).read_bytes())
        block = spec.mlProgram.functions["main"].block_specializations["CoreML8"]
        op = block.operations[0]

        lut_type = op.inputs["lut"].arguments[0].value.type.tensorType

        def set_shape(t, shape):
            t.ClearField("dimensions")
            for dim in shape:
                t.dimensions.add().constant.size = dim

        # Vector palettization (vector size 2) must be named.
        set_shape(lut_type, [2, 1, 16, 2])
        with self.assertRaisesRegex(DepalettizeError, "vector palettization"):
            expand_lut_constants(spec, weights, weights / "weight.bin")

        # Non-divisible indices dim must be named.
        set_shape(lut_type, [2, 1, 16, 1])
        set_shape(
            op.inputs["indices"].arguments[0].value.type.tensorType, [7, 12]
        )
        with self.assertRaisesRegex(DepalettizeError, "not divisible"):
            expand_lut_constants(spec, weights, weights / "weight.bin")

    def test_bad_sentinel_is_named(self):
        path = self.root / "weight.bin"
        with path.open("wb") as handle:
            handle.write(b"\x00" * (2 * ALIGNMENT))
            handle.seek(0)
            handle.write(STORAGE_HEADER.pack(1, 2))
            handle.seek(ALIGNMENT)
            handle.write(BLOB_METADATA.pack(1, 1, 0, 0, 0) + BLOB_RESERVED)
        with path.open("rb") as handle:
            with self.assertRaisesRegex(DepalettizeError, "sentinel"):
                read_blob_metadata(handle, ALIGNMENT)


if __name__ == "__main__":
    unittest.main()
