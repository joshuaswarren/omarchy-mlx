# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""The numpy MIL evaluator reproduces the authenticated golden encoder output
inside the reference lock's frozen contract."""

import hashlib
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "overlay" / "tools"
sys.path.insert(0, str(TOOLS))

from coreml.depalettize import DepalettizeError, _BlobAppender
from coreml.mil_adapter import emit_mlpackage
from coreml.mil_numpy import MilError, Program, validate
from coreml.reference import ReferenceLock, default_cache_root

LOCK_PATH = TOOLS / "coreml" / "parakeet-reference.lock"
LOCK = ReferenceLock.load(LOCK_PATH)
PACKAGE = default_cache_root() / LOCK.model_repo / LOCK.model_revision / "encoder.mlpackage"
CAPTURE = (
    default_cache_root()
    / "captures"
    / "b650695c-75aec2a"
    / "20260912T154759Z-librispeech"
    / "ane"
)
GOLDEN = ("encoder_input_features.npy", "encoder_input_mask.npy", "encoder_hidden.npy")


def test_encoder_matches_golden_capture(tmp_path):
    if not PACKAGE.is_dir():
        pytest.skip("pinned encoder.mlpackage is not installed")
    if not all((CAPTURE / name).is_file() for name in GOLDEN):
        pytest.skip("authenticated LibriSpeech ANE capture is not installed")
    for name in GOLDEN:
        digest = hashlib.sha256((CAPTURE / name).read_bytes()).hexdigest()
        assert digest == LOCK.macos_reference_paths[name], name

    source = emit_mlpackage(PACKAGE, tmp_path / "source", LOCK_PATH)
    report = validate(Path(source.mil_path).parent, CAPTURE, LOCK.to_dict()["numerical_contract"])

    assert report["ops_executed"] == 3351
    assert report["encoder_hidden_shape"] == [1, 375, 640]
    assert report["encoder_mask_equal"]
    assert report["within_frozen_contract"], report["measured"]


def _program(tmp_path, statements, signature=""):
    path = tmp_path / "model.mil"
    path.write_text(f"program(1.3) {{\nfunc main<ios18>({signature}) {{\n{statements}\n}} -> (y);\n}}")
    return Program(path, tmp_path)


@pytest.mark.parametrize(("decl", "literal", "expected"), [
    ("bool", "bool(false)", False),
    ("bool", "bool(true)", True),
    ("int32", "int32(256)", 256),
    ("string", 'string("custom")', "custom"),
    ("fp16", "fp16(0x1.002p+0)", np.float16(1)),
    ("fp32", "fp32(0x1.002p+0)", np.float32(float.fromhex("0x1.002p+0"))),
])
def test_original_scalar_constructors(tmp_path, decl, literal, expected):
    program = _program(tmp_path, f'{decl} y = const()[val = {literal}];')
    got = program.run({}, {"y"})["y"]
    assert got == expected
    if decl != "string":
        assert np.asarray(got).dtype == np.asarray(expected, dtype=decl.replace("fp", "float")).dtype


@pytest.mark.parametrize("literal", ["bool(maybe)", "bool(0)", "string(42)", "fp16(nope)"])
def test_malformed_scalar_is_refused(tmp_path, literal):
    program = _program(tmp_path, f'bool y = const()[val = {literal}];')
    with pytest.raises((MilError, ValueError)):
        program.run({}, {"y"})


def test_greater_broadcast_and_equality(tmp_path):
    program = _program(tmp_path, """
tensor<fp16, [3]> x = const()[val = tensor<fp16, [3]>([1, 2, 3])];
tensor<fp16, [1]> limit = const()[val = tensor<fp16, [1]>([2])];
tensor<bool, [3]> y = greater(x = x, y = limit);
""")
    got = program.run({}, {"y"})["y"]
    np.testing.assert_array_equal(got, [False, False, True])
    assert got.dtype == np.bool_


def _palette_program(tmp_path, *, indices_dtype="uint4", indices_shape=(2, 4),
                     lut_shape=(2, 1, 16, 1), indices_storage=11, lut_storage=1,
                     output_shape=(2, 4), output_dtype="fp16", extra=""):
    weights = tmp_path / "weights"
    weights.mkdir(exist_ok=True)
    appender = _BlobAppender(weights / "weight.bin")
    try:
        indices_offset = appender.append(bytes([0x10, 0x32, 0x54, 0x76]), indices_storage)
        lut = np.array([*range(16), *range(100, 116)], dtype="<f2")
        lut_offset = appender.append(lut.tobytes(), lut_storage)
    finally:
        appender.close()
    indices = f'tensor<{indices_dtype}, {list(indices_shape)}>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64({indices_offset})))'
    table = f'tensor<fp16, {list(lut_shape)}>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64({lut_offset})))'
    return _program(tmp_path, f'tensor<{output_dtype}, {list(output_shape)}> y = constexpr_lut_to_dense(indices = {indices}, lut = {table}{extra});')


def test_original_uint4_palette_copies_exact_fp16_bits(tmp_path):
    program = _palette_program(tmp_path)
    got = program.run({}, {"y"})["y"]
    expected = np.array([[0, 1, 2, 3], [104, 105, 106, 107]], dtype="<f2")
    assert got.tobytes() == expected.tobytes()
    assert got.shape == expected.shape and got.dtype == expected.dtype


@pytest.mark.parametrize("changes", [
    {"indices_dtype": "uint8"},
    {"indices_storage": 3},
    {"lut_storage": 2},
    {"lut_shape": (2, 16, 1)},
    {"lut_shape": (3, 1, 16, 1)},
    {"lut_shape": (0, 1, 16, 1)},
    {"lut_shape": (2, 1, 8, 2)},
    {"indices_shape": (2, 8)},
    {"output_shape": (4, 2)},
    {"output_dtype": "fp32"},
    {"extra": ", vector_axis = tensor<int32, []>(0)"},
])
def test_palette_rejects_invalid_dtype_shape_or_payload(tmp_path, changes):
    program = _palette_program(tmp_path, **changes)
    with pytest.raises((MilError, DepalettizeError)):
        program.run({}, {"y"})


def test_palette_rejects_truncated_blob(tmp_path):
    program = _palette_program(tmp_path)
    blob = tmp_path / "weights" / "weight.bin"
    with blob.open("r+b") as stream:
        stream.seek(192)
        _, _, size, payload = struct.unpack("<IIQQ", stream.read(24))
        stream.truncate(payload + size - 1)
    with pytest.raises((MilError, DepalettizeError)):
        program.run({}, {"y"})


def test_supported_linear_accumulates_fp32_then_rounds(tmp_path):
    program = _program(tmp_path, """
tensor<fp16, [1, 3]> x = const()[val = tensor<fp16, [1, 3]>([2048, 1, -2048])];
tensor<fp16, [1, 3]> w = const()[val = tensor<fp16, [1, 3]>([1, 1, 1])];
tensor<fp16, [1, 1]> y = linear(x = x, weight = w);
""")
    np.testing.assert_array_equal(program.run({}, {"y"})["y"], np.array([[1]], dtype=np.float16))


def test_standalone_cli_retains_its_existing_entrypoint():
    result = subprocess.run(
        [sys.executable, "-B", str(TOOLS / "coreml" / "mil_numpy.py"), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--reference-lock" in result.stdout


def test_supported_convolution_and_scalar_chain(tmp_path):
    program = _program(tmp_path, """
string pad_type = const()[val = string("custom")];
int32 groups = const()[val = int32(1)];
tensor<int32, [2]> steps = const()[val = tensor<int32, [2]>([1, 1])];
tensor<int32, [4]> pad = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];
tensor<fp16, [1, 1, 2, 2]> x = const()[val = tensor<fp16, [1, 1, 2, 2]>([1, 2, 3, 4])];
tensor<fp16, [1, 1, 2, 2]> w = const()[val = tensor<fp16, [1, 1, 2, 2]>([1, 0, 0, 1])];
tensor<fp16, [1, 1, 1, 1]> y = conv(x = x, weight = w, strides = steps, dilations = steps, groups = groups, pad_type = pad_type, pad = pad);
""")
    got = program.run({}, {"y"})["y"]
    assert got.dtype == np.float16 and got.shape == (1, 1, 1, 1)
    assert got.item() == 5


@pytest.mark.parametrize("parenthesized", [False, True], ids=["original", "adapter"])
def test_split_outputs_feed_both_glu_consumers(tmp_path, parenthesized):
    results = "tensor<fp16, [1, 2, 2]> left, tensor<fp16, [1, 2, 2]> right"
    if parenthesized:
        results = f"({results})"
    program = _program(tmp_path, f"""
tensor<fp16, [1, 4, 2]> x = const()[val = tensor<fp16, [1, 4, 2]>([2, 4, 6, 8, 0, 0, 0, 0])];
int32 axis = const()[val = int32(1)];
int32 count = const()[val = int32(2)];
{results} = split(axis = axis, num_splits = count, x = x)[name = string("split")];
tensor<fp16, [1, 2, 2]> gate = sigmoid(x = right);
tensor<fp16, [1, 2, 2]> y = mul(x = left, y = gate);
""")
    got = program.run({}, {"left", "right", "y"})
    np.testing.assert_array_equal(got["left"], np.array([[[2, 4], [6, 8]]], dtype=np.float16))
    np.testing.assert_array_equal(got["right"], np.zeros((1, 2, 2), dtype=np.float16))
    np.testing.assert_array_equal(got["y"], np.array([[[1, 2], [3, 4]]], dtype=np.float16))
    assert all(value.dtype == np.float16 for value in got.values())


@pytest.mark.parametrize("declaration", [
    "(tensor<fp16, [1]> left, tensor<fp16, [1]> right",
    "tensor<fp16, [1]> left, tensor<fp16, [1]> right)",
    "tensor<fp64, [1]> y",
    "uint64 y",
])
def test_unsupported_result_declaration_fails_before_execution(tmp_path, declaration):
    with pytest.raises(MilError, match="declaration|dtype"):
        _program(tmp_path, f"{declaration} = split(x = missing);")
