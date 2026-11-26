# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Strict constant access for the pinned Parakeet decoder and joint packages."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .mlpackage import open_mlpackage
from .proto import count_unknown_fields, load_model, mil_dtype_name
from .reference import ReferenceLock

_COMPONENTS = {"decoder", "joint"}
_IMMEDIATE = {
    "BOOL": ("bools", np.bool_),
    "INT32": ("ints", np.int32),
    "FLOAT32": ("floats", np.float32),
    "STRING": ("strings", np.str_),
}
_BLOB = {"FLOAT16": (1, np.dtype("<f2"))}
_BLOB_MAGIC = 0xDEADBEEF


class PinnedComponentError(ValueError):
    """The component differs from the pinned Parakeet package contract."""


@dataclass
class PinnedComponent:
    """One hash-verified MIL block with immutable named constant snapshots."""

    package_path: Path
    block: Any
    _constants: dict[str, np.ndarray] = field(repr=False)

    def constant(self, name: str) -> np.ndarray:
        """Return one named MIL ``const`` output as an immutable NumPy array."""
        try:
            return self._constants[name]
        except KeyError as exc:
            raise PinnedComponentError(f"pinned component has no constant {name!r}") from exc


class _SnapshotBlobReader:
    def __init__(self, files: dict[str, bytes], model_directory: PurePosixPath):
        self._files = files
        self._model_directory = model_directory

    def payload(self, value: Any) -> tuple[int, memoryview]:
        blob = value.blobFileValue
        if not blob.fileName.startswith("@model_path/"):
            raise PinnedComponentError(
                f"blob path must start with '@model_path/', got {blob.fileName!r}"
            )
        relative = PurePosixPath(blob.fileName.removeprefix("@model_path/"))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise PinnedComponentError(f"blob path escapes model root: {blob.fileName!r}")
        package_relative = self._model_directory.joinpath(*relative.parts).as_posix()
        try:
            data = self._files[package_relative]
        except KeyError as exc:
            raise PinnedComponentError(
                f"blob path does not resolve inside package: {blob.fileName!r}"
            ) from exc
        offset = int(blob.offset)
        if offset < 64 or offset + 64 > len(data):
            raise PinnedComponentError(
                f"blob offset {blob.offset} is not a record header in {blob.fileName!r}"
            )
        header = data[offset : offset + 64]
        magic, storage_code, payload_size, payload_offset = struct.unpack(
            "<IIQQ", header[:24]
        )
        if magic != _BLOB_MAGIC or any(header[24:]):
            raise PinnedComponentError(
                f"blob file has an invalid record at {blob.offset}"
            )
        payload_end = payload_offset + payload_size
        if payload_offset != offset + 64 or payload_end > len(data):
            raise PinnedComponentError(
                f"blob file has an invalid payload at {blob.offset}"
            )
        return storage_code, memoryview(data)[payload_offset:payload_end]


def _decode_constant(value: Any, blobs: _SnapshotBlobReader) -> np.ndarray:
    tensor_type = value.type.tensorType
    shape = []
    for dimension in tensor_type.dimensions:
        if dimension.WhichOneof("dimension") != "constant":
            raise PinnedComponentError("pinned component constant has a dynamic shape")
        shape.append(int(dimension.constant.size))
    shape = tuple(shape)
    dtype = mil_dtype_name(tensor_type.dataType)
    storage = value.WhichOneof("value")
    if storage == "blobFileValue":
        storage_code, payload = blobs.payload(value)
        try:
            expected_code, numpy_dtype = _BLOB[dtype]
        except KeyError as exc:
            raise PinnedComponentError(
                f"pinned component blob dtype {dtype} is unsupported"
            ) from exc
        expected_size = int(np.prod(shape, dtype=np.int64)) * numpy_dtype.itemsize
        if storage_code != expected_code or len(payload) != expected_size:
            raise PinnedComponentError(
                f"pinned component blob does not match {dtype}{shape}"
            )
        result = np.frombuffer(payload, dtype=numpy_dtype)
    elif storage == "immediateValue":
        tensor = value.immediateValue.tensor
        field_name = tensor.WhichOneof("value")
        if dtype == "FLOAT16":
            if field_name != "bytes":
                raise PinnedComponentError("pinned FLOAT16 immediate is not byte encoded")
            result = np.frombuffer(bytes(tensor.bytes.values), dtype=np.dtype("<f2"))
        else:
            try:
                expected_field, numpy_dtype = _IMMEDIATE[dtype]
            except KeyError as exc:
                raise PinnedComponentError(
                    f"pinned component immediate dtype {dtype} is unsupported"
                ) from exc
            if field_name != expected_field:
                raise PinnedComponentError(
                    f"pinned {dtype} immediate uses {field_name!r}, expected {expected_field!r}"
                )
            decoded = np.asarray(
                getattr(tensor, field_name).values, dtype=numpy_dtype
            )
            result = np.frombuffer(decoded.tobytes(), dtype=decoded.dtype)
    else:
        raise PinnedComponentError(
            f"pinned component constant storage {storage!r} is unsupported"
        )
    expected_count = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if result.size != expected_count:
        raise PinnedComponentError(
            f"pinned component constant has {result.size} values for shape {shape}"
        )
    result = result.reshape(shape)
    result.setflags(write=False)
    return result


def _read_package_snapshot(package: Any, component: str, lock: ReferenceLock) -> dict[str, bytes]:
    prefix = f"{component}.mlpackage/"
    expected = {
        entry.path[len(prefix) :]: entry
        for entry in lock.files
        if entry.path.startswith(prefix)
    }
    actual = {}
    for path in package.path.rglob("*"):
        if path.is_symlink():
            raise PinnedComponentError(
                f"{component}.mlpackage contains a symbolic link"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise PinnedComponentError(
                f"{component}.mlpackage contains a non-regular entry"
            )
        resolved = path.resolve()
        if not resolved.is_relative_to(package.path):
            raise PinnedComponentError(
                f"{component}.mlpackage contains an escaping file"
            )
        actual[path.relative_to(package.path).as_posix()] = resolved
    if set(actual) != set(expected):
        raise PinnedComponentError(
            f"{component}.mlpackage files differ from the pinned reference lock"
        )
    snapshot = {}
    for relative, path in actual.items():
        pin = expected[relative]
        data = path.read_bytes()
        if len(data) != pin.size or hashlib.sha256(data).hexdigest() != pin.sha256:
            raise PinnedComponentError(
                f"{component}.mlpackage/{relative} differs from the pinned reference lock"
            )
        snapshot[relative] = data
    return snapshot


def load_pinned_component(package_path: Path, component: str) -> PinnedComponent:
    """Load only the exact pinned decoder or joint package and its main MIL block."""
    if component not in _COMPONENTS:
        raise PinnedComponentError(f"unsupported pinned component {component!r}")
    package_path = Path(package_path).resolve()
    expected_name = f"{component}.mlpackage"
    if package_path.name != expected_name:
        raise PinnedComponentError(
            f"expected {expected_name}, got {package_path.name or str(package_path)!r}"
        )
    package = open_mlpackage(package_path)
    files = _read_package_snapshot(package, component, ReferenceLock.load())
    model_relative = package.model_path.relative_to(package.path).as_posix()
    model = load_model(files[model_relative])
    if model.specificationVersion != 9 or model.WhichOneof("Type") != "mlProgram":
        raise PinnedComponentError("pinned component is not the expected Core ML 9 MIL program")
    if count_unknown_fields(model):
        raise PinnedComponentError("pinned component contains unknown protobuf fields")
    if set(model.mlProgram.functions) != {"main"}:
        raise PinnedComponentError("pinned component must contain only the main function")
    function = model.mlProgram.functions["main"]
    if function.opset != "CoreML8" or set(function.block_specializations) != {"CoreML8"}:
        raise PinnedComponentError("pinned component must contain one CoreML8 block")
    block = function.block_specializations["CoreML8"]
    values = {}
    for operation in block.operations:
        if operation.type != "const":
            continue
        if len(operation.outputs) != 1 or "val" not in operation.attributes:
            raise PinnedComponentError("pinned component has an invalid const operation")
        output = operation.outputs[0]
        if output.name in values or output.type != operation.attributes["val"].type:
            raise PinnedComponentError("pinned component has an inconsistent const output")
        values[output.name] = operation.attributes["val"]
    blobs = _SnapshotBlobReader(files, PurePosixPath(model_relative).parent)
    constants = {name: _decode_constant(value, blobs) for name, value in values.items()}
    return PinnedComponent(package_path, block, constants)
