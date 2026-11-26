#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Validate a pinned Core ML ML Program and emit ordinary textual MIL inputs.

The adapter preserves the protobuf operation order and typed SSA bindings. It
normalizes CoreML8 scalar UINT4 ``constexpr_lut_to_dense`` operations into
standard FP16 BLOBFILE constants without materializing a complete dense weight.
Compilation is an explicit CLI action and a compiler failure remains a failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coreml.mlpackage import MlPackage, MlPackageError, open_mlpackage
from coreml.proto import count_unknown_fields, feature_dtype_name, load_model, mil_dtype_name
from coreml.reference import ReferenceError, ReferenceLock, validate_lock

EXPECTED_REPOSITORY = "mweinbach1/parakeet-tdt-0.6b-v3-coreml"
EXPECTED_REVISION = "b650695c2322ee5281dff48d7345b2f3a58ff018"
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_@]*\Z")
_MAGIC = 0xDEADBEEF
_BLOB_VERSION = 2
_FP16_STORAGE = 1
_STORAGE_CODE = {"FLOAT16": _FP16_STORAGE, "UINT4": 11, "INT32": 14}
_DTYPE = {
    "BOOL": "bool",
    "FLOAT16": "fp16",
    "FLOAT32": "fp32",
    "INT8": "int8",
    "INT32": "int32",
    "STRING": "string",
    "UINT4": "uint4",
    "UINT64": "uint64",
}
_IMMEDIATE_FIELD = {
    "BOOL": "bools",
    "FLOAT16": "bytes",
    "INT32": "ints",
    "STRING": "strings",
}


class AdapterError(RuntimeError):
    """The package cannot be represented by the supported textual MIL subset."""


@dataclass(frozen=True)
class BlobRecord:
    path: Path
    header_offset: int
    storage_code: int
    payload_size: int
    payload_offset: int


@dataclass(frozen=True)
class EmittedRecord:
    name: str
    header_offset: int
    payload_size: int
    payload_sha256: str


@dataclass(frozen=True)
class Emission:
    source_model_sha256: str
    source_weight_sha256: dict[str, str]
    source_repository: str
    source_revision: str
    program_version: int
    function_opset: str
    operation_count: int
    operation_result_count: int
    split_count: int
    split_result_count: int
    normalized_constexpr_count: int
    dense_fp16_payload_bytes: int
    fp16_immediate_count: int
    fp16_immediate_payload_bytes: int
    mil_path: str
    mil_bytes: int
    mil_sha256: str
    model_root: str
    normalized_blob_path: str
    normalized_blob_bytes: int
    normalized_blob_sha256: str
    source_weight_materialization: dict[str, str]
    normalized_records: list[EmittedRecord]
    immediate_records: list[EmittedRecord]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CoreMLBlobWriter:
    """Small standard Core ML blob-v2 writer for exact FP16 payload bytes."""

    def __init__(self, path: Path, record_count: int):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("xb")
        self._record_count = record_count
        self._written_records = 0
        self._stream.write(struct.pack("<II", record_count, _BLOB_VERSION))
        self._stream.write(b"\0" * 56)

    def add_fp16(self, payload_size: int, write_payload: Callable[[Callable[[bytes], None]], None]) -> tuple[int, str]:
        if payload_size < 0 or payload_size % 2:
            raise AdapterError(f"FP16 blob payload has invalid size {payload_size}")
        padding = (-self._stream.tell()) % 64
        if padding:
            self._stream.write(b"\0" * padding)
        header_offset = self._stream.tell()
        payload_offset = header_offset + 64
        self._stream.write(
            struct.pack("<IIQQ", _MAGIC, _FP16_STORAGE, payload_size, payload_offset)
        )
        self._stream.write(b"\0" * 40)
        digest = hashlib.sha256()
        written = 0

        def sink(chunk: bytes) -> None:
            nonlocal written
            if not isinstance(chunk, bytes):
                raise AdapterError("blob payload writer produced non-bytes data")
            written += len(chunk)
            if written > payload_size:
                raise AdapterError("blob payload writer exceeded its declared byte count")
            digest.update(chunk)
            self._stream.write(chunk)

        write_payload(sink)
        if written != payload_size:
            raise AdapterError(
                f"blob payload writer produced {written} bytes, expected {payload_size}"
            )
        self._written_records += 1
        return header_offset, digest.hexdigest()


    def __enter__(self) -> CoreMLBlobWriter:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stream.close()
        if exc_type is None and self._written_records != self._record_count:
            raise AdapterError(
                f"wrote {self._written_records} blob records, expected {self._record_count}"
            )


class _Adapter:
    def __init__(self, package: MlPackage, model: Any):
        self.package = package
        self.model = model
        self.program = model.mlProgram
        self.function: Any = None
        self.block: Any = None
        self._blob_files: dict[Path, dict[int, BlobRecord]] = {}
        self._normalized: list[EmittedRecord] = []
        self._immediates: list[EmittedRecord] = []
        self._dense_bytes = 0
        self._immediate_bytes = 0
        self._writer: CoreMLBlobWriter | None = None

    def validate(self) -> None:
        unknown = count_unknown_fields(self.model)
        if unknown:
            raise AdapterError(f"model contains {unknown} unknown protobuf field(s)")
        if self.model.WhichOneof("Type") != "mlProgram":
            raise AdapterError("model type must be mlProgram")
        if self.model.specificationVersion != 9:
            raise AdapterError(
                f"model specification version must be 9, got {self.model.specificationVersion}"
            )
        if self.model.isUpdatable:
            raise AdapterError("updatable model semantics are not representable")
        if self.model.description.state:
            raise AdapterError("state features are not representable")
        if self.program.version != 1:
            raise AdapterError(f"program version must be 1, got {self.program.version}")
        if self.program.docString:
            raise AdapterError("program docString is not representable")
        if self.program.attributes:
            raise AdapterError("program attributes are not representable")
        if set(self.program.functions) != {"main"}:
            raise AdapterError("exactly one function named 'main' is required")
        self.function = self.program.functions["main"]
        if self.function.opset != "CoreML8":
            raise AdapterError(
                f"function opset must be CoreML8, got {self.function.opset!r}"
            )
        if self.function.attributes:
            raise AdapterError("function attributes are not representable")
        if set(self.function.block_specializations) != {"CoreML8"}:
            raise AdapterError("function must have only the CoreML8 specialization")
        self.block = self.function.block_specializations["CoreML8"]
        if self.block.inputs:
            raise AdapterError("block inputs are not representable")
        if self.block.attributes:
            raise AdapterError("block attributes are not representable")

        defined: set[str] = set()
        defined_types: dict[str, Any] = {}
        for value in self.function.inputs:
            self._validate_identifier(value.name, "function input")
            if value.name in defined:
                raise AdapterError(f"duplicate SSA name {value.name!r}")
            self._type_text(value.type)
            defined.add(value.name)
            defined_types[value.name] = value.type

        for index, op in enumerate(self.block.operations):
            self._validate_identifier(op.type, f"operation #{index} type")
            if op.blocks:
                raise AdapterError(
                    f"operation #{index} {op.type!r} contains nested blocks"
                )
            if not op.outputs:
                raise AdapterError(f"operation #{index} {op.type!r} has no outputs")
            for key, argument in op.inputs.items():
                self._validate_identifier(key, f"operation #{index} input key")
                if len(argument.arguments) != 1:
                    raise AdapterError(
                        f"operation #{index} {op.type!r} input {key!r} has "
                        f"{len(argument.arguments)} bindings; only one is representable"
                    )
                binding = argument.arguments[0]
                kind = binding.WhichOneof("binding")
                if kind == "name":
                    if binding.name not in defined:
                        raise AdapterError(
                            f"operation #{index} {op.type!r} input {key!r} "
                            f"references undefined SSA name {binding.name!r}"
                        )
                elif kind == "value":
                    self._validate_value(binding.value)
                else:
                    raise AdapterError(
                        f"operation #{index} {op.type!r} input {key!r} is unset"
                    )
            for key, value in op.attributes.items():
                self._validate_identifier(key, f"operation #{index} attribute key")
                self._validate_value(value)
            for output in op.outputs:
                self._validate_identifier(output.name, f"operation #{index} output")
                self._type_text(output.type)
                if output.name in defined:
                    raise AdapterError(f"duplicate SSA name {output.name!r}")
                defined.add(output.name)
                defined_types[output.name] = output.type
            if op.type == "constexpr_lut_to_dense":
                self._validate_lut_op(op, index)

        for output in self.block.outputs:
            self._validate_identifier(output, "block output")
            if output not in defined:
                raise AdapterError(f"block output {output!r} is not defined")
        self._validate_model_boundary(defined_types)

    def _validate_model_boundary(self, defined_types: dict[str, Any]) -> None:
        for label, features, names in (
            ("input", self.model.description.input, [item.name for item in self.function.inputs]),
            ("output", self.model.description.output, list(self.block.outputs)),
        ):
            if [item.name for item in features] != names:
                raise AdapterError(f"model {label} names do not match the MIL function")
            for feature in features:
                if feature.type.WhichOneof("Type") != "multiArrayType":
                    raise AdapterError(f"model {label} {feature.name!r} is not a multi-array")
                if feature.type.isOptional:
                    raise AdapterError(f"optional model {label} {feature.name!r} is not representable")
                array = feature.type.multiArrayType
                if array.WhichOneof("ShapeFlexibility") is not None:
                    raise AdapterError(f"dynamic model {label} {feature.name!r} is not representable")
                if array.WhichOneof("defaultOptionalValue") is not None:
                    raise AdapterError(f"default model {label} {feature.name!r} is not representable")
                value_type = defined_types[feature.name]
                if tuple(array.shape) != self._shape(value_type):
                    raise AdapterError(f"model {label} {feature.name!r} shape differs from MIL")
                if feature_dtype_name(array.dataType) != mil_dtype_name(
                    value_type.tensorType.dataType
                ):
                    raise AdapterError(f"model {label} {feature.name!r} dtype differs from MIL")

    @staticmethod
    def _validate_identifier(name: str, context: str) -> None:
        if not _IDENTIFIER.fullmatch(name):
            raise AdapterError(f"{context} has invalid MIL identifier {name!r}")

    def _shape(self, value_type: Any) -> tuple[int, ...]:
        if value_type.WhichOneof("type") != "tensorType":
            raise AdapterError(
                f"value type {value_type.WhichOneof('type')!r} is not representable"
            )
        tensor = value_type.tensorType
        if tensor.rank != len(tensor.dimensions):
            raise AdapterError(
                f"tensor rank {tensor.rank} does not match {len(tensor.dimensions)} dimensions"
            )
        shape = []
        for dimension in tensor.dimensions:
            if dimension.WhichOneof("dimension") != "constant":
                raise AdapterError("dynamic tensor dimensions are not representable")
            shape.append(int(dimension.constant.size))
        return tuple(shape)

    def _type_text(self, value_type: Any) -> str:
        shape = self._shape(value_type)
        dtype_name = mil_dtype_name(value_type.tensorType.dataType)
        try:
            dtype = _DTYPE[dtype_name]
        except KeyError as exc:
            raise AdapterError(f"MIL dtype {dtype_name} is not representable") from exc
        return f"tensor<{dtype}, [{', '.join(map(str, shape))}]>"

    @staticmethod
    def _element_count(shape: tuple[int, ...]) -> int:
        result = 1
        for dimension in shape:
            result *= dimension
        return result

    def _validate_value(self, value: Any) -> None:
        if value.docString:
            raise AdapterError("value docString is not representable")
        shape = self._shape(value.type)
        dtype = mil_dtype_name(value.type.tensorType.dataType)
        storage = value.WhichOneof("value")
        if storage == "blobFileValue":
            record = self._blob_record(value)
            expected_code = _STORAGE_CODE.get(dtype)
            if expected_code is None or record.storage_code != expected_code:
                raise AdapterError(
                    f"blob storage code {record.storage_code} does not match dtype {dtype}"
                )
            elements = self._element_count(shape)
            expected_size = (elements + 1) // 2 if dtype == "UINT4" else elements * (2 if dtype == "FLOAT16" else 4)
            if record.payload_size != expected_size:
                raise AdapterError(
                    f"blob payload has {record.payload_size} bytes for {dtype}{shape}, "
                    f"expected {expected_size}"
                )
            return
        if storage != "immediateValue":
            raise AdapterError(f"value storage {storage!r} is not representable")
        immediate = value.immediateValue
        if immediate.WhichOneof("value") != "tensor":
            raise AdapterError(
                f"immediate {immediate.WhichOneof('value')!r} is not representable"
            )
        tensor = immediate.tensor
        field = tensor.WhichOneof("value")
        expected = _IMMEDIATE_FIELD.get(dtype)
        if field != expected:
            raise AdapterError(
                f"immediate field {field!r} does not exactly represent dtype {dtype}"
            )
        count = self._element_count(shape)
        if field == "bytes":
            actual = len(tensor.bytes.values)
            expected_bytes = count * 2
            if actual != expected_bytes:
                raise AdapterError(
                    f"FP16 immediate has {actual} bytes, expected {expected_bytes}"
                )
        else:
            values = getattr(tensor, field).values
            if len(values) != count:
                raise AdapterError(
                    f"{dtype} immediate has {len(values)} values, expected {count}"
                )
            if field == "strings":
                for text in values:
                    try:
                        text.encode("ascii")
                    except UnicodeEncodeError as exc:
                        raise AdapterError("non-ASCII MIL strings are not representable") from exc

    def _blob_record(self, value: Any) -> BlobRecord:
        blob = value.blobFileValue
        if not blob.fileName.startswith("@model_path/"):
            raise AdapterError(
                f"blob path must start with '@model_path/', got {blob.fileName!r}"
            )
        relative = PurePosixPath(blob.fileName.removeprefix("@model_path/"))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise AdapterError(f"blob path escapes model root: {blob.fileName!r}")
        path = (self.package.model_path.parent / Path(*relative.parts)).resolve()
        if not path.is_relative_to(self.package.path) or not path.is_file():
            raise AdapterError(f"blob path does not resolve inside package: {blob.fileName!r}")
        records = self._blob_files.get(path)
        if records is None:
            records = self._read_blob_file(path)
            self._blob_files[path] = records
        try:
            return records[int(blob.offset)]
        except KeyError as exc:
            raise AdapterError(
                f"blob offset {blob.offset} is not a record header in {blob.fileName!r}"
            ) from exc

    @staticmethod
    def _read_blob_file(path: Path) -> dict[int, BlobRecord]:
        size = path.stat().st_size
        if size < 64:
            raise AdapterError(f"blob file {path} is shorter than its global header")
        records: dict[int, BlobRecord] = {}
        with path.open("rb") as stream:
            global_header = stream.read(64)
            count, version = struct.unpack("<II", global_header[:8])
            if version != _BLOB_VERSION or any(global_header[8:]):
                raise AdapterError(f"blob file {path} has an invalid global header")
            offset = 64
            last_end = 64
            for _ in range(count):
                if offset + 64 > size:
                    raise AdapterError(f"blob file {path} has a truncated record header")
                stream.seek(offset)
                header = stream.read(64)
                magic, code, payload_size, payload_offset = struct.unpack(
                    "<IIQQ", header[:24]
                )
                if magic != _MAGIC or any(header[24:]):
                    raise AdapterError(f"blob file {path} has an invalid record at {offset}")
                if payload_offset != offset + 64 or payload_offset + payload_size > size:
                    raise AdapterError(f"blob file {path} has an invalid payload at {offset}")
                records[offset] = BlobRecord(
                    path, offset, code, payload_size, payload_offset
                )
                last_end = payload_offset + payload_size
                offset = (last_end + 63) & ~63
            if count == 0 or last_end != size:
                raise AdapterError(
                    f"blob file {path} record count does not cover the complete file"
                )
        return records

    def _validate_lut_op(self, op: Any, index: int) -> None:
        if set(op.inputs) != {"indices", "lut"}:
            raise AdapterError(
                f"constexpr_lut_to_dense #{index} requires only indices and lut"
            )
        if set(op.attributes) != {"name"}:
            raise AdapterError(
                f"constexpr_lut_to_dense #{index} has unrepresentable attributes"
            )
        if len(op.outputs) != 1:
            raise AdapterError(
                f"constexpr_lut_to_dense #{index} must have exactly one output"
            )
        values = {}
        for key in ("indices", "lut"):
            binding = op.inputs[key].arguments[0]
            if binding.WhichOneof("binding") != "value":
                raise AdapterError(
                    f"constexpr_lut_to_dense #{index} {key} must be an inline constant"
                )
            if binding.value.WhichOneof("value") != "blobFileValue":
                raise AdapterError(
                    f"constexpr_lut_to_dense #{index} {key} must use BLOBFILE storage"
                )
            values[key] = binding.value
        indices_shape = self._shape(values["indices"].type)
        lut_shape = self._shape(values["lut"].type)
        output_shape = self._shape(op.outputs[0].type)
        if mil_dtype_name(values["indices"].type.tensorType.dataType) != "UINT4":
            raise AdapterError(f"constexpr_lut_to_dense #{index} indices must be UINT4")
        if mil_dtype_name(values["lut"].type.tensorType.dataType) != "FLOAT16":
            raise AdapterError(f"constexpr_lut_to_dense #{index} lut must be FP16")
        if mil_dtype_name(op.outputs[0].type.tensorType.dataType) != "FLOAT16":
            raise AdapterError(f"constexpr_lut_to_dense #{index} output must be FP16")
        if len(indices_shape) != 2 or len(lut_shape) != 4:
            raise AdapterError(
                f"constexpr_lut_to_dense #{index} requires rank-2 scalar palettization"
            )
        if lut_shape[-2:] != (16, 1):
            raise AdapterError(
                f"constexpr_lut_to_dense #{index} requires UINT4 scalar palettes"
            )
        if output_shape != indices_shape:
            raise AdapterError(
                f"constexpr_lut_to_dense #{index} output shape differs from indices"
            )
        if any(i == 0 or l == 0 or i % l for i, l in zip(indices_shape, lut_shape[:2])):
            raise AdapterError(
                f"constexpr_lut_to_dense #{index} has invalid block geometry"
            )
        if indices_shape[1] % 2 or (indices_shape[1] // lut_shape[1]) % 2:
            raise AdapterError(
                f"constexpr_lut_to_dense #{index} has unrepresentable UINT4 row packing"
            )

    def _value_text(self, value: Any) -> str:
        value_type = self._type_text(value.type)
        storage = value.WhichOneof("value")
        if storage == "blobFileValue":
            blob = value.blobFileValue
            rendered = (
                f"BLOBFILE(path = string({self._quote(blob.fileName)}), "
                f"offset = uint64({int(blob.offset)}))"
            )
            return f"{value_type}({rendered})"
        tensor = value.immediateValue.tensor
        dtype = mil_dtype_name(value.type.tensorType.dataType)
        field = tensor.WhichOneof("value")
        if field == "bytes":
            raw = bytes(tensor.bytes.values)
            assert self._writer is not None
            offset, digest = self._writer.add_fp16(len(raw), lambda sink: sink(raw))
            self._immediate_bytes += len(raw)
            self._immediates.append(
                EmittedRecord("immediate", offset, len(raw), digest)
            )
            rendered = (
                'BLOBFILE(path = string("@model_path/weights/normalized.bin"), '
                f"offset = uint64({offset}))"
            )
        else:
            values = list(getattr(tensor, field).values)
            if dtype == "BOOL":
                text = ["true" if item else "false" for item in values]
            elif dtype == "STRING":
                text = [self._quote(item) for item in values]
            else:
                text = [str(item) for item in values]
            rendered = text[0] if len(text) == 1 else f"[{', '.join(text)}]"
        return f"{value_type}({rendered})"

    @staticmethod
    def _quote(text: str) -> str:
        text.encode("ascii")
        return json.dumps(text, ensure_ascii=True)

    def _binding_text(self, binding: Any) -> str:
        if binding.WhichOneof("binding") == "name":
            return binding.name
        return self._value_text(binding.value)

    def _map_text(self, mapping: Any) -> str:
        return ", ".join(
            f"{name} = {self._binding_text(mapping[name].arguments[0])}"
            for name in sorted(mapping)
        )

    def _attributes_text(self, mapping: Any) -> str:
        return ", ".join(
            f"{name} = {self._value_text(mapping[name])}" for name in sorted(mapping)
        )

    def _result_text(self, op: Any) -> str:
        results = [f"{self._type_text(out.type)} {out.name}" for out in op.outputs]
        return results[0] if len(results) == 1 else f"({', '.join(results)})"

    def _normalize_lut(self, op: Any) -> str:
        import numpy as np

        indices_value = op.inputs["indices"].arguments[0].value
        lut_value = op.inputs["lut"].arguments[0].value
        indices_record = self._blob_record(indices_value)
        lut_record = self._blob_record(lut_value)
        indices_shape = self._shape(indices_value.type)
        lut_shape = self._shape(lut_value.type)
        payload_size = self._element_count(indices_shape) * 2
        assert self._writer is not None

        def write_dense(sink: Callable[[bytes], None]) -> None:
            rows, columns = indices_shape
            group_rows, group_columns = lut_shape[:2]
            row_block = rows // group_rows
            column_block = columns // group_columns
            with lut_record.path.open("rb") as source:
                source.seek(lut_record.payload_offset)
                lut_raw = source.read(lut_record.payload_size)
            if len(lut_raw) != lut_record.payload_size:
                raise AdapterError("truncated FP16 LUT payload")
            lut = np.frombuffer(lut_raw, dtype="<u2").reshape(
                group_rows, group_columns, 16
            )
            with indices_record.path.open("rb") as source:
                for row in range(rows):
                    source.seek(indices_record.payload_offset + row * columns // 2)
                    packed = source.read(columns // 2)
                    if len(packed) != columns // 2:
                        raise AdapterError("truncated UINT4 indices payload")
                    packed_array = np.frombuffer(packed, dtype=np.uint8)
                    indices = np.empty(columns, dtype=np.uint8)
                    indices[0::2] = packed_array & 0x0F
                    indices[1::2] = packed_array >> 4
                    dense = np.empty(columns, dtype="<u2")
                    row_group = row // row_block
                    for column_group in range(group_columns):
                        start = column_group * column_block
                        end = start + column_block
                        dense[start:end] = lut[
                            row_group, column_group, indices[start:end]
                        ]
                    sink(dense.tobytes())

        offset, digest = self._writer.add_fp16(payload_size, write_dense)
        self._dense_bytes += payload_size
        self._normalized.append(
            EmittedRecord(op.outputs[0].name, offset, payload_size, digest)
        )
        attributes = dict(op.attributes)
        rendered_attributes = self._attributes_text(attributes)
        if rendered_attributes:
            rendered_attributes += ", "
        rendered_attributes += (
            f"val = {self._type_text(op.outputs[0].type)}(BLOBFILE("
            'path = string("@model_path/weights/normalized.bin"), '
            f"offset = uint64({offset})))"
        )
        return f"    {self._result_text(op)} = const()[{rendered_attributes}];"

    def emit(self, root: Path) -> Emission:
        self.validate()
        operations = list(self.block.operations)
        normalized_count = sum(op.type == "constexpr_lut_to_dense" for op in operations)
        immediate_count = sum(
            1
            for op in operations
            for value in self._operation_values(op)
            if value.WhichOneof("value") == "immediateValue"
            and value.immediateValue.WhichOneof("value") == "tensor"
            and value.immediateValue.tensor.WhichOneof("value") == "bytes"
        )
        model_root = root / "model-root"
        materialization = self._materialize_weights(model_root)
        normalized_blob = model_root / "weights" / "normalized.bin"
        lines = [
            f"program({int(self.program.version)})",
            "{",
            "  func main<CoreML8>("
            + ", ".join(
                f"{self._type_text(item.type)} {item.name}"
                for item in self.function.inputs
            )
            + ") {",
        ]
        with CoreMLBlobWriter(
            normalized_blob, normalized_count + immediate_count
        ) as writer:
            self._writer = writer
            for op in operations:
                if op.type == "constexpr_lut_to_dense":
                    lines.append(self._normalize_lut(op))
                else:
                    lines.append(
                        f"    {self._result_text(op)} = {op.type}({self._map_text(op.inputs)})"
                        f"[{self._attributes_text(op.attributes)}];"
                    )
            self._writer = None
        lines.extend([f"  }} -> ({', '.join(self.block.outputs)});", "}", ""])
        text = "\n".join(lines)
        mil_path = root / "model.mil"
        mil_path.write_text(text, encoding="utf-8")
        source_model_sha256 = hashlib.sha256(
            self.package.model_path.read_bytes()
        ).hexdigest()
        weight_hashes = {item.relative: item.sha256 for item in self.package.weight_files}
        split_ops = [op for op in operations if op.type == "split"]
        return Emission(
            source_model_sha256=source_model_sha256,
            source_weight_sha256=weight_hashes,
            source_repository=EXPECTED_REPOSITORY,
            source_revision=EXPECTED_REVISION,
            program_version=int(self.program.version),
            function_opset=self.function.opset,
            operation_count=len(operations),
            operation_result_count=sum(len(op.outputs) for op in operations),
            split_count=len(split_ops),
            split_result_count=sum(len(op.outputs) for op in split_ops),
            normalized_constexpr_count=len(self._normalized),
            dense_fp16_payload_bytes=self._dense_bytes,
            fp16_immediate_count=len(self._immediates),
            fp16_immediate_payload_bytes=self._immediate_bytes,
            mil_path=str(mil_path),
            mil_bytes=len(text.encode()),
            mil_sha256=hashlib.sha256(text.encode()).hexdigest(),
            model_root=str(model_root),
            normalized_blob_path=str(normalized_blob),
            normalized_blob_bytes=normalized_blob.stat().st_size,
            normalized_blob_sha256=self._file_sha256(normalized_blob),
            source_weight_materialization=materialization,
            normalized_records=self._normalized,
            immediate_records=self._immediates,
        )

    @staticmethod
    def _operation_values(op: Any):
        for argument in op.inputs.values():
            for binding in argument.arguments:
                if binding.WhichOneof("binding") == "value":
                    yield binding.value
        yield from op.attributes.values()

    def _materialize_weights(self, model_root: Path) -> dict[str, str]:
        materialization: dict[str, str] = {}
        for weight in self.package.weight_files:
            try:
                relative = weight.path.relative_to(self.package.model_path.parent)
            except ValueError as exc:
                raise AdapterError(
                    f"weight file {weight.path} is outside the model root"
                ) from exc
            destination = model_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(weight.path, destination)
                materialization[relative.as_posix()] = "hardlink"
            except OSError:
                shutil.copyfile(weight.path, destination)
                materialization[relative.as_posix()] = "copy"
        return materialization

    @staticmethod
    def _file_sha256(path: Path) -> str:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_locked_encoder(package: MlPackage, lock_path: Path) -> None:
    lock = ReferenceLock.load(lock_path)
    validate_lock(lock)
    if lock.model_repo != EXPECTED_REPOSITORY or lock.model_revision != EXPECTED_REVISION:
        raise AdapterError(
            "reference lock does not identify the pinned Parakeet repository revision"
        )
    prefix = "encoder.mlpackage/"
    expected = {
        item.path.removeprefix(prefix): item
        for item in lock.files
        if item.path.startswith(prefix)
    }
    actual_paths: dict[str, Path] = {}
    for path in package.path.rglob("*"):
        if path.is_symlink():
            raise AdapterError(f"locked package contains symlink {path}")
        if path.is_file():
            actual_paths[path.relative_to(package.path).as_posix()] = path
    if set(actual_paths) != set(expected):
        missing = sorted(set(expected) - set(actual_paths))
        extra = sorted(set(actual_paths) - set(expected))
        raise AdapterError(f"locked encoder file set differs: missing={missing}, extra={extra}")
    for relative, path in actual_paths.items():
        item = expected[relative]
        if path.stat().st_size != item.size:
            raise AdapterError(f"locked encoder size mismatch for {relative}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item.sha256:
            raise AdapterError(f"locked encoder SHA-256 mismatch for {relative}")


def emit_mlpackage(package_path: Path, output: Path, lock_path: Path) -> Emission:
    package = open_mlpackage(package_path)
    verify_locked_encoder(package, lock_path)
    model = load_model(package.model_path.read_bytes())
    output = output.resolve()
    if output.exists():
        raise AdapterError(f"output path already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    )
    try:
        emission = _Adapter(package, model).emit(temporary)
        temporary.rename(output)
        payload = emission.to_dict()
        payload["mil_path"] = str(output / "model.mil")
        payload["model_root"] = str(output / "model-root")
        payload["normalized_blob_path"] = str(
            output / "model-root" / "weights" / "normalized.bin"
        )
        return Emission(
            **{
                **payload,
                "normalized_records": [
                    EmittedRecord(**item) for item in payload["normalized_records"]
                ],
                "immediate_records": [
                    EmittedRecord(**item) for item in payload["immediate_records"]
                ],
            }
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _require_compiler(path: Path) -> Path:
    if not path.is_absolute():
        raise AdapterError("--compiler must be an explicit absolute path")
    resolved = path.resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise AdapterError(f"compiler is not an executable file: {resolved}")
    return resolved


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit validated textual MIL for the pinned Parakeet encoder"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("emit", "compile"):
        command = subparsers.add_parser(name)
        command.add_argument("package", type=Path)
        command.add_argument("output", type=Path, help="new compiler-source directory")
        command.add_argument("--reference-lock", type=Path, required=True)
        if name == "compile":
            command.add_argument(
                "--compiler",
                type=Path,
                required=True,
                help="absolute path to mil-hwxc",
            )
            command.add_argument(
                "--compiler-output", type=Path, required=True, help="new H13 output path"
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        compiler = _require_compiler(args.compiler) if args.command == "compile" else None
        emission = emit_mlpackage(args.package, args.output, args.reference_lock)
        print(json.dumps(emission.to_dict(), indent=2, sort_keys=True), flush=True)
        if compiler is None:
            return 0
        invocation = [
            str(compiler),
            "--mil",
            emission.mil_path,
            "--model-root",
            emission.model_root,
            "--output",
            str(args.compiler_output.resolve()),
            "--target",
            "H13",
            "--format",
            "anec",
        ]
        print("compiler invocation: " + " ".join(invocation), file=sys.stderr, flush=True)
        result = subprocess.run(invocation, check=False)
        if result.returncode:
            print(
                f"mil_adapter: compiler failed with exit {result.returncode}",
                file=sys.stderr,
            )
        return result.returncode
    except (AdapterError, MlPackageError, ReferenceError, OSError, ValueError) as exc:
        print(f"mil_adapter: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
