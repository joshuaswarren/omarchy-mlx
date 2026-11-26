# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Access to the official Core ML protobuf schema (vendored bindings).

The wire-format reader that used to live in this module was a guess at
the Core ML message layouts and it got the dtype enums wrong (it labeled
MIL ``BOOL=1`` as ``float16``). It is gone. This module loads model
specifications through the vendored, officially generated bindings in
:mod:`coreml.schema` (Apple coremltools 9.0 protobuf schema, see
``schema/VENDORED.json``) and the ``protobuf`` runtime.

Everything the inspector reports about dtypes, shapes, and values comes
from the generated enums and messages — no hand-maintained name tables.

Only the messages the schema actually defines are interpreted here.
Values inside constant tensors are never materialized or decoded: the
inspector reports types, shapes, and weight-file references, not data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import MIL_pb2, Model_pb2

# ---------------------------------------------------------------------------
# Vendored schema provenance
# ---------------------------------------------------------------------------

_SCHEMA_DIR = Path(__file__).resolve().parent / "schema"
VENDORED_MANIFEST = _SCHEMA_DIR / "VENDORED.json"


def schema_info() -> dict[str, Any]:
    """Provenance of the vendored official schema (no network access)."""
    manifest = json.loads(VENDORED_MANIFEST.read_text(encoding="utf-8"))
    entry = manifest["vendored_schema"]
    return {
        "upstream": entry["upstream_repository"],
        "tag": entry["upstream_tag"],
        "pypi_artifact": entry["pypi_artifact"],
        "sdist_sha256": entry["pypi_sdist_sha256"],
        "license": entry["license"],
        "excluded": entry["excluded"],
        "runtime_requirement": entry["runtime_requirement"],
        "pinned_files": len(entry["files"]),
    }


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ModelSpecError(ValueError):
    """Raised when the model.mlmodel bytes are not a parseable Model spec."""


def load_model(data: bytes) -> Model_pb2.Model:
    """Parse model.mlmodel bytes into the official ``Model`` message.

    Raises :class:`ModelSpecError` (never a raw protobuf exception) when
    the payload is truncated, corrupted, or not a Core ML Model.
    """
    model = Model_pb2.Model()
    try:
        model.ParseFromString(data)
    except Exception as exc:  # protobuf.DecodeError and friends
        raise ModelSpecError(
            f"model.mlmodel is not a parseable Core ML Model "
            f"specification ({type(exc).__name__}: {exc})"
        ) from exc
    return model


# ---------------------------------------------------------------------------
# Enum naming — straight from the generated descriptors
# ---------------------------------------------------------------------------


def _enum_name(enum_type: Any, value: int) -> str:
    """Name for an enum value; unknown values are explicit, never blank."""
    try:
        return enum_type.Name(value)
    except ValueError:
        return f"UNRECOGNIZED({value})"


def mil_dtype_name(value: int) -> str:
    """Name of a ``MILSpec.DataType`` value (e.g. FLOAT16=10, BOOL=1)."""
    return _enum_name(MIL_pb2.DataType, value)


def feature_dtype_name(value: int) -> str:
    """Name of a ``FeatureTypes.ArrayFeatureType.ArrayDataType`` value.

    This is the model-description (FeatureType) enum, which is a
    different numbering from the MIL dtype enum (e.g. FLOAT32=65568
    here vs FLOAT32=11 in MIL). The two are never interchangeable and
    the inspector keeps them distinct everywhere.
    """
    return _enum_name(Model_pb2.ArrayFeatureType.ArrayDataType, value)


# ---------------------------------------------------------------------------
# Type summaries
# ---------------------------------------------------------------------------


def dimension_summary(dim: Any) -> dict[str, Any]:
    """One ``MILSpec.Dimension``: constant size or explicit unknown."""
    kind = dim.WhichOneof("dimension")
    if kind == "constant":
        return {"constant": dim.constant.size}
    if kind == "unknown":
        return {"unknown": True, "variadic": dim.unknown.variadic}
    return {"unknown_kind": None}  # no oneof set: schema-legal but unnamed




def count_unknown_fields(message: Any) -> int:
    """Recursively count schema fields this vendored schema does not know.

    A package written by a newer Core ML than the vendored schema
    parses fine (protobuf forward compatibility) but its new fields
    are invisible. This scan makes that visibility limit an explicit,
    reported number instead of a silent blind spot.
    """
    from google.protobuf import unknown_fields as _unknown_fields
    from google.protobuf.descriptor import FieldDescriptor as _FD

    total = len(_unknown_fields.UnknownFieldSet(message))
    for field, value in message.ListFields():
        if field.type != _FD.TYPE_MESSAGE:
            continue
        entry = field.message_type
        if entry.GetOptions().map_entry:
            # Map field: scalar-valued maps cannot carry unknown
            # fields; recurse only into message-valued entries.
            if entry.fields_by_name["value"].type == _FD.TYPE_MESSAGE:
                for item in value.values():
                    total += count_unknown_fields(item)
        elif field.is_repeated:
            for item in value:
                total += count_unknown_fields(item)
        else:
            total += count_unknown_fields(value)
    return total


def tensor_type_summary(t: Any) -> dict[str, Any]:
    """``MILSpec.TensorType`` → dtype name, rank, dimensions."""
    out: dict[str, Any] = {
        "dtype": mil_dtype_name(t.dataType),
        "rank": t.rank,
        "dimensions": [dimension_summary(d) for d in t.dimensions],
    }
    if len(t.dimensions) != t.rank and t.rank >= 0:
        out["rank_mismatch"] = {
            "declared_rank": t.rank,
            "dimension_count": len(t.dimensions),
        }
    return out


def value_type_summary(vt: Any, _depth: int = 0) -> dict[str, Any]:
    """``MILSpec.ValueType`` → which kind it is, with the kind's details.

    Every schema kind is reported explicitly. If the oneof is unset or
    (defensively) deeper than the recursion bound, the summary says so
    instead of presenting a blank tensor reading.
    """
    kind = vt.WhichOneof("type")
    if kind == "tensorType":
        return {"kind": "tensor", **tensor_type_summary(vt.tensorType)}
    if kind == "listType":
        return {
            "kind": "list",
            "element": value_type_summary(vt.listType.type, _depth + 1)
            if _depth < 8
            else {"kind": "depth_limit"},
            "length": dimension_summary(vt.listType.length)
            if vt.listType.HasField("length")
            else None,
        }
    if kind == "tupleType":
        return {
            "kind": "tuple",
            "elements": [
                value_type_summary(t, _depth + 1) for t in vt.tupleType.types
            ]
            if _depth < 8
            else [{"kind": "depth_limit"}],
        }
    if kind == "dictionaryType":
        return {
            "kind": "dictionary",
            "key": value_type_summary(vt.dictionaryType.keyType, _depth + 1),
            "value": value_type_summary(vt.dictionaryType.valueType, _depth + 1),
        }
    if kind == "stateType":
        return {
            "kind": "state",
            "wrapped": value_type_summary(vt.stateType.wrappedType, _depth + 1),
        }
    return {"kind": "UNSET"}  # ValueType with no type set: report, don't guess


def value_summary(value: Any) -> dict[str, Any]:
    """``MILSpec.Value``: its type plus where its data lives.

    For compile-time tensor values only the storage location is
    reported (blob file + offset), never the tensor contents.
    """
    out: dict[str, Any] = {"type": value_type_summary(value.type)}
    stored = value.WhichOneof("value")
    if stored == "blobFileValue":
        out["storage"] = {
            "blob_file": value.blobFileValue.fileName,
            "offset": value.blobFileValue.offset,
        }
    elif stored == "immediateValue":
        out["storage"] = {"immediate": True}  # inline constant; contents not read
    else:
        out["storage"] = {"unset": True}
    return out


# ---------------------------------------------------------------------------
# Model-description (FeatureType) summaries
# ---------------------------------------------------------------------------


def size_range_summary(r: Any) -> dict[str, Any]:
    return {
        "lower": r.lowerBound,
        # Schema: negative upperBound means unbounded.
        "upper": r.upperBound,
        "unbounded": r.upperBound < 0,
    }


def array_feature_summary(arr: Any) -> dict[str, Any]:
    """``FeatureTypes.ArrayFeatureType`` with explicit shape flexibility.

    The default ``shape`` is always reported as-is. Flexibility
    (enumeratedShapes / shapeRange) is reported when present; a static
    shape with no flexibility is reported as static. Nothing here
    forces or assumes fp16 — the dtype is whatever the package says.
    """
    out: dict[str, Any] = {
        "feature_dtype": feature_dtype_name(arr.dataType),
        "shape": list(arr.shape),
    }
    flex = arr.WhichOneof("ShapeFlexibility")
    if flex == "enumeratedShapes":
        out["shape_flexibility"] = {
            "kind": "enumeratedShapes",
            "shapes": [list(s.shape) for s in arr.enumeratedShapes.shapes],
        }
    elif flex == "shapeRange":
        out["shape_flexibility"] = {
            "kind": "shapeRange",
            "size_ranges": [size_range_summary(r) for r in arr.shapeRange.sizeRanges],
        }
    else:
        out["shape_flexibility"] = {"kind": "static"}
    return out


def feature_type_summary(ft: Any) -> dict[str, Any]:
    """``FeatureTypes.FeatureType`` → explicit summary of which type it is."""
    kind = ft.WhichOneof("Type")
    if kind == "multiArrayType":
        return {"kind": "multiArrayType", **array_feature_summary(ft.multiArrayType)}
    if kind == "int64Type":
        return {"kind": "int64"}
    if kind == "doubleType":
        return {"kind": "double"}
    if kind == "stringType":
        return {"kind": "string"}
    if kind == "dictionaryType":
        return {"kind": "dictionary"}
    if kind == "sequenceType":
        return {"kind": "sequence"}
    if kind == "imageType":
        img = ft.imageType
        return {
            "kind": "imageType",
            "width": img.width,
            "height": img.height,
            "colorSpace": _enum_name(
                Model_pb2.ImageFeatureType.ColorSpace, img.colorSpace
            ),
        }
    if kind == "stateType":
        return {
            "kind": "stateType",
            **array_feature_summary(ft.stateType.arrayType),
        }
    if kind is None:
        return {"kind": "UNSET"}
    return {"kind": f"UNRECOGNIZED({kind})"}


def feature_description_summary(fd: Any) -> dict[str, Any]:
    """``FeatureDescription`` (model input/output/state entry)."""
    return {"name": fd.name, **feature_type_summary(fd.type)}
