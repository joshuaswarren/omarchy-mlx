# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Inspect Core ML packages using the vendored official protobuf schema.

Inventories describe types, bindings, nested operations, and blob references.
Weights are streamed for hashing, not decoded. Compiler eligibility is
assessed separately against the target compiler, never inferred here.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import proto
from .proto import load_model

INVENTORY_SCHEMA = "mlx-omarchy.coreml.inventory/1"


class MlPackageError(RuntimeError):
    """Raised when the mlpackage structure is invalid or inconsistent."""


# ---------------------------------------------------------------------------
# Package structure
# ---------------------------------------------------------------------------


@dataclass
class WeightFile:
    path: Path
    relative: str  # package-relative POSIX path, as reported
    size: int
    sha256: str


@dataclass
class MlPackage:
    path: Path
    file_format_version: str | None
    root_identifier: str | None
    manifest: dict
    model_path: Path
    weights_dir: Path | None
    weight_files: list[WeightFile] = field(default_factory=list)


def _check_manifest_path(raw: object, package: Path) -> str:
    """Validate one itemInfoEntries ``path`` and return it.

    Refuses absolute paths and traversal (``..``) — a manifest is
    untrusted input and must stay inside the package directory.
    """
    if not isinstance(raw, str) or not raw:
        raise MlPackageError(f"{package}: Manifest.json entry has no path")
    if raw.startswith(("/", "\\")) or Path(raw).is_absolute():
        raise MlPackageError(
            f"{package}: Manifest.json entry path is absolute: {raw!r}"
        )
    parts = Path(raw).parts
    if ".." in parts:
        raise MlPackageError(
            f"{package}: Manifest.json entry path escapes the package: {raw!r}"
        )
    return raw


def _contained(path: Path, package: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(package):
        raise MlPackageError(f"{path}: path escapes the package")
    return resolved


def open_mlpackage(path: Path) -> MlPackage:
    """Open and validate an ``.mlpackage`` directory.

    Raises :class:`MlPackageError` with a specific message for every
    structural problem: missing directory/manifest, unparsable
    manifest JSON, no itemInfoEntries, escaped or absolute manifest
    paths, missing/ambiguous model file, missing model bytes.
    """
    p = Path(path).resolve()
    if not p.is_dir():
        raise MlPackageError(f"{p}: not a directory (mlpackages are directories)")

    manifest_path = _contained(p / "Manifest.json", p)
    if not manifest_path.is_file():
        raise MlPackageError(f"{p}: missing Manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MlPackageError(f"{manifest_path}: unparsable manifest ({exc})") from exc
    if not isinstance(manifest, dict):
        raise MlPackageError(f"{manifest_path}: manifest is not a JSON object")

    entries = manifest.get("itemInfoEntries")
    if not isinstance(entries, dict) or not entries:
        raise MlPackageError(f"{manifest_path}: no itemInfoEntries")

    root_id = manifest.get("rootModelIdentifier")
    if not isinstance(root_id, str) or root_id not in entries:
        raise MlPackageError(
            f"{manifest_path}: rootModelIdentifier has no matching entry"
        )
    paths = {}
    weights_rel: list[str] = []
    for entry_id, info in entries.items():
        if not isinstance(info, dict):
            raise MlPackageError(
                f"{manifest_path}: itemInfoEntries[{entry_id!r}] is not an object"
            )
        rel = _check_manifest_path(info.get("path"), p)
        paths[entry_id] = _contained(p / "Data" / rel, p)
        if info.get("name") == "weights":
            weights_rel.append(rel)
    model_path = paths[root_id]
    if not model_path.is_file():
        raise MlPackageError(f"{model_path}: model file named by manifest is missing")

    weights_dir = None
    if weights_rel:
        if len(weights_rel) > 1:
            raise MlPackageError(
                f"{p}: manifest names {len(weights_rel)} weights entries"
            )
        weights_dir = _contained(p / "Data" / weights_rel[0], p)

    weight_files: list[WeightFile] = []
    if weights_dir is not None and weights_dir.is_dir():
        for wf in sorted(weights_dir.rglob("*")):
            target = _contained(wf, p)
            if target.is_file():
                with target.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                weight_files.append(
                    WeightFile(
                        path=target,
                        relative=wf.relative_to(p).as_posix(),
                        size=target.stat().st_size,
                        sha256=digest,
                    )
                )

    return MlPackage(
        path=p,
        file_format_version=str(manifest.get("fileFormatVersion", "")) or None,
        root_identifier=manifest.get("rootModelIdentifier"),
        manifest=manifest,
        model_path=model_path,
        weights_dir=weights_dir,
        weight_files=weight_files,
    )


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


def _binding_summary(binding: Any) -> dict:
    """One ``Argument.Binding``: a name reference or an inline value."""
    which = binding.WhichOneof("binding")
    if which == "name":
        return {"name": binding.name}
    if which == "value":
        return {"value": proto.value_summary(binding.value)}
    return {"binding": "UNSET"}


def _argument_summary(argument: Any) -> list[dict]:
    return [_binding_summary(b) for b in argument.arguments]


def _named_types_summary(named: list[Any]) -> list[dict]:
    return [{"name": t.name, "type": proto.value_type_summary(t.type)} for t in named]


def _op_summary(op: Any, index: int) -> dict:
    return {
        "index": index,
        "type": op.type,
        "bindings": {k: _argument_summary(a) for k, a in sorted(op.inputs.items())},
        "outputs": _named_types_summary(op.outputs),
        "attributes": {
            k: proto.value_summary(v) for k, v in sorted(op.attributes.items())
        },
        "blocks": [_block_summary(b) for b in op.blocks],
    }


def _block_summary(block: Any) -> dict:
    ops = [_op_summary(op, i) for i, op in enumerate(block.operations)]
    histogram = Counter(op["type"] for op in ops)
    return {
        "inputs": _named_types_summary(block.inputs),
        "outputs": list(block.outputs),
        "operations": ops,
        "op_count": len(ops),
        "op_types": sorted(histogram),
        "attributes": {
            k: proto.value_summary(v) for k, v in sorted(block.attributes.items())
        },
    }


def _function_summary(name: str, function: Any) -> dict:
    specializations = function.block_specializations
    blocks = {k: _block_summary(b) for k, b in sorted(specializations.items())}
    active = function.opset if function.opset in blocks else None
    histogram = Counter(op["type"] for op in _walk_ops(blocks.values()))
    opset_note = None
    if active is None:
        opset_note = (
            f"function.opset={function.opset!r} has no matching "
            "block_specializations entry"
            if function.opset
            else "function.opset is empty and has no default specialization"
        )
    active_block = blocks.get(active)
    return {
        "name": name,
        "opset": function.opset or None,
        "program_version": None,  # filled by the caller (program-level)
        "inputs": _named_types_summary(function.inputs),
        "outputs": active_block["outputs"] if active_block else [],
        "block_specializations": blocks,
        "active_block": active,
        "opset_consistency": opset_note or "ok",
        "op_histogram": dict(sorted(histogram.items())),
        "op_total": sum(histogram.values()),
        "control_flow_ops": sorted(
            {op["type"] for op in _walk_ops(blocks.values()) if op["blocks"]}
        ),
    }


def _walk_ops(blocks):
    for block in blocks:
        for op in block["operations"]:
            yield op
            yield from _walk_ops(op["blocks"])


def _const_op_blob_refs(op: dict):
    values = list(op["attributes"].values())
    values.extend(
        binding["value"]
        for bindings in op["bindings"].values()
        for binding in bindings
        if "value" in binding
    )
    for value in values:
        storage = value.get("storage", {})
        if "blob_file" in storage:
            yield storage


def describe_package(package: MlPackage) -> dict:
    """Full typed inventory of a validated package (JSON-ready)."""
    model = load_model(package.model_path.read_bytes())

    model_type = model.WhichOneof("Type")
    program = model.mlProgram if model_type == "mlProgram" else None

    functions = []
    program_version = None
    if program is not None:
        program_version = program.version or None
        for name, function in sorted(program.functions.items()):
            summary = _function_summary(name, function)
            summary["program_version"] = program_version
            functions.append(summary)

    inputs = [proto.feature_description_summary(f) for f in model.description.input]
    outputs = [proto.feature_description_summary(f) for f in model.description.output]
    state = [proto.feature_description_summary(f) for f in model.description.state]

    blob_refs: dict[str, dict] = {}
    constexpr_ops: list[dict] = []
    for fn in functions:
        for op in _walk_ops(fn["block_specializations"].values()):
            if op["type"].startswith("constexpr_"):
                constexpr_ops.append(
                    {
                        "function": fn["name"],
                        "op": op["type"],
                        "op_index": op["index"],
                        "inputs": sorted(op["bindings"]),
                        "value_types": {
                            k: v[0]["value"].get("type")
                            for k, v in op["bindings"].items()
                            if v and "value" in v[0]
                        },
                    }
                )
            for ref in _const_op_blob_refs(op):
                entry = blob_refs.setdefault(
                    ref["blob_file"], {"blob_file": ref["blob_file"], "offsets": []}
                )
                entry["offsets"].append(ref["offset"])

    weight_summary = []
    for wf in package.weight_files:
        weight_summary.append(
            {
                "file": wf.relative,
                "size": wf.size,
                "sha256": wf.sha256,
            }
        )

    referenced_blobs = []
    weights_by_path = {wf.path: wf for wf in package.weight_files}
    for entry in sorted(blob_refs.values(), key=lambda e: e["blob_file"]):
        name = entry["blob_file"]
        tail = name.removeprefix("@model_path/")
        rel = _check_manifest_path(tail, package.path)
        target = _contained(package.model_path.parent / rel, package.path)
        match = weights_by_path.get(target)
        offsets = sorted(set(entry["offsets"]))
        referenced_blobs.append(
            {
                "blob_file": name,
                "distinct_offsets": len(offsets),
                "offsets": offsets,
                "resolved_file": match.relative if match else None,
                "resolved": match is not None
                and all(0 <= offset < match.size for offset in offsets),
            }
        )

    total_ops = sum(fn["op_total"] for fn in functions)

    inventory: dict = {
        "inventory_schema": INVENTORY_SCHEMA,
        "package": {
            "path": str(package.path),
            "format": "mlpackage",
            "file_format_version": package.file_format_version,
            "root_model_identifier": package.root_identifier,
        },
        "model": {
            "specification_version": model.specificationVersion,
            "type": model_type or "UNSET",
            "is_updatable": model.isUpdatable,
            "has_ml_program": program is not None,
            "description": {
                "inputs": inputs,
                "outputs": outputs,
                "state": state,
            },
            "metadata": {
                "short_description": model.description.metadata.shortDescription
                or None,
                "version_string": model.description.metadata.versionString or None,
                "author": model.description.metadata.author or None,
                "license": model.description.metadata.license or None,
                "user_defined": dict(model.description.metadata.userDefined),
            },
        },
        "program": {
            "version": program_version,
            "functions": functions,
        },
        "weights": {
            "files": weight_summary,
            "blob_references": referenced_blobs,
        },
        "compression": {
            "representation": "constexpr_* operations (see schema semantics)"
            if constexpr_ops
            else "none observed",
            "constexpr_ops": constexpr_ops,
            "constexpr_op_count": len(constexpr_ops),
        },
        "op_histogram": _merged_histogram(functions),
        "op_total": total_ops,
        "control_flow": {
            "present": any(fn["control_flow_ops"] for fn in functions),
            "ops": sorted({t for fn in functions for t in fn["control_flow_ops"]}),
        },
        "validity": _validity(model, functions),
        "eligibility": {
            "assessed": False,
            "note": (
                "compiler/deployment eligibility is not assessed by this "
                "inspector; it requires the target compiler's coverage data. "
                "Only parse validity and the typed inventory above are claims."
            ),
        },
    }
    return inventory


def _merged_histogram(functions: list[dict]) -> dict[str, int]:
    merged: Counter = Counter()
    for fn in functions:
        merged.update(fn["op_histogram"])
    return dict(sorted(merged.items()))


def _block_ssa_findings(
    block: dict, inherited: set[str], location: str
) -> list[str]:
    """Return unresolved SSA bindings in one block and its nested blocks."""
    defined = set(inherited)
    defined.update(item["name"] for item in block["inputs"] if item["name"])
    findings: list[str] = []
    for op in block["operations"]:
        op_location = f"{location}/op[{op['index']}] {op['type']!r}"
        for parameter, bindings in op["bindings"].items():
            for binding in bindings:
                name = binding.get("name")
                if name is not None and name not in defined:
                    findings.append(
                        f"{op_location} parameter {parameter!r} consumes "
                        f"undefined value {name!r}"
                    )
                elif binding.get("binding") == "UNSET":
                    findings.append(
                        f"{op_location} parameter {parameter!r} has an unset binding"
                    )
        for index, child in enumerate(op["blocks"]):
            findings.extend(
                _block_ssa_findings(child, defined, f"{op_location}/block[{index}]")
            )
        defined.update(item["name"] for item in op["outputs"] if item["name"])
    for name in block["outputs"]:
        if name not in defined:
            findings.append(f"{location} returns undefined value {name!r}")
    return findings


def _ssa_findings(functions: list[dict]) -> list[str]:
    findings: list[str] = []
    for function in functions:
        inherited = {item["name"] for item in function["inputs"] if item["name"]}
        for specialization, block in function["block_specializations"].items():
            findings.extend(
                _block_ssa_findings(
                    block,
                    inherited,
                    f"function {function['name']!r} specialization {specialization!r}",
                )
            )
    return findings


def _validity(model: Any, functions: list[dict]) -> dict:
    """Report parse confidence without claiming semantic completeness."""
    notes: list[str] = []
    for fn in functions:
        if fn["opset_consistency"] != "ok":
            notes.append(f"function {fn['name']}: {fn['opset_consistency']}")
        if fn["program_version"] is None:
            notes.append(f"function {fn['name']}: program.version is unset (0)")
    ssa_findings = _ssa_findings(functions)
    notes.extend(ssa_findings)
    unknown = proto.count_unknown_fields(model)
    if unknown:
        notes.append(
            f"{unknown} field(s) written by a schema newer than the vendored "
            "one are present but not interpreted (forward-compatibility limit)"
        )
    return {
        "protobuf_parse": "ok",
        "model_type_set": model.WhichOneof("Type") is not None,
        "unknown_schema_fields": unknown,
        "ssa_bindings": "invalid" if ssa_findings else "ok",
        "notes": notes,
        "confidence": (
            "full structure read through the official schema, zero unknown "
            "schema fields encountered"
            if not notes and not unknown
            else "see notes"
        ),
    }


def inspect(path: Path) -> dict:
    """One-shot: open a package and return its inventory."""
    return describe_package(open_mlpackage(Path(path)))
