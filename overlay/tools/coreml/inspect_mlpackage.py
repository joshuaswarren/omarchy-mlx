#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Core ML .mlpackage inspector (human or JSON output, device-free).

Usage:
    python3 overlay/tools/coreml/inspect_mlpackage.py inspect PATH [--json] [--strict]
    python3 -m coreml.inspect_mlpackage inspect PATH [--json] [--strict]

Reports the plan's section-14 checklist for one ``.mlpackage``:
structure, model type, functions/ops (typed), weights and blob
references, compression representation, versions, control flow, and
parse validity. Optional source-matched static compiler coverage is reported
with its provenance, never as proof of compilation. No ANE device, GPU,
or compiler is opened; runs on any Linux host.

Exit codes: 0 = inspected, 1 = invalid package or usage error,
2 = ``--strict`` validation failure (missing weight files, unset
model type, opset inconsistencies, or unresolved SSA bindings).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):  # plain-script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coreml.mlpackage import MlPackage, MlPackageError, describe_package, open_mlpackage
from coreml.proto import ModelSpecError, schema_info


def coverage_evidence(path: Path, package: MlPackage, inv: dict) -> dict:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        with package.model_path.open("rb") as stream:
            model_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if report["model_sha256"] != model_hash:
            raise ValueError("model SHA-256 does not match coverage evidence")
        weights = {wf.relative: wf.sha256 for wf in package.weight_files}
        if report["weights"] != weights:
            raise ValueError("weight SHA-256 map does not match coverage evidence")
        compiler = report["compiler"]
        if (
            not isinstance(compiler, dict)
            or compiler.get("target") != "H13"
            or not isinstance(compiler.get("repository"), str)
            or not compiler["repository"].strip()
            or not isinstance(compiler.get("commit"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", compiler["commit"])
            or not isinstance(report["source"], str)
            or not report["source"].strip()
        ):
            raise ValueError(
                "coverage requires a source and exact H13 compiler identity"
            )
        counts = report["counts"]
        categories = {
            "direct",
            "direct-alias",
            "direct-const",
            "normalization-needed",
            "missing-envelope",
            "missing",
        }
        if (
            not isinstance(counts, dict)
            or set(counts) != categories
            or any(type(n) is not int or n < 0 for n in counts.values())
            or sum(counts.values()) != inv["op_total"]
            or inv["op_total"] == 0
        ):
            raise ValueError(
                "coverage counts must be nonnegative and sum to the inventory operation count"
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise MlPackageError(f"{path}: invalid compiler coverage: {exc}") from exc
    return {
        "assessed": True,
        "scope": "source-matched static report; classifier not executed",
        "compiler": {key: compiler[key] for key in ("repository", "commit", "target")},
        "source": report["source"],
        "counts": counts,
        "compilable": None,
        "note": "Package bytes match the supplied report; its provenance is declared, not authenticated. Compilation and ANE execution are not proven.",
    }


def _fmt_type(t: dict) -> str:
    if t.get("kind") == "multiArrayType":
        flex = t.get("shape_flexibility", {})
        suffix = ""
        if flex.get("kind") == "shapeRange":
            suffix = " range"
        elif flex.get("kind") == "enumeratedShapes":
            suffix = f" enumerated{len(flex.get('shapes', []))}"
        return f"{t.get('feature_dtype')} {t.get('shape')}{suffix}"
    return str(t.get("kind"))


def human_report(inv: dict) -> str:
    lines: list[str] = []
    pkg = inv["package"]
    model = inv["model"]
    lines.append(f"Package: {pkg['path']}")
    lines.append(
        f"Format: {pkg['format']} (fileFormatVersion {pkg['file_format_version']})"
    )
    lines.append(f"Model type: {model['type']}")
    lines.append(f"Specification version: {model['specification_version']}")
    for key, entries in (
        ("Inputs", model["description"]["inputs"]),
        ("Outputs", model["description"]["outputs"]),
        ("State", model["description"]["state"]),
    ):
        lines.append(f"{key}: {len(entries)}")
        for e in entries:
            lines.append(f"  {e['name']}: {_fmt_type(e)}")
    for fn in inv["program"]["functions"]:
        lines.append(
            f"Function {fn['name']}: opset={fn['opset']} "
            f"blocks={len(fn['block_specializations'])} ops={fn['op_total']}"
        )
        for name, dtype in sorted(fn["op_histogram"].items(), key=lambda kv: -kv[1])[
            :12
        ]:
            lines.append(f"  {name}: {dtype}")
        if len(fn["op_histogram"]) > 12:
            lines.append(f"  ... {len(fn['op_histogram']) - 12} more op types")
    comp = inv["compression"]
    lines.append(
        f"Compression: {comp['representation']}"
        + (
            f" ({comp['constexpr_op_count']} constexpr ops)"
            if comp["constexpr_op_count"]
            else ""
        )
    )
    lines.append("Weights:")
    for wf in inv["weights"]["files"]:
        lines.append(f"  {wf['file']}: {wf['size']} bytes sha256={wf['sha256']}")
    for ref in inv["weights"]["blob_references"]:
        status = "ok" if ref["resolved"] else "MISSING FILE"
        lines.append(
            f"  blob ref {ref['blob_file']}: {ref['distinct_offsets']} offsets [{status}]"
        )
    if inv["control_flow"]["present"]:
        lines.append(f"Control flow: {', '.join(inv['control_flow']['ops'])}")
    else:
        lines.append("Control flow: none")
    validity = inv["validity"]
    lines.append(f"Parse validity: {validity['protobuf_parse']}")
    lines.append(f"SSA bindings: {validity['ssa_bindings']}")
    for note in validity["notes"]:
        lines.append(f"  note: {note}")
    eligibility = inv["eligibility"]
    if eligibility["assessed"]:
        compiler = eligibility["compiler"]
        lines.append(f"H13 compiler coverage: static report for {compiler['commit']}")
        for category, count in eligibility["counts"].items():
            lines.append(f"  {category}: {count}")
        lines.append(f"Evidence: {eligibility['source']}")
        lines.append("ANE compilability: not proven (static report only)")
    else:
        lines.append("Compiler eligibility: not assessed (see JSON eligibility.note)")
    lines.append(
        f"Schema: official Core ML protobuf, vendored from {schema_info()['upstream']} "
        f"tag {schema_info()['tag']}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="inspect_mlpackage", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_inspect = sub.add_parser("inspect", help="inspect one .mlpackage")
    p_inspect.add_argument("path", type=Path, help="path to the .mlpackage directory")
    p_inspect.add_argument(
        "--json", action="store_true", help="machine-readable output"
    )
    p_inspect.add_argument(
        "--strict",
        action="store_true",
        help="exit 2 if unresolved weight files, unset model type, opset "
        "inconsistencies, or unresolved SSA bindings are found",
    )
    p_inspect.add_argument(
        "--compiler-coverage",
        type=Path,
        help="attach source-matched static H13 coverage evidence (not a compiler run)",
    )
    args = parser.parse_args(argv)

    try:
        package = open_mlpackage(args.path)
        inv = describe_package(package)
        if args.compiler_coverage:
            inv["eligibility"] = coverage_evidence(args.compiler_coverage, package, inv)
    except (MlPackageError, ModelSpecError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(inv, indent=2, sort_keys=False))
    else:
        print(human_report(inv))

    if args.strict:
        problems = list(inv["validity"]["notes"])
        problems += [
            f"unresolved blob reference {r['blob_file']}"
            for r in inv["weights"]["blob_references"]
            if not r["resolved"]
        ]
        if not inv["model"]["type"] or inv["model"]["type"] == "UNSET":
            problems.append("model Type oneof is unset")
        if problems:
            for problem in problems:
                print(f"strict: {problem}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
