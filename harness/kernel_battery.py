#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""KernelBattery harness: extract every mx.fast.metal_kernel call from the
pinned oMLX and TensorFold trees, classify each, and run the ones the
translator accepts through a dev-box C++ test driver.

Inputs:
  --omlx DIR   Path to the pinned oMLX checkout (omlx/ subdir of v0.7.0).
  --tf   DIR   Path to the pinned TensorFold checkout (src/tensorfold/ of
               main=v0.6.5).
  --translator PATH  Path to the translate_msl driver binary
                     (omarchy_translate_msl_driver built from
                     overlay/mlx/backend/omarchy/custom_kernel.cpp).
  --out FILE   CSV output path. Columns:
                 source, file, name, kind, gmsl_features, ttf_status,
                 ttf_error, run_status, run_error
                 where:
                   source   = 'omlx' | 'tf'
                   kind     = 'mtk' (mx.fast.metal_kernel) or other
                   gmsl_features  = '|' separated keywords found
                   ttf_status = 'accept' | 'reject' | 'noconstructor'
                                 | 'defer_m2' (deferred to M2 run)
                   ttf_error  = the rejection text or empty
                   run_status = 'not-run' | 'gpu-pass' | 'gpu-fail'
                   run_error  = empty or 'pass:fraction' or 'fail:msg'
  --limit N    Stop after N kernels (debug).

Discriminators:
  * Static AST extraction: ast.parse, walk the tree, find calls to
    mx.fast.metal_kernel, capture source= and the surrounding
    call-chain arguments.
  * For kernels with template= / f-strings, reproduce the template by
    reading the call site and substituting constexpr values from the
    enclosing scope (best-effort; many are not extractable).
  * Heuristic pass: per source, list the metal-only / translator-only
    keywords that the omarchy translator refuses ('simdgroup_matrix',
    'mpp::tensor_ops', 'matmul2d_descriptor', 'texture', 'sampler',
    '[[texture]]', 'metal::' deep paths).
  * TTF (translator table fetch) = invoke the driver; record pass or
    refusal text.
  * Run = mark 'defer_m2' on this dev box; the driver only runs the
    translation, not the dispatch (which needs a real device and the
    M2 GPU tickets).

This script is the static analysis lane; the GPU dispatch is owned by
the M2 lane ticket (Main's schedule, post-23:50Z, family-lane slot).
"""
from __future__ import annotations

import argparse
import ast
import re
import csv
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


KEYWORDS_REFUSED = (
    "simdgroup_matrix",
    "matmul2d_descriptor",
    "mpp::tensor_ops",
    "texture",
    "sampler",
    "imageblock",
    "raytracing",
    "quadgroup",
    "visible_function",
    "intersection_function",
    "object_data",
)

# device-pointer alias classification (translate_device_pointer_aliases,
# landed e22eaf199): scalar-pointee aliases rewrite to base[(off)+(i)];
# vector-pointee aliases and pointer arrays stay exact-error (an alias
# index spans a vector; arrays need a wider rewrite).
PTR_ALIAS = re.compile(
    r"\bdevice\s+(?:const\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\*\s*"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);")
PTR_ARRAY = re.compile(
    r"\bdevice\s+(?:const\s+)?[A-Za-z_][A-Za-z0-9_]*\s*\*\s*"
    r"[A-Za-z_][A-Za-z0-9_]*\s*\[")
VEC_SUFFIX = re.compile(r"[234]$")


def ptr_alias_class(msl: str) -> str:
    if PTR_ARRAY.search(msl):
        return "ptr-array"
    decls = PTR_ALIAS.findall(msl)
    if not decls:
        return "none"
    return "vector" if any(VEC_SUFFIX.search(t) for t, _, _ in decls) \
        else "scalar"


@dataclass
class Kernel:
    source: str
    file: str
    name: str
    msllib: str
    gmsl_features: list[str] = field(default_factory=list)
    template_args: dict[str, object] = field(default_factory=dict)
    line: int = 0


def walk_calls(tree: ast.AST) -> Iterator[ast.Call]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


def call_str(node: ast.Call) -> str:
    """Best-effort reconstruction of `callee(args)`.
    Returns a short string like `mx.fast.metal_kernel(...)`."""
    func = node.func
    parts = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    parts.reverse()
    return ".".join(parts) + "(...)"


def extract_kernels(py_file: Path, source_tag: str) -> Iterator[Kernel]:
    try:
        tree = ast.parse(py_file.read_text(), filename=str(py_file))
    except (SyntaxError, UnicodeDecodeError) as exc:
        return
    # Build a name -> string-literal map for local variables that hold an
    # MSL source (r"""...""" or "..."). This lets us resolve `source=source`
    # patterns where the literal is assigned to a name earlier in the file.
    # Iterate assignments in source order (by line number) so concatenations
    # of the form `source = _HEADER + _PROLOGUE + _EPILOGUE` resolve even
    # when the helpers appear in any order in the file.
    str_consts: dict[str, str] = {}
    assigns = [n for n in ast.walk(tree)
               if isinstance(n, ast.Assign) and len(n.targets) == 1
               and isinstance(n.targets[0], ast.Name)]
    assigns.sort(key=lambda n: n.lineno)
    for _ in range(50):
        progress = False
        for stmt in assigns:
            tgt = stmt.targets[0]
            if tgt.id in str_consts:
                continue
            value = stmt.value
            # Direct string constant
            if (isinstance(value, ast.Constant)
                    and isinstance(value.value, str)):
                text = value.value
                # MSL heuristics are intentionally broad: any string that
                # looks like a GPU shader header or body. _HEADER-only
                # strings in TensorFold carry inline helpers and a
                # `_Pragma` token; PROLOGUE/BODY strings carry
                # thread_position / device / threadgroup / [[buffer
                # markup. Either family is correctly identified.
                if ("thread_position" in text or "[[buffer" in text or
                        "device " in text or "threadgroup " in text or
                        "metal::" in text or "kernel void" in text or
                        "as_type" in text or
                        ("_Pragma" in text and "inline" in text) or
                        ("uint" in text and "int" in text and "(" in text)):
                    str_consts[tgt.id] = text
                progress = True
                continue
            # Binary string concat (left + right)
            if (isinstance(value, ast.BinOp)
                    and isinstance(value.op, ast.Add)):
                left, right = value.left, value.right
                if (isinstance(left, ast.Constant)
                        and isinstance(left.value, str)
                        and isinstance(right, ast.Name)
                        and right.id in str_consts):
                    str_consts[tgt.id] = left.value + str_consts[right.id]
                    progress = True
                elif (isinstance(left, ast.Name)
                        and left.id in str_consts
                        and isinstance(right, ast.Constant)
                        and isinstance(right.value, str)):
                    str_consts[tgt.id] = str_consts[left.id] + right.value
                    progress = True
                elif (isinstance(left, ast.Name)
                        and left.id in str_consts
                        and isinstance(right, ast.Name)
                        and right.id in str_consts):
                    str_consts[tgt.id] = str_consts[left.id] + str_consts[right.id]
                    progress = True
        if not progress:
            break
    for call in walk_calls(tree):
        if not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr != "metal_kernel":
            continue
        if not (isinstance(call.func.value, ast.Attribute) and
                call.func.value.attr == "fast"):
            continue
        # Extract name= and source=
        name = ""
        msllib = ""
        template_args: dict[str, object] = {}
        for kw in call.keywords:
            if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                name = str(kw.value.value)
            elif kw.arg == "name" and isinstance(kw.value, ast.JoinedStr):
                # f-string: capture as best-effort without evaluation
                name = "<fstring>"
            elif kw.arg == "source" and isinstance(kw.value, ast.Constant):
                msllib = str(kw.value.value)
            elif kw.arg == "source" and isinstance(kw.value, ast.Name):
                msllib = str_consts.get(kw.value.id, "")
            elif kw.arg == "template" and isinstance(kw.value, ast.Dict):
                for k, v in zip(kw.value.keys, kw.value.values):
                    if isinstance(k, ast.Constant):
                        try:
                            template_args[str(k.value)] = ast.literal_eval(v) \
                                if isinstance(v, (ast.Constant, ast.List, ast.Tuple)) \
                                else "<expr>"
                        except (ValueError, SyntaxError):
                            template_args[str(k.value)] = "<expr>"
        if not msllib:
            continue  # dynamic or external source string; skip
        gmsl_features = [k for k in KEYWORDS_REFUSED if k in msllib]
        yield Kernel(
            source=source_tag,
            file=str(py_file),
            name=name,
            msllib=msllib,
            gmsl_features=gmsl_features,
            template_args=template_args,
            line=call.lineno,
        )


def extract_from_tree(root: Path, source_tag: str) -> Iterator[Kernel]:
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield from extract_kernels(path, source_tag)


def invoke_translator(driver: Path, msllib: str) -> tuple[str, str]:
    """Run the translator driver. Returns (status, error_text)."""
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".msl", delete=False) as in_f:
        in_f.write(msllib)
        in_path = in_f.name
    try:
        proc = subprocess.run(
            [str(driver), in_path],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode == 0:
            return "accept", ""
        return "reject", proc.stderr.strip().splitlines()[-1] if proc.stderr else ""
    except subprocess.TimeoutExpired:
        return "reject", "translator timeout"
    finally:
        os.unlink(in_path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--omlx", type=Path, required=False)
    p.add_argument("--tf", type=Path, required=False)
    p.add_argument("--translator", type=Path, required=False,
                   help="Path to omarchy_translate_msl_driver binary. "
                        "If omitted, only static classification runs.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, int]] = set()
    n_total = 0

    feeds: list[tuple[str, Path]] = []
    if args.omlx:
        feeds.append(("omlx", args.omlx))
    if args.tf:
        feeds.append(("tf", args.tf))

    for tag, root in feeds:
        for kern in extract_from_tree(root, tag):
            key = (kern.source, kern.file, kern.line)
            if key in seen:
                continue
            seen.add(key)
            n_total += 1
            if args.limit and n_total > args.limit:
                break
            row = {
                "source": kern.source,
                "file": kern.file,
                "name": kern.name,
                "line": str(kern.line),
                "kind": "mtk",
                "gmsl_features": "|".join(kern.gmsl_features),
                "ptr_alias": ptr_alias_class(kern.msllib),
                "msl_size": str(len(kern.msllib)),
                "ttf_status": "skip",
                "ttf_error": "",
                "run_status": "defer_m2",
                "run_error": "",
            }
            if kern.gmsl_features:
                # pre-classify as a refusal without invoking the driver
                row["ttf_status"] = "reject-by-keyword"
                row["ttf_error"] = "found Metal-only: " + ",".join(kern.gmsl_features)
            elif args.translator:
                status, err = invoke_translator(args.translator, kern.msllib)
                row["ttf_status"] = status
                row["ttf_error"] = err
            rows.append(row)
        if args.limit and n_total > args.limit:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [
            "source", "file", "name", "line", "kind",
            "gmsl_features", "msl_size", "ttf_status",
            "ttf_error", "run_status", "run_error"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} rows to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
