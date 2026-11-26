#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Wrap an H13 v2 ANEC package as a schema-4 bundle.

``h13_package_to_bundle`` rejects unknown tensor-object fields
(``tensors.cond.dtype`` on the attn-select island). Schema-4 infers
dtype from program bindings, so this tool drops only those tensor-object
keys and leaves binding dtypes — including ``bool`` — intact. The
original package is not rewritten.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import shutil
import sys
import tempfile
from pathlib import Path

_ANE_EXPORT = Path(__file__).resolve().parents[1] / "ane-export"
if str(_ANE_EXPORT) not in sys.path:
    sys.path.insert(0, str(_ANE_EXPORT))
import h13_package_to_bundle as _h13  # noqa: E402

# Matches h13_package_to_bundle tensor-object allow-list.
TENSOR_OBJECT_FIELDS = frozenset(
    {"accumulation", "aliasOf", "logicalBytes", "role", "shape"}
)

AdapterError = _h13.AdapterError


def drop_unknown_tensor_fields(source: dict) -> tuple[dict, list[tuple[str, str]]]:
    """Copy ``source`` and strip tensor-object keys schema-4 rejects."""
    out = copy.deepcopy(source)
    dropped: list[tuple[str, str]] = []
    tensors = out.get("tensors")
    if not isinstance(tensors, dict):
        return out, dropped
    for name, tensor in tensors.items():
        if not isinstance(tensor, dict):
            continue
        for key in list(tensor):
            if key not in TENSOR_OBJECT_FIELDS:
                del tensor[key]
                dropped.append((str(name), key))
    return out, dropped


def convert(package: Path, output: Path, identity: dict) -> tuple[dict, list[tuple[str, str]]]:
    source = _h13.load_object(package / "manifest.json")
    cleaned, dropped = drop_unknown_tensor_fields(source)
    with tempfile.TemporaryDirectory() as directory:
        staged = Path(directory) / "package"
        shutil.copytree(package, staged)
        (staged / "manifest.json").write_text(
            json.dumps(cleaned, indent=2, sort_keys=True) + "\n"
        )
        manifest = _h13.adapt(staged, output, identity)
    return manifest, dropped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--graph-hash", required=True)
    parser.add_argument("--compiler-host-build", required=True)
    parser.add_argument("--compiler-toolchain", required=True)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--exported-at", default=datetime.date.today().isoformat())
    args = parser.parse_args()
    identity = {
        "name": args.name,
        "graph_hash": args.graph_hash,
        "compiler_host_build": args.compiler_host_build,
        "compiler_toolchain": args.compiler_toolchain,
        "source_repo": args.source_repo,
        "source_commit": args.source_commit,
        "exported_at": args.exported_at,
        "model": args.model,
    }
    try:
        manifest, dropped = convert(args.package, args.out_dir, identity)
    except (AdapterError, OSError) as error:
        print(f"h13_v2_to_schema4: error: {error}", file=sys.stderr)
        return 1
    print(
        f"h13_v2_to_schema4: PASS programs={len(manifest['programs'])} "
        f"dropped={dropped} output={args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
