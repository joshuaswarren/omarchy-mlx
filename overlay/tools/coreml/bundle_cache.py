# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Cached bundle production for the ANE worker path (sections 28-29).

Wraps the validated package→bundle adaptation
(``overlay/tools/ane-export/h13_package_to_bundle.py``) in the
content-addressed compiled cache (``coreml.compiled_cache``): the
producer runs at most once per key, every hit re-hashes the stored
bundle, and the worker loads bundles from the cache path. The key binds
the model artifacts, the compiler identity from the package's pinned
receipt, the operation set, the bundle schema, the driver ABI, the
firmware identity, and the frontend version — so any change on any of
those axes misses the cache.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent.parent
for entry in (str(_TOOLS), str(_TOOLS / "coreml"), str(_TOOLS / "ane-export")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from coreml.compiled_cache import (  # noqa: E402
    CacheKey,
    CompiledCache,
    CompilerIdentity,
    hash_model_artifacts,
)

FRONTEND_VERSION = "mlx-omarchy-coreml-1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class BundleCacheError(RuntimeError):
    """The bundle cache cannot be used; the reason is named."""


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise BundleCacheError(f"{path}: {error}") from error
    if not isinstance(value, dict):
        raise BundleCacheError(f"{path}: expected a JSON object")
    return value


def bundle_cache_key(
    package: Path,
    *,
    static_input_shapes: dict[str, tuple[int, ...]],
    driver_abi_major: int = 1,
    firmware_identity: str = "T8103-H13",
    frontend_version: str = FRONTEND_VERSION,
) -> tuple[CacheKey, dict]:
    """Build the cache key for one compiler package.

    The compiler identity comes from the package's pinned
    ``source.json`` receipt (compiler commit, binary sha, schema, host
    build); the operation set comes from the compiler manifest.
    """
    package = Path(package)
    receipt = _load_json(package / "source.json")
    manifest = _load_json(package / "manifest.json")
    for field in (
        "compiler_executable_source_commit",
        "compiler_binary_sha256",
        "compiler_host_build",
    ):
        if not isinstance(receipt.get(field), str) or not receipt[field]:
            raise BundleCacheError(
                f"package receipt is missing '{field}'"
            )
    if not _SHA256_RE.match(receipt["compiler_binary_sha256"]):
        raise BundleCacheError(
            "receipt compiler_binary_sha256 must be 64-char hex"
        )
    schema = manifest.get("schema")
    target = manifest.get("target")
    if not isinstance(schema, str) or not isinstance(target, str):
        raise BundleCacheError("compiler manifest must name schema and target")
    operations = {
        program.get("operation", "anec")
        for program in manifest.get("programs", [])
        if isinstance(program, dict)
    }
    operations.add("anec")
    from coreml.compiled_cache import _hash_tree

    key = CacheKey.create(
        model_artifacts=_hash_tree(Path(package), "compiler package"),
        selected_function="main",
        static_input_shapes=static_input_shapes,
        compiler=CompilerIdentity(
            name="mil-hwxc",
            version=receipt["compiler_executable_source_commit"][:12],
            commit=receipt["compiler_executable_source_commit"],
            target=target,
            package_schema=schema,
        ),
        operations=operations,
        bundle_schema=4,
        driver_abi_major=driver_abi_major,
        firmware_identity=firmware_identity,
        frontend_version=frontend_version,
    )
    return key, receipt


def cached_bundle(
    package: Path,
    cache_root: Path,
    *,
    graph_source: Path,
    compiler_source: Path,
    name: str,
    source_repo: str,
    source_commit: str,
    model: str,
    static_input_shapes: dict[str, tuple[int, ...]],
    exported_at: str = "1970-01-01",
    firmware_identity: str = "T8103-H13",
) -> tuple[Path, bool]:
    """Return ``(bundle_path, cache_hit)`` for the adapted bundle.

    On the first call the package is adapted inside the cache's staging
    area and atomically published; later calls with the same key return
    the verified cached entry without re-adapting.
    """
    import h13_package_to_bundle as adapter

    key, receipt = bundle_cache_key(
        package, static_input_shapes=static_input_shapes,
        firmware_identity=firmware_identity,
    )
    cache = CompiledCache(Path(cache_root))

    def produce(bundle: Path) -> None:
        identity = {
            "name": name,
            "graph_hash": receipt["graph_source_sha256"],
            "compiler_host_build": receipt["compiler_host_build"],
            "compiler_toolchain": (
                f"mil-hwxc {receipt['compiler_executable_source_commit']} "
                f"sha256:{receipt['compiler_binary_sha256']}"
            ),
            "source_repo": source_repo,
            "source_commit": source_commit,
            "exported_at": exported_at,
            "model": model,
        }
        adapter.adapt(Path(package), bundle, identity)

    result = cache.get_or_create(key, produce)
    return result.bundle, result.hit
