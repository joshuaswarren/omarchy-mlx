# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Content-addressed cache for Linux-compiled Core ML ANE bundles."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator, Mapping, Sequence

CACHE_KEY_SCHEMA = "mlx-omarchy.coreml-cache-key.v1"
CACHE_ENTRY_SCHEMA = "mlx-omarchy.coreml-cache-entry.v1"


class CompiledCacheError(RuntimeError):
    """Base error for compiled-cache failures."""


class CacheContractError(CompiledCacheError, ValueError):
    """The compatibility key or artifact tree is not cacheable."""


class CacheCorruptionError(CompiledCacheError):
    """A stored entry failed metadata or payload verification."""


class CacheProducerError(CompiledCacheError):
    """The producer did not create a cacheable compiled bundle."""


@dataclass(frozen=True, order=True)
class ArtifactDigest:
    path: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        relative = PurePosixPath(self.path)
        if not self.path or relative.is_absolute() or ".." in relative.parts:
            raise CacheContractError(f"artifact path is not relative: {self.path!r}")
        _require_sha256(self.sha256, f"artifact {self.path!r} SHA-256")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise CacheContractError(f"artifact {self.path!r} size must be non-negative")

    def document(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}


@dataclass(frozen=True)
class CompilerIdentity:
    name: str
    version: str
    commit: str
    target: str
    package_schema: str

    def __post_init__(self) -> None:
        for field in ("name", "version", "target", "package_schema"):
            _require_text(getattr(self, field), f"compiler {field}")
        if len(self.commit) != 40 or any(c not in "0123456789abcdef" for c in self.commit):
            raise CacheContractError("compiler commit must be 40 lowercase hex characters")

    def document(self) -> dict[str, str]:
        return {
            "name": self.name,
            "version": self.version,
            "commit": self.commit,
            "target": self.target,
            "package_schema": self.package_schema,
        }


@dataclass(frozen=True)
class CacheKey:
    model_artifacts: tuple[ArtifactDigest, ...]
    selected_function: str
    static_input_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    compiler: CompilerIdentity
    operations: tuple[str, ...]
    bundle_schema: int
    driver_abi_major: int
    firmware_identity: str
    frontend_version: str

    @classmethod
    def create(
        cls,
        *,
        model_artifacts: Sequence[ArtifactDigest],
        selected_function: str,
        static_input_shapes: Mapping[str, Sequence[int]],
        compiler: CompilerIdentity,
        operations: Sequence[str] | set[str],
        bundle_schema: int,
        driver_abi_major: int,
        firmware_identity: str,
        frontend_version: str,
    ) -> "CacheKey":
        shapes = tuple(
            sorted((name, tuple(dimensions)) for name, dimensions in static_input_shapes.items())
        )
        return cls(
            model_artifacts=tuple(sorted(model_artifacts)),
            selected_function=selected_function,
            static_input_shapes=shapes,
            compiler=compiler,
            operations=tuple(sorted(set(operations))),
            bundle_schema=bundle_schema,
            driver_abi_major=driver_abi_major,
            firmware_identity=firmware_identity,
            frontend_version=frontend_version,
        )

    def __post_init__(self) -> None:
        if not self.model_artifacts:
            raise CacheContractError("model artifact hashes must not be empty")
        if tuple(sorted(self.model_artifacts)) != self.model_artifacts:
            raise CacheContractError("model artifact hashes must be sorted")
        if len({artifact.path for artifact in self.model_artifacts}) != len(self.model_artifacts):
            raise CacheContractError("model artifact paths must be unique")
        _require_text(self.selected_function, "selected function")
        if not self.static_input_shapes:
            raise CacheContractError("static input shapes must not be empty")
        if tuple(sorted(self.static_input_shapes)) != self.static_input_shapes:
            raise CacheContractError("static input shapes must be sorted")
        if len({name for name, _ in self.static_input_shapes}) != len(self.static_input_shapes):
            raise CacheContractError("static input names must be unique")
        for name, dimensions in self.static_input_shapes:
            _require_text(name, "static input name")
            if not dimensions or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
                for dimension in dimensions
            ):
                raise CacheContractError(
                    f"static input shape for {name!r} must contain positive integers"
                )
        if not self.operations:
            raise CacheContractError("operation set must not be empty")
        if tuple(sorted(set(self.operations))) != self.operations:
            raise CacheContractError("operation set must be sorted and unique")
        for operation in self.operations:
            _require_text(operation, "operation")
        _require_positive_int(self.bundle_schema, "bundle schema")
        _require_positive_int(self.driver_abi_major, "driver ABI major")
        _require_text(self.firmware_identity, "firmware compatibility identity")
        _require_text(self.frontend_version, "frontend version")

    @property
    def document(self) -> dict[str, object]:
        return {
            "schema": CACHE_KEY_SCHEMA,
            "model_artifacts": [artifact.document() for artifact in self.model_artifacts],
            "selected_function": self.selected_function,
            "static_input_shapes": {
                name: list(dimensions) for name, dimensions in self.static_input_shapes
            },
            "compiler": self.compiler.document(),
            "operations": list(self.operations),
            "abi": {
                "bundle_schema": self.bundle_schema,
                "driver_abi_major": self.driver_abi_major,
                "firmware_identity": self.firmware_identity,
            },
            "frontend_version": self.frontend_version,
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.document)).hexdigest()


@dataclass(frozen=True)
class CacheResult:
    bundle: Path
    hit: bool


def default_cache_root() -> Path:
    override = os.environ.get("MLX_OMARCHY_CACHE_DIR")
    if override:
        return Path(override).expanduser() / "coreml"
    return Path.home() / ".cache" / "mlx-omarchy" / "coreml"


def hash_model_artifacts(package: Path) -> tuple[ArtifactDigest, ...]:
    """Hash every regular source artifact in a canonical .mlpackage tree."""
    package = Path(package)
    if package.suffix != ".mlpackage" or not package.is_dir():
        raise CacheContractError(
            "canonical Core ML source must be a .mlpackage directory"
        )
    return _hash_tree(package, "model package")


class CompiledCache:
    def __init__(self, root: Path | None = None):
        self.root = (Path(root) if root is not None else default_cache_root()).expanduser()

    def lookup(self, key: CacheKey) -> Path | None:
        with self._key_lock(key):
            return self._lookup_unlocked(key)

    def invalidate(self, key: CacheKey) -> None:
        with self._key_lock(key):
            self._remove_entry(key)

    def get_or_create(
        self,
        key: CacheKey,
        producer: Callable[[Path], None],
    ) -> CacheResult:
        """Return a verified hit or atomically publish one producer result."""
        with self._key_lock(key):
            cached = self._lookup_unlocked(key)
            if cached is not None:
                return CacheResult(cached, True)
            with tempfile.TemporaryDirectory(
                prefix=f".{key.digest}.", dir=self.root
            ) as temporary:
                staging = Path(temporary) / "entry"
                staging.mkdir()
                bundle = staging / "bundle"
                producer(bundle)
                if not bundle.is_dir() or bundle.is_symlink():
                    raise CacheProducerError("producer must create a regular bundle directory")
                try:
                    artifacts = _hash_tree(bundle, "compiled bundle")
                except CacheContractError as error:
                    raise CacheProducerError(str(error)) from error
                metadata = {
                    "schema": CACHE_ENTRY_SCHEMA,
                    "key_sha256": key.digest,
                    "key": key.document,
                    "bundle_artifacts": [artifact.document() for artifact in artifacts],
                }
                (staging / "cache.json").write_bytes(
                    json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8") + b"\n"
                )
                destination = self._entry(key)
                if destination.exists() or destination.is_symlink():
                    self._remove_entry(key)
                os.rename(staging, destination)
            verified = self._verify_entry(key)
            return CacheResult(verified, False)

    def _entry(self, key: CacheKey) -> Path:
        return self.root / key.digest

    @contextmanager
    def _key_lock(self, key: CacheKey) -> Iterator[None]:
        try:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor = os.open(
                self.root / f".{key.digest}.lock",
                os.O_CREAT | os.O_RDWR,
                0o600,
            )
        except OSError as error:
            raise CompiledCacheError(f"cannot prepare Core ML cache {self.root}: {error}") from error
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            os.close(descriptor)

    def _lookup_unlocked(self, key: CacheKey) -> Path | None:
        entry = self._entry(key)
        if not entry.exists() and not entry.is_symlink():
            return None
        try:
            return self._verify_entry(key)
        except CacheCorruptionError:
            self._remove_entry(key)
            return None

    def _verify_entry(self, key: CacheKey) -> Path:
        entry = self._entry(key)
        metadata_path = entry / "cache.json"
        bundle = entry / "bundle"
        if entry.is_symlink() or not entry.is_dir():
            raise CacheCorruptionError("cache entry is not a regular directory")
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise CacheCorruptionError("cache entry metadata is missing")
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CacheCorruptionError(f"cache entry metadata is unreadable: {error}") from error
        required = {"schema", "key_sha256", "key", "bundle_artifacts"}
        if not isinstance(metadata, dict) or set(metadata) != required:
            raise CacheCorruptionError("cache entry metadata fields are invalid")
        if metadata["schema"] != CACHE_ENTRY_SCHEMA:
            raise CacheCorruptionError("cache entry schema is incompatible")
        if metadata["key_sha256"] != key.digest or metadata["key"] != key.document:
            raise CacheCorruptionError("cache entry compatibility key does not match")
        try:
            actual = _hash_tree(bundle, "compiled bundle")
        except CacheContractError as error:
            raise CacheCorruptionError(str(error)) from error
        expected = [artifact.document() for artifact in actual]
        if metadata["bundle_artifacts"] != expected:
            raise CacheCorruptionError("cache entry payload hashes do not match")
        return bundle

    def _remove_entry(self, key: CacheKey) -> None:
        entry = self._entry(key)
        try:
            if entry.is_symlink() or entry.is_file():
                entry.unlink()
            elif entry.exists():
                shutil.rmtree(entry)
        except OSError as error:
            raise CompiledCacheError(f"cannot invalidate Core ML cache entry: {error}") from error


def _hash_tree(root: Path, label: str) -> tuple[ArtifactDigest, ...]:
    if root.is_symlink():
        raise CacheContractError(f"{label} must not be a symlink")
    if root.is_file():
        return (ArtifactDigest(root.name, _sha256(root), root.stat().st_size),)
    if not root.is_dir():
        raise CacheContractError(f"{label} must be a file or directory")
    artifacts: list[ArtifactDigest] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CacheContractError(f"{label} contains a symlink: {path.relative_to(root)}")
        if path.is_file():
            artifacts.append(
                ArtifactDigest(
                    path.relative_to(root).as_posix(),
                    _sha256(path),
                    path.stat().st_size,
                )
            )
        elif not path.is_dir():
            raise CacheContractError(
                f"{label} contains a non-regular artifact: {path.relative_to(root)}"
            )
    if not artifacts:
        raise CacheContractError(f"{label} contains no artifacts")
    return tuple(artifacts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise CacheContractError(f"cannot hash artifact {path}: {error}") from error
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _require_text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise CacheContractError(f"{label} must be a non-empty string")


def _require_sha256(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CacheContractError(f"{label} must be 64 lowercase hex characters")


def _require_positive_int(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CacheContractError(f"{label} must be a positive integer")
