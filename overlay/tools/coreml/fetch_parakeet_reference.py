# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Reproducible downloader for the pinned public Parakeet reference.

Commands::

    python3 overlay/tools/coreml/fetch_parakeet_reference.py download [--force]
    python3 overlay/tools/coreml/fetch_parakeet_reference.py verify
    python3 overlay/tools/coreml/fetch_parakeet_reference.py path
    python3 overlay/tools/coreml/fetch_parakeet_reference.py info

``download`` places every pinned file of the pinned Hugging Face
revision into the predictable cache and SHA-256-verifies all content.
Existing cache entries are re-hashed (never trusted); anything that
does not match the pin is re-fetched; freshly fetched content that
still mismatches the pin is a hard error. ``verify`` re-hashes the
cache and exits non-zero on any mismatch. ``path`` prints the cache
directory. ``info`` prints the lock summary including licenses.

Integrity contract: file size alone is never accepted as integrity.
Before any download, the live HF tree is cross-checked against the pin
(LFS oids are sha-256 content hashes; non-LFS files are compared by
git blob id). A moved upstream revision is a hard error. When the
cache already verifies complete, no network access happens at all
(fully offline reproducibility).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

if __package__ in (None, ""):  # plain-script invocation
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from reference import (  # noqa: E402
    ReferenceError,
    ReferenceLock,
    _verify_cache_full,
    default_cache_root,
    default_lock_path,
    git_blob_sha1,
    model_cache_dir,
    model_url,
    record_stamps,
    sha256_file,
    tree_url,
    validate_lock,
    verify_cache,
    verify_cache_with_stamp,
)

_CHUNK = 1 << 20


def _http_json(url: str, timeout: float = 60.0):
    request = urllib.request.Request(
        url, headers={"User-Agent": "mlx-omarchy-parakeet-reference"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return json.load(resp)


def _download_to(url: str, dest: Path, timeout: float = 300.0) -> None:
    """Stream ``url`` into ``dest`` via a temp file + atomic rename."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "mlx-omarchy-parakeet-reference"}
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), prefix=".fetch-")
    tmp = Path(tmp_name)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp, open(
            fd, "wb"
        ) as out:
            while True:
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def check_upstream_tree(lock: ReferenceLock, cache_dir: Path) -> None:
    """Cross-check the live HF tree against the pin; hard error on drift.

    LFS files carry their sha-256 as the LFS oid. Non-LFS files expose
    only the git blob sha1, so the pinned cache copy's blob id is
    compared instead (the cache copy itself is separately verified
    against the pinned sha-256).
    """
    tree = _http_json(tree_url(lock.model_repo, lock.model_revision))
    upstream = {
        entry["path"]: entry for entry in tree if entry.get("type") == "file"
    }
    for f in lock.files:
        entry = upstream.get(f.path)
        if entry is None:
            raise ReferenceError(
                f"upstream tree is missing pinned file {f.path!r}; "
                f"revision {lock.model_revision} does not match the lock"
            )
        lfs = entry.get("lfs") or {}
        if lfs.get("oid"):
            if lfs["oid"] != f.sha256:
                raise ReferenceError(
                    f"upstream content drift for {f.path}: lock pins "
                    f"{f.sha256}, revision now exposes LFS oid {lfs['oid']}"
                )
        else:
            local = cache_dir / f.path
            if not local.is_file():
                continue  # nothing pinned locally; the fetch path covers it
            blob = git_blob_sha1(local)
            if blob != entry["oid"]:
                raise ReferenceError(
                    f"upstream content drift for {f.path}: local pinned copy "
                    f"blob {blob} != revision blob {entry['oid']}"
                )


def cmd_download(lock: ReferenceLock, cache_dir: Path, force: bool) -> int:
    validate_lock(lock)

    if not force:
        ok, _ = verify_cache(cache_dir, lock)
        if ok:
            print(f"cache already verifies: {cache_dir}")
            return 0

    # Online path: pin drift check first, then fetch only what is
    # missing or mismatched, then verify the whole cache again.
    check_upstream_tree(lock, cache_dir)

    _, _, bad_paths = _verify_cache_full(cache_dir, lock)

    for f in lock.files:
        if f.path not in bad_paths:
            continue
        url = model_url(lock.model_repo, lock.model_revision, f.path)
        print(f"fetching {f.path} ({f.size} bytes)")
        _download_to(url, cache_dir / f.path)
        actual = sha256_file(cache_dir / f.path)
        if actual != f.sha256:
            raise ReferenceError(
                f"fetched content for {f.path} does not match the pin: "
                f"expected {f.sha256}, got {actual} (refusing to use it)"
            )

    ok, mismatches = verify_cache(cache_dir, lock)
    if not ok:
        for m in mismatches:
            print(f"MISMATCH: {m}", file=sys.stderr)
        raise ReferenceError("cache failed post-download verification")

    record_stamps(cache_dir, lock)
    print(f"verified {len(lock.files)} files: {cache_dir}")
    return 0


def cmd_verify(lock: ReferenceLock, cache_dir: Path) -> int:
    validate_lock(lock)
    ok, mismatches = verify_cache(cache_dir, lock)
    if ok:
        record_stamps(cache_dir, lock)
        print(f"OK: {len(lock.files)} files verified in {cache_dir}")
        return 0
    for m in mismatches:
        print(f"MISMATCH: {m}", file=sys.stderr)
    return 1


def cmd_path(cache_dir: Path) -> int:
    print(cache_dir)
    return 0


def cmd_info(lock: ReferenceLock, cache_dir: Path) -> int:
    ok, mismatches = verify_cache(cache_dir, lock)
    info = {
        "reference_repo": lock.reference_repo,
        "reference_commit": lock.reference_commit,
        "model_repo": lock.model_repo,
        "model_revision": lock.model_revision,
        "model_license": lock.model_license,
        "model_quantization": lock.model_quantization,
        "audio": {
            "url": lock.audio.url,
            "sha256": lock.audio.sha256,
            "license": lock.audio.license,
            "duration_seconds": lock.audio.duration_seconds,
        },
        "mel": lock.mel.__dict__,
        "tdt": {
            "blank_token_id": lock.tdt.blank_token_id,
            "durations": lock.tdt.durations,
            "max_symbols_per_step": lock.tdt.max_symbols_per_step,
            "vocab_size": lock.tdt.vocab_size,
        },
        "cache_dir": str(cache_dir),
        "cache_ok": ok,
        "cache_mismatches": mismatches,
        "numerical_contract_frozen": lock.numerical_contract is not None,
        "golden_captured": bool(lock.macos_reference_paths),
        "macos_reference_environment": (
            lock.macos_reference_environment.__dict__
            if lock.macos_reference_environment
            else None
        ),
    }
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fetch_parakeet_reference",
        description="Download/verify the pinned public Parakeet reference.",
    )
    parser.add_argument(
        "command",
        choices=["download", "verify", "path", "info"],
        help="action to run",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even when the cache already verifies",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=None,
        help="path to parakeet-reference.lock (default: next to this module)",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="override the cache root (default: $MLX_OMARCHY_CACHE_DIR or ~/.cache)",
    )
    args = parser.parse_args(argv)

    lock = ReferenceLock.load(args.lock or default_lock_path())
    if args.cache_root:
        # same semantics as $MLX_OMARCHY_CACHE_DIR: the reference cache
        # lives under <root>/parakeet-reference/<repo>/<revision>
        root = args.cache_root / "parakeet-reference"
    else:
        root = default_cache_root()
    cache_dir = model_cache_dir(root, lock.model_repo, lock.model_revision)
    try:
        if args.command == "download":
            return cmd_download(lock, cache_dir, force=args.force)
        if args.command == "verify":
            return cmd_verify(lock, cache_dir)
        if args.command == "path":
            return cmd_path(cache_dir)
        if args.command == "info":
            return cmd_info(lock, cache_dir)
    except ReferenceError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 2
    except urllib.error.URLError as ex:
        print(f"network error: {ex}", file=sys.stderr)
        return 3
    raise AssertionError("unreachable")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
