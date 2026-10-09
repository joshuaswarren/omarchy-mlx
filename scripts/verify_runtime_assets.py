#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Verify installed/staged Parakeet runtime assets against the runtime pin.

Usage:
    python3 verify_runtime_assets.py SHARE_DIR [SHARE_DIR ...]

Each SHARE_DIR is a ``mlx/share/mlx-omarchy/parakeet-1`` tree. When it
carries ``parakeet-runtime-pin.json``, every pinned bundle payload and the
strict libane must exist with exactly the pinned SHA-256, and every file
under ``bundles/`` and ``libane/`` must be covered by the pin — a tree that
ships bytes the pin does not name is exactly how a swapped ANE userspace
reaches a package silently (the 2026-10-02 omarchy-mlx 0.7.10-1 defect: the
package stage carried a locally rebuilt libane-strict.so/worker beside the
wheel's untouched pin, and the worker seal refused at first transcribe).

The fd-protocol ANE worker ships outside the share tree, at
``mlx/bin/mlx-omarchy-ane-worker`` beside it. It compiles inside the wheel
build, so its digest is knowable only after the build:
``scripts/build-wheel.sh`` stamps the built worker's SHA-256 into
``assets.worker`` of the pin the wheel ships. The checker therefore refuses
an installed tree that ships a worker its pin does not name (wheels cut
before worker pinning fail here; repin to a release built by the stamping
pipeline), and a build tree that has not built the worker yet still passes.

Exits 0 when every present pin manifest verifies; exits 1 with
expected/actual digests and the two legal fixes otherwise. Trees without a
pin manifest (non-aarch64 installs) are skipped.
"""

import hashlib
import json
import sys
from pathlib import Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mismatches(share: Path) -> list[str]:
    pin = json.loads((share / "parakeet-runtime-pin.json").read_text())
    problems: list[str] = []
    pinned: dict[str, str] = {}
    for bundle, files in sorted(pin["assets"].get("bundles", {}).items()):
        for name, digest in sorted(files.items()):
            pinned[f"bundles/{bundle}/{name}"] = digest
    for name, digest in sorted(pin["assets"].get("libane", {}).items()):
        pinned[f"libane/{name}"] = digest

    for rel, expected in sorted(pinned.items()):
        path = share / rel
        if not path.is_file():
            problems.append(f"MISSING {rel}: pin requires it (expected {expected})")
        else:
            actual = _sha256_file(path)
            if actual != expected:
                problems.append(
                    f"MISMATCH {rel}: expected {expected}, got {actual}"
                )

    shipped_roots = [share / "bundles", share / "libane"]
    for root in sorted(filter(None, shipped_roots)):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = str(path.relative_to(share))
            if rel not in pinned:
                problems.append(
                    f"UNPINNED {rel}: shipped but not named by the pin "
                    f"(sha256 {_sha256_file(path)})"
                )

    # The ANE worker lives at mlx/bin beside the share tree. Release builds
    # stamp its digest into assets.worker; a shipped-but-unnamed worker is
    # the same defect class as an unpinned libane.
    if len(share.parents) > 2:
        bin_dir = share.parents[2] / "bin"
        named_workers = set(pin["assets"].get("worker", {}))
        for name, digest in sorted(pin["assets"].get("worker", {}).items()):
            path = bin_dir / name
            if not path.is_file():
                problems.append(
                    f"MISSING bin/{name}: pin requires it (expected {digest})"
                )
            else:
                actual = _sha256_file(path)
                if actual != digest:
                    problems.append(
                        f"MISMATCH bin/{name}: expected {digest}, got {actual}"
                    )
        worker_bin = bin_dir / "mlx-omarchy-ane-worker"
        if worker_bin.is_file() and "mlx-omarchy-ane-worker" not in named_workers:
            problems.append(
                f"UNPINNED bin/mlx-omarchy-ane-worker: shipped but not named "
                f"by the pin (sha256 {_sha256_file(worker_bin)}); release "
                f"builds stamp the built worker sha into assets.worker "
                f"(scripts/build-wheel.sh)"
            )
    return problems


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    failures = 0
    for arg in argv:
        share = Path(arg)
        if not (share / "parakeet-runtime-pin.json").is_file():
            print(f"skip: no pin manifest in {share}")
            continue
        problems = _mismatches(share)
        if problems:
            failures += 1
            print(f"FAIL {share}: installed runtime assets disagree with the pin:")
            for line in problems:
                print(f"  {line}")
            print(
                "  fix: rebuild the runtime assets from the pinned omarchy-ane "
                "commit (see provenance in parakeet-runtime-pin.json) or move "
                "the pin in the same change — never ship a tree whose bytes "
                "its own pin does not name"
            )
        else:
            print(f"OK: {share}: pinned runtime assets verified")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
