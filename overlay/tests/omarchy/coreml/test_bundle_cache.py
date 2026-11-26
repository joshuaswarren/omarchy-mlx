# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host tests for the worker-path bundle cache (sections 28-29)."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

_TOOLS_DIR = Path(_TOOLS).resolve()
for _entry in (
    str(_TOOLS_DIR),
    str(_TOOLS_DIR / "coreml"),
    str(_TOOLS_DIR / "ane-export"),
):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from coreml.bundle_cache import bundle_cache_key, cached_bundle  # noqa: E402

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "ane"
    / "fixtures"
    / "h13-explicit-chain-add-mul"
)
# The adapter requires a git repository containing the pinned compiler
# commit; the prepared compiler tree is one when it exists.
_COMPILER_COMMIT = "b12b03f17619c76b77c48a83510847de2245ea1a"
_WITNESS_COMMIT = "aa688df66cbc2110e0df94f0d50fb72c7fa30a18"
_SHAPES = {"audio": (1, 64, 1, 1)}
_WITNESS_CANDIDATES = (
    _TOOLS_DIR.parents[1] / ".work" / "ane-compiler" / "mil-hwx-compiler",
    Path.home() / "src" / "mil-hwx-compiler",
)


def _witness_with_commit() -> Path | None:
    import subprocess

    for candidate in _WITNESS_CANDIDATES:
        if not candidate.is_dir():
            continue
        probe = subprocess.run(
            [
                "git",
                "-C",
                str(candidate),
                "cat-file",
                "-e",
                f"{_COMPILER_COMMIT}^{{commit}}",
            ],
            capture_output=True,
        )
        if probe.returncode == 0:
            return candidate
    return None


_WITNESS = _witness_with_commit() or _WITNESS_CANDIDATES[0]


def _witness_available() -> bool:
    return _witness_with_commit() is not None


def _load(root: Path):
    return cached_bundle(
        _FIXTURE,
        root / "cache",
        graph_source=_FIXTURE / "model.mil",
        compiler_source=_WITNESS,
        name="cached-add-mul",
        source_repo="mil-hwxc",
        source_commit=_WITNESS_COMMIT,
        model="model.mil",
        static_input_shapes=_SHAPES,
    )


class BundleCacheTest(unittest.TestCase):
    def test_key_binds_package_and_compiler_identity(self):
        key, receipt = bundle_cache_key(_FIXTURE, static_input_shapes=_SHAPES)
        self.assertEqual(
            receipt["compiler_executable_source_commit"], _COMPILER_COMMIT
        )
        self.assertEqual(key.compiler.commit, _COMPILER_COMMIT)
        self.assertEqual(key.compiler.target, "H13")
        self.assertEqual(
            key.compiler.package_schema, "mil-hwxc.h13-anec-package.v2"
        )
        self.assertEqual(key.bundle_schema, 4)
        self.assertEqual(key.driver_abi_major, 1)
        # Compatibility axes must change the digest.
        other, _ = bundle_cache_key(
            _FIXTURE,
            static_input_shapes=_SHAPES,
            firmware_identity="T8103-H13-fw-99",
        )
        self.assertNotEqual(key.digest, other.digest)
        reshaped, _ = bundle_cache_key(
            _FIXTURE, static_input_shapes={"audio": (1, 128, 1, 1)}
        )
        self.assertNotEqual(key.digest, reshaped.digest)

    def test_second_load_hits_cache(self):
        if not _witness_available():
            self.skipTest("prepared mil-hwx-compiler witness not present")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, hit = _load(root)
            self.assertFalse(hit)
            self.assertTrue((bundle / "manifest.json").is_file())
            manifest = json.loads((bundle / "manifest.json").read_text())
            self.assertEqual(len(manifest["programs"]), 2)

            again, hit2 = _load(root)
            self.assertTrue(hit2)
            self.assertEqual(again, bundle)

            # The cache entry records the key and re-verifies artifacts
            # on the hit path (compiled_cache contract).
            entry = bundle.parent / "cache.json"
            self.assertTrue(entry.is_file())
            document = json.loads(entry.read_text())
            self.assertEqual(
                document["schema"], "mlx-omarchy.coreml-cache-entry.v1"
            )


if __name__ == "__main__":
    unittest.main()
