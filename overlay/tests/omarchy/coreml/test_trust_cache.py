# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host tests for the opt-in stamp-aware cache verify on `transcribe`.

`download` / `verify` keep the strict re-hash contract: presence, size
or prior stamps are never trusted. `transcribe` may opt in to a
sidecar-stamp short-circuit via ``MLX_OMARCHY_PK_TRUST_CACHE=1``; the
stamp must agree with the lock AND the on-disk file size, and any
disagreement falls through to a real hash so a corrupted cache is
refused with the same message `verify_cache` would have produced.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

_TOOLS_DIR = Path(_TOOLS).resolve()
for _entry in (str(_TOOLS_DIR), str(_TOOLS_DIR / "coreml")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from coreml.reference import (  # noqa: E402
    CACHE_HASHES_NAME,
    CACHE_STAMP_NAME,
    CACHE_VERIFIED_NAME,
    record_stamps,
    record_verified_hashes,
    trust_cache,
    verify_cache,
    verify_cache_with_stamp,
)


def _lock_with_files(files):
    """A minimal ``ReferenceLock``-shaped namespace for the stamp tests.

    Only ``lock.files`` is read by ``trust_cache`` / ``verify_cache`` /
    ``record_stamps``. ``LockedFile`` is the real shape, but the suite
    runs without a real lock file in the source tree, so we mock just
    enough surface to drive the path.
    """

    class _StubLock:
        def __init__(self, files):
            self.files = files

    return _StubLock(files)


def _locked_file(path, size, sha256):
    from coreml.reference import LockedFile

    return LockedFile(path=path, size=size, sha256=sha256)


def _populate_cache(cache_dir, contents):
    """Write each ``(path, bytes)`` pair under ``cache_dir``."""
    for rel, data in contents.items():
        p = cache_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


class TrustCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="trust-cache-"))
        self.cache = self.tmp / "cache"
        self.cache.mkdir()
        # Two tiny files at known sizes + sha256.
        self.a = (b"hello world", "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9")
        self.b = (b"second file", "54811cbc6c86311729b0a33e26c89087881b36b9ca3217d15cb5196e35f9a7e3")
        contents = {
            "alpha.bin": self.a[0],
            "beta.bin": self.b[0],
        }
        _populate_cache(self.cache, contents)
        self.lock = _lock_with_files([
            _locked_file("alpha.bin", len(self.a[0]), self.a[1]),
            _locked_file("beta.bin", len(self.b[0]), self.b[1]),
        ])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_trust_cache_pass_on_valid_stamps(self):
        record_stamps(self.cache, self.lock)
        ok, mismatches = trust_cache(self.cache, self.lock)
        self.assertTrue(ok, mismatches)
        self.assertEqual(mismatches, [])

    def test_trust_cache_fails_without_stamps(self):
        ok, mismatches = trust_cache(self.cache, self.lock)
        self.assertFalse(ok)
        self.assertTrue(any("missing-stamp" in m for m in mismatches), mismatches)

    def test_trust_cache_fails_on_size_drift(self):
        record_stamps(self.cache, self.lock)
        # Replace alpha with a same-path file of a different size. The
        # stamp recorded the original size, so trust_cache must refuse
        # before the hash is even read.
        (self.cache / "alpha.bin").write_bytes(b"a" * (len(self.a[0]) + 1))
        ok, mismatches = trust_cache(self.cache, self.lock)
        self.assertFalse(ok)
        self.assertTrue(any("size-mismatch" in m for m in mismatches), mismatches)

    def test_trust_cache_fails_on_sha_stamp_vs_lock(self):
        record_stamps(self.cache, self.lock)
        # Forge the verified-hashes sidecar for alpha to a different
        # sha256 — the file mtime is untouched (within the same verify
        # cycle), so only the recorded-vs-lock mismatch catches it.
        verified = self.cache / CACHE_VERIFIED_NAME
        text = verified.read_text(encoding="utf-8")
        forged = text.replace(self.a[1], "0" * 64)
        # Keep the sidecar mtime newer than the cache files; only the
        # recorded hash disagrees with the lock.
        verified.write_text(forged, encoding="utf-8")
        verified.touch()
        ok, mismatches = trust_cache(self.cache, self.lock)
        self.assertFalse(ok)
        self.assertTrue(any("sha-stamp-mismatch" in m for m in mismatches),
                        mismatches)

    def test_trust_cache_fails_on_missing_file(self):
        record_stamps(self.cache, self.lock)
        (self.cache / "beta.bin").unlink()
        ok, mismatches = trust_cache(self.cache, self.lock)
        self.assertFalse(ok)
        self.assertTrue(any("missing: beta.bin" == m for m in mismatches),
                        mismatches)

    def test_trust_cache_detects_post_verify_write(self):
        # record_stamps writes the sidecar with mtime >= cache files.
        # Replacing alpha (any size) bumps alpha's mtime past the
        # sidecar; trust must refuse without re-hashing the bytes.
        record_stamps(self.cache, self.lock)
        (self.cache / "alpha.bin").write_bytes(b"12345678901")
        ok, mismatches = trust_cache(self.cache, self.lock)
        self.assertFalse(ok)
        self.assertTrue(any("newer-than-stamp" in m for m in mismatches),
                        mismatches)

    def test_verify_cache_with_stamp_off_is_strict(self):
        # Without trust_stamp the wrapper is a straight verify_cache;
        # the stamps are ignored.
        ok, mismatches = verify_cache_with_stamp(
            self.cache, self.lock, trust_stamp=False,
        )
        self.assertTrue(ok, mismatches)

    def test_verify_cache_with_stamp_on_passes_without_hash(self):
        record_stamps(self.cache, self.lock)
        ok, mismatches = verify_cache_with_stamp(
            self.cache, self.lock, trust_stamp=True,
        )
        self.assertTrue(ok, mismatches)

    def test_verify_cache_with_stamp_on_falls_through_on_drift(self):
        record_stamps(self.cache, self.lock)
        # Replace alpha with a same-name, same-size file of different
        # bytes — the verified-hashes sidecar still records the old
        # SHA, so the fall-through hits verify_cache which re-hashes and
        # refuses.
        (self.cache / "alpha.bin").write_bytes(b"12345678901")
        ok, mismatches = verify_cache_with_stamp(
            self.cache, self.lock, trust_stamp=True,
        )
        self.assertFalse(ok)
        self.assertTrue(any("sha256-mismatch" in m for m in mismatches),
                        mismatches)

    def test_verify_cache_with_stamp_refreshes_stamps_after_fallthrough(self):
        # Force a fall-through: drop all stamps, ask for trust_stamp.
        # The wrapper re-hashes via verify_cache, succeeds, and leaves
        # fresh verified-hashes / manifest-stamp / sha256sums behind.
        (self.cache / CACHE_STAMP_NAME).unlink(missing_ok=True)
        (self.cache / CACHE_HASHES_NAME).unlink(missing_ok=True)
        (self.cache / CACHE_VERIFIED_NAME).unlink(missing_ok=True)
        ok, mismatches = verify_cache_with_stamp(
            self.cache, self.lock, trust_stamp=True,
        )
        self.assertTrue(ok, mismatches)
        self.assertTrue((self.cache / CACHE_STAMP_NAME).is_file())
        self.assertTrue((self.cache / CACHE_HASHES_NAME).is_file())
        self.assertTrue((self.cache / CACHE_VERIFIED_NAME).is_file())

    def test_record_verified_hashes_writes_actual_sha(self):
        # record_verified_hashes captures the on-disk bytes' SHA, not
        # the lock's pinned value. A tamper between verifies becomes a
        # recorded-vs-lock disagreement that trust_cache refuses.
        record_verified_hashes(self.cache, self.lock)
        verified = self.cache / CACHE_VERIFIED_NAME
        self.assertTrue(verified.is_file())
        text = verified.read_text(encoding="utf-8")
        self.assertIn(f"{self.a[1]}  alpha.bin", text)
        self.assertIn(f"{self.b[1]}  beta.bin", text)


if __name__ == "__main__":
    unittest.main()