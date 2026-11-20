"""Tests for the installed-binary provenance gate (mlx_provenance.py).

Stdlib unittest only, no network, no mlx import: the hash-comparison
core runs against a fake install tree built in a temp directory.
"""

import sys
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import mlx_provenance as prov  # noqa: E402


class ProvenanceSelfTest(unittest.TestCase):
    def test_self_test_covers_the_comparison_core(self):
        # _self_test builds a fake site-packages and proves: matching
        # tree verifies clean, swapped binary refuses, compiled-vs-dist
        # version disagreement refuses, positive control detects a
        # foreign library, and --expect-wheel compares members.
        self.assertIsNone(prov._self_test())


class ProvenanceLineTests(unittest.TestCase):
    def test_line_carries_version_and_hashes(self):
        line = prov.provenance_line({
            "dist": "mlx-omarchy", "dist_version": "1.0",
            "mx_version": "1.0", "verified": "match",
            "files": [{"path": "mlx/lib/libmlx.so",
                       "sha256": "a" * 64}],
        })
        self.assertIn("mlx-omarchy 1.0", line)
        self.assertIn("verified=match", line)
        self.assertIn("libmlx.so=sha256:" + "a" * 16, line)

    def test_control_string_is_the_baked_in_literal(self):
        self.assertEqual(prov.CONTROL_STRING, b"MLX_DISABLE_COMPILE")


class UpstreamCpuProvenanceTests(unittest.TestCase):
    def test_split_wheel_records_must_own_the_loaded_binaries(self):
        with TemporaryDirectory() as tmp:
            core, lib = Path(tmp) / "core.so", Path(tmp) / "libmlx.so"
            core.write_bytes(b"binding")
            lib.write_bytes(b"library")
            mlx = SimpleNamespace(core=SimpleNamespace(__version__="1.0"))
            dists = {
                name: SimpleNamespace(version="1.0", read_text=lambda _: "present",
                                      locate_file=lambda rel: Path(tmp) / rel)
                for name in ("mlx", "mlx-cpu")
            }
            hashes = {
                "mlx": {core.name: prov.sha256_file(core)},
                "mlx-cpu": {lib.name: prov.sha256_file(lib)},
            }
            with (
                patch.dict(sys.modules, {"mlx": mlx, "mlx.core": mlx.core}),
                patch.object(prov, "_loaded_linux_cpu_binaries", return_value=(core, lib)),
                patch.object(prov.importlib.metadata, "distribution", side_effect=dists.__getitem__),
                patch.object(prov, "_record_hashes", side_effect=lambda dist: hashes[
                    "mlx" if dist is dists["mlx"] else "mlx-cpu"]),
            ):
                result = prov.native_provenance(upstream_cpu=True)
                self.assertEqual(result["verified"], "match")
                self.assertEqual([f["match"] for f in result["files"]], [True, True])
                hashes["mlx"][lib.name] = hashes["mlx-cpu"].pop(lib.name)
                self.assertEqual(prov.native_provenance(upstream_cpu=True)["verified"], "unverified")
                hashes["mlx-cpu"][lib.name] = "0" * 64
                self.assertEqual(prov.native_provenance(upstream_cpu=True)["verified"], "mismatch")


if __name__ == "__main__":
    unittest.main()
