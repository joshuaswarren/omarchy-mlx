"""packaging/stage-whole-bundle.sh: the wheel build's bundle gate.

Drives the sourced function in a temp tree with generated pins and
bundles — no network, no build. Cases: refusing without a bundle dir,
refusing wrong bytes, staging exact bytes, skipping an undeclaring pin,
and the removed opt-out staying removed.
"""

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GATE = REPO / "packaging" / "stage-whole-bundle.sh"

MANIFEST_BYTES = b'{"name": "parakeet-encoder-whole", "v": 1}\n'
PROGRAM_BYTES = b"\x00\x01anec-bytes"


class StageWholeBundleTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.work = self.tmp / "work"
        self.share = self.work / "mlx/tools/mlx-omarchy-parakeet/share/mlx-omarchy/parakeet-1"
        self.share.mkdir(parents=True)
        self.bundle = self.tmp / "bundle"
        self.bundle.mkdir()
        (self.bundle / "manifest.json").write_bytes(MANIFEST_BYTES)
        (self.bundle / "program-0.anec").write_bytes(PROGRAM_BYTES)

    def write_pin(self, *, declares: bool) -> Path:
        pin = {"assets": {"bundles": {}}}
        if declares:
            pin["assets"]["bundles"]["parakeet-encoder-whole"] = {
                "manifest.json": hashlib.sha256(MANIFEST_BYTES).hexdigest(),
                "program-0.anec": hashlib.sha256(PROGRAM_BYTES).hexdigest(),
            }
        path = self.share / "parakeet-runtime-pin.json"
        path.write_text(json.dumps(pin))
        return path

    def run_gate(self, *, bundle_dir: str | None = None) -> subprocess.CompletedProcess:
        script = (
            "set -euo pipefail\n"
            f"source {GATE}\n"
            f"stage_whole_bundle {self.work}\n"
        )
        env = {"PATH": os.environ["PATH"],
               "MLX_OMARCHY_WHOLE_BUNDLE_DIR": bundle_dir if bundle_dir is not None else str(self.bundle)}
        return subprocess.run(["bash", "-c", script], env=env,
                              capture_output=True, text=True)

    def staged(self) -> Path:
        return self.share / "bundles/parakeet-encoder-whole"


class GateRefusalTest(StageWholeBundleTest):
    def test_missing_bundle_dir_refuses(self):
        self.write_pin(declares=True)
        result = self.run_gate(bundle_dir="")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MLX_OMARCHY_WHOLE_BUNDLE_DIR is unset", result.stderr)

    def test_wrong_hash_refuses(self):
        self.write_pin(declares=True)
        (self.bundle / "program-0.anec").write_bytes(b"tampered")
        result = self.run_gate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("program-0.anec sha mismatch", result.stderr)
        self.assertFalse(self.staged().exists())

    def test_missing_file_refuses(self):
        self.write_pin(declares=True)
        (self.bundle / "manifest.json").unlink()
        result = self.run_gate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not contain manifest.json", result.stderr)


class GateAcceptTest(StageWholeBundleTest):
    def test_exact_bytes_are_staged(self):
        self.write_pin(declares=True)
        result = self.run_gate()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[bundle] staged", result.stdout)
        self.assertEqual((self.staged() / "manifest.json").read_bytes(), MANIFEST_BYTES)
        self.assertEqual((self.staged() / "program-0.anec").read_bytes(), PROGRAM_BYTES)

    def test_undeclaring_pin_skips_stage(self):
        self.write_pin(declares=False)
        result = self.run_gate()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("does not declare parakeet-encoder-whole", result.stdout)
        self.assertFalse(self.staged().exists())

    def test_no_skip_opt_out_remains(self):
        """The removed MLX_OMARCHY_WHOLE_BUNDLE_SKIP trap must be gone: the
        gate never accepted it, but its documented promise made a worker
        lane record a false environment fact (AbiVerify, 2026-09-30)."""
        text = GATE.read_text()
        self.assertNotIn("WHOLE_BUNDLE_SKIP", text)


if __name__ == "__main__":
    unittest.main()
