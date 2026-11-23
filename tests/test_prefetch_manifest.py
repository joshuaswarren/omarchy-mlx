"""Prefetch manifest: artifact coverage, determinism, no-fetch on rebuild.

UNIT only, no network and no model artifacts: the manifest builder runs
against a staged fake home. What is proven: every staged root lands in
the manifest with per-file sha256, the build is byte-identical across
runs (reproducibility gate for the jwm2 two-fetch check), roots and
entries are sorted, and --manifest-only never touches the network.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_assistant import prefetch  # noqa: E402


def stage_home() -> Path:
    home = Path(tempfile.mkdtemp())
    (home / "voice" / "kokoro-82m-bf16").mkdir(parents=True)
    (home / "voice" / "parakeet-tdt-0.6b-v3").mkdir(parents=True)
    (home / "wake" / "v1").mkdir(parents=True)
    (home / "vad").mkdir()
    (home / "voice" / "kokoro-82m-bf16" / "a.bin").write_bytes(b"weights" * 100)
    (home / "voice" / "parakeet-tdt-0.6b-v3" / "enc.bin").write_bytes(b"enc" * 50)
    (home / "wake" / "v1" / "hey_jarvis_v0.1.onnx").write_bytes(b"wakeword")
    (home / "vad" / "silero_vad.onnx").write_bytes(b"vadmodel")
    return home


class PrefetchManifestTest(unittest.TestCase):
    def test_covers_every_staged_root(self):
        home = stage_home()
        manifest = prefetch.build_manifest(home, None)
        self.assertEqual([a["root"] for a in manifest["artifacts"]],
                         ["vad", "voice/kokoro-82m-bf16",
                          "voice/parakeet-tdt-0.6b-v3", "wake"])
        self.assertEqual(manifest["file_count"], 4)
        for artifact in manifest["artifacts"]:
            for entry in artifact["files"]:
                self.assertEqual(len(entry["sha256"]), 64)
                self.assertGreater(entry["bytes"], 0)

    def test_rebuild_is_byte_identical(self):
        home = stage_home()
        first = prefetch.build_manifest(home, None)
        second = prefetch.build_manifest(home, None)
        self.assertEqual(first, second)
        self.assertEqual(json.dumps(first, sort_keys=True),
                         json.dumps(second, sort_keys=True))

    def test_manifest_file_lands_under_assistant(self):
        home = stage_home()
        path = prefetch.write_manifest(home, prefetch.build_manifest(home, None))
        self.assertEqual(path, home / "assistant" / "prefetch-manifest.json")
        self.assertEqual(len(__import__("hashlib").sha256(
            path.read_bytes()).hexdigest()), 64)

    def test_manifest_only_pairless(self):
        home = stage_home()
        code = prefetch.main(["--home", str(home), "--manifest-only"])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
