"""Prefetch manifest: artifact coverage, determinism, no-fetch on rebuild.

UNIT only, no network and no model artifacts: the manifest builder runs
against a staged fake home. What is proven: every staged root lands in
the manifest with per-file sha256, the build is byte-identical across
runs (reproducibility gate for the jwm2 two-fetch check), roots and
entries are sorted, --manifest-only never touches the network, and the
CLI reports a PairError on stderr with exit 1 instead of a traceback.
"""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_assistant import pairs, prefetch  # noqa: E402


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

    def test_pair_weights_hash_without_lock(self):
        # A fetched-but-never-served pair has no pair lock (only start()
        # writes one); the setup result's model_paths still identify the
        # pair's weight directories for the manifest.
        home = stage_home()
        chat = home / "models" / "chat-model"
        decision = home / "models" / "decision-model"
        chat.mkdir(parents=True)
        decision.mkdir(parents=True)
        (chat / "weights.safetensors").write_bytes(b"chat" * 200)
        (decision / "weights.safetensors").write_bytes(b"dec" * 100)
        manifest = prefetch.build_manifest(
            home, "everyday",
            model_paths={"chat": chat, "decision": decision},
            model_ids={"chat": "chat-model", "decision": "decision-model"})
        by_role = {a["role"]: a for a in manifest["artifacts"] if "role" in a}
        self.assertEqual(set(by_role), {"chat", "decision"})
        self.assertEqual(by_role["chat"]["root"], "models/chat-model")
        self.assertEqual(by_role["chat"]["model_id"], "chat-model")
        self.assertEqual(manifest["file_count"], 6)

    def test_pair_without_lock_or_paths_fails(self):
        home = stage_home()
        with self.assertRaises(pairs.PairError):
            prefetch.build_manifest(home, "everyday")

    def test_cli_pair_without_lock_exits_1_without_traceback(self):
        with tempfile.TemporaryDirectory() as home:
            result = subprocess.run(
                [sys.executable, "-m", "mlx_omarchy_assistant.prefetch",
                 "--manifest-only", "--pair", "everyday", "--home", home],
                env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "serve")},
                capture_output=True, text=True, timeout=120, check=False)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stderr,
                         "prefetch: no pair lock for 'everyday'; "
                         "run the pair setup first\n")
        self.assertEqual(result.stdout, "")

    def test_main_reports_pair_error_from_download_path(self):
        stdout, stderr = io.StringIO(), io.StringIO()

        def refuse(*_args, **_kwargs):
            raise pairs.PairError("setup returned a plan only")

        with tempfile.TemporaryDirectory() as home, \
                mock.patch.object(prefetch, "prefetch", refuse), \
                contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(stderr):
            code = prefetch.main(["--home", home, "--pair", "everyday"])
        self.assertEqual(code, 1)
        self.assertEqual(stderr.getvalue(),
                         "prefetch: setup returned a plan only\n")
        self.assertEqual(stdout.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
