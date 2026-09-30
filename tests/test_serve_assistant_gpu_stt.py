# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""GPU speech-recognition backend: download gate, qualification, worker protocol.

Host-logic tests run everywhere. The resampler test needs mlx and skips
without it.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_assistant import gpu_stt, recognition  # noqa: E402


def _small_pack(tmp: Path):
    """A two-file stand-in for the pinned model, served by a fake fetch."""
    blobs = {"config.json": b'{"model_type": "parakeet"}',
             "model.safetensors": b"weights" * 100}
    files = [{"name": name, "bytes": len(data),
              "sha256": hashlib.sha256(data).hexdigest()}
             for name, data in blobs.items()]

    def fetch(url, dest, corrupt=False):
        name = url.rsplit("/", 1)[-1]
        Path(dest).write_bytes(b"x" + blobs[name] if corrupt else blobs[name])

    return files, fetch


class DownloadGateTest(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.files, self.fetch = _small_pack(self.home)
        self.pin = patch.dict(gpu_stt.GPU_STT_MODEL, {"files": self.files})
        self.pin.start()
        self.addCleanup(self.pin.stop)
        gpu_stt._verified.cache_clear()

    def test_refuses_without_approval_and_names_size_licence_revision(self):
        result = gpu_stt.prepare(self.home, approve_download=False, fetch=self.fetch)
        self.assertFalse(result["verified"])
        for fact in (str(gpu_stt.GPU_STT_MODEL["weights_bytes"]),
                     gpu_stt.GPU_STT_MODEL["license"],
                     gpu_stt.GPU_STT_MODEL["revision"]):
            self.assertIn(fact, result["reason"])
        self.assertFalse(gpu_stt.model_dir(self.home).exists())

    def test_approved_download_verifies_and_probe_sees_it(self):
        result = gpu_stt.prepare(self.home, approve_download=True, fetch=self.fetch)
        self.assertTrue(result["verified"])
        self.assertTrue(gpu_stt.model_verified(self.home))
        self.assertIn(gpu_stt.probe_runtime(self.home)["facts"]["cache"], ("present",))

    def test_sha_mismatch_is_refused_and_leaves_no_file(self):
        corrupt = lambda url, dest: self.fetch(url, dest, corrupt=True)  # noqa: E731
        with self.assertRaisesRegex(recognition.RecognitionUnavailable, "sha256 mismatch"):
            gpu_stt.prepare(self.home, approve_download=True, fetch=corrupt)
        self.assertFalse(gpu_stt.model_verified(self.home))
        self.assertEqual(list(gpu_stt.model_dir(self.home).glob("*.part")), [])

    def test_replaced_file_after_download_is_not_verified(self):
        gpu_stt.prepare(self.home, approve_download=True, fetch=self.fetch)
        weights = gpu_stt.model_dir(self.home) / "model.safetensors"
        data = bytearray(weights.read_bytes())
        data[0] ^= 1
        swapped = weights.with_name("swap")
        swapped.write_bytes(bytes(data))
        os.replace(swapped, weights)
        self.assertFalse(gpu_stt.model_verified(self.home))


PASSING = dict(
    wer_by_subset={"test-clean": 0.031, "test-other": 0.034, "accented": 0.048,
                   "mixed_0dB": 0.227},
    empty_rate_by_subset={"silence": 1.0, "noise": 1.0},
    latency_ms={"p50": 457.1, "p95": 468.1},
    cpu_tensor_events=0, receipt_source="eval-results.json", host="test")


class QualificationReceiptTest(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.identity = {"mlx_backend": "backend-a", "mlx_backend_detail": None,
                         "model_sha256": "model-a"}
        runtime = patch.object(gpu_stt, "_runtime_identity",
                               side_effect=lambda home: dict(self.identity))
        runtime.start()
        self.addCleanup(runtime.stop)

    def test_receipt_reads_back_only_for_the_runtime_that_wrote_it(self):
        gpu_stt.write_acceptance_receipt(self.home, **PASSING)
        self.assertEqual(gpu_stt.read_acceptance_receipt(self.home)["mlx_backend"], "backend-a")
        self.identity["mlx_backend"] = "backend-b"
        self.assertIsNone(gpu_stt.read_acceptance_receipt(self.home))
        self.identity.update(mlx_backend="backend-a", model_sha256=None)
        self.assertIsNone(gpu_stt.read_acceptance_receipt(self.home))

    def test_each_frozen_threshold_blocks_the_receipt(self):
        failing = [
            ("wer_by_subset", {**PASSING["wer_by_subset"], "test-clean": 0.061}),
            ("wer_by_subset", {**PASSING["wer_by_subset"], "mixed_0dB": 0.31}),
            ("empty_rate_by_subset", {"silence": 0.94, "noise": 1.0}),
            ("latency_ms", {"p50": 1.0, "p95": 2000.1}),
            ("cpu_tensor_events", 1),
        ]
        for key, value in failing:
            with self.subTest(key=key, value=value):
                with self.assertRaisesRegex(ValueError, "acceptance refused"):
                    gpu_stt.write_acceptance_receipt(self.home, **{**PASSING, key: value})
        self.assertIsNone(gpu_stt.read_acceptance_receipt(self.home))

    def test_unverified_runtime_cannot_write(self):
        self.identity["mlx_backend"] = None
        with self.assertRaisesRegex(ValueError, "runtime identity unverified"):
            gpu_stt.write_acceptance_receipt(self.home, **PASSING)


class BackendSelectionTest(unittest.TestCase):
    def test_failed_ane_probe_falls_back_to_gpu_and_pin_ane_refuses(self):
        ane = type("Ane", (), {"probe_runtime": staticmethod(
            lambda: {"ok": False, "reasons": ["platform: no /dev/accel/accel0"], "facts": {}})})
        with patch.object(recognition, "_locate_tools_root", return_value=Path("/nowhere")), \
                patch.dict(sys.modules, {"coreml": type(sys)("coreml")}):
            sys.modules["coreml"].parakeet_dictation = ane
            with patch.dict(os.environ, {"MLX_OMARCHY_RECOGNITION_BACKEND": "auto"}):
                self.assertIs(recognition._load_dictation_module(), gpu_stt)
            with patch.dict(os.environ, {"MLX_OMARCHY_RECOGNITION_BACKEND": "ane"}):
                with self.assertRaisesRegex(recognition.RecognitionUnavailable, "accel0"):
                    recognition._load_dictation_module()


FAKE_WORKER = textwrap.dedent(r"""
    import json, struct, sys, time
    out, inp = sys.stdout.buffer, sys.stdin.buffer
    def send(h):
        b = json.dumps(dict(h, payload_bytes=0)).encode()
        out.write(struct.pack("<Q", len(b)) + b); out.flush()
    def read(n):
        b = b""
        while len(b) < n:
            c = inp.read(n - len(b))
            if not c: sys.exit(0)
            b += c
        return b
    send({"ok": True, "event": "ready"})
    while True:
        h = json.loads(read(struct.unpack("<Q", read(8))[0]))
        payload = read(h["payload_bytes"]) if h["payload_bytes"] else b""
        time.sleep(h.get("sleep", 0))
        send({"ok": True, "id": h["id"], "transcript": f"clip {len(payload) // 4}"})
""")


class WorkerFramingTest(unittest.TestCase):
    """The real handle against a stand-in worker process."""

    def setUp(self):
        script = Path(tempfile.mkdtemp()) / "fake_worker.py"
        script.write_text(FAKE_WORKER)
        real_popen = subprocess.Popen

        def popen(args, **kwargs):
            return real_popen([sys.executable, str(script)], **kwargs)

        with patch.object(gpu_stt.subprocess, "Popen", side_effect=popen):
            self.handle = gpu_stt._WorkerHandle(Path("/unused"))
        self.addCleanup(self.handle.kill)

    def test_first_request_gets_its_own_answer_not_the_boot_frame(self):
        for samples in (16_000, 8_000, 4_000):
            response = self.handle.request({"op": "transcribe", "sample_rate": 16_000},
                                           b"\0" * (4 * samples), timeout=10)
            self.assertEqual(response["transcript"], f"clip {samples}")

    def test_an_unused_cancel_event_does_not_delay_the_answer(self):
        self.handle.request({"op": "transcribe"}, b"", timeout=10)
        started = time.monotonic()
        response = self.handle.request({"op": "transcribe", "sample_rate": 16_000},
                                       b"\0" * 64, timeout=10, cancel=threading.Event())
        self.assertEqual(response["transcript"], "clip 16")
        self.assertLess(time.monotonic() - started, 1.0)

    def test_cancel_mid_request_raises_and_confirms_the_worker_exited(self):
        cancel = threading.Event()
        threading.Timer(0.3, cancel.set).start()
        started = time.monotonic()
        with self.assertRaises(recognition.RecognitionCancelled):
            self.handle.request({"op": "transcribe", "sample_rate": 16_000, "sleep": 60},
                                b"\0" * 64, timeout=90, cancel=cancel)
        self.assertLess(time.monotonic() - started, 3.0)
        self.assertFalse(self.handle.alive)
        self.assertFalse(self.handle._group_alive())


class ResampleTest(unittest.TestCase):
    def test_48k_sine_resamples_to_16k_on_device(self):
        try:
            import mlx.core as mx
        except ImportError:
            self.skipTest("mlx is not importable on this host")
        import numpy as np
        t = np.arange(48_000) / 48_000
        tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        out = np.asarray(gpu_stt.resample_to_16k(mx.array(tone), 48_000))
        self.assertEqual(out.shape, (16_000,))
        expected = np.sin(2 * np.pi * 440 * np.arange(16_000) / 16_000)
        self.assertLess(np.abs(out[200:-200] - expected[200:-200]).max(), 1e-2)


class BarePackageTest(unittest.TestCase):
    def test_submodule_imports_without_running_the_package_init(self):
        from mlx_omarchy_assistant import gpu_stt_worker
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp, "stt_families_probe")
            (pkg / "wanted").mkdir(parents=True)
            (pkg / "__init__.py").write_text("raise RuntimeError('package __init__ ran')\n")
            (pkg / "wanted" / "__init__.py").write_text("VALUE = 7\n")
            sys.path.insert(0, tmp)
            try:
                gpu_stt_worker._register_bare_package("stt_families_probe")
                import stt_families_probe.wanted as wanted
                self.assertEqual(wanted.VALUE, 7)
            finally:
                sys.path.remove(tmp)
                for name in ("stt_families_probe.wanted", "stt_families_probe"):
                    sys.modules.pop(name, None)


if __name__ == "__main__":
    unittest.main()
