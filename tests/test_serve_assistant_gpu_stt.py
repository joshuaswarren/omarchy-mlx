# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Focused tests for the GPU speech-recognition backend.

Host-logic tests (manifest, sha256 verification, probe, receipt I/O,
resampler) run everywhere; full-pipeline tests run only where the
pinned GPU STT model is present, the live mlx binary is importable,
and the GPU is available — they skip elsewhere with the probe's
named reasons.
"""

import json
import os
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_assistant import gpu_stt  # noqa: E402


def sine_int16(freq: int, rate: int, seconds: float, amplitude: float = 0.5):
    import math
    import numpy as np
    t = np.arange(int(rate * seconds), dtype=np.float64) / rate
    return (amplitude * 32767.0 * np.sin(2 * math.pi * freq * t)).astype(np.int16)


def wav_bytes(samples, rate: int = 16_000, *, tag: int = 1,
              bits: int = 16, channels: int = 1) -> bytes:
    import numpy as np
    payload = np.asarray(samples, dtype="<i2").tobytes()
    width = bits // 8
    fmt = struct.pack("<HHIIHH", tag, channels, rate, rate * width,
                      channels * width, bits)
    body = (b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"data" + struct.pack("<I", len(payload)) + payload)
    return (
        b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body
    )


class GpuSttManifestTest(unittest.TestCase):
    """The pinned manifest is auditable and stable."""

    def test_manifest_keys(self):
        m = gpu_stt.GPU_STT_MODEL
        self.assertEqual(m["id"], "parakeet-tdt-0.6b-v3")
        self.assertEqual(m["repo"], "mlx-community/parakeet-tdt-0.6b-v3")
        self.assertTrue(m["license"])
        self.assertEqual(len(m["weights_sha256"]), 64)
        self.assertEqual(len(m["config_sha256"]), 64)

    def test_backend_kind_is_gpu(self):
        self.assertEqual(gpu_stt.BACKEND_KIND, "gpu")

    def test_resample_passthrough(self):
        import numpy as np
        x = np.linspace(-1, 1, 1600, dtype=np.float32)
        self.assertIs(gpu_stt.resample_to_16k(x, 16000), x)

    def test_resample_to_16k_changes_length(self):
        import numpy as np
        x = np.zeros(48000, dtype=np.float32)
        y = gpu_stt.resample_to_16k(x, 48000)
        self.assertEqual(y.shape[0], 16000)
        self.assertEqual(y.dtype, np.float32)

    def test_resample_rejects_invalid_rate(self):
        import numpy as np
        with self.assertRaises(gpu_stt.GpuSttInputError):
            gpu_stt.resample_to_16k(np.zeros(16, dtype=np.float32), 0)


class GpuSttReceiptTest(unittest.TestCase):
    """Hardware acceptance receipt I/O is strict on the facts it requires."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.home = Path(self._tmp.name)

    def test_missing_receipt_is_none(self):
        self.assertIsNone(gpu_stt.read_acceptance_receipt(self.home))

    def test_write_acceptance_receipt_round_trips(self):
        receipt = gpu_stt.write_acceptance_receipt(
            self.home,
            transcript="hello",
            emissions={"tokens": 1},
            cpu_tensor_events=0,
            latency_ms={"p50": 80.0, "p95": 130.0},
            mlx_binary_sha="0" * 64,
            model_sha=gpu_stt.GPU_STT_MODEL["weights_sha256"],
            host="self-test",
        )
        self.assertTrue(receipt["listener_verified"])
        self.assertEqual(receipt["cpu_tensor_events"], 0)
        self.assertEqual(receipt["transcript"], "hello")
        loaded = gpu_stt.read_acceptance_receipt(self.home)
        self.assertEqual(loaded["transcript"], "hello")

    def test_write_refuses_nonzero_cpu_events(self):
        with self.assertRaises(ValueError):
            gpu_stt.write_acceptance_receipt(
                self.home,
                transcript="x",
                emissions={},
                cpu_tensor_events=1,
                latency_ms={"p50": 1.0, "p95": 2.0},
                mlx_binary_sha="0" * 64,
                model_sha="0" * 64,
                host="self-test",
            )

    def test_write_requires_latency_p50_and_p95(self):
        with self.assertRaises(ValueError):
            gpu_stt.write_acceptance_receipt(
                self.home,
                transcript="x",
                emissions={},
                cpu_tensor_events=0,
                latency_ms={"p50": 1.0},
                mlx_binary_sha="0" * 64,
                model_sha="0" * 64,
                host="self-test",
            )


class GpuSttProbeTest(unittest.TestCase):
    """probe_runtime is honest and stable when the runtime is partial."""

    def test_probe_facts_present(self):
        probe = gpu_stt.probe_runtime(verify_cache=False)
        facts = probe["facts"]
        self.assertEqual(facts["kind"], "gpu_stt")
        self.assertEqual(facts["model"], gpu_stt.GPU_STT_MODEL["id"])
        self.assertIn("dependencies", facts)
        self.assertIn("accelerator", facts)


class GpuSttRecognitionTest(unittest.TestCase):
    """End-to-end only when the pinned model is installed and the GPU is live."""

    @classmethod
    def setUpClass(cls):
        cls.probe = gpu_stt.probe_runtime(verify_cache=True)
        if not cls.probe["ok"]:
            raise unittest.SkipTest(
                "qualified GPU STT runtime unavailable: "
                + "; ".join(cls.probe["reasons"])
            )

    def test_silence_transcribes_to_empty_string(self):
        from mlx_omarchy_assistant import recognition
        from mlx_omarchy_assistant.recognition import decode_wav
        import numpy as np
        wav = wav_bytes(np.zeros(16000, dtype=np.int16), rate=16_000)
        rec = recognition.Recognition(Path(tempfile.mkdtemp()))
        try:
            samples, rate = decode_wav(wav)
            assert rate == 16_000
            text = rec.transcribe(wav)
            self.assertIsInstance(text, str)
            self.assertEqual(text.strip(), "")
        finally:
            rec.close()

    def test_resampler_runs_for_48k_input(self):
        from mlx_omarchy_assistant import recognition
        import numpy as np
        wav = wav_bytes(sine_int16(440, 48_000, 0.5), rate=48_000)
        rec = recognition.Recognition(Path(tempfile.mkdtemp()))
        try:
            self.recognition = rec
            text = rec.transcribe(wav)
            self.assertIsInstance(text, str)
        finally:
            rec.close()


if __name__ == "__main__":
    unittest.main()