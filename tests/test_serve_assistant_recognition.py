# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Focused tests for the assistant recognition voice adapter.

Host-logic tests (strict WAV bounds, status truthfulness, worker
protocol, cancellation) run everywhere; resampling and full-pipeline
tests run only where MLX / the qualified Parakeet runtime is installed
and skip elsewhere with the probe's named reasons.
"""

import math
import os
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_assistant import recognition  # noqa: E402


def wav_bytes(samples: np.ndarray, rate: int = 16_000, *, tag: int = 1,
              bits: int = 16, channels: int = 1, block_align: int | None = None,
              declared_size: int | None = None,
              riff_size: int | None = None,
              pad_odd: bool = True) -> bytes:
    """Build a minimal mono WAV from int16/float32/int32 samples or raw bytes."""
    if isinstance(samples, (bytes, bytearray)):
        payload = bytes(samples)
    elif tag == 1 and bits == 16:
        payload = np.asarray(samples, dtype="<i2").tobytes()
    elif tag == 1 and bits == 32:
        payload = np.asarray(samples, dtype="<i4").tobytes()
    elif tag == 3:
        payload = np.asarray(samples, dtype="<f4").tobytes()
    else:
        payload = bytes(samples)
    width = bits // 8
    align = block_align if block_align is not None else channels * width
    fmt = struct.pack(
        "<HHIIHH", tag, channels, rate,
        rate * align, align, bits,
    )
    data_size = declared_size if declared_size is not None else len(payload)
    body = (
        b"fmt " + struct.pack("<I", len(fmt)) + fmt
        + b"data" + struct.pack("<I", data_size) + payload
    )
    if data_size % 2 and pad_odd:
        body += b"\x00"  # RIFF pads odd chunks to even boundaries
    total = 4 + len(body) + (len(body) % 2)
    return (
        b"RIFF" + struct.pack("<I", riff_size if riff_size is not None
                              else total) + b"WAVE" + body
    )


def sine_int16(freq: int, rate: int, seconds: float, amplitude: float = 0.8):
    t = np.arange(int(rate * seconds), dtype=np.float64) / rate
    return (amplitude * 32767.0 * np.sin(2 * math.pi * freq * t)).astype(np.int16)


class DecodeWavBoundsTest(unittest.TestCase):
    """Strict header rejections happen by name, before sample conversion."""

    def test_rejects_non_wave_payload(self):
        with self.assertRaisesRegex(
            recognition.RecognitionInputError, "RIFF"
        ):
            recognition.decode_wav(b"this is not a wave file at all")

    def test_rejects_riff_size_mismatch(self):
        samples = sine_int16(440, 16_000, 0.05)
        correct = 4 + (8 + 16) + (8 + len(samples) * 2)
        with self.assertRaisesRegex(
            recognition.RecognitionInputError, "RIFF header declares"
        ):
            recognition.decode_wav(wav_bytes(samples, riff_size=correct + 2))
        with self.assertRaisesRegex(
            recognition.RecognitionInputError, "RIFF header declares"
        ):
            recognition.decode_wav(
                wav_bytes(samples, riff_size=0xFFFFFFFF)
            )

    def test_rejects_truncated_declared_chunk(self):
        samples = sine_int16(440, 16_000, 0.1)
        with self.assertRaisesRegex(
            recognition.RecognitionInputError, "truncated"
        ):
            recognition.decode_wav(
                wav_bytes(samples, declared_size=len(samples) * 2 * 100)
            )

    def test_rejects_duplicate_fmt_and_data_chunks(self):
        samples = sine_int16(440, 16_000, 0.05)
        width, rate = 2, 16_000
        fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * width, width, 16)
        payload = samples.astype("<i2").tobytes()
        body = (
            b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"data" + struct.pack("<I", len(payload)) + payload
        )
        blob = (
            b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body
        )
        with self.assertRaisesRegex(
            recognition.RecognitionInputError, "duplicate fmt"
        ):
            recognition.decode_wav(blob)
        body = (
            b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"data" + struct.pack("<I", len(payload)) + payload
            + b"data" + struct.pack("<I", len(payload)) + payload
        )
        blob = (
            b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body
        )
        with self.assertRaisesRegex(
            recognition.RecognitionInputError, "duplicate data"
        ):
            recognition.decode_wav(blob)

    def test_rejects_wrong_block_align(self):
        samples = sine_int16(440, 16_000, 0.05)
        with self.assertRaisesRegex(
            recognition.RecognitionInputError, "block align"
        ):
            recognition.decode_wav(wav_bytes(samples, block_align=4))

    def test_rejects_odd_data_bytes(self):
        odd = b"\x01\x00\x01"  # three bytes: not whole 16-bit samples
        with self.assertRaisesRegex(
            recognition.RecognitionInputError, "whole number"
        ):
            recognition.decode_wav(wav_bytes(odd, declared_size=3, pad_odd=True))

    def test_rejects_stereo(self):
        samples = sine_int16(440, 16_000, 0.1)
        with self.assertRaisesRegex(
            recognition.RecognitionInputError, "mono"
        ):
            recognition.decode_wav(wav_bytes(samples, channels=2))

    def test_rejects_invalid_rates(self):
        samples = sine_int16(440, 16_000, 0.01)
        for rate in (0, 500_000):
            with self.assertRaisesRegex(
                recognition.RecognitionInputError, "invalid sample rate"
            ):
                recognition.decode_wav(wav_bytes(samples, rate=rate))

    def test_rejects_unsupported_formats(self):
        samples = sine_int16(440, 16_000, 0.01)
        with self.assertRaisesRegex(
            recognition.RecognitionInputError, "format tag 7"
        ):
            recognition.decode_wav(wav_bytes(samples, tag=7))
        with self.assertRaisesRegex(
            recognition.RecognitionInputError, "sample width 8"
        ):
            recognition.decode_wav(wav_bytes(b"\x00" * 800, bits=8))

    def test_rejects_over_duration_before_conversion(self):
        samples = sine_int16(440, 8_000, 40.0)
        with self.assertRaisesRegex(
            recognition.RecognitionInputError, "at most 30 s"
        ):
            recognition.decode_wav(wav_bytes(samples, rate=8_000))

    def test_rejects_missing_chunks(self):
        fmt = struct.pack("<HHIIHH", 1, 1, 16_000, 32_000, 2, 16)
        header = (
            b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt)) + b"WAVE"
            + b"fmt " + struct.pack("<I", len(fmt)) + fmt
        )
        with self.assertRaisesRegex(
            recognition.RecognitionInputError, "no data chunk"
        ):
            recognition.decode_wav(header)
        data_only = (
            b"RIFF" + struct.pack("<I", 4 + 8 + 4) + b"WAVE"
            + b"data" + struct.pack("<I", 4) + b"\x00\x00\x00\x00"
        )
        with self.assertRaisesRegex(
            recognition.RecognitionInputError, "no fmt chunk"
        ):
            recognition.decode_wav(data_only)

    def test_rejects_empty_audio(self):
        with self.assertRaisesRegex(
            recognition.RecognitionInputError, "no audio samples"
        ):
            recognition.decode_wav(wav_bytes(np.zeros(0, dtype=np.int16)))

    def test_rejects_nonfinite_float_samples(self):
        floats = np.array([0.1, np.nan, 0.2, -np.inf], dtype=np.float32)
        with self.assertRaisesRegex(
            recognition.RecognitionInputError, "non-finite"
        ):
            recognition.decode_wav(wav_bytes(floats, tag=3, bits=32))

    def test_converts_formats_to_float32(self):
        floats = np.array([0.5, -0.25, 1.0, -1.0], dtype=np.float32)
        decoded, rate = recognition.decode_wav(wav_bytes(floats, tag=3, bits=32))
        self.assertEqual(rate, 16_000)
        np.testing.assert_allclose(
            decoded, np.array([0.5, -0.25, 1.0, -1.0], dtype=np.float32)
        )
        wide = np.array([0x7FFF0000, -0x80000000], dtype=np.int32)
        decoded, rate = recognition.decode_wav(wav_bytes(wide, bits=32))
        np.testing.assert_allclose(
            decoded,
            np.array([2147418112 / 2147483648, -1.0], dtype=np.float32),
            atol=1e-7,
        )

    def test_accepts_s16_mono_scaled(self):
        samples = sine_int16(440, 16_000, 0.05)
        decoded, rate = recognition.decode_wav(wav_bytes(samples))
        self.assertEqual(rate, 16_000)
        np.testing.assert_allclose(
            decoded, samples.astype(np.float32) / 32768.0, rtol=0, atol=0
        )


class RecognitionStatusTest(unittest.TestCase):
    """Contract behavior that must hold with or without the ANE."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.recognition = recognition.Recognition(Path(self._tmp.name))

    def test_status_states_are_receipt_gated(self):
        status = self.recognition.status()
        self.assertIn(
            status["state"], ("ready", "usable", "unqualified", "missing")
        )
        self.assertEqual(status["max_duration_seconds"], 30.0)
        self.assertEqual(status["sample_rate"], 16_000)
        self.assertIn("float32", status["encoding"])
        self.assertEqual(status["ready"], status["state"] == "ready")
        self.assertEqual(status["qualified"], "acceptance" in status)
        memory = status["memory"]
        self.assertGreater(memory["runtime_estimate_bytes"], 0)
        self.assertTrue(memory["estimate"])
        self.assertIn("conservative_margin_bytes", memory["estimate_basis"])
        if status["state"] == "usable":
            self.assertFalse(status["ready"])
        if not status["ready"] and status["state"] != "usable":
            self.assertTrue(status["reasons"], status)

    def test_cancelled_before_any_work(self):
        cancel = threading.Event()
        cancel.set()
        samples = sine_int16(440, 16_000, 0.05)
        with self.assertRaises(recognition.RecognitionCancelled):
            self.recognition.transcribe(wav_bytes(samples), cancel)

    def test_transcribe_is_named_or_real(self):
        status = self.recognition.status()
        silence = wav_bytes(np.zeros(16_000, dtype=np.int16))
        if status["state"] == "ready":
            self.assertEqual(self.recognition.transcribe(silence), "")
        else:
            with self.assertRaises(recognition.RecognitionUnavailable) as caught:
                self.recognition.transcribe(silence)
            self.assertTrue(str(caught.exception), "refusal must be named")

    def test_non_16k_audio_reaches_runtime_gate(self):
        """48 kHz input passes parsing and hits the runtime, not the parser."""
        status = self.recognition.status()
        audio = wav_bytes(sine_int16(440, 48_000, 0.2), 48_000)
        if status["state"] == "ready":
            transcript = self.recognition.transcribe(audio)
            self.assertIsInstance(transcript, str)
        else:
            with self.assertRaises(recognition.RecognitionUnavailable) as caught:
                self.recognition.transcribe(audio)
            self.assertNotIn("sample rate", str(caught.exception))
            self.assertNotIn("RIFF", str(caught.exception))

    def test_close_confirms_worker_exit_and_is_idempotent(self):
        self.recognition.close()
        self.recognition.close()
        self.assertIsNone(self.recognition._worker)


class WorkerProtocolTest(unittest.TestCase):
    """The owned subprocess answers, stops on request, and exits cleanly."""

    def setUp(self):
        module = recognition._load_dictation_module()
        self.module = module
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.handle = recognition._WorkerHandle(
            recognition._locate_tools_root()
        )
        self.addCleanup(self.handle.shutdown)

    def test_worker_answers_unknown_op_with_input_error(self):
        with self.assertRaises(recognition.RecognitionInputError):
            self.handle.request({"op": "frobnicate"}, timeout=30.0)

    def test_transcribe_refusal_is_named_without_runtime(self):
        import platform

        if platform.machine().lower() in ("aarch64", "arm64"):
            self.skipTest("ANE platform present; refusal path not applicable")
        payload = np.zeros(1600, dtype="<f4").tobytes()
        with self.assertRaises(recognition.RecognitionUnavailable) as caught:
            self.handle.request(
                {"op": "transcribe", "sample_rate": 16_000,
                 "deadline_ms": 90_000},
                payload,
                timeout=60.0,
            )
        self.assertTrue(str(caught.exception))

    def test_shutdown_confirms_child_exit(self):
        self.handle.shutdown()
        self.assertIsNotNone(self.handle._process.poll())

    def test_worker_runs_in_its_own_session(self):
        self.assertNotEqual(
            os.getpgid(self.handle._process.pid), os.getpgrp()
        )
        self.assertEqual(
            os.getpgid(self.handle._process.pid), self.handle._pgid
        )

    def test_kill_confirms_empty_process_group(self):
        self.handle.kill()
        self.assertIsNotNone(self.handle._process.poll())
        self.assertFalse(self.handle._group_alive())
        with self.assertRaises(ProcessLookupError):
            os.killpg(self.handle._pgid, 0)

    def test_dead_worker_request_is_named(self):
        self.handle.kill()
        with self.assertRaises(recognition.RecognitionUnavailable):
            self.handle.request({"op": "frobnicate"}, timeout=5.0)

    def test_installed_venv_layout_resolves_tools_root(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp)
            root = venv / "lib" / "python3.11" / "site-packages" / "mlx"
            (root / "coreml").mkdir(parents=True)
            (root / "coreml" / "reference.py").write_text("# probe\n")
            (root / "coreml" / "parakeet_dictation.py").write_text("# dictation\n")
            previous = os.environ.get("MLX_OMARCHY_VENV")
            os.environ["MLX_OMARCHY_VENV"] = str(venv)
            try:
                located = recognition._locate_tools_root()
            finally:
                if previous is None:
                    os.environ.pop("MLX_OMARCHY_VENV", None)
                else:
                    os.environ["MLX_OMARCHY_VENV"] = previous
            self.assertEqual(located, root)
    def test_transfer_package_resolves_tools_without_checkout(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp) / "site-packages"
            (site / "coreml").mkdir(parents=True)
            (site / "coreml" / "reference.py").write_text("# installed tools\n")
            (site / "coreml" / "parakeet_dictation.py").write_text("# dictation\n")
            installed = site / "mlx_omarchy_assistant" / "recognition.py"
            with patch.object(recognition, "__file__", str(installed)), \
                    patch.object(recognition, "_installed_roots", return_value=[]), \
                    patch.object(recognition.shutil, "which", return_value=None), \
                    patch.dict(os.environ, {"MLX_OMARCHY_TOOLS": ""}):
                self.assertEqual(recognition._locate_tools_root(), site)

    def test_mlx_coreml_without_dictation_is_not_the_tools_root(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            mlx_pkg = base / "venv" / "lib" / "python3.14" / "site-packages" / "mlx"
            (mlx_pkg / "coreml").mkdir(parents=True)
            (mlx_pkg / "coreml" / "reference.py").write_text("# mlx coreml\n")
            tools = base / "overlay" / "tools"
            (tools / "coreml").mkdir(parents=True)
            (tools / "coreml" / "parakeet_dictation.py").write_text("# dictation\n")
            fake_file = base / "serve" / "mlx_omarchy_assistant" / "recognition.py"
            fake_file.parent.mkdir(parents=True)
            with patch.object(recognition, "__file__", str(fake_file)), \
                    patch.object(recognition.shutil, "which", return_value=None), \
                    patch.dict(os.environ, {"MLX_OMARCHY_TOOLS": "", "MLX_OMARCHY_VENV": str(base / "venv")}):
                self.assertEqual(recognition._locate_tools_root(), tools)


class HardwareRecognitionTest(unittest.TestCase):
    """Full pipeline on hosts with the qualified runtime; skips elsewhere."""

    @classmethod
    def setUpClass(cls):
        module = recognition._load_dictation_module()
        cls.module = module
        cls.probe = module.probe_runtime(verify_cache=True)
        if not cls.probe["ok"]:
            raise unittest.SkipTest(
                "qualified Parakeet runtime unavailable: "
                + "; ".join(cls.probe["reasons"])
            )

    def test_arbitrary_wav_matches_fixture_oracle(self):
        """Same audio through WAV ingress, not the pinned-file path."""
        cli = self.module.load_cli_module()
        pin = cli._load_pin()
        lock = cli.ReferenceLock.load()
        cache_dir = cli._cache_dir(lock)
        fixture = cache_dir / "audio" / "fixture.flac"
        pcm, rate, _decoder = cli._decode_flac(fixture)
        self.assertEqual(rate, 16_000)

        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        recognition_instance = recognition.Recognition(Path(home.name))
        try:
            transcript = recognition_instance.transcribe(
                wav_bytes(np.asarray(pcm, dtype=np.int16), rate)
            )
            self.assertEqual(transcript, pin["e2e"]["transcript"])
            parity = self.module.DictationTranscriber()
            try:
                waveform = np.asarray(pcm, dtype=np.float32) / 32768.0
                self.assertEqual(
                    parity.transcribe_waveform(waveform, rate),
                    pin["e2e"]["transcript"],
                )
            finally:
                parity.close()
        finally:
            recognition_instance.close()

    def test_acceptance_receipt_moves_status_to_ready(self):
        cli = self.module.load_cli_module()
        pin = cli._load_pin()
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        home_path = Path(home.name)
        self.module.write_acceptance_receipt(
            home_path,
            transcript=pin["e2e"]["transcript"],
            emissions=pin["e2e"]["emissions"],
            host="hardware-acceptance-run",
        )
        status = recognition.Recognition(home_path).status()
        self.assertTrue(status["qualified"])
        if self.probe["facts"].get("cache") in ("present", "verified"):
            self.assertIn(status["state"], ("usable", "ready"))
            self.assertEqual(status["acceptance"]["emissions"],
                             pin["e2e"]["emissions"])


if __name__ == "__main__":
    unittest.main()
