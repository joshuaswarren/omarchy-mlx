# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Focused tests for the arbitrary-audio Parakeet dictation module.

Chunk planning, waveform validation, cancellation-before-start, probe
facts, acceptance receipts, and the resampler's host-visible contract
are covered without the ANE. The full-pipeline tests run only where
the qualified ANE/Vulkan runtime is installed and skip elsewhere with
named reasons; the fixture oracle transcript must match through the
dictation path exactly as it does through the CLI.
"""

import math
import sys
import threading
import unittest
from pathlib import Path

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

import numpy as np

from coreml import parakeet_dictation as pd


class PlanChunksTest(unittest.TestCase):
    def test_splits_at_chunk_samples(self):
        self.assertEqual(pd.plan_chunks(1), [1])
        self.assertEqual(pd.plan_chunks(480_000), [480_000])
        self.assertEqual(pd.plan_chunks(480_001), [480_000, 1])
        self.assertEqual(pd.plan_chunks(960_000), [480_000, 480_000])
        self.assertEqual(
            pd.plan_chunks(960_001), [480_000, 480_000, 1]
        )

    def test_rejects_nonpositive_counts(self):
        for count in (0, -5):
            with self.assertRaisesRegex(ValueError, "sample_count"):
                pd.plan_chunks(count)


class TranscribeInputValidationTest(unittest.TestCase):
    """Refusals happen before the runtime is ever touched."""

    def setUp(self):
        self.transcriber = pd.DictationTranscriber()

    def test_rejects_non_float32_waveform(self):
        with self.assertRaisesRegex(ValueError, "numpy float32"):
            self.transcriber.transcribe_waveform(
                np.zeros(100, dtype=np.int16), 16_000
            )

    def test_rejects_two_dimensional_or_empty_waveform(self):
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            self.transcriber.transcribe_waveform(
                np.zeros((10, 10), dtype=np.float32), 16_000
            )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.transcriber.transcribe_waveform(
                np.zeros(0, dtype=np.float32), 16_000
            )

    def test_rejects_wrong_sample_rate(self):
        with self.assertRaisesRegex(ValueError, "16000"):
            self.transcriber.transcribe_waveform(
                np.zeros(100, dtype=np.float32), 48_000
            )


class CancelBeforeStartTest(unittest.TestCase):
    def test_cancelled_event_short_circuits(self):
        cancel = threading.Event()
        cancel.set()
        transcriber = pd.DictationTranscriber()
        with self.assertRaisesRegex(
            pd.DictationCancelled, "before transcription"
        ):
            transcriber.transcribe_waveform(
                np.zeros(100, dtype=np.float32), 16_000, cancel
            )


class ProbeTest(unittest.TestCase):
    def test_probe_reports_named_facts(self):
        probe = pd.probe_runtime()
        self.assertIsInstance(probe["ok"], bool)
        facts = probe["facts"]
        self.assertIn("cli", facts)
        self.assertEqual(facts["sample_rate"], 16_000)
        self.assertEqual(facts["chunk_samples"], 480_000)
        if not probe["ok"]:
            self.assertTrue(probe["reasons"], probe)
            for reason in probe["reasons"]:
                self.assertIsInstance(reason, str)

    def test_probe_names_platform_blocker_without_ane(self):
        import platform

        machine = platform.machine().lower()
        if machine in ("aarch64", "arm64"):
            self.skipTest(f"ANE platform present ({machine})")
        probe = pd.probe_runtime()
        self.assertFalse(probe["ok"])
        self.assertTrue(
            any(reason.startswith("platform:") for reason in probe["reasons"]),
            probe["reasons"],
        )


class CliLocatorTest(unittest.TestCase):
    def test_loads_pinned_product_module(self):
        cli = pd.load_cli_module()
        self.assertTrue(callable(cli._load_pin))
        self.assertTrue(callable(cli._verify_assets))

    def test_loads_extensionless_installed_script(self):
        """The installed CLI has no .py suffix; the loader must cope."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "mlx-omarchy-parakeet"
            script.write_text(
                "VALUE = 41 + 1\n"
                "def _load_pin():\n"
                "    return 'pin'\n",
                encoding="utf-8",
            )
            module = pd._load_script_module("extensionless_probe", script)
            self.assertEqual(module.VALUE, 42)
            self.assertEqual(module._load_pin(), "pin")

    def test_installed_venv_layout_is_a_cli_candidate(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp)
            root = venv / "lib" / "python3.14" / "site-packages" / "mlx"
            (root / "bin").mkdir(parents=True)
            (root / "bin" / "mlx-omarchy-parakeet").write_text("x = 1\n")
            previous = os.environ.get("OMARCHY_MLX_VENV")
            os.environ["OMARCHY_MLX_VENV"] = str(venv)
            try:
                candidates = list(pd._script_candidates())
            finally:
                if previous is None:
                    os.environ.pop("OMARCHY_MLX_VENV", None)
                else:
                    os.environ["OMARCHY_MLX_VENV"] = previous
            self.assertIn(root / "bin" / "mlx-omarchy-parakeet", candidates)


class AcceptanceReceiptTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.home = self._tmp.name

    def test_receipt_round_trip_and_revision_guard(self):
        cli = pd.load_cli_module()
        try:
            pin = cli._load_pin()
        except Exception as error:
            self.skipTest(
                f"runtime pin not resolvable without the installed product: "
                f"{error}"
            )
        revision = cli.ReferenceLock.load().model_revision
        path = pd.write_acceptance_receipt(
            self.home, transcript=pin["e2e"]["transcript"],
            emissions=pin["e2e"]["emissions"], host="unit-test",
        )
        receipt = pd.read_acceptance_receipt(self.home)
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt["model_revision"], revision)
        self.assertEqual(
            receipt["transcript_sha256"], pin["e2e"]["transcript_sha256"]
        )
        import json

        tampered = json.loads(path.read_text())
        tampered["model_revision"] = "0" * 40
        path.write_text(json.dumps(tampered))
        self.assertIsNone(pd.read_acceptance_receipt(self.home))

    def test_missing_receipt_is_none(self):
        self.assertIsNone(pd.read_acceptance_receipt(self.home))


class PeakFrequencyEstimatorTest(unittest.TestCase):
    """Pure-numpy guard for the estimator used by the gated resampler tests."""

    @staticmethod
    def peak_frequency_hz(samples: np.ndarray, rate: int) -> float:
        """Dominant frequency in Hz via rfftfreq (bins are not Hz)."""
        spectrum = np.abs(np.fft.rfft(samples.astype(np.float64)))
        frequencies = np.fft.rfftfreq(samples.size, d=1.0 / rate)
        return float(frequencies[int(np.argmax(spectrum))])

    def test_estimator_is_hz_accurate_on_cropped_windows(self):
        t = np.arange(15_800, dtype=np.float64) / 16_000
        sine = np.sin(2 * math.pi * 440.0 * t)
        self.assertAlmostEqual(
            self.peak_frequency_hz(sine, 16_000), 440.0, delta=1.0
        )
        t = np.arange(15_600, dtype=np.float64) / 16_000
        sine = np.sin(2 * math.pi * 440.0 * t)
        self.assertAlmostEqual(
            self.peak_frequency_hz(sine, 16_000), 440.0, delta=1.0
        )


class ResamplerTest(unittest.TestCase):
    """GPU polyphase resampling; MLX import gates the tests."""

    @classmethod
    def setUpClass(cls):
        try:
            import mlx.core  # noqa: F401

            cls.mx_missing = False
        except ImportError:
            raise unittest.SkipTest("mlx is not importable on this host")

    peak_frequency_hz = PeakFrequencyEstimatorTest.peak_frequency_hz

    def resample(self, samples, rate):
        return pd.resample_to_16k(samples, rate)

    def test_16k_is_identity(self):
        samples = np.zeros(1_600, dtype=np.float32)
        out = self.resample(samples, 16_000)
        np.testing.assert_array_equal(out, samples)

    def test_48k_sine_keeps_frequency_and_gain(self):
        import math

        t = np.arange(48_000, dtype=np.float64) / 48_000
        sine = (0.8 * np.sin(2 * math.pi * 440.0 * t)).astype(np.float32)
        resampled = self.resample(sine, 48_000)
        self.assertEqual(resampled.size, 16_000)
        self.assertEqual(resampled.dtype, np.float32)
        core = resampled[100:-100]
        self.assertAlmostEqual(
            self.peak_frequency_hz(core, 16_000), 440.0, delta=2.0
        )
        peak = float(np.abs(core.astype(np.float64)).max())
        self.assertAlmostEqual(peak / 0.8, 1.0, delta=0.02)

    def test_44100_sine_maps_to_exact_length(self):
        import math

        t = np.arange(44_100, dtype=np.float64) / 44_100
        sine = (0.8 * np.sin(2 * math.pi * 440.0 * t)).astype(np.float32)
        resampled = self.resample(sine, 44_100)
        self.assertEqual(resampled.size, 16_000)
        core = resampled[200:-200]
        self.assertAlmostEqual(
            self.peak_frequency_hz(core, 16_000), 440.0, delta=4.0
        )
        peak = float(np.abs(core.astype(np.float64)).max())
        self.assertAlmostEqual(peak / 0.8, 1.0, delta=0.02)

    def test_dc_level_is_preserved(self):
        flat = np.full(48_000, 0.25, dtype=np.float32)
        resampled = self.resample(flat, 48_000)
        middle = resampled[100:-100].astype(np.float64)
        self.assertAlmostEqual(float(middle.mean()), 0.25, delta=0.001)


class HardwareDictationTest(unittest.TestCase):
    """Real ANE/Vulkan execution; skips with named reasons elsewhere."""

    @classmethod
    def setUpClass(cls):
        cls.probe = pd.probe_runtime(verify_cache=True)
        if not cls.probe["ok"]:
            raise unittest.SkipTest(
                "qualified Parakeet runtime unavailable: "
                + "; ".join(cls.probe["reasons"])
            )
        cls.transcriber = pd.DictationTranscriber()

    @classmethod
    def tearDownClass(cls):
        cls.transcriber.close()

    def _fixture_waveform(self):
        cli = pd.load_cli_module()
        lock = cli.ReferenceLock.load()
        fixture = cli._fixture_path(cli._cache_dir(lock))
        pcm, rate, _decoder = cli._decode_flac(fixture)
        waveform = np.asarray(pcm, dtype=np.float32) / 32768.0
        return waveform, rate

    def test_fixture_audio_matches_pinned_oracle(self):
        cli = pd.load_cli_module()
        pin = cli._load_pin()
        waveform, rate = self._fixture_waveform()
        self.assertEqual(rate, 16_000)
        transcript = self.transcriber.transcribe_waveform(waveform, rate)
        self.assertEqual(transcript, pin["e2e"]["transcript"])

    def test_silence_produces_no_text(self):
        transcript = self.transcriber.transcribe_waveform(
            np.zeros(16_000, dtype=np.float32), 16_000
        )
        self.assertEqual(transcript, "")

    def test_multi_chunk_carries_recurrent_state(self):
        waveform, rate = self._fixture_waveform()
        spaced = np.concatenate(
            [waveform, np.zeros(16_000, dtype=np.float32), waveform]
        )
        self.assertGreater(spaced.size, 480_000)
        transcript = self.transcriber.transcribe_waveform(spaced, rate)
        self.assertIsInstance(transcript, str)
        self.assertTrue(transcript.strip(), "two speech segments must decode")

    def test_close_releases_and_refuses_further_work(self):
        transcriber = pd.DictationTranscriber()
        transcriber.close()
        transcriber.close()  # idempotent
        with self.assertRaises(pd.DictationRefusal):
            transcriber.transcribe_waveform(
                np.zeros(100, dtype=np.float32), 16_000
            )


if __name__ == "__main__":
    unittest.main()
