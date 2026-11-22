"""Per-stage timers on the GPU speech path (card V2).

The speech path must report where the RTFx goes: load, resample, mel,
encoder, decoder (the TDT step loop), joint and post. The timers must not
change the transcript, the retry behavior, or the response's existing keys.
"""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_assistant import gpu_stt_worker  # noqa: E402

# Resolved at import, before any test can stub mlx into sys.modules.
HAS_MLX = importlib.util.find_spec("mlx") is not None


class _StubTDTModel:
    """The ParakeetTDT shape instrument() wraps, driving the real stage
    order (mel -> encoder -> TDT step loop) from generate()."""

    def __init__(self):
        self.mel_calls = 0
        self.encoder_calls = 0
        self.step_calls = 0
        self.joint_calls = 0
        self.generate_calls = 0
        self.encoder = self
        self.joint = self

    def __call__(self, *args, **kwargs):
        return args

    def _log_mel_spectrogram(self, audio):
        self.mel_calls += 1
        return audio

    def _tdt_step(self, feature, current_token, hidden, cell):
        self.step_calls += 1
        return feature, feature, hidden, cell

    def generate(self, audio, verbose=False):
        self.generate_calls += 1
        mel = self._log_mel_spectrogram(audio)
        self.encoder(mel)
        self._tdt_step(mel, 1, audio, audio)
        self.joint(mel, mel)
        return SimpleNamespace(text="hello stage timers")


@unittest.skipUnless(HAS_MLX, "mlx is required for the resample stage")
class StageTimerResponseTest(unittest.TestCase):
    def _transcribe_once(self, model):
        payload = np.ones(16_000, dtype="<f4").tobytes()
        return gpu_stt_worker._transcribe(
            model, payload, 16_000, None, load_ms=12.5)

    def test_response_reports_every_named_stage_in_milliseconds(self):
        model = _StubTDTModel()
        response = self._transcribe_once(model)
        self.assertTrue(response["ok"], response)
        self.assertEqual(response["transcript"], "hello stage timers")
        self.assertEqual(model.generate_calls, 1)
        stage_ms = response["stage_ms"]
        for stage in ("load", "resample", "mel", "encoder",
                      "decoder", "joint", "post"):
            self.assertIn(stage, stage_ms)
            self.assertIsInstance(stage_ms[stage], float)
            self.assertGreaterEqual(stage_ms[stage], 0.0)
        self.assertEqual(response["load_ms"], 12.5)
        # The old keys survive untouched.
        self.assertIn("peak_memory_bytes", response)

    def test_stages_route_through_the_clock_exactly_once_per_request(self):
        model = _StubTDTModel()
        clock = gpu_stt_worker.StageClock()
        gpu_stt_worker.instrument(model, clock)
        self._transcribe_once(model)
        self._transcribe_once(model)
        self.assertEqual(model.mel_calls, 2)
        self.assertGreater(clock.ms["mel"], 0.0)
        self.assertGreater(clock.ms["decoder"], 0.0)
        self.assertGreater(clock.ms["joint"], 0.0)

    def test_instrument_delegates_attribute_access_and_is_idempotent(self):
        model = _StubTDTModel()
        clock = gpu_stt_worker.StageClock()
        self.assertIs(gpu_stt_worker.instrument(model, clock), model)
        # The seam proxies still reach the wrapped callables.
        self.assertIs(model.encoder._target, model)
        self.assertIs(gpu_stt_worker.instrument(model, clock), model)
        self.assertIs(model.encoder._target, model)


if __name__ == "__main__":
    unittest.main()
