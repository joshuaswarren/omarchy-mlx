"""Offline wake evaluator: curve building, the grid post-processing that
mirrors the detector's fire rules, and the results artifact. Scripted
sessions only — real openWakeWord inference is the receipt run, not CI.
"""

import json
import math
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_assistant import wake_eval, wake_word  # noqa: E402


class ScriptedSession:
    """Score stand-in: RMS above 1000 scores high, silence scores low."""

    def __init__(self, high=0.9, low=0.1):
        self.high, self.low = high, low
        self.calls = 0

    def predict(self, pcm):
        self.calls += 1
        rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
        return {"hey_jarvis_v0.1": self.high if rms > 1000 else self.low}

    def reset(self):
        pass


class SilenceFiresSession:
    """Scripted session that scores silence high and speech low, so a wake
    fires inside the joining silence rather than inside a clip."""

    def predict(self, pcm):
        rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
        return {"hey_jarvis_v0.1": 0.9 if rms <= 1000 else 0.1}

    def reset(self):
        pass


def write_clip(path: Path, loud_seconds=0.8, silent_seconds=1.2):
    rate = wake_eval.SAMPLE_RATE
    loud = (np.sin(np.arange(int(loud_seconds * rate)) * 0.05) * 8000).astype("int16")
    silent = np.zeros(int(silent_seconds * rate), dtype="int16")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(np.concatenate([loud, silent]).tobytes())


def write_wav(path: Path, frames: np.ndarray, rate: int):
    """Write int16 frames shaped (samples, channels) as a PCM WAV."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(frames.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames.astype("<i2").tobytes())


def fake_audio_modules(overshoot=1.0, float_source=None):
    """soundfile and scipy.signal stand-ins; CI installs neither.

    read() decodes 16-bit WAV like soundfile: (samples, channels) unless
    mono without always_2d, int16 or float scaled by 1/32768. Given
    float_source (frames, rate), it plays a float file instead, which
    soundfile returns unscaled when asked for int16. resample_poly
    decimates along axis 0, scaled by `overshoot` to model the filter's
    ringing past full scale on a loud edge.
    """
    def read(path, dtype="float64", always_2d=False):
        if float_source is not None:
            data, rate = float_source
            channels = data.shape[1]
            if dtype == "int16":
                data = np.clip(np.rint(data), -32768, 32767).astype("int16")
        else:
            with wave.open(str(path), "rb") as handle:
                channels, rate = handle.getnchannels(), handle.getframerate()
                data = np.frombuffer(handle.readframes(handle.getnframes()),
                                     dtype="<i2").reshape(-1, channels)
            if dtype != "int16":
                data = data / 32768.0
        if channels == 1 and not always_2d:
            data = data[:, 0]
        return data, rate

    def resample_poly(x, up, down, axis=0):
        g = math.gcd(up, down)
        assert up // g == 1 and axis == 0
        return np.asarray(x, dtype=np.float64)[::down // g] * overshoot

    signal = types.SimpleNamespace(resample_poly=resample_poly)
    return {"soundfile": types.SimpleNamespace(read=read),
            "scipy": types.SimpleNamespace(signal=signal),
            "scipy.signal": signal}


class ResampleTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def load(self, frames, rate, overshoot=1.0):
        write_wav(self.dir / "clip.wav", frames, rate)
        with mock.patch.dict(sys.modules, fake_audio_modules(overshoot)):
            pcm = wake_eval.load_pcm(self.dir / "clip.wav")
        return np.frombuffer(pcm, dtype="<i2")

    def test_stereo_16k_mixes_down_to_mono(self):
        n = int(2.5 * wake_eval.SAMPLE_RATE)
        frames = np.stack([np.full(n, 8000), np.full(n, -2000)], axis=1)
        out = self.load(frames, wake_eval.SAMPLE_RATE)
        self.assertEqual(len(out), n)
        self.assertTrue(np.all(out == 3000))

    def test_stereo_48k_mixes_down_then_resamples(self):
        n = int(2.5 * 48000)
        frames = np.stack([np.full(n, 6000), np.full(n, 2000)], axis=1)
        out = self.load(frames, 48000)
        self.assertEqual(len(out), int(2.5 * wake_eval.SAMPLE_RATE))
        self.assertTrue(np.all(out == 4000))

    def test_resample_overshoot_clips_instead_of_wrapping(self):
        n = int(2.5 * 48000)
        mono = np.where(np.arange(n) < n // 2, 32767, -32768)[:, None]
        out = self.load(mono, 48000, overshoot=1.09)
        half = len(out) // 2
        self.assertTrue(np.all(out[:half] == 32767))
        self.assertTrue(np.all(out[half:] == -32768))

    def test_float_source_is_scaled_and_clipped(self):
        path = self.dir / "clip.flac"  # not a WAV: load_pcm hands it on
        path.write_bytes(b"fLaC" + bytes(64))
        n = wake_eval.SAMPLE_RATE
        frames = np.repeat([0.5, 1.5, -1.5], n)[:, None]
        sources = fake_audio_modules(float_source=(frames, wake_eval.SAMPLE_RATE))
        with mock.patch.dict(sys.modules, sources):
            out = np.frombuffer(wake_eval.load_pcm(path), dtype="<i2")
        self.assertEqual(len(out), 3 * n)
        self.assertTrue(np.all(out[:n] == 16384))
        self.assertTrue(np.all(out[n:2 * n] == 32767))
        self.assertTrue(np.all(out[2 * n:] == -32768))


class WakeEvalTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        write_clip(self.dir / "pos-a.wav")
        write_clip(self.dir / "pos-b.wav")
        write_clip(self.dir / "neg-a.wav", loud_seconds=0.0, silent_seconds=1.6)
        self.manifest = [
            {"file": "pos-a.wav", "variant": "m1", "text": "hey jarvis", "label": "positive"},
            {"file": "pos-b.wav", "variant": "f1", "text": "hey, jarvis", "label": "positive"},
            {"file": "neg-a.wav", "variant": "m2", "text": "silence", "label": "negative"},
        ]

    def test_grid_counts_tpr_fpr_at_default(self):
        curves = wake_eval.build_curves(
            self.manifest, self.dir, "unused", session=ScriptedSession())
        row = wake_eval.grid_row(curves, threshold=0.5, floor=0.0)
        self.assertEqual((row["tp"], row["fn"]), (2, 0))
        self.assertEqual((row["fp"], row["tn"]), (0, 1))
        self.assertEqual(row["tpr"], 1.0)
        self.assertEqual(row["fpr"], 0.0)

    def test_vad_floor_gates_silence_without_losing_positives(self):
        curves = wake_eval.build_curves(
            self.manifest, self.dir, "unused", session=ScriptedSession())
        row = wake_eval.grid_row(curves, threshold=0.5, floor=200.0)
        self.assertEqual((row["tp"], row["fp"], row["tn"], row["fn"]), (2, 0, 1, 0))
        # The all-silent clip must show peak 0: the gate suppressed every chunk.
        neg = wake_eval.sweep(curves[2:3], thresholds=[0.5], floors=[200.0])[0]
        self.assertEqual(neg["tp"], 0)

    def test_curve_row_mirrors_refractory(self):
        curve = {"rms": [10000.0] * 40, "scores": [0.9] * 40}
        row = wake_eval.curve_row(curve, threshold=0.5, floor=0.0)
        self.assertTrue(row["fired"])  # one fire, refractory eats the rest

    def test_curve_row_vetoed_chunks_never_fire_but_still_score(self):
        curve = {"rms": [0.0] * 40, "scores": [0.9] * 40}
        row = wake_eval.curve_row(curve, threshold=0.5, floor=200.0)
        self.assertFalse(row["fired"])
        self.assertEqual(row["peak"], 0.9)

    def test_sweep_covers_the_grid(self):
        curves = wake_eval.build_curves(
            self.manifest, self.dir, "unused", session=ScriptedSession())
        rows = wake_eval.sweep(curves, thresholds=[0.5], floors=[0.0, 200.0])
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["vad_rms_floor"] for r in rows], [0.0, 200.0])

    def test_direct_row_counts_through_the_shipped_detector(self):
        row = wake_eval.direct_row(
            self.manifest, self.dir, "unused", threshold=0.5, floor=0.0,
            session=ScriptedSession())
        self.assertEqual((row["tp"], row["fn"]), (2, 0))
        self.assertEqual((row["fp"], row["tn"]), (0, 1))
        self.assertEqual(row["tpr"], 1.0)

    def test_main_writes_results_json(self):
        out = self.dir / "results.json"
        (self.dir / "m.json").write_text(json.dumps(self.manifest))
        rc = wake_eval.main([
            "--manifest", str(self.dir / "m.json"),
            "--clips-dir", str(self.dir),
            "--model", "unused",
            "--thresholds", "0.5",
            "--floors", "0",
            "--out", str(out),
        ], session=ScriptedSession())
        self.assertEqual(rc, 0)
        payload = json.loads(out.read_text())
        self.assertEqual(len(payload["rows"]), 1)
        self.assertEqual(payload["rows"][0]["tp"], 2)
        self.assertEqual(payload["curves"], [])

    def test_main_curve_mode_keeps_the_scored_curves(self):
        out = self.dir / "results-curve.json"
        (self.dir / "m.json").write_text(json.dumps(self.manifest))
        rc = wake_eval.main([
            "--manifest", str(self.dir / "m.json"),
            "--clips-dir", str(self.dir),
            "--model", "unused",
            "--thresholds", "0.5",
            "--floors", "0",
            "--mode", "curve",
            "--out", str(out),
        ], session=ScriptedSession())
        self.assertEqual(rc, 0)
        payload = json.loads(out.read_text())
        self.assertEqual(len(payload["curves"]), 3)


    def test_corpus_row_counts_wake_events_per_hour(self):
        row = wake_eval.corpus_row(
            [self.dir / "pos-a.wav", self.dir / "neg-a.wav"], "unused",
            threshold=0.5, floor=0.0, session=ScriptedSession())
        self.assertEqual(row["wakes"], 1)  # one loud clip, one silent clip
        self.assertEqual(row["files"], 2)
        self.assertGreater(row["hours"], 0.0)
        # both clips pad to MIN_SECONDS: 4 s of audio, 1 wake
        self.assertEqual(row["wakes_per_hour"], 900.0)

    def test_corpus_row_refractory_limits_one_wake_per_window(self):
        row = wake_eval.corpus_row(
            [self.dir / "pos-a.wav"], "unused", threshold=0.5, floor=0.0,
            session=ScriptedSession())
        self.assertEqual(row["wakes"], 1)  # one phrase, one wake

    def test_corpus_row_counts_wakes_that_fire_in_the_joining_silence(self):
        loud = self.dir / "loud-full.wav"
        write_clip(loud, loud_seconds=2.5, silent_seconds=0.0)
        row = wake_eval.corpus_row(
            [loud], "unused", threshold=0.5, floor=0.0, gap_seconds=1.0,
            session=SilenceFiresSession())
        self.assertEqual(row["wakes"], 1)  # fires in the gap, not in the clip

    def test_main_corpus_mode_counts_wakes(self):
        out = self.dir / "results-corpus.json"
        files = sorted(str(p) for p in self.dir.glob("*.wav"))
        (self.dir / "corpus.txt").write_text("\n".join(files) + "\n")
        rc = wake_eval.main([
            "--mode", "corpus",
            "--corpus-list", str(self.dir / "corpus.txt"),
            "--model", "unused",
            "--thresholds", "0.5",
            "--floors", "0",
            "--out", str(out),
        ], session=ScriptedSession())
        self.assertEqual(rc, 0)
        payload = json.loads(out.read_text())
        self.assertEqual(len(payload["rows"]), 1)
        self.assertEqual(payload["rows"][0]["wakes"], 2)


if __name__ == "__main__":
    unittest.main()
