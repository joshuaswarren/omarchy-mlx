"""Offline wake evaluator: curve building, the grid post-processing that
mirrors the detector's fire rules, and the results artifact. Scripted
sessions only — real openWakeWord inference is the receipt run, not CI.
"""

import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path

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


def write_clip(path: Path, loud_seconds=0.8, silent_seconds=1.2):
    rate = wake_eval.SAMPLE_RATE
    loud = (np.sin(np.arange(int(loud_seconds * rate)) * 0.05) * 8000).astype("int16")
    silent = np.zeros(int(silent_seconds * rate), dtype="int16")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(np.concatenate([loud, silent]).tobytes())


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


if __name__ == "__main__":
    unittest.main()
