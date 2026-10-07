"""Unit tests for scripts/kokoro_objective_ab.py on synthetic signals.

The pre-registered acceptance requires: a click injected at a known chunk
boundary must be detected against the whole-file distribution; identical
files must give zero deltas; the autocorrelation F0 estimator must recover
a known sine. These tests never touch the lab artifacts.
"""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import kokoro_objective_ab as ab  # noqa: E402

SR = ab.SR


def tone(seconds: float, hz: float = 150.0, seed: int = 7) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    x = 0.5 * np.sin(2 * np.pi * hz * t) + 0.01 * rng.standard_normal(n)
    return x.astype(np.float32)


HAS_LIBROSA = importlib.util.find_spec("librosa") is not None


@unittest.skipUnless(HAS_LIBROSA, "needs librosa (spectral flux)")
class IdenticalFilesTest(unittest.TestCase):
    def test_zero_deltas_and_trivial_pass(self):
        x = tone(1.0)
        out = ab.pair_metrics(x, x)
        self.assertEqual(out["duration_delta"], 0.0)
        self.assertEqual(out["mcd_db"], 0.0)
        self.assertTrue(out["f0"]["pass"])
        self.assertEqual(out["f0"]["median_cents"], 0.0)
        self.assertEqual(out["f0"]["corr"], 1.0)

    def test_click_free_boundaries_pass(self):
        x = tone(1.5)
        report = ab.click_report(x, x, [{"pos": SR // 2}])
        self.assertTrue(report["worst"]["step_pass"])
        self.assertTrue(report["worst"]["flux_pass"])


@unittest.skipUnless(HAS_LIBROSA, "needs librosa (spectral flux)")
class ClickDetectionTest(unittest.TestCase):
    def test_injected_click_is_detected(self):
        whole = tone(1.5)
        stream = whole.copy()
        pos = SR // 2
        stream[pos - 40:pos] = 0.0        # abrupt step at the boundary
        stream[pos] = 0.9
        edges = [{"pos": pos, "class": "utterance"}]
        report = ab.click_report(stream, whole, edges)
        self.assertFalse(report["worst"]["step_pass"])
        self.assertGreater(report["boundaries"][0]["step"],
                           ab.CLICK_MARGIN * report["reference"]["p999_step"])

    def test_unclicked_signal_keeps_passing(self):
        whole = tone(1.5)
        stream = whole.copy()
        edges = [{"pos": SR // 2, "class": "utterance"}]
        report = ab.click_report(stream, whole, edges)
        self.assertTrue(report["worst"]["step_pass"])
        self.assertTrue(report["worst"]["flux_pass"])


class F0EstimatorTest(unittest.TestCase):
    def test_autocorr_recovers_known_sine(self):
        f0, voiced = ab.f0_autocorr(tone(1.0, hz=150.0))
        self.assertTrue(voiced.any())
        err = np.median(np.abs(1200 * np.log2(f0[voiced] / 150.0)))
        self.assertLess(err, 5.0)

    def test_f0_metrics_bars(self):
        t = np.arange(100)
        good = 150.0 + 8.0 * np.sin(t / 9.0)
        near = good * 1.005                       # ~8.6 cents, corr 1.0
        voiced = np.ones(100, dtype=bool)
        self.assertTrue(ab.f0_metrics(good, voiced, near, voiced)["pass"])
        far = good * 1.30
        self.assertFalse(ab.f0_metrics(good, voiced, far, voiced)["pass"])
        self.assertFalse(ab.f0_metrics(good, voiced, np.zeros(100), voiced)["pass"])


class ProbeBoundariesTest(unittest.TestCase):
    def test_chunk_edges_and_classes(self):
        probe = "\n".join(json.dumps(e) for e in [
            {"event": "item", "item": "s",
             "chunks": [
                 {"seq": 1, "t": 0.0, "dur": 0.6},
                 {"seq": 1, "t": 0.224, "dur": 0.6},      # window join @28800
                 {"seq": 1, "t": 0.448, "dur": 0.15},     # utterance tail
                 {"seq": 1, "t": 1.4, "dur": 0.3},        # utterance join @45600
                 {"seq": 2, "t": 2.6, "dur": 0.6},        # sentence join @60000
             ]}])
        path = Path(__file__).parent / "_probe_fixture.jsonl"
        path.write_text(probe)
        try:
            bounds = ab.boundaries_from_probe(path)
        finally:
            path.unlink()
        self.assertEqual([e["pos"] for e in bounds["s"]["window"]], [14400, 28800])
        self.assertEqual([e["pos"] for e in bounds["s"]["utterance"]], [32400])
        self.assertEqual([e["pos"] for e in bounds["s"]["sentence"]], [39600])


class SilenceTest(unittest.TestCase):
    def test_pad_silence_measured(self):
        x = tone(1.0)
        x[SR // 4:SR // 4 + int(0.05 * SR)] = 0.0
        run = ab.longest_silence(x, SR // 4)
        self.assertAlmostEqual(run, 0.05, delta=0.005)


if __name__ == "__main__":
    unittest.main()
