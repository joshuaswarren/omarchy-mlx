"""Frozen generator statistics for windowed Kokoro decoding.

The windowed decoder normalises the generator's AdaIN layers with
statistics calibrated per voice (scripts/kokoro_gen_stats.py). These tests
pin the shipped file to the pinned pack and its voices, and the loader's
refusals: statistics from another pack revision, or a voice without
statistics, must fail loudly instead of decoding with the wrong numbers.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "serve"))

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy ships with the voice runtime
    np = None

from mlx_omarchy_assistant import kokoro_stream  # noqa: E402
from mlx_omarchy_assistant.synthesis import KOKORO_PACK  # noqa: E402


@unittest.skipIf(np is None, "numpy not installed")
class FrozenStatsTests(unittest.TestCase):
    def _write(self, path, revision, voices, layers=("generator.a", "generator.b")):
        arrays = {}
        for voice in voices:
            for layer in layers:
                arrays[f"{voice}|{layer}|mean"] = np.zeros(4, np.float32)
                arrays[f"{voice}|{layer}|var"] = np.ones(4, np.float32)
        np.savez(path, meta=np.array(json.dumps({"pack_revision": revision})), **arrays)

    def test_shipped_stats_cover_every_pack_voice_at_the_pinned_revision(self):
        stats = kokoro_stream.load_stats(kokoro_stream.STATS_FILE, KOKORO_PACK["voices"],
                                         KOKORO_PACK["revision"])
        self.assertEqual(set(stats), set(KOKORO_PACK["voices"]))
        layer_sets = {frozenset(table) for table in stats.values()}
        self.assertEqual(len(layer_sets), 1, "every voice covers the same generator layers")
        for table in stats.values():
            for layer, (mean, var) in table.items():
                self.assertTrue(layer.startswith("generator."), layer)
                self.assertEqual(mean.shape, var.shape)
                self.assertTrue(np.isfinite(mean).all() and np.isfinite(var).all(), layer)
                self.assertTrue((var >= 0).all(), layer)

    def test_stats_from_another_pack_revision_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stats.npz"
            self._write(path, "0" * 40, KOKORO_PACK["voices"])
            with self.assertRaisesRegex(ValueError, "pack revision"):
                kokoro_stream.load_stats(path, KOKORO_PACK["voices"], KOKORO_PACK["revision"])

    def test_a_voice_without_stats_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stats.npz"
            self._write(path, KOKORO_PACK["revision"], KOKORO_PACK["voices"][:1])
            with self.assertRaisesRegex(ValueError, KOKORO_PACK["voices"][1]):
                kokoro_stream.load_stats(path, KOKORO_PACK["voices"], KOKORO_PACK["revision"])


class PhonemeSegmentTests(unittest.TestCase):
    """Utterance cuts: every word kept in order, cuts at word boundaries,
    punctuation preferred, no tiny tail utterance."""

    PS = ("hˈWɛvəɹ jʊɹ mˈitɪŋ stˈɑɹts æt nˈIn, ænd ðə ɹəvjˈu fˈɑlOz æt "
          "əlˈɛvən ɪn ðə kˈɔnfəɹəns ɹˈum ɑn ðə θˈɜɹd flˈɔɹ.")

    def test_words_survive_in_order(self):
        segments = kokoro_stream.phoneme_segments(self.PS)
        self.assertGreater(len(segments), 1)
        self.assertEqual(" ".join(segments).split(), self.PS.split())

    def test_segments_stay_near_the_budget(self):
        budget = kokoro_stream.SEGMENT_PHONEMES
        longest = max(len(w) for w in self.PS.split())
        for segment in kokoro_stream.phoneme_segments(self.PS)[:-1]:
            size = sum(len(w) for w in segment.split())
            self.assertLess(size, budget + longest, segment)

    def test_cut_after_punctuation_once_past_sixty_percent(self):
        segments = kokoro_stream.phoneme_segments(self.PS)
        self.assertTrue(segments[0].endswith("nˈIn,"), segments[0])

    def test_short_tail_joins_the_previous_utterance(self):
        ps = " ".join(["wˈʌnwˈʌn"] * 4) + " ˈO"
        segments = kokoro_stream.phoneme_segments(ps, budget=29)
        self.assertTrue(segments[-1].endswith("ˈO"))
        self.assertGreater(sum(len(w) for w in segments[-1].split()), 29 / 3)

    def test_short_input_is_one_utterance(self):
        self.assertEqual(kokoro_stream.phoneme_segments("hˈɛlO."), ["hˈɛlO."])


class CutFrameTests(unittest.TestCase):
    """Frame-level utterance cuts from the whole-sentence predicted
    durations: vocab-filtered token mapping and full coverage."""

    def test_boundaries_derive_phoneme_segments(self):
        ps = "wˈɜrd wˈʌnx tuː ðɪːŋz ɐnd ʌp."  # 30 chars, multi-word
        cuts = kokoro_stream.segment_boundaries(ps, budget=10)
        rebuilt = kokoro_stream.phoneme_segments(ps, budget=10)
        edges = [0] + cuts + [len(ps)]
        self.assertEqual(rebuilt,
                         [ps[a:b].strip() for a, b in zip(edges, edges[1:])
                          if ps[a:b].strip()])
        self.assertTrue(all(0 < c < len(ps) for c in cuts))
        self.assertEqual(ps[cuts[0]], " ")    # cut lands between words
        self.assertNotEqual(ps[cuts[0] - 1], " ")

    def test_cut_frames_map_vocab_filtered_tokens(self):
        vocab = {"a": 1, "b": 2, "c": 3, "d": 4}
        ps = "ab.cd"                       # every char in vocab, no spaces
        pred_dur = [9, 1, 2, 3, 4, 7]      # pad + a,b,c,d + pad
        frames = kokoro_stream.cut_frames(ps, vocab, pred_dur, [2])
        self.assertEqual(frames, [12])     # 9 pad + 1 + 2
        self.assertEqual(kokoro_stream.utterance_ranges([12], 26),
                         [(0, 12), (12, 26)])

    def test_ranges_cover_every_frame(self):
        self.assertEqual(kokoro_stream.utterance_ranges([], 5), [(0, 5)])
        self.assertEqual(kokoro_stream.utterance_ranges([3, 3, 0, 9, 2], 9),
                         [(0, 2), (2, 3), (3, 9)])
        covered = [r for rs in kokoro_stream.utterance_ranges([4, 7], 9)
                   for r in rs]
        self.assertEqual(sorted(set(covered)), [0, 4, 7, 9])


if __name__ == "__main__":
    unittest.main()
