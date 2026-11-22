#!/usr/bin/env python3
"""Tests for the V27 word-level seam merge.

The V24 merge only dropped an exact repeating prefix, so any transcription
difference inside the 2 s overlap duplicated the whole overlap. V27 keeps the
exact fast path for clean seams and falls back to fuzzy word alignment for
broken ones. Fixtures cover Main's required case: a word straddling the cut.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from earnings22_seam_merge import merge


class MergeTests(unittest.TestCase):
    def test_clean_repeat_dropped_like_v24(self):
        # The whole overlap repeats exactly: drop it.
        merged = ["good", "day", "and", "ladies", "and", "gentlemen,", "a", "qa", "session"]
        words = ["a", "qa", "session.", "today's", "call"]
        out, dropped, _matched = merge(merged, words)
        self.assertEqual(out, merged + ["today's", "call"])
        self.assertEqual(dropped, 3)
        self.assertEqual(_matched, 3)

    def test_word_straddles_the_cut(self):
        # Window A ended mid-word ("twenty"), window B starts with the full
        # phrase: the straddled word must come from B, once.
        merged = ["third", "quarter", "twenty"]
        words = ["twenty", "twenty-one", "earnings", "call"]
        out, dropped, _matched = merge(merged, words)
        self.assertEqual(out, ["third", "quarter", "twenty", "twenty-one", "earnings", "call"])
        self.assertEqual(dropped, 1)
        self.assertEqual(_matched, 1)

    def test_overlap_repeat_with_one_changed_word(self):
        # The duplicated-operator-line failure V24 actually hit: the whole
        # overlap repeats but one word differs, so the exact match fails and
        # fuzzy alignment must still drop the overlap.
        merged = ["there", "will", "be", "a", "qa", "session"]
        words = ["marks,", "there", "will", "be", "a", "qa", "session", "today"]
        out, dropped, _matched = merge(merged, words)
        self.assertEqual(out, merged + ["today"])
        self.assertEqual(dropped, 7)
        self.assertEqual(_matched, 6)

    def test_no_overlap_appends_everything(self):
        # Audio changed across the cut and nothing aligns: keep both sides
        # (V24 behavior), do not guess a drop.
        merged = ["the", "annual", "report"]
        words = ["welcome", "to", "the", "call"]
        out, dropped, _matched = merge(merged, words)
        self.assertEqual(out, merged + words)
        self.assertEqual(dropped, 0)
        self.assertEqual(_matched, 0)

    def test_case_and_punctuation_are_ignored_for_matching(self):
        merged = ["session", "today"]
        words = ["Session.", "Today", "we", "begin"]
        out, dropped, _matched = merge(merged, words)
        self.assertEqual(out, ["session", "today", "we", "begin"])
        self.assertEqual(dropped, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
