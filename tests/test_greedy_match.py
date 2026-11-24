# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Gate semantics for tools/torch-mlx-vulkan/greedy_match.py.

gen_qwen3.py's MATCH used exact id equality and failed at Qwen3-0.6B
token 5, where the fp32 reference top-2 gap is 0.0242 — below bf16 logit
resolution (0.0625 ulp at logit ~10), so the compiled path legitimately
computes an exact tie and argmax lands on whichever side the build's
fusion order picks. A wheel bisect showed pre-window and head builds flip
identically, so the equality gate reads a coin flip as a regression.

These tests pin the tie-aware contract:

- a divergence at a reference near-tie (gap < TIE_TOL) toward the
  reference's other top-2 candidate is tolerated, and the reference token
  is fed forward so later steps still compare on the same context;
- a divergence when the reference was confident (gap >= TIE_TOL), or
  toward a token outside the reference top-2, is a real failure;
- the harness output names tolerated ties rather than hiding them.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "torch-mlx-vulkan"))

import greedy_match as gm


def row(tok, alt, gap):
    return {"tok": tok, "alt": alt, "gap": gap}


class GuidedTokenTest(unittest.TestCase):
    def test_same_argmax_is_match_and_feeds_it(self):
        feed, mark = gm.guided_token(row(15344, 9625, 0.024), row(15344, 9625, 0.0242))
        self.assertEqual((feed, mark), (15344, "match"))

    def test_near_tie_flip_is_tolerated_and_feeds_reference(self):
        # The measured token-5 case: ref was split 15344/9625 by 0.0242,
        # the device computed an exact tie and took 9625. The reference
        # token must be fed forward so step 6+ compare on the same context.
        feed, mark = gm.guided_token(row(9625, 15344, 0.0), row(15344, 9625, 0.0242))
        self.assertEqual(mark, "tie")
        self.assertEqual(feed, 15344)

    def test_confident_reference_still_fails(self):
        feed, mark = gm.guided_token(row(9625, 100, 5.0), row(15344, 9625, 0.19))
        self.assertEqual((feed, mark), (9625, "diff"))

    def test_third_candidate_at_a_near_tie_fails(self):
        # Device wandered outside the reference top-2: not a tie flip.
        feed, mark = gm.guided_token(row(999, 15344, 0.0), row(15344, 9625, 0.01))
        self.assertEqual((feed, mark), (999, "diff"))

    def test_boundary_gap_is_confident(self):
        _, mark = gm.guided_token(row(9625, 15344, 0.0), row(15344, 9625, gm.TIE_TOL))
        self.assertEqual(mark, "diff")


class MatchReportTest(unittest.TestCase):
    def test_all_match(self):
        rep = gm.match_report(["match"] * 16)
        self.assertEqual(rep, {"match": True, "ties": [], "first_diff": -1})

    def test_ties_pass_and_are_named(self):
        marks = ["match"] * 5 + ["tie"] + ["match"] * 10
        rep = gm.match_report(marks)
        self.assertTrue(rep["match"])
        self.assertEqual(rep["ties"], [5])
        self.assertEqual(rep["first_diff"], -1)

    def test_diff_fails_with_index(self):
        marks = ["match"] * 5 + ["tie", "diff"] + ["match"] * 9
        rep = gm.match_report(marks)
        self.assertFalse(rep["match"])
        self.assertEqual(rep["ties"], [5])
        self.assertEqual(rep["first_diff"], 6)

    def test_exact_equality_would_fail_the_tolerated_case(self):
        # Why the gate needed fixing: the old ref_ids == comp_ids check
        # returns False on exactly the sequence the tie-aware gate passes.
        ref_ids = [12095, 13, 576, 6722, 315, 15344, 374]
        comp_ids = [12095, 13, 576, 6722, 315, 9625, 374]
        self.assertFalse(ref_ids == comp_ids)  # old gate: regression reported
        marks = [gm.guided_token(row(c, -1, 0.0), row(r, c, 0.0242))[1]
                 if i == 5 else "match" for i, (r, c) in enumerate(zip(ref_ids, comp_ids))]
        self.assertTrue(gm.match_report(marks)["match"])


if __name__ == "__main__":
    unittest.main()
