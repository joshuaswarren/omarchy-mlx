"""Frozen held-out routing suite. Spent: evaluated once on 2026-09-30 with
routing policy 3 (receipts/2026-09-30-routing-gate). Automatic routing stays
off. The bytes are pinned by sha256; never edit the cases."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "serve"))
from mlx_omarchy_assistant.coordinator import Coordinator  # noqa: E402

SUITE = Path(__file__).resolve().parent / "fixtures" / "routing_held_out.json"
FROZEN_SHA256 = "09a37b602336e308df6ece8ae9135d7771c9f9407826c89ab92b51b3198906f0"
REQUIRED = {"ordinary", "decisions", "ambiguity", "negation", "injection", "oversized"}
ROUTES = {"conversation", "structured_decision", "clarify"}


class RoutingSuiteTests(unittest.TestCase):
    def setUp(self):
        self.doc = json.loads(SUITE.read_text())

    def test_suite_is_frozen_and_covers_the_required_categories(self):
        self.assertEqual(hashlib.sha256(SUITE.read_bytes()).hexdigest(), FROZEN_SHA256)
        cases = self.doc["cases"]
        self.assertGreaterEqual(len(cases), 100)
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        self.assertTrue(REQUIRED <= {case["category"] for case in cases})
        for case in cases:
            self.assertIn(case["expected"], ROUTES)
            self.assertTrue(case["text"].strip())
            self.assertTrue(case["note"].strip())
        for case in cases:
            if case["category"] == "negation":
                lowered = case["text"].lower()
                self.assertTrue(any(word in lowered for word in ("not", "never", "don't")))
            if case["category"] == "oversized":
                self.assertGreater(len(case["text"]), 2000)

    def test_automatic_mode_is_rejected(self):
        class Stopped:
            def stop(self):
                return {"stopped": True, "retained": []}

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        app = Coordinator(Path(directory.name), Stopped())
        self.addCleanup(app.close)
        with self.assertRaises(ValueError):
            app.submit("unused", {"text": "Pick the right option.", "mode": "auto"})
