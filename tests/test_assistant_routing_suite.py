"""Frozen held-out routing suite. Spent: evaluated once on 2026-09-30 with
routing policy 3 (receipts/2026-09-30-routing-gate) and once on 2026-10-03
with the rope-table head (receipts/2026-10-02-laya-head-latency). It is
spent for this candidate and is never re-evaluated here; only the bytes
are pinned by sha256. Automatic routing is ON by default since the owner
decision of 2026-10-03 (receipts/2026-10-03-routing-on)."""

import hashlib
import json
import sys
import tempfile
import time
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

    def test_automatic_mode_is_admitted_by_default(self):
        """Owner decision 2026-10-03: routing is ON by default. A turn the
        deterministic stage resolves on its own (choice wording, fewer
        than two options) routes to clarify and never calls the head."""
        class Stopped:
            def status(self):
                return None
            def start(self):
                return {"model_paths": {"decision": "/tmp/fake-decision"},
                        "decision_url": "http://127.0.0.1:1/",
                        "chat_url": "http://127.0.0.1:1/",
                        "context_tokens": 4096}
            def stop(self):
                return {"stopped": True, "retained": []}
            def ensure_context(self, required_tokens):
                return {"ok": True, "context_tokens": max(required_tokens, 4096),
                        "requested": required_tokens, "changed": False}

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        app = Coordinator(Path(directory.name), Stopped())
        self.addCleanup(app.close)

        class NoDecision:
            calls = 0
            def count(self, path, messages):
                return 1
            def decision(self, pair, payload):
                NoDecision.calls += 1
                return {}
            def close_connection(self):
                pass

        app.models = app.router.models = NoDecision()
        cid = app.store.create()["id"]
        app.submit(cid, {"text": "Pick the right option.", "mode": "auto"})
        deadline = time.monotonic() + 2.0
        routing = None
        while time.monotonic() < deadline:
            events = app.store.events(cid, 0)
            routing = next((e for e in events if e["type"] == "routing"), None)
            if routing is not None:
                break
            time.sleep(0.02)
        self.assertIsNotNone(routing, "routing event missing")
        self.assertEqual(routing["data"]["route"], "clarify")
        self.assertEqual(NoDecision.calls, 0)
