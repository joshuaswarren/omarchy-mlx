"""Coordinator card-promotion contract: model fence wins, promotion only
fires on plain markdown replies, hostile input is inert, large inputs are
bounded, and a validation rejection drops the card without dropping the
prose.  Uses the SSE wire-protocol harness from test_assistant_generation."""

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_assistant import coordinator as coord  # noqa: E402
from mlx_omarchy_assistant import components  # noqa: E402
from mlx_omarchy_assistant import card_promotion  # noqa: E402
from mlx_omarchy_assistant.card_promotion import extract_text  # noqa: E402

from tests.test_assistant_generation import (
    GenerationTests, FakeWorker, VALID_ENVELOPE, delta, finish,
)


def _card_events(record):
    return [e for e in record["messages"][-1].get("components") or []]


class CardPromotionTests(GenerationTests):
    """Reuses GenerationTests.setUp (fake SSE worker + fake pair manager)."""

    def _run_chat(self, cid, worker_scripts, user_text="please help",
                  chat_model="fake", max_tokens=700):
        self.worker.scripts = worker_scripts
        # Force a known chat_model id so the schema-policy branch is
        # exercised and stays reproducible across catalog changes.
        original_start = self.manager.start
        def start_with_model():
            result = original_start()
            result["chat_model"] = chat_model
            return result
        self.manager.start = start_with_model
        turn, record = self.run_turn(
            cid, {"text": user_text, "max_tokens": max_tokens})
        return turn, record

    def test_model_fence_wins_over_promotion(self):
        # Model emits a valid fenced block; the parser would also be able
        # to derive a checklist from the prose, but the fence must win.
        reply = "Here is the list.\n\n```assistant-ui\n" + VALID_ENVELOPE + "\n```\n"
        cid = self.cid()
        turn, record = self._run_chat(
            cid,
            [[(0, delta(reply)), (0, finish("stop"))]],
            user_text="give me a checklist of steps",
        )
        cards = _card_events(record)
        self.assertEqual(len(cards), 1, "model fence must produce exactly one card")
        self.assertEqual(cards[0]["type"], "checklist")
        # The card must NOT carry the assistant-built title suffix.
        self.assertFalse((cards[0].get("title") or "").endswith(" (from reply)"),
                         "model-emitted card must not be tagged as derived")

    def test_promotion_off_for_plain_prose(self):
        cid = self.cid()
        turn, record = self._run_chat(
            cid,
            [[(0, delta("The capital of France is Paris.")), (0, finish("stop"))]],
            user_text="What is the capital of France?",
        )
        self.assertEqual(_card_events(record), [])

    def test_promotion_derives_checklist_when_user_asks(self):
        # No fence, plain bullet list, but the user asked for a checklist.
        reply = "- one\n- two\n- three\n- four\n"
        cid = self.cid()
        turn, record = self._run_chat(
            cid,
            [[(0, delta(reply)), (0, finish("stop"))]],
            user_text="give me a checklist of steps",
        )
        cards = _card_events(record)
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["type"], "checklist")
        self.assertEqual(len(cards[0]["items"]), 4)
        self.assertTrue(cards[0]["title"].endswith(" (from reply)"))

    def test_promotion_derives_comparison_from_pipe_table(self):
        reply = (
            "| Option | Cost |\n|---|---|\n"
            "| Cat | low |\n| Dog | high |\n"
        )
        cid = self.cid()
        turn, record = self._run_chat(
            cid,
            [[(0, delta(reply)), (0, finish("stop"))]],
            user_text="compare cats and dogs",
        )
        cards = _card_events(record)
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["type"], "comparison")
        self.assertEqual(len(cards[0]["rows"]), 2)
        self.assertTrue(cards[0]["title"].endswith(" (from reply)"))

    def test_promotion_derives_timeline_from_day_markers(self):
        reply = "Day 1: gather\nDay 2: design\nDay 3: build\nDay 4: ship\n"
        cid = self.cid()
        turn, record = self._run_chat(
            cid,
            [[(0, delta(reply)), (0, finish("stop"))]],
            user_text="plan the milestones",
        )
        cards = _card_events(record)
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["type"], "timeline")
        self.assertEqual(len(cards[0]["entries"]), 4)
        self.assertEqual(cards[0]["entries"][0]["when"], "Day 1")

    def test_hostile_markdown_inside_code_block_is_inert(self):
        # A code block contains what looks like a task list.  The parser
        # must not promote it.
        reply = "```\n- [ ] fake\n- [ ] fake\n- [ ] fake\n```\n"
        cid = self.cid()
        turn, record = self._run_chat(
            cid,
            [[(0, delta(reply)), (0, finish("stop"))]],
            user_text="explain with code",
        )
        self.assertEqual(_card_events(record), [])

    def test_one_megabyte_input_is_bounded(self):
        # 2 MiB pad: over the 1 MiB cap, must refuse.
        huge = "x" * (1024 * 1024 + 16)
        comp = extract_text(huge, "")
        self.assertIsNone(comp)
        # A legal-size reply still parses.
        legal = "\n".join(f"- step {i + 1}" for i in range(4)) + "\n"
        comp = extract_text(legal, "give me a checklist")
        self.assertIsNotNone(comp)

    def test_validate_components_rejection_drops_card_keeps_prose(self):
        # Hand-craft a candidate the validator would reject (items=0),
        # monkeypatch extract_text to return it, and check the coordinator
        # does not emit a component event AND keeps the prose.
        bad = {"type": "checklist", "items": []}
        with mock.patch.object(card_promotion, "extract_text", return_value=bad):
            cid = self.cid()
            turn, record = self._run_chat(
                cid,
                [[(0, delta("Some prose stays visible.")), (0, finish("stop"))]],
                user_text="anything",
            )
        self.assertEqual(_card_events(record), [])
        self.assertIn("Some prose stays visible.", record["messages"][-1]["content"])

    def test_invalid_promotion_does_not_emit_status_event(self):
        # The model-fenced-block path emits an invalid_component status on
        # bad fences; the promotion path is silent when its candidate is
        # rejected.  This is intentional: promotion is best-effort.
        bad = {"type": "comparison", "columns": [{"id": "c1", "label": "x"}],
               "rows": [{"id": "r1", "label": "y", "values": []}]}
        with mock.patch.object(card_promotion, "extract_text", return_value=bad):
            cid = self.cid()
            turn, record = self._run_chat(
                cid,
                [[(0, delta("ok")), (0, finish("stop"))]],
                user_text="anything",
            )
        self.assertEqual(_card_events(record), [])
        status = [e for e in self.coord.store.events(cid, 0)
                  if e["turn_id"] == turn and e["type"] == "status"]
        self.assertFalse(any(s["data"].get("state") == "invalid_component"
                             for s in status),
                        "promotion failures stay silent")

    def test_promotion_skipped_in_compare_mode(self):
        # compare/decide modes do their own thing; promotion must not fire.
        cid = self.cid()
        self.worker.decision_response = {
            "answers": {"comparison": {
                "type": "choice", "choice": "a",
                "probabilities": {"a": 0.7, "b": 0.3},
                "rl_agent": {"act_probability": 0.9}, "confidence": 0.7}},
        }
        self.worker.scripts = [[(0, delta("Some explanation text.")), (0, finish("stop"))]]
        turn, record = self.run_turn(
            cid, {"text": "compare a and b", "mode": "compare",
                  "options": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
                  "criteria": "which is faster"})
        # The decision event was emitted; no promoted checklist.
        self.assertEqual(_card_events(record), [])

    def test_user_requested_full_schema_overrides_markdown_promotion(self):
        # A "charts" keyword forces the full schema and the chat path
        # continues to work as before (no regression).
        cid = self.cid()
        self._run_chat(
            cid,
            [[(0, delta("ok")), (0, finish("stop"))]],
            user_text="give me a chart of monthly sales",
        )
        request = self.worker.calls[0]
        self.assertIn(components.SCHEMA_PROMPT, request["messages"][0]["content"])


class CardPromotionUnitTests(unittest.TestCase):
    """Direct tests of the extract_text rules (no coordinator, no model)."""

    def test_dev_set_tuning_targets(self):
        # Load the DEV set and assert the parser is conservative enough:
        # >= 80% of card-worthy prompts produce a valid card, and the
        # 6 plain prompts + 6 near-miss prompts produce no card.
        path = REPO_ROOT / "tests" / "fixtures" / "cards_dev.json"
        data = json.loads(path.read_text())
        hits = 0
        expected_cards = 0
        spurious = 0
        for prompt in data["prompts"]:
            # The "model" emits just the structured markdown (no fence).
            # We hand-craft the typical 2B shape per kind.
            reply = _fake_model_reply(prompt)
            user_text = prompt["text"]
            comp = extract_text(reply, user_text)
            if prompt["category"] == "card-worthy":
                expected_cards += 1
                if comp is not None:
                    try:
                        components.validate_components(
                            {"version": 1, "components": [comp]})
                        hits += 1
                    except Exception:
                        pass
            else:
                if comp is not None:
                    try:
                        components.validate_components(
                            {"version": 1, "components": [comp]})
                        spurious += 1
                    except Exception:
                        pass
        self.assertGreaterEqual(hits, int(0.8 * expected_cards),
                                f"dev hits {hits} / {expected_cards} < 80%")
        self.assertEqual(spurious, 0,
                         f"dev produced {spurious} spurious cards")


def _fake_model_reply(prompt):
    """Render the kind of markdown the model would produce for this prompt.

    These are the same shapes the v0.7.6 qualification recorded: the 2B
    writes rich markdown and the 27B writes a markdown checklist/table
    when it does not emit JSON."""
    kind = prompt.get("kind")
    text = prompt["text"]
    if kind == "checklist":
        return "\n".join(f"- [ ] step {i + 1} for {text[:40]}"
                          for i in range(4)) + "\n"
    if kind == "comparison":
        # table 2x2
        return "| Option | Detail |\n|---|---|\n| A | x |\n| B | y |\n"
    if kind == "timeline":
        return "\n".join(f"Day {i + 1}: phase {i + 1}"
                          for i in range(4)) + "\n"
    if kind == "facts":
        return ("- fact one\n- fact two\n- fact three\n- fact four\n")
    if kind == "none":
        return ("The answer is straightforward: " + text + "\n")
    if kind == "list-in-prose":
        return ("Three things come to mind: " + ", ".join(
            f"item {i}" for i in range(3)) + ". That's the answer.\n")
    if kind == "short-list":
        return "\n".join(f"- {i}" for i in range(3)) + "\n"
    if kind == "code-only":
        return ("```python\nprint('hi')\n```\n")
    return "Sure, here you go.\n"


if __name__ == "__main__":
    unittest.main()
