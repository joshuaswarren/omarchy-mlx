"""Card promotion contract.

Coordinator tests drive the SSE wire-protocol harness from
test_assistant_generation (fake worker, fake pair manager): the model fence
wins, text streams before the card, a validator rejection keeps the prose,
compare turns never promote.  Rule tests call extract_text directly:
promotion needs a user request for the artifact, prose requests and fenced
code are inert, and hostile input costs linear time.
"""

import sys
import time
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_assistant import card_promotion, components  # noqa: E402
from mlx_omarchy_assistant.card_promotion import extract_text, requested_kinds  # noqa: E402

from tests import test_assistant_generation as harness  # noqa: E402

delta, finish = harness.delta, harness.finish


def _valid(component):
    return components.validate_components({"version": 1, "components": [component]})


class CoordinatorPromotionTests(unittest.TestCase):
    setUp = harness.GenerationTests.setUp
    cid = harness.GenerationTests.cid
    tearDown = harness.GenerationTests.tearDown
    run_turn = harness.GenerationTests.run_turn

    def chat(self, reply, user_text):
        cid = self.cid()
        self.worker.scripts = [[(0, delta(reply)), (0, finish("stop"))]]
        turn, record = self.run_turn(cid, {"text": user_text})
        return cid, turn, record

    def cards(self, record):
        return record["messages"][-1].get("components") or []

    def test_model_fence_wins_over_promotion(self):
        reply = ("- one\n- two\n- three\n\n```assistant-ui\n"
                 + harness.VALID_ENVELOPE + "\n```\n")
        _, _, record = self.chat(reply, "give me a checklist")
        cards = self.cards(record)
        self.assertEqual(len(cards), 1)
        self.assertEqual([i["text"] for i in cards[0]["items"]], ["x"])
        self.assertNotIn("(from reply)", cards[0].get("title", ""))

    def test_card_follows_the_streamed_text(self):
        reply = "Pack these:\n- [ ] tent\n- [x] stove\n- [ ] water\n"
        cid, turn, record = self.chat(reply, "packing list for camping")
        kinds = [e["type"] for e in self.coord.store.events(cid, 0)
                 if e["turn_id"] == turn and e["type"] in ("text", "component")]
        self.assertEqual(kinds[-1], "component")
        self.assertIn("text", kinds[:-1])
        card = self.cards(record)[0]
        self.assertEqual(card["type"], "checklist")
        self.assertEqual(card["title"], "Checklist (from reply)")
        self.assertEqual([i["done"] for i in card["items"]], [False, True, False])
        self.assertEqual(record["messages"][-1]["content"], reply)

    def test_explanatory_table_stays_prose(self):
        reply = "| Protocol | Ordered |\n|---|---|\n| TCP | yes |\n| UDP | no |\n"
        _, _, record = self.chat(reply, "What is the difference between TCP and UDP?")
        self.assertEqual(self.cards(record), [])

    def test_validator_rejection_drops_card_and_keeps_prose(self):
        bad = {"type": "comparison", "columns": [{"id": "c1", "label": "x", "kind": "text"}],
               "rows": [{"id": "r1", "label": "y", "values": ["a", "b"]}], "title": "T"}
        with mock.patch.object(card_promotion, "extract_text", return_value=bad):
            cid, turn, record = self.chat("Some prose stays visible.", "compare a and b")
        self.assertEqual(self.cards(record), [])
        self.assertEqual(record["messages"][-1]["content"], "Some prose stays visible.")
        self.assertEqual(record["messages"][-1]["status"], "complete")
        self.assertFalse(any(e["data"].get("state") == "invalid_component"
                             for e in self.coord.store.events(cid, 0)
                             if e["turn_id"] == turn and e["type"] == "status"))

    def test_compare_mode_never_promotes(self):
        cid = self.cid()
        self.worker.decision_response = {"answers": {"comparison": {
            "type": "choice", "choice": "a", "probabilities": {"a": 0.7, "b": 0.3},
            "rl_agent": {"act_probability": 0.9}, "confidence": 0.7}}}
        self.worker.scripts = [[(0, delta("- one\n- two\n- three\n")), (0, finish("stop"))]]
        _, record = self.run_turn(cid, {
            "text": "compare a and b, as a checklist", "mode": "compare",
            "options": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
            "criteria": "which is faster"})
        self.assertFalse(any(c.get("title", "").endswith("(from reply)")
                             for c in self.cards(record)))


class RuleTests(unittest.TestCase):
    def test_request_words_select_kinds_in_mention_order(self):
        self.assertEqual(requested_kinds("Compare iOS and Android, then a rollout timeline"),
                         ["comparison", "timeline"])
        self.assertEqual(requested_kinds("What are the steps to bake bread?"), ["checklist"])
        self.assertEqual(requested_kinds("Explain how DNS works, with the key steps"), [])
        self.assertEqual(requested_kinds("Give me the steps to make tea in a paragraph."), [])
        self.assertEqual(requested_kinds("Why is the sky blue?"), [])
        self.assertEqual(requested_kinds("Describe the stages and phases of sleep"), [])
        self.assertEqual(requested_kinds("Outline the project phases"), ["timeline"])
        self.assertEqual(requested_kinds("Plan my week, without using a list"), [])

    def test_plain_list_needs_a_request(self):
        reply = "- red\n- blue\n- green\n"
        self.assertIsNone(extract_text(reply, "Name three primary colours."))
        card = extract_text(reply, "Make a checklist of colours to buy")
        self.assertEqual(card["type"], "checklist")
        _valid(card)

    def test_comparison_from_headed_sections(self):
        by_option = ("### 1. The first plan\n*   **Price:** low\n*   **Speed:** slow\n\n"
                     "### 2. The second plan\n*   **Price:** high\n*   **Speed:** fast\n"
                     "*   **Extras:** many\n")
        card = extract_text(by_option, "compare the two plans")
        self.assertEqual([c["label"] for c in card["columns"]], ["Item", "Price", "Speed"])
        self.assertEqual(card["rows"][1]["values"], ["The second plan", "high", "fast"])
        by_criterion = ("**Cost**\n- **Kindle:**\n    - device is expensive\n    - books are cheap\n"
                        "- **Paperback:** no device\n\n**Weight**\n- **Kindle:** light\n"
                        "- **Paperback:** heavy\n")
        card = extract_text(by_criterion, "Kindle versus paperback")
        self.assertEqual(card["rows"][0]["values"],
                         ["Cost", "device is expensive books are cheap", "no device"])
        unshared = "## Setup\n- quick\n## Support\n- forums\n- paid plans\n"
        card = extract_text(unshared, "contrast the two tools")
        self.assertEqual(card["rows"][1]["values"], ["Support", "forums; paid plans"])
        for reply, user in ((by_option, "compare the two plans"),
                            (by_criterion, "Kindle versus paperback"),
                            (unshared, "contrast the two tools")):
            _valid(extract_text(reply, user))
        self.assertIsNone(extract_text(by_option, "Explain how the plans work"))
        self.assertIsNone(extract_text("### Only one\n- **Price:** low\n", "compare it"))

    def test_comparison_needs_a_table_or_sections(self):
        self.assertIsNone(extract_text("- cats: cheap\n- dogs: costly\n- fish: cheap\n",
                                       "compare cats, dogs and fish"))
        table = ("| Pet | Cost | Noise |\n|:--|--:|---|\n| **Cat** | low | quiet |\n"
                 "| Dog | high | loud |\n| Fish | low | |\n")
        card = extract_text(table, "compare cats, dogs and fish")
        self.assertEqual([c["label"] for c in card["columns"]], ["Pet", "Cost", "Noise"])
        self.assertEqual(card["rows"][0]["values"], ["Cat", "low", "quiet"])
        _valid(card)

    def test_ragged_or_one_column_tables_are_refused(self):
        ragged = "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 |\n"
        self.assertIsNone(extract_text(ragged, "put it in a table"))
        narrow = "| a |\n|---|\n| 1 |\n| 2 |\n"
        self.assertIsNone(extract_text(narrow, "put it in a table"))

    def test_timeline_from_labels_markers_tables_and_headings(self):
        labelled = ("1. **Discovery** - interview users\n2. **Pilot**: two teams\n"
                    "3. **Rollout** \u2013 everyone\n4. **Cleanup**: remove flags\n")
        card = extract_text(labelled, "Schedule the migration in phases")
        self.assertEqual([e["when"] for e in card["entries"]],
                         ["Discovery", "Pilot", "Rollout", "Cleanup"])
        timed = "9:00 standup\n11:30 code review\n12:00 lunch\n"
        card = extract_text(timed, "plan my monday")
        self.assertEqual([e["when"] for e in card["entries"]], ["9:00", "11:30", "12:00"])
        self.assertEqual(card["entries"][0]["text"], "standup")
        table = ("| Week | Goal |\n|---|---|\n| Weeks 1-4 | plan |\n"
                 "| Weeks 5-10 | build |\n| Weeks 11-12 | review |\n")
        card = extract_text(table, "OKR cycle timeline")
        self.assertEqual(card["entries"][1], {"id": card["entries"][1]["id"],
                                              "when": "Weeks 5-10", "text": "build"})
        headings = "## Week 1: internal\n...\n## Week 2: beta\n...\n## Week 3: launch\n"
        card = extract_text(headings, "roadmap for launch")
        self.assertEqual([e["when"] for e in card["entries"]], ["Week 1", "Week 2", "Week 3"])
        for text, user in ((labelled, "phases"), (timed, "plan my monday"),
                           (table, "timeline"), (headings, "roadmap")):
            _valid(extract_text(text, user))

    def test_facts_accept_two_items(self):
        card = extract_text("- Moon: no atmosphere\n- Distance: 384,400 km\n",
                            "two facts about the moon")
        self.assertEqual([c["heading"] for c in card["cards"]], ["Moon", "Distance"])
        _valid(card)

    def test_fenced_code_and_unterminated_fences_are_inert(self):
        inside = "```\n- [ ] a\n- [ ] b\n- [ ] c\n```\n"
        self.assertIsNone(extract_text(inside, "checklist please"))
        open_fence = "~~~~md\n- [ ] a\n- [ ] b\n- [ ] c\n"
        self.assertIsNone(extract_text(open_fence, "checklist please"))
        after = "```\ncode\n```\n- [ ] a\n- [ ] b\n- [ ] c\n"
        self.assertEqual(len(extract_text(after, "checklist please")["items"]), 3)

    def test_markup_stays_text(self):
        reply = ("- <script>alert(1)</script>\n- <img src=x onerror=alert(1)>\n"
                 "- [link](javascript:alert(1))\n")
        card = extract_text(reply, "to-do list")
        self.assertEqual(card["items"][0]["text"], "<script>alert(1)</script>")
        _valid(card)

    def test_oversize_items_are_truncated_to_validator_bounds(self):
        reply = "\n".join(f"- {'x' * 5000} {i}" for i in range(80))
        card = extract_text(reply, "checklist")
        self.assertEqual(len(card["items"]), 50)
        _valid(card)
        wide = "| " + " | ".join(f"h{i}" for i in range(9)) + " |\n|" + "---|" * 9 + "\n"
        wide += ("| " + " | ".join("v" for _ in range(9)) + " |\n") * 3
        self.assertIsNone(extract_text(wide, "table"))

    def test_input_over_one_mebibyte_is_refused(self):
        self.assertIsNone(extract_text("- a\n" * 300_000, "checklist"))

    def test_hostile_one_mebibyte_inputs_are_linear(self):
        near = card_promotion.MAX_INPUT_BYTES - 64
        cases = [
            "- a" + " " * near + "b",
            "| a" + " " * near + "b",
            "a | b\n" + "|" + " " * near + "-",
            "# " + "x " * (near // 2),
            "```\n" * (near // 4),
            "- [ ] " + "[" * (near - 8),
            "9:" * (near // 2),
            "**" * (near // 2),
        ]
        users = ["checklist, table, timeline and facts", "shopping" + "g" * 200_000 + " list"]
        for reply in cases:
            for user in users:
                started = time.monotonic()
                extract_text(reply, user)
                self.assertLess(time.monotonic() - started, 2.0, reply[:20])


if __name__ == "__main__":
    unittest.main()
