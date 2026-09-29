"""Dev-set tuner for card_promotion rules.

Uses simulated model outputs (the shapes the v0.7.6 qualification
recorded) to exercise the rules against the 24-prompt DEV set.  This
lets us tune the parser without holding the GPU.  The HELD-OUT set is
the source of truth for the final gate.
"""
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_assistant import components  # noqa: E402
from mlx_omarchy_assistant.card_promotion import extract_text  # noqa: E402


# Map kind -> simulated model reply (the shape the actual model emits
# per the v0.7.6 quality logs).
def _reply_for(prompt):
    kind = prompt.get("kind")
    if kind == "checklist":
        # The 2B writes rich markdown checklists and ignores the fence;
        # the 27B does the same plus sometimes emits JSON.  We model the
        # markdown-only path here (the worst case for promotion).
        return "\n".join(f"- [ ] step {i + 1}"
                         for i in range(5)) + "\n"
    if kind == "comparison":
        return ("| Option | Cost | Space |\n"
                "|---|---|---|\n"
                "| Cat | low | small |\n"
                "| Dog | high | large |\n")
    if kind == "timeline":
        return "\n".join(f"Day {i + 1}: phase {i + 1}"
                         for i in range(5)) + "\n"
    if kind == "facts":
        return ("- fact one from my note\n"
                "- fact two from my note\n"
                "- fact three from my note\n")
    if kind == "none":
        return "The answer is " + prompt["text"].split("?")[0] + ".\n"
    if kind == "list-in-prose":
        return ("Three things come to mind: a, b, c. That's the answer.\n")
    if kind == "short-list":
        return "\n".join(f"- {x}" for x in ("red", "blue", "green")) + "\n"
    if kind == "code-only":
        return "```python\nprint('hi')\n```\n"
    return "Sure.\n"


class DevTunerTests(unittest.TestCase):
    def test_dev_set_conservative_targets(self):
        path = REPO_ROOT / "tests" / "fixtures" / "cards_dev.json"
        data = json.loads(path.read_text())
        valid_card_kinds = {"checklist", "comparison", "timeline", "facts"}
        hits = 0
        expected_cards = 0
        spurious = 0
        per_kind = {}
        for prompt in data["prompts"]:
            reply = _reply_for(prompt)
            comp = extract_text(reply, prompt["text"])
            # Use the rules' per-kind intent to score.
            user_text = prompt["text"]
            if prompt["category"] == "card-worthy":
                expected_cards += 1
                if comp is not None:
                    try:
                        components.validate_components(
                            {"version": 1, "components": [comp]})
                        hits += 1
                        per_kind[prompt["kind"]] = (
                            per_kind.get(prompt["kind"], 0) + 1)
                    except Exception as exc:
                        print(f"validation failed for {prompt['id']}: {exc}")
            else:
                if comp is not None:
                    try:
                        components.validate_components(
                            {"version": 1, "components": [comp]})
                        if comp.get("type") in valid_card_kinds:
                            spurious += 1
                    except Exception:
                        pass
        # Targets on the simulated-reply DEV set.
        # >= 80% of card-worthy promote, zero spurious.
        self.assertGreaterEqual(
            hits, int(0.8 * expected_cards),
            f"dev hits {hits} / {expected_cards} < 80%")
        self.assertEqual(spurious, 0, f"spurious {spurious} > 0")
        # Per-kind sanity: all four kinds fire at least once.
        for kind in ("checklist", "comparison", "timeline", "facts"):
            self.assertGreater(per_kind.get(kind, 0), 0,
                               f"kind {kind} never produced a card")


if __name__ == "__main__":
    unittest.main()
