"""DEV-set tuning check for card_promotion (no model, no GPU).

Every DEV prompt is paired with every reply shape the chat models were seen
to write for its kind, including the failure classes from the spent
held-out runs: explanatory answers with a table or bullets, bullets despite
a request for a paragraph, timelines written as a table or as labelled
phases without dates.  Card-worthy prompts must promote on at least 80% of
(prompt, shape) pairs; no other prompt may ever produce a card.  The real
replies recorded by the chat-model bench (GSM8K and instruction-following
turns) must stay prose.  The HELD-OUT suite, not this file, is the gate.
"""
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_assistant import components  # noqa: E402
from mlx_omarchy_assistant.card_promotion import extract_text  # noqa: E402

EXPLAIN = ("Here is how it works in practice.\n\n"
           "| Aspect | First | Second |\n|---|---|---|\n"
           "| Speed | fast | slow |\n| Cost | low | high |\n\n"
           "Key points:\n- it is simple\n- it is common\n- it scales\n")
SHAPES = {
    "checklist": [
        "## Before you start\n- [ ] book the site\n- [ ] check weather\n"
        "## Gear\n- [ ] tent\n- [x] stove\n",
        "1. **Prepare** the workspace\n2. **Build** the thing\n3. **Test** it\n4. **Ship** it\n",
        "Here you go:\n\n- Sleeping bag\n  - rated to 0 C\n- Water filter\n- Headlamp\n",
    ],
    "comparison": [
        "| Option | Cost | Space |\n|---|---|---|\n| A | low | small |\n| B | high | large |\n",
        "Short answer first.\n\n| Criterion | **A** | **B** |\n|:--|:-:|--:|\n"
        "| Setup | easy | hard |\n| Speed | ok | fast |\n| Ecosystem | big | small |\n\n"
        "- A suits beginners\n- B suits experts\n",
        "Here is how they compare:\n\n### 1. Cost\n*   **Option A:** cheaper up front\n"
        "*   **Option B:** cheaper to run\n\n### 2. Upkeep\n*   **Option A:**\n"
        "    *   needs yearly service\n*   **Option B:** almost none\n\n**Verdict:** it depends.\n",
        "#### The first choice\n- **Price:** low\n- **Speed:** slow\n\n"
        "#### The second choice\n- **Price:** high\n- **Speed:** fast\n",
        "**Setup**\n- quick for one, slow for the other\n\n**Support**\n- community forums\n- paid plans\n",
    ],
    "timeline": [
        "| Week | Milestone |\n|---|---|\n| 1-2 | discovery |\n| 3-6 | build |\n"
        "| 7-8 | pilot |\n| 9 | launch |\n",
        "1. **Discovery** - interviews\n2. **Design** - mockups\n3. **Build** - code\n"
        "4. **Pilot** - two teams\n",
        "### Day 1\nArrive.\n### Day 2\nHike.\n### Day 3\nLeave.\n",
        "- 9:00 standup\n- 11:00 code review\n- 12:00 lunch\n- 14:00 planning\n",
    ],
    "facts": [
        "- Atmosphere: none\n- Distance: 384,400 km\n- Orbit: 27.3 days\n",
        "Here are the facts:\n\n- It is old.\n- It is large.\n",
    ],
    "mixed": [
        "| Metric | A | B |\n|---|---|---|\n| Speed | 1.2 | 0.8 |\n| Quality | high | medium |\n\n- [ ] book the venue\n- [ ] send invites\n",
        "## Plan\n- **First:** research\n- **Then:** draft\n\n## Goals\n- **Quality:** high\n- **Speed:** fast\n",
    ],
    "adversarial": [
        EXPLAIN,
        "- one thing\n- another thing\n- a third thing\n",
        "```python\ndef f():\n    return 1\n```\n- returns one\n- takes no args\n- is pure\n",
        "> Quoted text here.\n\nIt permits:\n- use\n- copy\n- modify\n",
        "It has four stages:\n1. **Start** \u2013 things begin\n2. **Middle** \u2013 things grow\n"
        "3. **Late** \u2013 things slow\n4. **End** \u2013 things stop\n",
        "## How it works\n- **Input:** air comes in\n- **Output:** heat leaves\n\n"
        "## Why it matters\n- **Input:** less energy\n- **Output:** lower bills\n",
        "### Pros\n- flexible hours\n- no commute\n### Cons\n- isolation\n- blurred boundaries\n",
    ],
}


def _card(reply, user_text):
    component = extract_text(reply, user_text)
    if component is None:
        return None
    return components.validate_components({"version": 1, "components": [component]})


class DevTuningTests(unittest.TestCase):
    def test_dev_prompts_across_reply_shapes(self):
        data = json.loads((REPO_ROOT / "tests" / "fixtures" / "cards_dev.json").read_text())
        hits = pairs = 0
        spurious = []
        for prompt in data["prompts"]:
            worthy = prompt["category"] == "card-worthy"
            for reply in SHAPES[prompt["kind"] if worthy else "adversarial"]:
                card = _card(reply, prompt["text"])
                if worthy:
                    pairs += 1
                    hits += card is not None
                elif card is not None:
                    spurious.append(prompt["id"])
        self.assertEqual(spurious, [])
        self.assertGreaterEqual(hits / pairs, 0.8, f"{hits}/{pairs}")

    def test_recorded_bench_replies_stay_prose(self):
        prompts_dir = REPO_ROOT / "scripts" / "bench" / "prompts"
        questions = {
            "gsm": {r["index"]: r["q"] for r in map(json.loads, (prompts_dir / "gsm8k_20.jsonl").read_text().splitlines())},
            "ife": {r["index"]: r["instruction"]
                    for r in map(json.loads, (prompts_dir / "ife_20.jsonl").read_text().splitlines())},
        }
        replies = 0
        for path in sorted((REPO_ROOT / "receipts" / "2026-09-30-chat-model-bench" / "v3").glob("*.json")):
            record = json.loads(path.read_text())
            for part, by_index in questions.items():
                for index, turn in (record.get(part) or {}).items():
                    if turn.get("text"):
                        replies += 1
                        self.assertIsNone(_card(turn["text"], by_index[int(index)]),
                                          f"{path.name} {part} {index}")
        self.assertGreater(replies, 100)


if __name__ == "__main__":
    unittest.main()
