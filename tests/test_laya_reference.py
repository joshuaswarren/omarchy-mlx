"""CPU reference tests: the mlx port must reproduce pinned upstream semantics.

Reference side: the pristine pinned upstream files (rl_common.py,
rl_agent_api.py from convaiinnovations/laya revision
1c5edc17a7acd8701df6fc341c0d179f1c62c982) run on torch CPU exactly as
upstream ships them — fp32 compute over the fp16 checkpoint tensors, which
is what RLAgent does on CPU (autocast is CUDA-only upstream).

Port side: this package's LayaEngine on mlx CPU with --allow-cpu semantics
(require_gpu=False, float32).

Both sides must produce identical public envelopes up to fp op-order noise:
exact input token counts, exact choice argmax, 4-decimal probabilities
within 2e-4 of upstream, act_probability within 2e-2 (the action head is
the one upstream-sensitive output; the upstream pack's own validation
accepts the same class of drift, see aac6fef/laya-mlx validation.json).

Environment:
  LAYA_UPSTREAM_DIR  dir holding pinned upstream rl_common.py/rl_agent_api.py
  LAYA_REF_CKPT_DIR  dir holding the pristine HF snapshot (root variant)
  LAYA_MLX_CKPT      converted checkpoint dir (mlx_omarchy_laya.convert)
Optional:
  LAYA_PACK_DIR      aac6fef/laya-mlx pack snapshot (same layout); if set,
                     the same battery must also pass loading the pack directly.
"""

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "serve"))

import pytest  # noqa: E402

pytest.importorskip("mlx.core")  # the laya engine imports mlx
from mlx_omarchy_laya.api import LayaEngine  # noqa: E402

UPSTREAM = os.environ.get("LAYA_UPSTREAM_DIR")
REF_CKPT = os.environ.get("LAYA_REF_CKPT_DIR")
CONVERTED = os.environ.get("LAYA_MLX_CKPT")
PACK = os.environ.get("LAYA_PACK_DIR")

PROB_TOL = 2e-4
ACT_TOL = 2e-2

FIXTURE_STATE = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
}
FIXTURE_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "low", "medium", "high", "critical"],
    },
    "is_refund_request": {"type": "noul", "instructions": "Is this a refund request?", "criteria": None},
}

BATTERY = [
    ("email_json_state", FIXTURE_STATE, FIXTURE_QUESTIONS),
    ("plain_string_state", "The server has been returning 503s for the EU checkout fleet since 09:12 UTC; "
                           "revenue impact is roughly 4k EUR per hour and growing.", {
        "severity": {"type": "score", "instructions": "Rate the severity.",
                     "criteria": ["trivial", "minor", "moderate", "major", "catastrophic"]},
        "is_outage": {"type": "noul", "instructions": "Is production currently degraded?",
                      "criteria": {"false": "no degradation", "true": "active degradation"}},
    }),
    ("empty_state_two_choice", "", {
        "coin": {"type": "choice", "instructions": "Pick one.", "criteria": ["heads", "tails"]},
        "color": {"type": "choice", "instructions": "Pick a color.",
                  "criteria": {"red": None, "green": None, "blue": None}},
    }),
    ("mask_token_and_unicode", "ユーザーは [MASK] トークンを含むテキストを送りました — attention: réclamation nº 7",
     {"lang": {"type": "choice", "instructions": "Which language is this?", "criteria": {
         "japanese": None, "french": None, "mixed": None, "other": None}},
      "complaint": {"type": "noul", "instructions": "Is this a complaint?", "criteria": None}}),
    ("long_state_truncation", " ".join("filler token %d" % i for i in range(2000)), {
        "topic": {"type": "choice", "instructions": "What is the dominant topic?",
                  "criteria": {"technology": None, "business": None, "sports": None, "noise": None}},
    }),
    ("dict_instructions_and_many_options", {"note": "triage"}, {
        "routing": {"type": "choice",
                    "instructions": {"goal": "route", "allow_multiple": False},
                    "criteria": {("opt%d" % i): ("description %d" % i) for i in range(12)}},
        "quality": {"type": "score", "instructions": "Score the writing quality.",
                    "criteria": ["poor", "fair", "good", "excellent"]},
    }),
]


def _upstream_envelope(upstream_dir, ckpt, state, questions):
    sys.path.insert(0, str(upstream_dir))
    for mod in [m for m in list(sys.modules) if m.startswith("rl_")]:
        del sys.modules[mod]
    import rl_agent_api  # noqa: PLC0415  (pinned upstream file, loaded per test)

    agent = rl_agent_api.RLAgent(ckpt, device="cpu")
    return agent.system_one(state, questions)


def _assert_envelopes_close(test, ref, got, context):
    test.assertEqual(got["usage"]["input_tokens"], ref["usage"]["input_tokens"],
                     "%s: input token count diverged" % context)
    test.assertEqual(got["usage"]["output_tokens"], 0, "%s: output_tokens must stay 0" % context)
    test.assertEqual(got["model"], "rl-agent")
    worst_prob, worst_act = 0.0, 0.0
    for qid, rq in ref["answers"].items():
        gq = got["answers"][qid]
        test.assertEqual(gq["type"], rq["type"], "%s/%s: answer type diverged" % (context, qid))
        if rq["type"] == "choice":
            keys = list(rq["probabilities"])
            test.assertEqual(list(gq["probabilities"]), keys, "%s/%s: label order diverged" % (context, qid))
            test.assertEqual(gq["choice"], rq["choice"], "%s/%s: argmax diverged" % (context, qid))
        if rq["type"] == "score":
            test.assertEqual(gq["score"], rq["score"], "%s/%s: expected score diverged at 4dp" % (context, qid))
            test.assertEqual(gq["legend"], rq["legend"])
        if rq["type"] == "noul":
            test.assertAlmostEqual(gq["noul"], rq["noul"], delta=PROB_TOL * 3,
                                   msg="%s/%s: noul diverged" % (context, qid))
        if rq["type"] in ("choice", "score"):  # upstream noul answers carry no probabilities block
            for k, v in rq["probabilities"].items():
                g = gq["probabilities"][k]
                d = abs(g - v)
                worst_prob = max(worst_prob, d)
                if d > PROB_TOL + 5e-4:  # 4dp rounding can flip on ~1e-4 boundaries
                    test.fail("%s/%s prob %s: ref %s got %s (delta %.6f)" % (context, qid, k, v, g, d))
        if rq["type"] == "score":
            test.assertAlmostEqual(gq["score"], rq["score"], delta=PROB_TOL * 3, msg=context)
        if rq["type"] in ("choice", "score"):  # upstream noul answers carry no confidence block
            test.assertAlmostEqual(gq["confidence"], rq["confidence"], delta=PROB_TOL * 3,
                                   msg="%s/%s: confidence diverged" % (context, qid))
        d_act = abs(gq["rl_agent"]["act_probability"] - rq["rl_agent"]["act_probability"])
        worst_act = max(worst_act, d_act)
        test.assertLessEqual(d_act, ACT_TOL, "%s/%s: act_probability diverged" % (context, qid))
    return worst_prob, worst_act


@unittest.skipUnless(UPSTREAM and REF_CKPT and CONVERTED and
                     all(Path(p).exists() for p in (UPSTREAM, REF_CKPT, CONVERTED)),
                     "set LAYA_UPSTREAM_DIR, LAYA_REF_CKPT_DIR, LAYA_MLX_CKPT "
                     "(see CONTRACT.md) to run the upstream comparison")
class UpstreamReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = LayaEngine(CONVERTED, dtype="float32", require_gpu=False)

    def test_battery_matches_upstream(self):
        worst = (0.0, 0.0)
        for name, state, questions in BATTERY:
            ref = _upstream_envelope(UPSTREAM, REF_CKPT, state, questions)
            got = self.engine.system_one(state, questions)
            worst = tuple(max(w, t) for w, t in zip(worst, _assert_envelopes_close(self, ref, got, name)))
        print("\nlaya reference battery: worst prob delta %.2e, worst act delta %.2e" % worst)

    def test_json_state_and_string_state_token_accounting(self):
        # usage.input_tokens is the padded-batch real-token total upstream; a
        # single-question call must therefore match a re-run token-for-token.
        state = {"a": 1}
        q = {"x": {"type": "noul", "instructions": "i", "criteria": None}}
        ref = _upstream_envelope(UPSTREAM, REF_CKPT, state, q)
        got = self.engine.system_one(state, q)
        self.assertEqual(got["usage"]["input_tokens"], ref["usage"]["input_tokens"])


@unittest.skipUnless(PACK and Path(PACK).exists(), "set LAYA_PACK_DIR to also verify the aac6fef/laya-mlx pack")
class PackLoadTests(unittest.TestCase):
    """The catalog pack must load directly and agree with the source checkpoint."""

    @classmethod
    def setUpClass(cls):
        cls.engine = LayaEngine(PACK, dtype="float32", require_gpu=False)

    def test_pack_matches_upstream_on_email_case(self):
        ref = _upstream_envelope(UPSTREAM, REF_CKPT, FIXTURE_STATE, FIXTURE_QUESTIONS)
        got = self.engine.system_one(FIXTURE_STATE, FIXTURE_QUESTIONS)
        _assert_envelopes_close(self, ref, got, "pack-email")


if __name__ == "__main__":
    unittest.main()
