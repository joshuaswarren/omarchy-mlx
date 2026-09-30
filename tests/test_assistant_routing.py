"""Focused unit tests for the routing module.

The held-out suite test (`tests/test_assistant_routing_suite.py`) pins
the suite shape and rejects automatic mode by default. This file
exercises the routing module's internal logic with a fake Laya worker:
deadline behavior, timeout accounting, invalid output, over-budget
material, threshold logic, flag off/on, and that ordinary chat latency
is unaffected (the routing call returns before the GPU is acquired).
"""

import json
import sys
import tempfile
import uuid
import threading
import time
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "serve"))

from mlx_omarchy_assistant import routing  # noqa: E402
from mlx_omarchy_assistant.routing import (  # noqa: E402
    ROUTING_POLICY,
    RoutingPolicy,
    SyncWorker,
    WARM_DEADLINE_SECONDS,
    evaluate_route,
    fit_route_question,
    pending_outcome,
)
from mlx_omarchy_assistant.coordinator import (  # noqa: E402
    Coordinator,
    _routing_gate_enabled,
)


# Inject a tiny fake tokenizer before any routing import so production
# code path stays unchanged: every evaluate_route test passes the
# tokenizer explicitly through `fit_route_question` keyword. Without it,
# the default path tries to scan the live venv's pair lock for a
# converted Laya checkpoint and refuses routing.
class _FakeTokenizer:
    mask_token = "[MASK]"

    def __init__(self):
        self.cls_token_id = 0
        self.sep_token_id = 1
        self.mask_token_id = 2
        self.pad_token_id = 3

    def encode(self, text):
        # Cheap stub: ~1 token per whitespace-delimited word plus
        # ~1 sub-token per 4 chars of punctuation. Real Laya uses BPE
        # with vocab ~30000 so this stays well within the 512-token cap
        # for normal-length routing instructions.
        tokens = []
        for word in text.split():
            tokens.append(word[:8])  # truncate long tokens to 1 id
            if len(word) > 8:
                # sub-tokens for the rest
                rest = word[8:]
                while rest:
                    tokens.append(rest[:6])
                    rest = rest[6:]
        if not tokens and text:
            tokens = [text[:8]]
        return tokens


_FAKE_TOKENIZER = _FakeTokenizer()


# Patch the loader so production code never tries to scan disk during tests.
routing._load_default_tokenizer = lambda: _FAKE_TOKENIZER  # type: ignore[assignment]


# ---------------------------------------------------------------- fake worker


class FakeWorker:
    """A worker that returns a fixed response and records the call."""

    def __init__(self, response=None, raise_after=None, sleep_seconds=0.0):
        self.response = response or self._default_response()
        self.raise_after = raise_after
        self.sleep_seconds = sleep_seconds
        self.calls = []

    def _default_response(self):
        return {
            "answers": {
                "route": {
                    "type": "choice",
                    "choice": "conversation",
                    "probabilities": {"conversation": 0.7, "structured_decision": 0.2, "clarify": 0.1},
                    "rl_agent": {"act_probability": 0.9},
                    "confidence": 0.5,
                }
            }
        }

    def call(self, payload, deadline_seconds):
        self.calls.append((payload, deadline_seconds))
        if self.raise_after is not None:
            time.sleep(min(self.raise_after, deadline_seconds))
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        if self.raise_after is not None and time.monotonic() - self.last_start >= self.raise_after:
            raise RuntimeError("deadline")
        return self.response


def _good_response(route, prob, margin=0.4, act=0.7):
    """Construct a valid Laya-style choice response."""
    other = (1.0 - prob) / 2.0
    probs = {"conversation": other, "structured_decision": other, "clarify": other}
    probs[route] = prob
    if margin:
        ordered = sorted(probs.values(), reverse=True)
        diff = ordered[0] - ordered[1]
        if diff < margin:
            probs[route] += margin - diff
            total = sum(probs.values())
            probs = {k: v / total for k, v in probs.items()}
    return {
        "answers": {
            "route": {
                "type": "choice",
                "choice": route,
                "probabilities": probs,
                "rl_agent": {"act_probability": act},
                "confidence": 0.5,
            }
        }
    }


def _bad_response(reason: str):
    if reason == "missing_choice":
        return {"answers": {"route": {"type": "choice"}}}
    if reason == "wrong_type":
        return {"answers": {"route": {"type": "score"}}}
    if reason == "bad_probs_keys":
        return {"answers": {"route": {"type": "choice", "choice": "conversation",
                                      "probabilities": {"foo": 1.0}}}}
    if reason == "negative_prob":
        return {"answers": {"route": {"type": "choice", "choice": "conversation",
                                      "probabilities": {"conversation": 1.2,
                                                          "structured_decision": -0.1,
                                                          "clarify": -0.1}}}}
    if reason == "sum_off":
        return {"answers": {"route": {"type": "choice", "choice": "conversation",
                                      "probabilities": {"conversation": 0.5,
                                                          "structured_decision": 0.2,
                                                          "clarify": 0.2}}}}
    if reason == "missing_act":
        return {"answers": {"route": {"type": "choice", "choice": "conversation",
                                      "probabilities": {"conversation": 0.7,
                                                          "structured_decision": 0.2,
                                                          "clarify": 0.1}}}}
    if reason == "bad_choice":
        return {"answers": {"route": {"type": "choice", "choice": "garbage",
                                      "probabilities": {"conversation": 0.7,
                                                          "structured_decision": 0.2,
                                                          "clarify": 0.1},
                                      "rl_agent": {"act_probability": 0.7}}}}
    return {}


# ---------------------------------------------------------------- fit checks


class FitRouteQuestionTests(unittest.TestCase):
    def test_empty_input_refuses(self):
        self.assertEqual(fit_route_question(""), (False, None))
        self.assertEqual(fit_route_question("   "), (False, None))
        self.assertEqual(fit_route_question(None), (False, None))  # type: ignore[arg-type]

    def test_huge_input_refuses(self):
        huge = "x" * (1024 * 1024 + 1)
        self.assertEqual(fit_route_question(huge), (False, None))


# ---------------------------------------------------------------- evaluate_route


class EvaluateRouteTests(unittest.TestCase):
    def test_warm_deadline_default(self):
        self.assertEqual(WARM_DEADLINE_SECONDS, 0.250)

    def test_returns_route_on_clean_pass(self):
        worker = FakeWorker(_good_response("structured_decision", prob=0.6, margin=0.3, act=0.7))
        out = evaluate_route("Pick between X and Y. Options: X; Y. Criteria: lowest cost.",
                             worker=worker)
        self.assertEqual(out.route, "structured_decision")
        self.assertEqual(out.reason, "")
        self.assertGreaterEqual(out.act_probability, ROUTING_POLICY.act_min)
        self.assertGreaterEqual(out.runner_up_margin, ROUTING_POLICY.margin_min)

    def test_threshold_miss_on_low_selected_prob(self):
        worker = FakeWorker(_good_response("conversation", prob=0.4, margin=0.3, act=0.9))
        out = evaluate_route("Tell me a story.", worker=worker)
        self.assertIsNone(out.route)
        self.assertEqual(out.reason, "threshold_miss")

    def test_threshold_miss_on_tight_margin(self):
        response = _good_response("structured_decision", prob=0.5, margin=0.05, act=0.7)
        worker = FakeWorker(response)
        out = evaluate_route("Pick between X and Y. Options: X; Y. Criteria: lowest cost.",
                             worker=worker)
        self.assertIsNone(out.route)
        self.assertEqual(out.reason, "threshold_miss")

    def test_threshold_miss_on_low_act_probability(self):
        worker = FakeWorker(_good_response("conversation", prob=0.7, margin=0.4, act=0.3))
        out = evaluate_route("Tell me a story.", worker=worker)
        self.assertIsNone(out.route)
        self.assertEqual(out.reason, "threshold_miss")

    def test_does_not_treat_confidence_as_correctness(self):
        # Even with confidence=0.0 the thresholds above matter; act < act_min fails.
        response = _good_response("conversation", prob=0.7, margin=0.4, act=0.3)
        response["answers"]["route"]["confidence"] = 0.0
        worker = FakeWorker(response)
        out = evaluate_route("Tell me a story.", worker=worker)
        self.assertEqual(out.reason, "threshold_miss")

    def test_does_not_invert_act_probability(self):
        # act_probability high means answer; low means escalate. Make sure
        # the policy uses act directly, not 1 - act.
        low_act = _good_response("structured_decision", prob=0.7, margin=0.4, act=0.1)
        worker = FakeWorker(low_act)
        out = evaluate_route("Pick between X and Y. Options: X; Y. Criteria: cost.", worker=worker)
        self.assertIsNone(out.route)
        self.assertEqual(out.reason, "threshold_miss")

    def test_invalid_output_returns_skip(self):
        for reason in ("missing_choice", "wrong_type", "bad_probs_keys", "negative_prob",
                       "missing_act", "bad_choice"):
            worker = FakeWorker(_bad_response(reason))
            out = evaluate_route("anything", worker=worker)
            self.assertIsNone(out.route, msg=reason)
            self.assertEqual(out.reason, "invalid_output", msg=reason)

    def test_sum_off_is_invalid_distribution(self):
        worker = FakeWorker(_bad_response("sum_off"))
        out = evaluate_route("anything", worker=worker)
        self.assertIsNone(out.route)
        self.assertEqual(out.reason, "invalid_distribution")

    def test_timed_out_response_returns_skip_with_timed_out_flag(self):
        worker = FakeWorker({"timed_out": True})
        out = evaluate_route("anything", worker=worker)
        self.assertIsNone(out.route)
        self.assertTrue(out.timed_out)
        self.assertEqual(out.reason, "deadline_miss")

    def test_over_budget_refuses_with_no_call(self):
        worker = FakeWorker(_good_response("conversation", prob=0.7))
        huge = "x" * (1024 * 1024 + 1)
        out = evaluate_route(huge, worker=worker)
        self.assertIsNone(out.route)
        self.assertEqual(out.reason, "material_does_not_fit")
        self.assertEqual(worker.calls, [])

    def test_over_budget_shortcuts_before_worker(self):
        # Anything that fit_route_question rejects never reaches the worker.
        worker = FakeWorker(_good_response("structured_decision", prob=0.9))
        out = evaluate_route("", worker=worker)
        self.assertEqual(out.reason, "material_does_not_fit")
        self.assertEqual(worker.calls, [])

    def test_policy_override_is_honored(self):
        # Strict policy with very high p_min: even a confident answer fails.
        strict = RoutingPolicy(version="x", question_text=ROUTING_POLICY.question_text,
                              p_min=0.99, margin_min=0.99, act_min=0.99)
        worker = FakeWorker(_good_response("conversation", prob=0.7, margin=0.5, act=0.9))
        out = evaluate_route("Tell me about cooking.", worker=worker, policy=strict)
        self.assertIsNone(out.route)
        self.assertEqual(out.reason, "threshold_miss")


# ---------------------------------------------------------------- pending_outcome


class PendingOutcomeTests(unittest.TestCase):
    """The async surface that the coordinator uses."""

    def test_async_fast_response(self):
        class FastWorker:
            def __init__(self):
                self.calls = 0
            def call_async(self, payload, deadline):
                self.calls += 1
                f = threading.Event()
                f.result = lambda: _good_response("structured_decision", prob=0.7, margin=0.4, act=0.8)
                f.done = lambda: True
                return f

        outcome, handle = pending_outcome(
            "Pick between X and Y. Options: X; Y. Criteria: lowest cost.",
            worker_factory=FastWorker,
        )
        self.assertIsNone(handle)
        self.assertEqual(outcome.route, "structured_decision")

    def test_async_timed_out_holds_handle(self):
        class SlowWorker:
            def call_async(self, payload, deadline):
                f = threading.Event()
                f.done = lambda: False
                f.result = lambda: _good_response("conversation", prob=0.7)
                return f

        outcome, handle = pending_outcome(
            "anything", worker_factory=SlowWorker, deadline_seconds=0.001
        )
        self.assertIsNotNone(handle)
        self.assertTrue(outcome.timed_out)
        self.assertEqual(outcome.reason, "deadline_miss")

    def test_async_over_budget_no_call(self):
        class CountingWorker:
            def __init__(self):
                self.calls = 0
            def call_async(self, payload, deadline):
                self.calls += 1
                return threading.Event()

        cw = CountingWorker()
        outcome, handle = pending_outcome("", worker_factory=lambda: cw)
        self.assertIsNone(handle)
        self.assertEqual(outcome.reason, "material_does_not_fit")
        self.assertEqual(cw.calls, 0)


# ---------------------------------------------------------------- coordinator gate


class _ManagerWithPair:
    """Minimal pair manager stub for the routing-gate helper."""

    def __init__(self, status):
        self._status = status

    def status(self):
        return self._status

    def start(self):
        return {"model_paths": {"decision": "/tmp/fake-decision"}, "decision_url": "http://127.0.0.1:1/"}


class RoutingGateFlagTests(unittest.TestCase):
    def test_gate_off_when_no_pair(self):
        manager = _ManagerWithPair(None)
        self.assertFalse(_routing_gate_enabled(manager))

    def test_gate_off_when_pair_has_no_evidence(self):
        manager = _ManagerWithPair({"extension": {}})
        self.assertFalse(_routing_gate_enabled(manager))

    def test_gate_off_when_evidence_lacks_routing_block(self):
        manager = _ManagerWithPair({"extension": {"selection_evidence": {"quality": 1.0}}})
        self.assertFalse(_routing_gate_enabled(manager))

    def test_gate_off_when_receipt_missing(self):
        manager = _ManagerWithPair({"extension": {"selection_evidence": {"routing": {
            "gate": "on", "suite_sha256": "abc", "policy_version": "1",
        }}}})
        self.assertFalse(_routing_gate_enabled(manager))

    def test_gate_off_when_policy_version_mismatch(self):
        manager = _ManagerWithPair({"extension": {"selection_evidence": {"routing": {
            "gate": "on", "suite_sha256": "abc", "receipt": "/p/r.md", "policy_version": "2",
        }}}})
        self.assertFalse(_routing_gate_enabled(manager))

    def test_gate_on_when_all_fields_present_and_match(self):
        manager = _ManagerWithPair({"extension": {"selection_evidence": {"routing": {
            "gate": "on", "suite_sha256": "abc", "receipt": "/p/r.md",
            "policy_version": ROUTING_POLICY.version,
        }}}})
        self.assertTrue(_routing_gate_enabled(manager))


class CoordinatorSubmitAutoTests(unittest.TestCase):
    """The submit() entrypoint refuses automatic mode until the gate is ON."""

    def _stopped_manager(self):
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

        return Stopped()

    def _on_manager(self):
        m = self._stopped_manager()
        m.status = lambda: {"extension": {"selection_evidence": {"routing": {
            "gate": "on", "suite_sha256": "abc", "receipt": "/p/r.md",
            "policy_version": ROUTING_POLICY.version,
        }}}}  # type: ignore[assignment]
        return m

    def test_auto_mode_rejected_when_gate_off(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        app = Coordinator(Path(directory.name), self._stopped_manager())
        self.addCleanup(app.close)
        with self.assertRaises(ValueError):
            app.submit("unused", {"text": "Pick the right option.", "mode": "auto"})

    def test_auto_mode_routes_to_chat_when_router_returns_conversation(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        manager = self._on_manager()

        class FakeModels:
            def __init__(self):
                self.calls = []
            def count(self, path, messages):
                return 1
            def decision(self, pair, payload):
                self.calls.append(payload)
                return _good_response("conversation", prob=0.7, margin=0.4, act=0.8)
            def close_connection(self):
                pass

        app = Coordinator(Path(directory.name), manager)
        self.addCleanup(app.close)
        app.models = FakeModels()  # type: ignore[assignment]

        cid = uuid.uuid4().hex
        turn = cid = app.store.create()["id"]
        app.submit(cid, {"text": "Tell me a story.", "mode": "auto"})
        # The turn ran with mode=chat; routing event was emitted.
        # Wait for the worker thread to emit its events.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            events = app.store.events(cid, 0)
            if any(e["type"] == "routing" for e in events):
                break
            time.sleep(0.02)
        events = app.store.events(cid, 0)
        self.assertTrue(any(e["type"] == "routing" for e in events),
                        "routing event missing")
        routing_event = next(e for e in events if e["type"] == "routing")
        self.assertEqual(routing_event["data"]["route"], "conversation")
        self.assertEqual(routing_event["data"]["policy_version"], ROUTING_POLICY.version)
        self.assertTrue(routing_event["data"]["use_chat_model_available"])

    def test_auto_mode_routes_to_compare_with_options(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        manager = self._on_manager()

        class FakeModels:
            def __init__(self):
                self.calls = []
            def count(self, path, messages):
                return 1
            def decision(self, pair, payload):
                self.calls.append(payload)
                return _good_response("structured_decision", prob=0.7, margin=0.4, act=0.8)
            def close_connection(self):
                pass

        app = Coordinator(Path(directory.name), manager)
        self.addCleanup(app.close)
        app.models = FakeModels()  # type: ignore[assignment]

        cid = uuid.uuid4().hex
        # Routing wants compare, but options are not supplied -> the
        # coordinator must downgrade to chat, never invent options.
        cid = app.store.create()["id"]
        app.submit(cid, {"text": "Pick between X and Y.", "mode": "auto"})

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            events = app.store.events(cid, 0)
            if any(e["type"] == "routing" for e in events):
                break
            time.sleep(0.02)
        events = app.store.events(cid, 0)
        routing_event = next(e for e in events if e["type"] == "routing")
        # The router said structured_decision; that's the routing record
        # but the actual dispatch was downgraded because options are absent.
        self.assertEqual(routing_event["data"]["route"], "structured_decision")

    def test_auto_mode_routes_to_compare_when_options_present(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        manager = self._on_manager()

        # Stub the routing module's evaluate_route so we can verify the
        # coordinator dispatches into the compare path without needing a
        # real Laya tokenizer on disk.
        from mlx_omarchy_assistant import routing as routing_module
        from mlx_omarchy_assistant import coordinator as coord_module
        original_evaluate = routing_module.evaluate_route
        original_coord_evaluate = coord_module.evaluate_route
        original_decision_request = coord_module.decision_request

        def stub_evaluate_route(text, *, worker, policy=None,
                                deadline_seconds=routing_module.WARM_DEADLINE_SECONDS):
            return routing_module.RoutingOutcome(
                route="structured_decision",
                reason="",
                probabilities={"conversation": 0.1, "structured_decision": 0.8, "clarify": 0.1},
                runner_up_margin=0.7,
                act_probability=0.9,
                latency_ms=1.0,
                timed_out=False,
            )

        def stub_decision_request(model_path, text, options, criteria):
            return {"state": text, "questions": {"comparison": {
                "type": "choice",
                "instructions": criteria,
                "criteria": {opt["id"]: opt["label"] for opt in options},
            }}}

        routing_module.evaluate_route = stub_evaluate_route
        coord_module.evaluate_route = stub_evaluate_route
        coord_module.decision_request = stub_decision_request
        try:
            class FakeManager:
                def status(self):
                    return {'extension': {'selection_evidence': {'routing': {'gate': 'on', 'suite_sha256': 'abc', 'receipt': '/p/r.md', 'policy_version': ROUTING_POLICY.version}}},
                            'model_paths': {'chat': '/tmp/c', 'decision': '/tmp/d'},
                            'chat_url': 'http://127.0.0.1:1/', 'decision_url': 'http://127.0.0.1:1/',
                            'context_tokens': 4096}
                def start(self):
                    return {'model_paths': {'chat':'/tmp/c', 'decision':'/tmp/d'},
                            'chat_url':'http://127.0.0.1:1/', 'decision_url':'http://127.0.0.1:1/',
                            'context_tokens': 4096}
                def stop(self):
                    return {'stopped': True, 'retained': []}
                def ensure_context(self, t):
                    return {'ok': True, 'context_tokens': max(t, 4096), 'requested': t, 'changed': False}

            class FakeModels:
                def count(self, path, messages):
                    return 1
                def decision(self, pair, payload):
                    qids = list((payload.get("questions") or {}).keys())
                    if "comparison" in qids:
                        crit = payload["questions"]["comparison"]["criteria"]
                        keys = list(crit.keys())
                        p_each = 1.0 / len(keys)
                        return {"answers": {"comparison": {
                            "type": "choice",
                            "choice": keys[0],
                            "probabilities": {k: p_each for k in keys},
                            "rl_agent": {"act_probability": 0.9},
                            "confidence": 0.5,
                        }}}
                    return {}
                def close_connection(self):
                    pass

            app = Coordinator(Path(directory.name), FakeManager())
            self.addCleanup(app.close)
            app.models = FakeModels()  # type: ignore[assignment]

            cid = app.store.create()["id"]
            app.submit(cid, {"text": "Pick between X and Y. Options: X; Y. Criteria: lowest cost.", "mode": "auto",
                               "options": [{"id": "X", "label": "X"}, {"id": "Y", "label": "Y"}],
                               "criteria": "lowest cost"})
        finally:
            routing_module.evaluate_route = original_evaluate
            coord_module.evaluate_route = original_coord_evaluate
            coord_module.decision_request = original_decision_request

        # The decision path was taken: a decision event is in the stream.
        deadline = time.monotonic() + 2.0
        decision_seen = False
        while time.monotonic() < deadline:
            events = app.store.events(cid, 0)
            if any(e["type"] == "decision" for e in events):
                decision_seen = True
                break
            time.sleep(0.02)
        self.assertTrue(decision_seen, "decision event not emitted on auto->compare")


class OrdinaryChatUnaffectedTests(unittest.TestCase):
    """Ordinary chat must not wait for a cold Laya."""

    def test_chat_submit_does_not_call_decision(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)

        class CountingManager:
            def __init__(self):
                self.decision_calls = 0
            def status(self):
                return {"model_paths": {"chat": "/tmp/fake-chat", "decision": "/tmp/fake-decision"},
                        "chat_url": "http://127.0.0.1:1/",
                        "decision_url": "http://127.0.0.1:1/",
                        "context_tokens": 4096}
            def start(self):
                return {"model_paths": {"chat": "/tmp/fake-chat", "decision": "/tmp/fake-decision"},
                        "chat_url": "http://127.0.0.1:1/",
                        "decision_url": "http://127.0.0.1:1/",
                        "context_tokens": 4096}
            def stop(self):
                return {"stopped": True, "retained": []}
            def ensure_context(self, required_tokens):
                return {"ok": True, "context_tokens": max(required_tokens, 4096),
                        "requested": required_tokens, "changed": False}

        class CountingModels:
            def __init__(self):
                self.decision_calls = 0
                self.chat_calls = 0
            def count(self, path, messages):
                return 1
            def decision(self, pair, payload):
                self.decision_calls += 1
                return {}
            def chat(self, pair, messages, max_tokens, cancel, yield_headers=None):
                self.chat_calls += 1
                yield ("delta", "ok")
                yield ("finish", "stop")
            def close_connection(self):
                pass

        mgr = CountingManager()
        app = Coordinator(Path(directory.name), mgr)
        self.addCleanup(app.close)
        app.models = CountingModels()  # type: ignore[assignment]

        # Chat mode: no routing call, no decision call.
        cid = uuid.uuid4().hex
        cid = app.store.create()["id"]
        app.submit(cid, {"text": "Hi", "mode": "chat"})

        # Wait briefly to see if any decision was issued.
        time.sleep(0.10)
        self.assertEqual(app.models.decision_calls, 0)
        self.assertGreaterEqual(app.models.chat_calls, 1)


if __name__ == "__main__":
    unittest.main()