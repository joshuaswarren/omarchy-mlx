"""Coordinator generation slice: schema repair, length stop, shutdown proofs.

UNIT ONLY, protocol-focused. There is no inference here and none is faked as
real: the pair manager is an orchestration fake (start/ensure_context/stop),
``LocalModels.count`` is stubbed to skip the transformers tokenizer, and the
chat worker is a real loopback HTTP server that speaks the SSE wire protocol
from a per-test script. What is exercised is the coordinator's protocol
behavior: prose preservation, one bounded repair attempt, finish_reason
handling, cancel semantics, and shutdown verification. TTS/synthesis is not
imported. Hardware numbers come from the parent's M1 runs, not this file.
"""

import http.server
import io
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_assistant import coordinator as coord  # noqa: E402
from mlx_omarchy_assistant import components  # noqa: E402
from mlx_omarchy_assistant.history import ConversationStore  # noqa: E402

VALID_ENVELOPE = json.dumps(
    {"version": 1,
     "components": [{"type": "checklist",
                     "items": [{"id": "a", "text": "x"}]}]})


def delta(text):
    return json.dumps({"choices": [{"delta": {"content": text}}]})


def finish(reason):
    return json.dumps({"choices": [{"delta": {}, "finish_reason": reason}]})


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 (stdlib API)
        server = self.server
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        if self.path == "/v1/internal/yield":
            # speech-yield control plane: never a chat call
            self.send_response(404)
            self.end_headers()
            return
        if self.path == "/v1/decisions":
            server.decision_calls.append(body)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(server.decision_response).encode())
            return
        server.calls.append(body)
        script = server.next_script()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            for step in script:
                delay, chunk = (step if isinstance(step, tuple) else (0, step))
                if delay:
                    time.sleep(delay)
                self.wfile.write(b"data: " + chunk.encode() + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


class FakeWorker(http.server.ThreadingHTTPServer):
    """Loopback SSE chat worker driven by per-request scripts."""

    daemon_threads = True

    def __init__(self, scripts):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.scripts = list(scripts)
        self.calls = []
        self.decision_calls = []
        self.decision_response = {"answers": {}}
        self.lock = threading.Lock()

    def next_script(self):
        with self.lock:
            if self.scripts:
                return self.scripts.pop(0)
            return [(0, finish("stop"))]

    @property
    def url(self):
        return "http://127.0.0.1:%d" % self.server_address[1]


class FakeManager:
    """Orchestration fake: no weights, no inference, no hardware."""

    def __init__(self, url):
        self.url = url
        self.stop_report = {"stopped": True, "retained": []}
        self.stop_calls = 0
        self.events = []
        self.refuse_after_first_admission = False
        self.context_tokens = 8192

    def start(self):
        return {"chat_url": self.url, "decision_url": self.url,
                "model_paths": {"chat": "/fake/chat", "decision": "/fake/dec"},
                "chat_model": "fake", "context_tokens": self.context_tokens}

    def ensure_context(self, required):
        refused = (self.refuse_after_first_admission
                   and self.events.count("ensure") >= 1)
        self.events.append("ensure")
        if refused or (isinstance(required, int) and required > self.context_tokens):
            return {"ok": False, "reason": "memory", "context_tokens": self.context_tokens}
        return {"ok": True, "context_tokens": self.context_tokens}

    def status(self):
        self.events.append("status")
        return dict(self.start(), ready_offline=True)

    def stop(self):
        self.stop_calls += 1
        return dict(self.stop_report)


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.worker = FakeWorker([])
        self.worker_thread = threading.Thread(target=self.worker.serve_forever,
                                              daemon=True)
        self.worker_thread.start()
        self.manager = FakeManager(self.worker.url)
        patcher = mock.patch.object(coord.LocalModels, "count",
                                    lambda self, path, messages: 16)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.worker.shutdown)
        self.addCleanup(self.worker.server_close)
        self.coord = coord.Coordinator(Path(self.tmp.name), self.manager)

    def tearDown(self):
        self.tmp.cleanup()

    def cid(self):
        return self.coord.store.create(save=False)["id"]

    def run_turn(self, cid, payload, timeout=10):
        turn = self.coord.submit(cid, payload)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            record = self.coord.store.get(cid)
            if record["active_turn"] is None:
                return turn, record
            time.sleep(0.01)
        raise AssertionError("turn did not finish within %ss" % timeout)

    def status_events_for(self, cid, turn):
        return [event for event in self.coord.store.events(cid, 0)
                if event["turn_id"] == turn and event["type"] == "status"]

    def test_prose_turn_completes(self):
        cid = self.cid()
        self.worker.scripts = [[(0, delta("Hello there.")), (0, finish("stop"))]]
        turn, record = self.run_turn(cid, {"text": "hi", "max_tokens": 1024})
        self.assertEqual(record["messages"][-1]["content"], "Hello there.")
        self.assertEqual(record["messages"][-1]["status"], "complete")
        done = [e for e in self.coord.store.events(cid, 0)
                if e["type"] == "done"][-1]
        self.assertFalse(done["data"]["stopped"])
        request = self.worker.calls[0]
        self.assertTrue(request["stream"])
        self.assertEqual(request["max_tokens"], 1024)

    def test_chat_request_serializes_model_path(self):
        cid = self.cid()
        self.worker.scripts = [[(0, delta("Hello there.")), (0, finish("stop"))]]
        path = Path("/tmp/qualify-chat")
        original = self.manager.start

        def start_with_path():
            result = original()
            result["model_paths"] = {"chat": path, "decision": path}
            return result

        self.manager.start = start_with_path
        self.run_turn(cid, {"text": "hi", "max_tokens": 128})
        self.assertEqual(self.worker.calls[0]["model"], str(path))
        self.assertIsInstance(self.worker.calls[0]["model"], str)
    def test_task_default_allowance_replaces_fixed_1024(self):
        cid = self.cid()
        self.worker.scripts = [[(0, delta("Hi.")), (0, finish("stop"))]]
        turn, record = self.run_turn(cid, {"text": "hi"})
        # chat task ceiling without an explicit allowance
        self.assertEqual(self.worker.calls[0]["max_tokens"], 2048)
        generating = [e for e in self.status_events_for(cid, turn)
                      if e["data"].get("state") == "generating"][-1]
        self.assertEqual(generating["data"]["output_tokens"], 2048)

    def test_auto_allowance_backed_by_admitted_context(self):
        cid = self.cid()
        self.worker.scripts = [[(0, delta("Short room left.")), (0, finish("stop"))]]
        patcher = mock.patch.object(coord.LocalModels, "count",
                                    lambda self, path, messages: 7000)
        patcher.start()
        self.addCleanup(patcher.stop)
        turn, record = self.run_turn(cid, {"text": "hi"})
        request = self.worker.calls[0]
        # 8192 admitted - 7000 prompt tokens: the promise shrinks to what the
        # admitted context actually holds, and the status event reports it.
        self.assertEqual(request["max_tokens"], 1192)
        events = self.coord.store.events(cid, 0)
        generating = [e for e in events if e["type"] == "status"
                      and e["data"].get("state") == "generating"][-1]
        self.assertEqual(generating["data"]["output_tokens"], 1192)
        self.assertEqual(record["messages"][-1]["status"], "complete")

    def test_explicit_allowance_refused_never_silently_reduced(self):
        cid = self.cid()
        self.manager.refuse_after_first_admission = True
        self.manager.events.append("ensure")  # arm: the very first admission refuses
        turn, record = self.run_turn(cid, {"text": "hi", "max_tokens": 4096})
        self.assertEqual(self.worker.calls, [], "a refused allowance must "
                         "never reach the worker with a reduced promise")
        errors = [e for e in self.coord.store.events(cid, 0) if e["type"] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertIn("not reduced silently", errors[0]["data"]["message"])
        self.assertIn("4096", errors[0]["data"]["message"])
        self.assertEqual(record["messages"][-1]["status"], "stopped")

    def test_length_finish_signals_truncated_continue(self):
        cid = self.cid()
        self.worker.scripts = [
            [(0, delta("Partial answer")), (0, finish("length"))],
            [(0, delta("more text")), (0, finish("stop"))],
        ]
        turn, record = self.run_turn(cid, {"text": "hi"})
        events = self.status_events_for(cid, turn)
        truncated = [e for e in events if e["data"].get("state") == "output_truncated"]
        self.assertEqual(len(truncated), 1)
        self.assertTrue(truncated[0]["data"]["continue"])
        self.assertEqual(record["messages"][-1]["status"], "complete")
        # Continue is a normal user turn on the same conversation.
        continue_text = "Continue."
        turn2, record = self.run_turn(cid, {"text": continue_text})
        self.assertNotEqual(turn2, turn)
        self.assertEqual(record["messages"][-1]["content"], "more text")
        self.assertEqual(len(record["messages"]), 4)  # two turns, user+assistant each

    def test_ordinary_chat_sends_the_compact_card_schema_and_charts_get_the_full_one(self):
        cid = self.cid()
        self.worker.scripts = [[(0, delta("ok")), (0, finish("stop"))]] * 2
        self.run_turn(cid, {"text": "Summarize the release notes in plain words."})
        self.run_turn(cid, {"text": "Show the numbers as a chart."})
        plain, chart = (call["messages"][0]["content"] for call in self.worker.calls)
        self.assertIn(components.SCHEMA_PROMPT_COMPACT, plain)
        self.assertNotIn(components.SCHEMA_PROMPT, plain)
        self.assertIn(components.SCHEMA_PROMPT, chart)
        # First-answer latency: the compact prompt was measured at ~3 ms per prompt token.
        self.assertLess(len(components.SCHEMA_PROMPT_COMPACT), 800)

    def test_invalid_component_repaired_once(self):
        cid = self.cid()
        bad_block = '{"version": 1, "components": [{"type": "nope"}]}'
        self.worker.scripts = [
            [(0, delta("Before.```assistant-ui\n" + bad_block + "```")),
             (0, finish("stop"))],
            [(0, delta("```assistant-ui\n" + VALID_ENVELOPE + "```")),
             (0, finish("stop"))],
        ]
        turn, record = self.run_turn(cid, {"text": "make a list"})
        self.assertEqual(len(self.worker.calls), 2,
                         "main call plus exactly one repair call")
        main, repair = self.worker.calls
        self.assertLessEqual(repair["max_tokens"], 2048)
        self.assertEqual(len(repair["messages"]), 1)
        self.assertIn("assistant-ui", repair["messages"][0]["content"])
        self.assertIn(bad_block, repair["messages"][0]["content"])
        self.assertNotIn("make a list", repair["messages"][0]["content"])
        self.assertEqual(len(record["messages"][-1]["components"]), 1)
        component = record["messages"][-1]["components"][0]
        self.assertEqual(component["type"], "checklist")
        self.assertEqual(component["turn_id"], turn)
        self.assertIn("Before.", record["messages"][-1]["content"])
        self.assertEqual(record["messages"][-1]["status"], "complete")

    def test_repair_failure_caps_chat_calls_at_two(self):
        cid = self.cid()
        bad_block = '{"version": 1, "components": [{"type": "nope"}]}'
        self.worker.scripts = [
            [(0, delta("Keep this.```assistant-ui\n" + bad_block + "```")),
             (0, finish("stop"))],
            [(0, delta("```assistant-ui\n" + bad_block + "```")),
             (0, finish("stop"))],
        ]
        turn, record = self.run_turn(cid, {"text": "hi"})
        self.assertEqual(len(self.worker.calls), 2)
        invalid = [e for e in self.status_events_for(cid, turn)
                   if e["data"].get("state") == "invalid_component"]
        self.assertEqual(len(invalid), 1)
        self.assertEqual(record["messages"][-1]["components"], [])
        self.assertIn("Keep this.", record["messages"][-1]["content"])
        self.assertEqual(record["messages"][-1]["status"], "complete")

    def test_incomplete_envelope_repaired(self):
        cid = self.cid()
        self.worker.scripts = [
            [(0, delta("A.```assistant-ui\n" + VALID_ENVELOPE[:20])),
             (0, finish("stop"))],
            [(0, delta("```assistant-ui\n" + VALID_ENVELOPE + "```")),
             (0, finish("stop"))],
        ]
        turn, record = self.run_turn(cid, {"text": "hi"})
        self.assertEqual(len(self.worker.calls), 2)
        self.assertEqual(len(record["messages"][-1]["components"]), 1)
        invalid = [e for e in self.status_events_for(cid, turn)
                   if e["data"].get("state") == "invalid_component"]
        self.assertEqual(invalid, [])
        self.assertIn("A.", record["messages"][-1]["content"])

    def test_repair_output_still_validated_never_rendered(self):
        cid = self.cid()
        schema_violation = json.dumps(
            {"version": 1, "components": [{"type": "checklist",
                                           "items": "not-an-array"}]})
        self.worker.scripts = [
            [(0, delta("```assistant-ui\n{\"version\": 1}```")),
             (0, finish("stop"))],
            [(0, delta("```assistant-ui\n" + schema_violation + "```")),
             (0, finish("stop"))],
        ]
        turn, record = self.run_turn(cid, {"text": "hi"})
        self.assertEqual(len(self.worker.calls), 2)
        self.assertEqual(record["messages"][-1]["components"], [])
        invalid = [e for e in self.status_events_for(cid, turn)
                   if e["data"].get("state") == "invalid_component"]
        self.assertEqual(len(invalid), 1)

    def test_valid_component_needs_no_repair(self):
        cid = self.cid()
        self.worker.scripts = [
            [(0, delta("Look:```assistant-ui\n" + VALID_ENVELOPE +
                       "``` Done.")),
             (0, finish("stop"))],
        ]
        turn, record = self.run_turn(cid, {"text": "hi"})
        self.assertEqual(len(self.worker.calls), 1)
        self.assertEqual(len(record["messages"][-1]["components"]), 1)
        self.assertIn("Look:", record["messages"][-1]["content"])
        self.assertIn("Done.", record["messages"][-1]["content"])

    def test_cancel_mid_stream_stops_and_verifies_shutdown(self):
        cid = self.cid()
        script = [(0, delta("part "))]
        script += [(0.2, delta("x")) for _ in range(30)]
        self.worker.scripts = [script]
        turn = self.coord.submit(cid, {"text": "hi"})
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if self.coord.store.get(cid)["messages"][-1]["content"]:
                break
            time.sleep(0.01)
        self.coord.cancel(cid, turn)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if self.coord.store.get(cid)["active_turn"] is None:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("cancelled turn did not finish")
        record = self.coord.store.get(cid)
        self.assertEqual(record["messages"][-1]["status"], "stopped")
        done = [e for e in self.coord.store.events(cid, 0)
                if e["type"] == "done"][-1]
        self.assertTrue(done["data"]["stopped"])
        self.assertTrue(record["messages"][-1]["content"])
        self.coord.close()  # verifies stop report; no orphan turn threads

    def test_repair_counts_admits_and_refreshes_pair(self):
        from mlx_omarchy_assistant import components
        cid = self.cid()
        bad_block = '{"version": 1, "components": [{"type": "nope"}]}'
        self.worker.scripts = [
            [(0, delta("```assistant-ui\n" + bad_block + "```")),
             (0, finish("stop"))],
            [(0, delta("```assistant-ui\n" + VALID_ENVELOPE + "```")),
             (0, finish("stop"))],
        ]
        turn, record = self.run_turn(cid, {"text": "hi"})
        self.assertEqual(len(self.worker.calls), 2)
        self.assertEqual(len(record["messages"][-1]["components"]), 1)
        # admission ran for main + repair; pair refreshed before the repair
        self.assertEqual(self.manager.events,
                         ["ensure", "status", "ensure", "status"])
        repair = self.worker.calls[1]
        self.assertIn(components.SCHEMA_PROMPT, repair["messages"][0]["content"])
        self.assertIn(bad_block, repair["messages"][0]["content"])
        self.assertLessEqual(repair["max_tokens"], 2048)

    def test_repair_admission_refusal_preserves_prose(self):
        cid = self.cid()
        bad_block = '{"version": 1, "components": [{"type": "nope"}]}'
        self.worker.scripts = [
            [(0, delta("Good prose.```assistant-ui\n" + bad_block + "```")),
             (0, finish("stop"))],
        ]
        self.manager.refuse_after_first_admission = True
        turn, record = self.run_turn(cid, {"text": "hi"})
        self.assertEqual(len(self.worker.calls), 1, "refused repair must not "
                         "reach the worker")
        self.assertIn("Good prose.", record["messages"][-1]["content"])
        invalid = [e for e in self.status_events_for(cid, turn)
                   if e["data"].get("state") == "invalid_component"]
        self.assertEqual(len(invalid), 1)
        self.assertEqual(record["messages"][-1]["status"], "complete")

    def test_close_verifies_stop_report_and_threads(self):
        cid = self.cid()
        self.manager.stop_report = {"stopped": False, "retained": ["chat"]}
        with self.assertRaises(RuntimeError) as ctx:
            self.coord.close()
        self.assertIn("shutdown unverified", str(ctx.exception))
        self.manager.stop_report = {"stopped": True, "retained": []}
        self.coord.close()  # second close: nothing left to stop

    def test_close_with_active_turn_cancels_and_joins(self):
        cid = self.cid()
        script = [(0, delta("part "))]
        script += [(0.2, delta("x")) for _ in range(60)]
        self.worker.scripts = [script]
        self.coord.submit(cid, {"text": "hi"})
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if self.coord.store.get(cid)["messages"][-1]["content"]:
                break
            time.sleep(0.01)
        jobs = dict(self.coord.turns)
        self.coord.close()
        for turn, job in jobs.items():
            self.assertFalse(job["thread"].is_alive(), turn)
        self.assertTrue(self.coord.store.get(cid)["active_turn"] is None)



    def test_compare_explanation_that_drops_the_laya_choice_is_marked(self):
        cid = self.cid()
        self.worker.decision_response = {"answers": {"comparison": {
            "type": "choice", "choice": "short",
            "probabilities": {"short": 0.8, "long": 0.2},
            "confidence": 0.6, "rl_agent": {"act_probability": 1.0}}}}
        self.worker.scripts = [[(0, delta("I choose the long answer.")), (0, finish("stop"))]]
        patcher = mock.patch.object(
            coord, "decision_request",
            lambda path, text, options, criteria: {"state": text, "questions": {"comparison": "choice"}})
        patcher.start()
        self.addCleanup(patcher.stop)
        options = [{"id": "short", "label": "short answer"},
                   {"id": "long", "label": "long answer"}]
        self.run_turn(cid, {"text": "latency note", "mode": "compare",
                            "options": options, "criteria": "lower latency"})
        events = self.coord.store.events(cid, 0)
        marked = [e for e in events if e["type"] == "status"
                  and e["data"].get("state") == "disagreement"]
        self.assertEqual(len(marked), 1)
        record = self.coord.store.get(cid)
        self.assertIn("The Laya result is unchanged.", record["messages"][-1]["content"])
        decision = next(e for e in events if e["type"] == "decision")["data"]
        self.assertEqual(decision["choice"], "short")

    def test_decide_mode_batches_typed_questions(self):
        cid = self.cid()
        self.worker.decision_response = {"answers": {
            "dept": {"type": "choice", "choice": "billing",
                     "probabilities": {"billing": 0.9, "technical": 0.1},
                     "confidence": 0.8, "rl_agent": {"act_probability": 1.0}},
            "urg": {"type": "score", "score": 0.5,
                    "legend": {"0": "low", "1": "high"},
                    "probabilities": {"0": 0.6, "1": 0.4},
                    "confidence": 0.2, "rl_agent": {"act_probability": 1.0}}}}
        self.worker.scripts = [[(0, delta("Explains both.")), (0, finish("stop"))]]
        # the real builder fit-checks against Laya's checkpoint tokenizer; the
        # batched-payload shape it returns is what this test pins.
        patcher = mock.patch.object(
            coord, "typed_questions_request",
            lambda path, text, spec: {"state": text,
                                      "questions": {item["id"]: item["type"] for item in spec}})
        patcher.start()
        self.addCleanup(patcher.stop)
        spec = [
            {"id": "dept", "type": "choice", "instructions": "Which department?",
             "options": ["billing", "technical"]},
            {"id": "urg", "type": "score", "instructions": "How urgent?",
             "levels": ["low", "high"]},
        ]
        self.run_turn(cid, {"text": "invoice text", "mode": "decide", "questions": spec})
        # one batched decision call carrying both questions
        self.assertEqual(len(self.worker.decision_calls), 1)
        self.assertEqual(sorted(self.worker.decision_calls[0]["questions"]),
                         ["dept", "urg"])
        self.assertEqual(len(self.worker.calls), 1, "explanation is one chat call")
        events = self.coord.store.events(cid, 0)
        kinds = [e["type"] for e in events]
        self.assertEqual(kinds, ["status", "decision", "status", "text", "done"])
        decision = next(e for e in events if e["type"] == "decision")["data"]
        self.assertEqual(decision["type"], "typed")
        self.assertEqual(decision["model"], "laya-mlx")
        self.assertEqual(len(decision["results"]), 2)
        dept = next(r for r in decision["results"] if r["id"] == "dept")
        self.assertFalse(dept["answer"]["abstained"])
        self.assertEqual(dept["answer"]["choice"], "billing")
        urg = next(r for r in decision["results"] if r["id"] == "urg")
        self.assertEqual(urg["answer"]["levels"], ["low", "high"])
        self.assertEqual(urg["answer"]["score"], 0.5)
        # the chat call explains the validated typed results, never raw output
        roles = [message["role"] for message in self.worker.calls[0]["messages"]]
        self.assertEqual(roles[0], "system")
        self.assertNotIn("system", roles[1:])
        explanation = self.worker.calls[0]["messages"][0]["content"]
        self.assertIn("Laya", explanation)
        self.assertIn("billing", explanation)

    def test_decide_rejects_changed_distribution(self):
        cid = self.cid()
        self.worker.decision_response = {"answers": {
            "dept": {"type": "choice", "choice": "billing",
                     "probabilities": {"billing": 0.5, "sales": 0.5},
                     "confidence": 0.5, "rl_agent": {"act_probability": 1.0}}}}
        spec = [{"id": "dept", "type": "choice", "instructions": "Which?",
                 "options": ["billing", "technical"]}]
        turn, record = self.run_turn(cid, {"text": "invoice text", "mode": "decide",
                                           "questions": spec})
        errors = [e for e in self.coord.store.events(cid, 0) if e["type"] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(record["messages"][-1]["status"], "stopped")

    def test_draft_turn_produces_editable_confirmation(self):
        cid = self.cid()
        draft_json = json.dumps({"criteria": "Travel weight",
                                 "options": [{"label": "Air M3"}, {"label": "Pro M3"}]})
        self.worker.scripts = [[(0, delta(draft_json)), (0, finish("stop"))]]
        turn, record = self.run_turn(cid, {"text": "deciding between two laptops",
                                           "mode": "draft"})
        self.assertEqual(len(self.worker.calls), 1)
        events = self.coord.store.events(cid, 0)
        kinds = [e["type"] for e in events]
        self.assertEqual(kinds, ["status", "text", "status", "done"])
        draft = [e for e in events if e["type"] == "status"
                 and e["data"].get("state") == "comparison_draft"][-1]["data"]
        self.assertEqual(draft["options"], [{"label": "Air M3"}, {"label": "Pro M3"}])
        self.assertEqual(draft["criteria"], "Travel weight")
        self.assertEqual(draft["source"], "deciding between two laptops",
                         "the originating material must ride back to the panel")
        self.assertIn("Air M3", record["messages"][-1]["content"])
        self.assertEqual(record["messages"][-1]["status"], "complete")

    def test_draft_repairs_once_then_reports_unavailable(self):
        cid = self.cid()
        self.worker.scripts = [
            [(0, delta("I think option A wins.")), (0, finish("stop"))],
            [(0, delta("Still prose, no JSON.")), (0, finish("stop"))],
        ]
        turn, record = self.run_turn(cid, {"text": "compare stuff", "mode": "draft"})
        self.assertEqual(len(self.worker.calls), 2, "one draft call plus one repair")
        repair = self.worker.calls[1]
        self.assertIn("not a valid draft", repair["messages"][0]["content"])
        unavailable = [e for e in self.status_events_for(cid, turn)
                       if e["data"].get("state") == "draft_unavailable"]
        self.assertEqual(len(unavailable), 1)
        self.assertIn("Compare options panel", unavailable[0]["data"]["message"])
        # no fabricated options anywhere
        self.assertEqual([e for e in self.coord.store.events(cid, 0)
                          if e["data"].get("options")], [])

    def test_selected_history_and_pinned_constraints_shape_prompt(self):
        cid = self.cid()
        self.worker.scripts = [
            [(0, delta("first answer")), (0, finish("stop"))],
            [(0, delta("second answer")), (0, finish("stop"))],
            [(0, delta("third answer")), (0, finish("stop"))],
        ]
        self.run_turn(cid, {"text": "first question"})
        self.run_turn(cid, {"text": "second question"})
        turn1 = self.coord.store.get(cid)["messages"][0]["turn_id"]
        with self.coord.store.lock:
            # record.context contract: null|list selection + pinned text.
            self.coord.store.records[cid]["context"] = {
                "selected_turn_ids": [turn1],
                "pinned_constraints": "Always answer in French"}
        turn, _record = self.run_turn(cid, {"text": "third question"})
        messages = self.worker.calls[-1]["messages"]
        contents = [m["content"] for m in messages]
        self.assertTrue(any("Always answer in French" in c for c in contents),
                        "pinned constraints ride in the prompt")
        self.assertIn("first question", contents)
        self.assertIn("first answer", contents)
        self.assertNotIn("second question", contents)
        self.assertNotIn("second answer", contents)
        self.assertEqual(messages[-1]["content"], "third question")

    def test_empty_selection_keeps_only_current_turn(self):
        cid = self.cid()
        self.worker.scripts = [
            [(0, delta("first answer")), (0, finish("stop"))],
            [(0, delta("second answer")), (0, finish("stop"))],
        ]
        self.run_turn(cid, {"text": "first question"})
        with self.coord.store.lock:
            self.coord.store.records[cid]["context"] = {
                "selected_turn_ids": [], "pinned_constraints": ""}
        self.run_turn(cid, {"text": "second question"})
        contents = [m["content"] for m in self.worker.calls[-1]["messages"]]
        self.assertNotIn("first question", contents)
        self.assertIn("second question", contents)

    def test_legacy_records_without_context_default_to_full_history(self):
        cid = self.cid()
        self.worker.scripts = [
            [(0, delta("first answer")), (0, finish("stop"))],
            [(0, delta("second answer")), (0, finish("stop"))],
        ]
        self.run_turn(cid, {"text": "first question"})
        self.run_turn(cid, {"text": "second question"})
        contents = [m["content"] for m in self.worker.calls[-1]["messages"]]
        self.assertIn("first question", contents)
        self.assertIn("first answer", contents)

    def test_parse_draft_schema_rejections(self):
        good = {"criteria": " weight ", "options": [{"label": " A "}, {"label": "B"}]}
        parsed = coord.parse_draft(json.dumps(good))
        self.assertEqual(parsed, {"criteria": "weight", "options": [{"label": "A"}, {"label": "B"}]})
        fenced = "```json\n" + json.dumps(good) + "\n```"
        self.assertEqual(coord.parse_draft(fenced)["options"][0]["label"], "A")
        for bad in ('{"options": [{"label": "A"}], "criteria": "x"}',
                    '{"options": [], "criteria": "x"}',
                    '{"options": [{"label": ""}, {"label": "B"}], "criteria": "x"}',
                    '{"options": [{"label": "A"}, {"label": "B"}], "criteria": ""}',
                    '{"options": [{"label": "A"}, {"label": "B"}], "criteria": "x", "extra": 1}',
                    '{"options": [{"label": "A"}, {"label": "a"}], "criteria": "x"}',
                    '{"options": [{"label": "A"}, {"label": "A"}], "criteria": "x"}',
                    'no json here'):
            with self.assertRaises((ValueError, TypeError)):
                coord.parse_draft(bad)

    def test_context_error_blocks_inference_until_repaired(self):
        cid = self.cid()
        with self.coord.store.lock:
            self.coord.store.records[cid]["context_error"] = "selection names unknown turn abc"
        turn, record = self.run_turn(cid, {"text": "hi"})
        self.assertEqual(self.worker.calls, [])
        self.assertEqual(self.worker.decision_calls, [])
        errors = [e for e in self.coord.store.events(cid, 0) if e["type"] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertIn("unknown turn", errors[0]["data"]["message"])
        self.assertEqual(record["messages"][-1]["status"], "stopped")


if __name__ == "__main__":
    unittest.main()
