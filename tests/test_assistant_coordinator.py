import json
import sys
import threading
import tempfile
import time
import unittest
import unittest.mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serve"))

from mlx_omarchy_assistant import coordinator as coordinator_module
from mlx_omarchy_assistant.coordinator import Coordinator, ConversationStore, BusyError, DecisionInputError, LocalModels, _RoutingWorker, validate_decision, validate_score_answer, explanation_disagrees, format_decision_event


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.home = Path(self.directory.name)
        self.store = ConversationStore(self.home)

    def test_history_requires_opt_in_and_delete_removes_saved_data(self):
        private = self.store.create()
        self.store.begin(private['id'], 'private message')
        self.assertEqual(ConversationStore(self.home).list(), [])
        saved = self.store.create(save=True)
        turn = self.store.begin(saved['id'], 'keep this')
        self.store.emit(saved['id'], turn, 'text', {'text': 'saved answer'})
        self.store.finish(saved['id'], turn)
        restored = ConversationStore(self.home)
        messages = restored.get(saved['id'])['messages']
        self.assertEqual([m['content'] for m in messages], ['keep this', 'saved answer'])
        restored.delete(saved['id'])
        self.assertEqual(ConversationStore(self.home).list(), [])
    def test_history_summary_counts_messages_and_tracks_updates(self):
        cid = self.store.create(save=True)['id']
        before = self.store.list()[0]
        self.assertEqual(before['message_count'], 0)
        turn = self.store.begin(cid, 'hello')
        self.store.emit(cid, turn, 'text', {'text': 'answer'})
        self.store.finish(cid, turn)
        after = ConversationStore(self.home).list()[0]
        self.assertEqual(after['message_count'], 2)
        self.assertGreaterEqual(after['updated_at'], before['updated_at'])

    def test_busy_cancel_and_late_result_do_not_overwrite_new_turn(self):
        cid = self.store.create()['id']
        first = self.store.begin(cid, 'first')
        with self.assertRaises(BusyError):
            self.store.begin(cid, 'overlap')
        self.store.emit(cid, first, 'text', {'text': 'partial'})
        self.store.cancel(cid, first)
        self.assertFalse(self.store.emit(cid, first, 'text', {'text': 'late'}))
        self.store.finish(cid, first, stopped=True)
        second = self.store.begin(cid, 'second')
        self.assertFalse(self.store.emit(cid, first, 'done', {}))
        self.store.emit(cid, second, 'text', {'text': 'new'})
        self.store.finish(cid, second)
        self.assertEqual([m['content'] for m in self.store.get(cid)['messages']], ['first', 'partial', 'second', 'new'])

    def test_resume_replays_events_without_starting_work(self):
        cid = self.store.create()['id']
        turn = self.store.begin(cid, 'hello')
        self.store.emit(cid, turn, 'text', {'text': 'one'})
        cursor = self.store.events(cid, 0)[-1]['sequence']
        self.store.emit(cid, turn, 'text', {'text': ' two'})
        resumed = self.store.events(cid, cursor)
        self.assertEqual([e['data']['text'] for e in resumed], [' two'])
        self.assertEqual(self.store.get(cid)['active_turn'], turn)

    def test_restart_marks_interrupted_turn_stopped(self):
        cid = self.store.create(save=True)['id']
        turn = self.store.begin(cid, 'question')
        self.store.emit(cid, turn, 'text', {'text': 'partial'})
        restored = ConversationStore(self.home).get(cid)
        self.assertIsNone(restored['active_turn'])
        self.assertEqual(restored['messages'][-1]['status'], 'stopped')

    def test_path_like_id_cannot_read_or_delete_other_files(self):
        for cid in ('../private', '/tmp/private', '', 'a' * 100):
            with self.assertRaises(ValueError):
                self.store.get(cid)


class DecisionTests(unittest.TestCase):
    INVALID_PROBABILITIES = '^Decision worker returned invalid probabilities$'
    INVALID_SCORE = '^Decision worker returned an invalid score$'

    def test_preserves_options_and_rejects_invalid_distribution(self):
        options = [{'id': 'a', 'label': 'Keep data'}, {'id': 'b', 'label': 'Delete data'}]
        result = {'type': 'choice', 'choice': 'a', 'probabilities': {'a': 0.7, 'b': 0.3},
                  'confidence': 0.1, 'rl_agent': {'act_probability': 0.9}}
        self.assertEqual(validate_decision(result, options)['choice'], 'a')
        for invalid in ({'a': 0.7, 'c': 0.3}, {'a': float('nan'), 'b': 0.3}, {'a': 0.9, 'b': 0.9}):
            with self.assertRaises(DecisionInputError):
                validate_decision(dict(result, probabilities=invalid), options)

    def test_abstention_signal_is_not_entropy_confidence(self):
        options = [{'id': 'a', 'label': 'A'}, {'id': 'b', 'label': 'B'}]
        result = {'type': 'choice', 'choice': 'a', 'probabilities': {'a': 1.0, 'b': 0.0},
                  'confidence': 1.0, 'rl_agent': {'act_probability': 0.01}}
        self.assertTrue(validate_decision(result, options)['abstained'])

    def test_rejects_malformed_rl_agent_and_huge_ints(self):
        options = [{'id': 'a', 'label': 'A'}, {'id': 'b', 'label': 'B'}]
        choice = {'type': 'choice', 'choice': 'a', 'probabilities': {'a': 0.7, 'b': 0.3},
                  'confidence': 0.5, 'rl_agent': {'act_probability': 0.9}}
        levels = ['low', 'high']
        score = {'type': 'score', 'score': 0.3, 'legend': {'0': 'low', '1': 'high'},
                 'probabilities': {'0': 0.7, '1': 0.3}, 'confidence': 0.5,
                 'rl_agent': {'act_probability': 0.9}}
        for fn, good, args in ((validate_decision, choice, options),
                               (validate_score_answer, score, levels)):
            for rl_agent in (None, [], 'bad', 1, True):
                with self.subTest(validator=fn.__name__, rl_agent=rl_agent):
                    with self.assertRaisesRegex(DecisionInputError, self.INVALID_PROBABILITIES):
                        fn(dict(good, rl_agent=rl_agent), args)
            first_key, second_key = list(good['probabilities'])
            for sign, huge in (('pos', 10 ** 400), ('neg', -(10 ** 400))):
                for field, malformed in (
                    ('probability', dict(good, probabilities={first_key: huge, second_key: 0.3})),
                    ('act_probability', dict(good, rl_agent={'act_probability': huge})),
                    ('confidence', dict(good, confidence=huge)),
                ):
                    with self.subTest(validator=fn.__name__, huge_int=field, sign=sign):
                        with self.assertRaisesRegex(DecisionInputError, self.INVALID_PROBABILITIES):
                            fn(malformed, args)
            for field, malformed in (
                ('act_probability', dict(good, rl_agent={'act_probability': float('nan')})),
                ('confidence', dict(good, confidence=float('nan'))),
            ):
                with self.subTest(validator=fn.__name__, nan=field):
                    with self.assertRaisesRegex(DecisionInputError, self.INVALID_PROBABILITIES):
                        fn(malformed, args)

    def test_score_range_accepts_bounds_and_rejects_malformed_values(self):
        scales = (
            (['low', 'high'],
             {'0': 'low', '1': 'high'},
             {'0': 0.5, '1': 0.5},
             (0, 1, 0.5),
             (-0.01, 1.01, 10 ** 400, -(10 ** 400), float('nan'), float('inf'), True)),
            (['low', 'mid', 'high'],
             {'0': 'low', '1': 'mid', '2': 'high'},
             {'0': 0.25, '1': 0.5, '2': 0.25},
             (2, 1.5),
             (2.01,)),
        )
        for levels, legend, probabilities, valid_scores, invalid_scores in scales:
            base = {'type': 'score', 'score': 0.5, 'legend': legend,
                    'probabilities': probabilities, 'confidence': 0.5,
                    'rl_agent': {'act_probability': 0.9}}
            for valid in valid_scores:
                with self.subTest(validator='validate_score_answer', levels=len(levels), valid_score=valid):
                    validated = validate_score_answer(dict(base, score=valid), levels)
                    self.assertEqual(validated['score'], valid)
                    self.assertIs(type(validated['score']), type(valid))
            for invalid in invalid_scores:
                with self.subTest(validator='validate_score_answer', levels=len(levels), invalid_score=invalid):
                    with self.assertRaisesRegex(DecisionInputError, self.INVALID_SCORE):
                        validate_score_answer(dict(base, score=invalid), levels)


class WorkerLifetimeTests(unittest.TestCase):
    STUB_PAIR = {
        "decision_url": "http://127.0.0.1:1",
        "chat_url": "http://127.0.0.1:1",
        "chat_model": "stub-chat",
        "context_tokens": 4096,
        "model_paths": {"decision": "/nonexistent/decision", "chat": "/nonexistent/chat"},
    }

    class _RecordingManager:
        def __init__(self):
            self.events = []

        def status(self):
            return None

        def start(self):
            self.events.append("start")
            return dict(WorkerLifetimeTests.STUB_PAIR)

        def stop(self):
            self.events.append("stop")
            return {"stopped": True}

    def _make_coordinator(self, manager):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        app = Coordinator(Path(directory.name), manager)
        self.addCleanup(app.close)
        return app

    def test_failed_shutdown_finishes_turn_but_blocks_new_work(self):
        class FailedManager:
            def start(self):
                raise RuntimeError('worker connection lost')

            def stop(self):
                raise RuntimeError('worker still alive')

        with tempfile.TemporaryDirectory() as directory:
            app = Coordinator(Path(directory), FailedManager())
            cid = app.store.create()['id']
            turn = app.store.begin(cid, 'hello')
            app.gpu.acquire()
            app._run(cid, turn, {}, 1, {'cancel': threading.Event()})
            self.assertIsNone(app.store.get(cid)['active_turn'])
            self.assertTrue(app.gpu.acquire(blocking=False))
            app.gpu.release()
            with self.assertRaisesRegex(RuntimeError, 'worker still alive'):
                app.submit(cid, {'text': 'must not overlap'})
            app.closed.set()
            app.watch.join(3)

    def test_closed_routing_worker_returns_error_dict(self):
        class StubModels:
            def decision(self, pair, payload):
                return {"answers": {}}

            def close_connection(self):
                pass

        worker = _RoutingWorker(manager=None)
        worker.models = StubModels()
        worker.close()
        reply = worker.call({}, {}, 1.0)
        self.assertIsInstance(reply, dict)
        self.assertEqual(list(reply.keys()), ["error"])
        self.assertIsInstance(reply["error"], str)
        self.assertTrue(reply["error"])

    def test_auto_submit_racing_close_does_not_restart_workers(self):
        events = []

        class RacingManager:
            def __init__(self):
                self.coordinator = None

            def status(self):
                return None

            def start(self):
                events.append("start")
                if events.count("start") == 1:
                    self.coordinator.close()
                return {
                    "decision_url": "http://127.0.0.1:1",
                    "chat_url": "http://127.0.0.1:1",
                    "chat_model": "stub-chat",
                    "model_paths": {"decision": "/nonexistent/decision", "chat": "/nonexistent/chat"},
                }

            def stop(self):
                events.append("stop")
                return {"stopped": True}

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        manager = RacingManager()
        app = Coordinator(Path(directory.name), manager)
        manager.coordinator = app
        self.addCleanup(app.close)
        cid = app.store.create()["id"]

        with unittest.mock.patch.dict("os.environ", {"MLX_OMARCHY_ROUTING": ""}), \
                unittest.mock.patch("mlx_omarchy_assistant.routing.fit_route_question",
                                    return_value=(True, None)) as fit_mock:
            with self.assertRaisesRegex(RuntimeError, "shutting down"):
                app.submit(cid, {
                    "mode": "auto",
                    "text": "Help me choose between the tram and the ferry.",
                })

        fit_mock.assert_called_once()
        with app.lock:
            threads = [job["thread"] for job in app.turns.values() if "thread" in job]
        for thread in threads:
            thread.join(timeout=2)
        time.sleep(0.05)

        self.assertIn("stop", events)
        self.assertNotIn("start", events[events.index("stop") + 1:], events)
        self.assertEqual(app.routing_status()["last_reason"], "worker_error")
        self.assertIsNone(app.store.get(cid)["active_turn"])
        self.assertTrue(app.gpu.acquire(blocking=False))
        app.gpu.release()

    def test_close_before_pair_start_blocks_worker_launch(self):
        closed_at = {}

        class StatusHookManager(self._RecordingManager):
            def __init__(self):
                super().__init__()
                self.on_status = None

            def status(self):
                if self.on_status is not None:
                    hook, self.on_status = self.on_status, None
                    hook()
                return None

        manager = StatusHookManager()
        app = self._make_coordinator(manager)
        cid = app.store.create()["id"]

        def close_once():
            app.close()
            closed_at["index"] = len(manager.events)

        manager.on_status = close_once
        with unittest.mock.patch.dict("os.environ", {"MLX_OMARCHY_ROUTING": ""}), \
                unittest.mock.patch("mlx_omarchy_assistant.routing.fit_route_question",
                                    return_value=(True, None)):
            with self.assertRaisesRegex(RuntimeError, "shutting down"):
                app.submit(cid, {
                    "mode": "auto",
                    "text": "Help me choose between the tram and the ferry.",
                })

        self.assertIn("index", closed_at)
        self.assertNotIn("start", manager.events[closed_at["index"]:], manager.events)
        self.assertIsNone(app.store.get(cid)["active_turn"])
        self.assertTrue(app.gpu.acquire(blocking=False))
        app.gpu.release()

    def test_close_before_turn_lock_refuses_without_writing_store(self):
        manager = self._RecordingManager()
        app = self._make_coordinator(manager)
        cid = app.store.create()["id"]

        caller = threading.current_thread()
        real_lock = app.lock
        triggered = [False]

        class RacingLock:
            def __enter__(self):
                if not triggered[0] and threading.current_thread() is caller:
                    triggered[0] = True
                    app.close()
                return real_lock.__enter__()

            def __exit__(self, *args):
                return real_lock.__exit__(*args)

            def __getattr__(self, name):
                return getattr(real_lock, name)

        app.lock = RacingLock()
        with self.assertRaisesRegex(RuntimeError, "shutting down"):
            app.submit(cid, {"mode": "chat", "text": "hello"})

        self.assertTrue(triggered[0])
        self.assertIn("stop", manager.events)
        self.assertNotIn("start", manager.events)
        record = app.store.get(cid)
        self.assertEqual(record["messages"], [])
        self.assertIsNone(record["active_turn"])
        self.assertTrue(app.gpu.acquire(blocking=False))
        app.gpu.release()

    def test_close_cancels_inflight_auto_route_pair_start(self):
        start_entered = threading.Event()
        second_cancel = threading.Event()
        release_start = threading.Event()
        close_done = threading.Event()

        class BlockingManager(self._RecordingManager):
            def __init__(self):
                super().__init__()
                self._cancel = threading.Event()

            def start(self):
                self._cancel.clear()
                pair = super().start()
                start_entered.set()
                release_start.wait(timeout=5)
                self.events.append("started")
                return pair

            def cancel(self):
                self.events.append("cancel")
                self._cancel.set()
                if self.events.count("cancel") >= 2:
                    second_cancel.set()
                return True

        manager = BlockingManager()
        app = self._make_coordinator(manager)
        self.addCleanup(release_start.set)
        self.addCleanup(close_done.set)
        cid = app.store.create()["id"]
        other_cid = app.store.create()["id"]

        submit_outcome = {}
        close_outcome = {}

        def run_submit():
            try:
                submit_outcome["turn"] = app.submit(cid, {
                    "mode": "auto",
                    "text": "Help me choose between the tram and the ferry.",
                })
            except Exception as exc:
                submit_outcome["error"] = exc

        def run_close():
            try:
                app.close()
            except Exception as exc:
                close_outcome["error"] = exc
            finally:
                close_done.set()

        def wait_for_close_then_fit(*args, **kwargs):
            close_done.wait(timeout=5)
            return (True, None)

        with unittest.mock.patch.dict("os.environ", {"MLX_OMARCHY_ROUTING": ""}), \
                unittest.mock.patch("mlx_omarchy_assistant.routing.fit_route_question",
                                    side_effect=wait_for_close_then_fit):
            submit_thread = threading.Thread(target=run_submit, daemon=True)
            submit_thread.start()
            self.assertTrue(start_entered.wait(timeout=5), "manager.start() was not reached")

            close_thread = threading.Thread(target=run_close, daemon=True)
            close_thread.start()
            try:
                self.assertTrue(second_cancel.wait(timeout=2), "close() did not repeat cancel for in-flight start")
                self.assertTrue(manager._cancel.is_set())
                self.assertTrue(close_thread.is_alive(), "close() returned while manager.start() was still in flight")
                self.assertTrue(app.closed.is_set(), "close() did not set closed before waiting")
                with self.assertRaisesRegex(RuntimeError, "shutting down"):
                    app.submit(other_cid, {"mode": "chat", "text": "hello"})
            finally:
                release_start.set()
                close_thread.join(timeout=5)
                submit_thread.join(timeout=5)

        self.assertFalse(close_thread.is_alive(), "close thread did not finish")
        self.assertFalse(submit_thread.is_alive(), "submit thread did not finish")
        self.assertNotIn("error", close_outcome)
        self.assertIsInstance(submit_outcome.get("error"), RuntimeError)
        self.assertRegex(str(submit_outcome["error"]), "shutting down")
        self.assertGreaterEqual(manager.events.count("cancel"), 2, manager.events)
        self.assertGreater(manager.events.index("stop"), manager.events.index("started"), manager.events)
        self.assertNotIn("start", manager.events[manager.events.index("stop") + 1:], manager.events)
        self.assertIsNone(app.store.get(cid)["active_turn"])
        self.assertIsNone(app.store.get(other_cid)["active_turn"])
        self.assertTrue(app.gpu.acquire(blocking=False))
        app.gpu.release()

    def test_close_during_turn_start_does_not_join_unstarted_thread(self):
        manager = self._RecordingManager()
        app = self._make_coordinator(manager)
        cid = app.store.create()["id"]

        close_outcome = {}
        close_threads = []
        closer_waited = []
        triggered = [False]

        def run_close():
            try:
                app.close()
            except Exception as exc:
                close_outcome["error"] = exc

        class RacingThread(threading.Thread):
            def start(self):
                if not triggered[0]:
                    triggered[0] = True
                    closer = threading.Thread(target=run_close, daemon=True)
                    close_threads.append(closer)
                    closer.start()
                    app.closed.wait(timeout=5)
                    closer.join(timeout=0.35)
                    closer_waited.append(closer.is_alive())
                super().start()

        with unittest.mock.patch.object(coordinator_module, "threading",
                                        wraps=threading) as mock_threading:
            mock_threading.Thread = RacingThread
            app.submit(cid, {"mode": "chat", "text": "hello"})

        self.assertEqual(len(close_threads), 1)
        closer = close_threads[0]
        closer.join(timeout=5)
        self.assertFalse(closer.is_alive(), "close thread did not finish")
        self.assertEqual(closer_waited, [True], "close() did not wait for turn thread start")
        self.assertNotIn("error", close_outcome)
        self.assertIn("stop", manager.events)
        self.assertNotIn("start", manager.events)
        self.assertIsNone(app.store.get(cid)["active_turn"])
        self.assertTrue(app.gpu.acquire(blocking=False))
        app.gpu.release()

    def test_submit_thread_start_failure_finishes_turn_and_releases_gpu(self):
        manager = self._RecordingManager()
        app = self._make_coordinator(manager)
        cid = app.store.create()["id"]

        class FailingThread(threading.Thread):
            def start(self):
                raise RuntimeError("can't start new thread")

        with unittest.mock.patch.object(coordinator_module, "threading",
                                        wraps=threading) as mock_threading:
            mock_threading.Thread = FailingThread
            with self.assertRaisesRegex(RuntimeError, "can't start new thread"):
                app.submit(cid, {"mode": "chat", "text": "hello"})

        self.assertIsNone(app.store.get(cid)["active_turn"])
        self.assertEqual(app.turns, {})
        self.assertTrue(app.gpu.acquire(blocking=False))
        app.gpu.release()

    def test_close_during_turn_pair_start_emits_no_error_event(self):
        start_entered = threading.Event()
        cancel_called = threading.Event()
        turn_stopped = threading.Event()

        class AbortOnCancelManager(self._RecordingManager):
            def start(self):
                super().start()
                start_entered.set()
                cancel_called.wait(timeout=2)
                raise RuntimeError("pair start cancelled")

            def cancel(self):
                self.events.append("cancel")
                cancel_called.set()
                turn_stopped.wait(timeout=2)
                return True

            def stop(self):
                turn_stopped.set()
                return super().stop()

        manager = AbortOnCancelManager()
        app = self._make_coordinator(manager)
        self.addCleanup(cancel_called.set)
        self.addCleanup(turn_stopped.set)
        cid = app.store.create()["id"]

        app.submit(cid, {"mode": "chat", "text": "hello"})
        self.assertTrue(start_entered.wait(timeout=5), "manager.start() was not reached")

        app.close()

        self.assertIn("cancel", manager.events)
        event_types = [event["type"] for event in app.store.events(cid, 0)]
        self.assertNotIn("error", event_types, event_types)
        self.assertIsNone(app.store.get(cid)["active_turn"])
        self.assertTrue(app.gpu.acquire(blocking=False))
        app.gpu.release()


class WorkerReplyTests(unittest.TestCase):
    """LocalModels refuses a malformed worker reply with a RuntimeError that
    names it, never a raw decoder, AttributeError or TypeError."""

    def serve(self, *bodies, status=200):
        """One loopback worker; request N gets `status` and bodies[N]."""
        pending = iter(bodies)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                body = next(pending)
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except OSError:  # the client stopped reading at a refused event
                    pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return "http://127.0.0.1:%d" % server.server_port

    def stream(self, *events):
        """An SSE body: one data line per event, then [DONE]."""
        return b"".join(b"data: " + event + b"\n\n" for event in events) + b"data: [DONE]\n\n"

    def chat(self, url):
        return list(LocalModels(None).chat({"chat_url": url, "chat_model": "m"}, [], 8, threading.Event()))

    def decision(self, url):
        return LocalModels(None).decision({"decision_url": url}, {})

    def test_role_only_usage_and_finish_chunks_carry_no_text(self):
        url = self.serve(self.stream(b'{"choices": [{"delta": {"role": "assistant", "content": ""}}]}',
                                     b'{"choices": [{"delta": {"content": null}}]}',
                                     b'{"choices": [{"delta": null}]}',
                                     b'{"choices": [{}]}',
                                     b'{"choices": [{"delta": {}, "finish_reason": null}]}',
                                     b'{"choices": [{"delta": {"content": "hi"}}]}',
                                     b'{"choices": [{"delta": {}, "finish_reason": "stop"}]}',
                                     b'{"choices": []}', b'{"choices": null}', b'{"usage": {}}'))
        self.assertEqual(self.chat(url), [("delta", "hi"), ("finish", "stop")])

    def test_a_malformed_chat_event_fails_clearly(self):
        self.addCleanup(sys.set_int_max_str_digits, sys.get_int_max_str_digits())
        sys.set_int_max_str_digits(4300)
        cases = (
            ("not JSON", b"{"), ("not JSON", b"nope"), ("not JSON", b"\xff"),
            ("not JSON", b'{"n": ' + b"1" * 5000 + b"}"),  # int digit cap
            ("not JSON", b'{"choices": "\r\x1b[2Kx"}'),
            ("malformed", b"null"), ("malformed", b"[]"), ("malformed", b"123"),
            ("malformed", b'"x"'), ("malformed", b"true"),
            ("malformed", b'{"choices": 1}'), ("malformed", b'{"choices": 0}'),
            ("malformed", b'{"choices": false}'), ("malformed", b'{"choices": ""}'),
            ("malformed", b'{"choices": {}}'), ("malformed", b'{"choices": "x"}'),
            ("malformed", b'{"choices": [null]}'), ("malformed", b'{"choices": [1]}'),
            ("malformed", b'{"choices": [[]]}'), ("malformed", b'{"choices": ["x"]}'),
            ("malformed", b'{"choices": [{"delta": 1}]}'), ("malformed", b'{"choices": [{"delta": "x"}]}'),
            ("malformed", b'{"choices": [{"delta": []}]}'), ("malformed", b'{"choices": [{"delta": false}]}'),
            ("malformed", b'{"choices": [{"delta": {"content": 123}}]}'),
            ("malformed", b'{"choices": [{"delta": {"content": ["a"]}}]}'),
            ("malformed", b'{"choices": [{"delta": {"content": {}}}]}'),
            ("malformed", b'{"choices": [{"delta": {"content": false}}]}'),
            ("malformed", b'{"choices": [{"delta": {"content": 0}}]}'),
            ("malformed", b'{"choices": [{"delta": {}, "finish_reason": 1}]}'),
            ("malformed", b'{"choices": [{"delta": {}, "finish_reason": ["stop"]}]}'),
            ("malformed", b'{"choices": [{"delta": {}, "finish_reason": true}]}'),
            ("malformed", b'{"choices": [{"delta": {}, "finish_reason": 0}]}'))
        url = self.serve(*(self.stream(b'{"choices": [{"delta": {"content": "ok"}}]}', event)
                           for _, event in cases))
        for index, (expected, event) in enumerate(cases):
            with self.subTest(i=index, event=event[:80]):
                got = []
                with self.assertRaisesRegex(RuntimeError, expected) as caught:
                    for item in LocalModels(None).chat({"chat_url": url, "chat_model": "m"}, [], 8,
                                                       threading.Event()):
                        got.append(item)
                message = str(caught.exception)
                self.assertNotIn("\r", message)
                self.assertNotIn("\x1b", message)
                self.assertTrue(message.endswith(": %r" % event[:200].decode("utf-8", "replace")),
                                message[:300])
                self.assertEqual(got, [("delta", "ok")])

    def test_a_malformed_decision_reply_fails_clearly(self):
        self.addCleanup(sys.set_int_max_str_digits, sys.get_int_max_str_digits())
        sys.set_int_max_str_digits(4300)
        cases = (
            ("not JSON", b"{"), ("not JSON", b""), ("not JSON", b"nope"), ("not JSON", b"\xff"),
            ("not JSON", b'{"answers": {"n": ' + b"1" * 5000 + b"}}"),  # int digit cap
            ("not JSON", b"\r\x1b[2K"),
            ("without an answers object", b"null"), ("without an answers object", b"[]"),
            ("without an answers object", b"1"), ("without an answers object", b'"x"'),
            ("without an answers object", b"true"), ("without an answers object", b"{}"),
            ("without an answers object", b'{"answers": null}'),
            ("without an answers object", b'{"answers": []}'),
            ("without an answers object", b'{"answers": "x"}'))
        url = self.serve(b'{"answers": {"route": 1}}', *(event for _, event in cases))
        self.assertEqual(self.decision(url), {"answers": {"route": 1}})
        for expected, event in cases:
            with self.subTest(event=event[:40]):
                with self.assertRaisesRegex(RuntimeError, expected) as caught:
                    self.decision(url)
                message = str(caught.exception)
                self.assertNotIn("\r", message)
                self.assertNotIn("\x1b", message)
                self.assertTrue(message.endswith(": %r" % event[:200].decode("utf-8", "replace")),
                                message[:300])

    def test_a_deeply_nested_reply_fails_clearly(self):
        """The 4 MiB request keeps the thread's stack small, and the
        200,000-deep reply overflows any thread stack below about 28 MiB,
        including a reused default-size one; it must fail as a clear
        RuntimeError."""
        deep = b'{"choices": ' + b"[" * 200_000 + b"]" * 200_000 + b"}"
        url = self.serve(self.stream(deep), deep)
        errors = []

        def run():
            try:
                with self.assertRaises(RecursionError):
                    json.loads(deep)
                for call in (self.chat, self.decision):
                    with self.assertRaisesRegex(RuntimeError, "not JSON") as caught:
                        call(url)
                    self.assertIsInstance(caught.exception.__cause__, RecursionError)
            except BaseException as exc:
                errors.append(exc)

        old = threading.stack_size(4 * 1024 * 1024)
        try:
            thread = threading.Thread(target=run)
            thread.start()
        finally:
            threading.stack_size(old)
        thread.join()
        if errors:
            raise errors[0]

    def test_a_decoder_recursion_error_fails_clearly(self):
        """Stack-independent twin of the deep-reply test."""
        url = self.serve(self.stream(b'{"choices": [{"delta": {"content": "hi"}}]}'), b'{"answers": {}}')
        with unittest.mock.patch.object(json, "loads", side_effect=RecursionError("too deep")):
            for call in (self.chat, self.decision):
                with self.assertRaisesRegex(RuntimeError, "not JSON") as caught:
                    call(url)
                self.assertIsInstance(caught.exception.__cause__, RecursionError)

    def test_non_200_worker_reply_escapes_and_bounds_body(self):
        body = b"overloaded\r\n\x1b[31mfake log\x07\x7f" + b"x" * 240
        url = self.serve(body, body, status=503)
        expected_tail = ": %r" % body[:200].decode("utf-8", "replace")
        for label, call, prefix in (
            ("decision", self.decision, "Decision worker refused the request (HTTP 503): "),
            ("chat", self.chat, "Chat worker refused the request (HTTP 503): "),
        ):
            with self.subTest(worker=label):
                with self.assertRaises(RuntimeError) as caught:
                    call(url)
                message = str(caught.exception)
                self.assertTrue(message.startswith(prefix), message[:300])
                self.assertFalse(any(ord(ch) < 0x20 or ord(ch) == 0x7f for ch in message),
                                 repr(message[:300]))
                self.assertTrue(message.endswith(expected_tail), message[:300])


class NestedChatConnectionTests(unittest.TestCase):
    """A nested LocalModels.chat() (the mid-stream component repair) must not
    hide the outer stream from cancel or leak its socket. Both framings: the
    pinned mlx-lm server sends close-delimited HTTP/1.0 SSE."""

    FRAMINGS = ("HTTP/1.0", "HTTP/1.1")

    def serve(self, protocol, hold_inner=False, decision=b'{"answers": {}}'):
        """Request 0 sends one delta and waits; with hold_inner, so do later ones.
        /v1/decisions answers `decision` at once. self.paths lists every POST."""
        release = threading.Event()
        lock = threading.Lock()
        state = {"count": 0}
        chunked = protocol == "HTTP/1.1"
        self.paths = paths = []

        def write(handler, payload):
            if chunked:
                payload = b"%x\r\n%s\r\n" % (len(payload), payload)
            handler.wfile.write(payload)
            handler.wfile.flush()

        def sse(text):
            return b"data: " + json.dumps({"choices": [{"delta": {"content": text}}]}).encode() + b"\n\n"

        class Handler(BaseHTTPRequestHandler):
            protocol_version = protocol

            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                paths.append(self.path)
                if self.path == "/v1/decisions":
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(decision)))
                    self.end_headers()
                    try:
                        self.wfile.write(decision)
                    except OSError:
                        pass
                    self.close_connection = True
                    return
                with lock:
                    index = state["count"]
                    state["count"] += 1
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                if chunked:
                    self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                try:
                    write(self, sse("outer" if index == 0 else "inner"))
                    if index == 0 or hold_inner:
                        release.wait(10)
                    write(self, b"data: [DONE]\n\n")
                    if chunked:
                        write(self, b"")
                except OSError:
                    pass
                self.close_connection = True

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(release.set)
        return {"chat_url": "http://127.0.0.1:%d" % server.server_port, "chat_model": "m"}, release

    def tracked_sockets(self, connected=None):
        """Every client socket opened from here on; fileno() is -1 once closed.
        connected() runs as each socket finishes connecting."""
        import socket
        opened = []
        real = socket.create_connection

        def create(*args, **kwargs):
            opened.append(real(*args, **kwargs))
            if connected:
                connected()
            return opened[-1]

        patcher = unittest.mock.patch("socket.create_connection", create)
        patcher.start()
        self.addCleanup(patcher.stop)
        return opened

    def blocked_reader(self, stream):
        """A thread already blocked in next(stream), as the turn thread is when cancel arrives."""
        outcome = {}

        def read_next():
            try:
                outcome["value"] = next(stream)
            except Exception as exc:  # keep no traceback: it would pin the stream's socket
                outcome["error"] = repr(exc)

        reader = threading.Thread(target=read_next, daemon=True)
        reader.start()
        reader.join(0.3)
        self.assertTrue(reader.is_alive(), "reader should be waiting on the worker")
        return reader, outcome

    def assert_cancel_unblocks(self, models, release, readers):
        started = time.monotonic()
        models.close_connection()
        for reader, _ in readers:
            reader.join(max(0.0, 3 - (time.monotonic() - started)))
        unblocked = (not any(reader.is_alive() for reader, _ in readers)
                     and time.monotonic() - started < 3)
        if not unblocked:
            release.set()
            for reader, _ in readers:
                reader.join(5)
        self.assertTrue(unblocked, "a reader stayed blocked after close_connection()")
        # _run treats any error after cancel as a stop; which one depends on the framing.
        for _, outcome in readers:
            self.assertIn("error", outcome)

    def test_cancel_after_nested_repair_unblocks_outer(self):
        for protocol in self.FRAMINGS:
            with self.subTest(protocol=protocol):
                pair, release = self.serve(protocol)
                opened = self.tracked_sockets()
                models = LocalModels(None)
                outer = models.chat(pair, [], 8, threading.Event())
                self.assertEqual(next(outer), ("delta", "outer"))
                self.assertEqual(list(models.chat(pair, [], 8, threading.Event())),
                                 [("delta", "inner"), ("finish", None)])
                self.assert_cancel_unblocks(models, release, [self.blocked_reader(outer)])
                self.assertEqual([sock.fileno() for sock in opened], [-1, -1])
                self.assertFalse(models.sockets)

    def test_cancel_while_both_streams_open_unblocks_both(self):
        for protocol in self.FRAMINGS:
            with self.subTest(protocol=protocol):
                pair, release = self.serve(protocol, hold_inner=True)
                opened = self.tracked_sockets()
                models = LocalModels(None)
                outer = models.chat(pair, [], 8, threading.Event())
                self.assertEqual(next(outer), ("delta", "outer"))
                inner = models.chat(pair, [], 8, threading.Event())
                self.assertEqual(next(inner), ("delta", "inner"))
                self.assert_cancel_unblocks(models, release,
                                            [self.blocked_reader(outer), self.blocked_reader(inner)])
                self.assertEqual([sock.fileno() for sock in opened], [-1, -1])
                self.assertFalse(models.sockets)

    def test_nested_repair_does_not_leak_outer_socket(self):
        for protocol in self.FRAMINGS:
            with self.subTest(protocol=protocol):
                pair, release = self.serve(protocol)
                opened = self.tracked_sockets()
                models = LocalModels(None)
                outer = models.chat(pair, [], 8, threading.Event())
                self.assertEqual(next(outer), ("delta", "outer"))
                self.assertEqual(list(models.chat(pair, [], 8, threading.Event())),
                                 [("delta", "inner"), ("finish", None)])
                release.set()
                self.assertEqual(list(outer), [("finish", None)])
                self.assertEqual([sock.fileno() for sock in opened], [-1, -1])
                self.assertFalse(models.sockets)

    def test_a_decision_during_a_stream_closes_only_its_own_socket(self):
        for protocol in self.FRAMINGS:
            with self.subTest(protocol=protocol):
                pair, release = self.serve(protocol)
                opened = self.tracked_sockets()
                models = LocalModels(None)
                outer = models.chat(pair, [], 8, threading.Event())
                self.assertEqual(next(outer), ("delta", "outer"))
                self.assertEqual(models.decision({"decision_url": pair["chat_url"]}, {}), {"answers": {}})
                self.assertEqual(opened[1].fileno(), -1)
                release.set()
                self.assertEqual(list(outer), [("finish", None)])
                self.assertEqual(opened[0].fileno(), -1)
                self.assertFalse(models.sockets)

    def test_a_failed_call_closes_its_socket_while_its_error_is_held(self):
        # A finished routing future keeps the exception, its traceback and the response.
        for protocol in self.FRAMINGS:
            with self.subTest(protocol=protocol):
                pair, _ = self.serve(protocol, decision=b"x" * (1024 * 1024 + 2))
                opened = self.tracked_sockets()
                held = []  # assertRaises would drop the traceback
                try:
                    LocalModels(None).decision({"decision_url": pair["chat_url"]}, {})
                except ValueError as error:
                    held.append(error)
                self.assertIn("too large", str(held[0]))
                self.assertIsNotNone(held[0].__traceback__)
                self.assertEqual([sock.fileno() for sock in opened], [-1])

    def test_an_already_cancelled_turn_opens_no_socket(self):
        pair, _ = self.serve("HTTP/1.0")
        opened = self.tracked_sockets()
        cancel = threading.Event()
        cancel.set()
        self.assertEqual(list(LocalModels(None).chat(pair, [], 8, cancel)), [])
        self.assertEqual((opened, self.paths), ([], []))

    def test_a_turn_cancelled_while_connecting_never_sends_its_request(self):
        for protocol in self.FRAMINGS:
            with self.subTest(protocol=protocol):
                pair, _ = self.serve(protocol)
                models = LocalModels(None)
                cancel = threading.Event()

                def cancel_turn():  # Coordinator.cancel, before the socket is recorded
                    cancel.set()
                    models.close_connection()

                opened = self.tracked_sockets(connected=cancel_turn)
                self.assertEqual(list(models.chat(pair, [], 8, cancel)), [])
                self.assertEqual(self.paths, [])
                self.assertEqual([sock.fileno() for sock in opened], [-1])
                self.assertFalse(models.sockets)

    def test_a_malformed_pair_opens_no_socket(self):
        opened = self.tracked_sockets()
        with self.assertRaises(KeyError):
            next(LocalModels(None).chat({"chat_url": "http://127.0.0.1:9"}, [], 8, threading.Event()))
        self.assertEqual(opened, [])

if __name__ == '__main__':
    unittest.main()


class DisagreementTests(unittest.TestCase):
    def decision(self, choice="short"):
        return {"type": "choice", "choice": choice, "abstained": False,
                "options": [{"id": "short", "label": "short answer"},
                            {"id": "long", "label": "long answer"}]}

    def test_explanation_disagreement_is_conservative(self):
        decision = self.decision()
        self.assertTrue(explanation_disagrees(decision, "I choose the long answer."))
        self.assertFalse(explanation_disagrees(decision, "I choose the short answer."))
        self.assertFalse(explanation_disagrees(decision, "long answer and short answer both exist."))
        self.assertFalse(explanation_disagrees(decision, "The long answer is mentioned."))
        abstained = dict(decision, abstained=True)
        self.assertFalse(explanation_disagrees(abstained, "I choose the long answer."))
        self.assertFalse(explanation_disagrees({"type": "typed"}, "I choose the long answer."))


    def test_terminal_decision_text_keeps_the_laya_choice(self):
        rendered = format_decision_event({
            "type": "choice", "choice": "short", "criteria": "lower latency",
            "options": [{"id": "short", "label": "short answer"},
                        {"id": "long", "label": "long answer"}],
            "model": "laya-mlx", "abstained": False})
        self.assertIn("* short answer", rendered)
        self.assertIn("long answer", rendered)
        self.assertNotIn("* long answer", rendered)
        self.assertIn("Decided by: laya-mlx", rendered)
        typed = format_decision_event({
            "type": "typed",
            "results": [{"type": "choice", "instructions": "Which?",
                         "answer": {"choice": "billing", "abstained": True}}]})
        self.assertIn("Which?: billing", typed)
        self.assertIn("abstained", typed)
        self.assertEqual(format_decision_event({"type": "other"}), "")
