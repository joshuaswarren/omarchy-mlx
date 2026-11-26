"""Loaded-model /v1/models route for the serve shim.

Upstream 0.31.3 fact (verified in the published wheel, server.py
handle_models_request): the endpoint scans the whole HF cache and appends
the --model entry LAST, so the first entry is whichever repo happened to
download most recently. A server serving gemma reported Qwen because the
Qwen download was newer; clients that read data[0].id pick the wrong
model. The route must report the LOADED model, and only it.
"""

import io
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_serve import _mlxlm_server as shim  # noqa: E402


class PayloadTests(unittest.TestCase):
    def test_repo_id_target_is_reported_verbatim(self):
        out = shim.loaded_models_payload("mlx-community/gemma-4-e4b-it-4bit")
        self.assertEqual(out["object"], "list")
        self.assertEqual(len(out["data"]), 1)
        self.assertEqual(out["data"][0]["id"], "mlx-community/gemma-4-e4b-it-4bit")
        self.assertEqual(out["data"][0]["object"], "model")

    def test_local_dir_target_resolves(self):
        out = shim.loaded_models_payload("/tmp")
        self.assertEqual(out["data"][0]["id"], str(Path("/tmp").resolve()))

    def test_missing_model_ref_is_unknown(self):
        out = shim.loaded_models_payload("")
        self.assertEqual(out["data"][0]["id"], "unknown")


class FakeArgs:
    def __init__(self, model):
        self.model = model


class FakeGenerator:
    def __init__(self, model):
        self.cli_args = FakeArgs(model)


class FakeHandler:
    """Stands in for mlx_lm.server.APIHandler in route tests."""

    def __init__(self, response_generator=None):
        self.response_generator = response_generator
        self.written = b""
        self.headers = []
        self.path = "/v1/models"
        self.wfile = io.BytesIO()

    def send_response(self, status):
        self.status = status

    def send_header(self, key, value):
        self.headers.append((key, value))

    def end_headers(self):
        pass

    def _set_completion_headers(self, status_code=200):
        """Mirror of the upstream method (server.py 0.31.3): response line,
        content type, then the CORS block driven by cli_args."""
        self.send_response(status_code)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.send_header("Access-Control-Allow-Headers", "*")

    def do_GET(self):
        raise AssertionError("upstream do_GET must not run for /v1/models")


class RouteTests(unittest.TestCase):
    def _server(self):
        import types

        class APIHandler(FakeHandler):
            def do_GET(self):
                self.written = b"upstream"

        return types.SimpleNamespace(APIHandler=APIHandler,
                                     ThreadingHTTPServer=object,
                                     _run_http_server=lambda host, port, response_generator,
                                          server_class=None, handler_class=None: (host, port))

    def test_models_route_answers_with_loaded_model(self):
        server = self._server()
        captured = {}

        def spy(host, port, response_generator, server_class=None, handler_class=None):
            captured["handler"] = handler_class
            return host, port

        server._run_http_server = spy
        shim.install_models_route(server)
        server._run_http_server(1, 2, None, server_class=None, handler_class=None)
        h = captured["handler"](FakeGenerator("mlx-community/gemma-4-e4b-it-4bit"))
        h.do_GET()
        self.assertEqual(h.status, 200)
        self.assertIn(b"gemma-4-e4b-it-4bit", h.wfile.getvalue())

    def test_route_wraps_incoming_handler_not_replaces(self):
        server = self._server()

        class Incoming(FakeHandler):
            def do_GET(self):
                self.written = b"upstream"

        captured = {}

        def spy(host, port, response_generator, server_class=None, handler_class=None):
            captured["handler"] = handler_class
            captured["server_class"] = server_class
            return host, port

        server._run_http_server = spy
        shim.install_models_route(server)
        server._run_http_server(1, 2, None, server_class=None, handler_class=Incoming)
        h = captured["handler"](FakeGenerator("some/model"))
        self.assertIsInstance(h, Incoming)
        h.do_GET()
        self.assertEqual(h.status, 200)
        self.assertIn(b"some/model", h.wfile.getvalue())

    def test_other_get_paths_fall_through(self):
        server = self._server()
        captured = {}

        def spy(host, port, response_generator, server_class=None, handler_class=None):
            captured["handler"] = handler_class
            return host, port

        server._run_http_server = spy
        shim.install_models_route(server)
        server._run_http_server(1, 2, None)
        h = captured["handler"](FakeGenerator("some/model"))
        h.path = "/health"
        h.do_GET()
        self.assertEqual(h.written, b"upstream")  # fall-through untouched


if __name__ == "__main__":
    unittest.main()


class FakeRepoFile:
    def __init__(self, name):
        self.file_path = type("P", (), {"name": name})()


class FakeRef:
    def __init__(self, files):
        self.files = [FakeRepoFile(f) for f in files]


class FakeRepo:
    def __init__(self, repo_id):
        self.repo_id = repo_id
        self.repo_type = "model"
        self.refs = {"main": FakeRef(
            ["config.json", "model.safetensors.index.json",
             "tokenizer_config.json"])}


class FakeCache:
    def __init__(self, repos):
        self.repos = repos


class RealUpstreamTests(unittest.TestCase):
    """Proves the route against the REAL mlx_lm.server handler: the
    cache-scan bug itself (upstream lists the cached repo first and the
    loaded model last) and the CORS headers the route must preserve.
    Skipped where mlx_lm is not importable."""

    def setUp(self):
        try:
            import mlx_lm.server as srv  # noqa: F401
        except ImportError:
            self.skipTest("mlx_lm not importable on this host")

    def _fake_self(self, srv, model):
        import io
        import types
        h = srv.APIHandler.__new__(srv.APIHandler)
        h.path = "/v1/models"
        h.created = 0
        h.wfile = io.BytesIO()
        h.headers = types.SimpleNamespace(get=lambda key: None)
        h.headers_sent = []
        h.response_generator = types.SimpleNamespace(cli_args=types.SimpleNamespace(
            model=model, allowed_origins=["*"]))
        h.send_response = lambda code, *a: setattr(h, "status", code)
        h.send_header = lambda k, v: h.headers_sent.append((k, v))
        h.end_headers = lambda: None
        return h

    def test_route_fixes_cache_scan_order_and_keeps_cors(self):
        import json
        import unittest.mock
        import mlx_lm.server as srv
        loaded = "mlx-community/gemma-4-e4b-it-4bit"
        cache = FakeCache([FakeRepo("mlx-community/Qwen2.5-1.5B-Instruct-4bit")])
        with unittest.mock.patch.object(srv, "scan_cache_dir",
                                        lambda *a, **k: cache):
            h = self._fake_self(srv, loaded)
            h.handle_models_request()
            upstream = json.loads(h.wfile.getvalue())["data"]
            self.assertEqual(upstream[0]["id"],
                             "mlx-community/Qwen2.5-1.5B-Instruct-4bit")
            self.assertNotEqual(upstream[0]["id"], loaded)

            # capture the handler class the route builds, then drive its
            # do_GET directly (the upstream do_GET would re-answer)
            captured = {}

            def spy(host, port, response_generator, server_class=None,
                    handler_class=None):
                captured["handler"] = handler_class
                return host, port

            with unittest.mock.patch.object(srv, "_run_http_server", spy):
                shim.install_models_route(srv)
                srv._run_http_server(1, 2, h.response_generator)
            cls = captured["handler"]
            # BaseRequestHandler.__init__ wants request/client/server; the
            # route only touches handler state, so borrow it from h
            h2 = cls.__new__(cls)
            h2.__dict__.update(h.__dict__)
            h.wfile = io.BytesIO()
            h2.wfile = h.wfile
            # drop the upstream call's headers so the CORS assertion below
            # can only pass through the route's own completion headers
            h.headers_sent.clear()
            h2.path = "/v1/models"
            h2.do_GET()
            fixed = json.loads(h2.wfile.getvalue())["data"]
            self.assertEqual([m["id"] for m in fixed], [loaded])
            self.assertIn(("Access-Control-Allow-Origin", "*"),
                          h.headers_sent)


class YieldOrderTests(unittest.TestCase):
    """The gate's yield route must survive either install order: post to
    YIELD_CONTROL_PATH through the stacked handler in both orders and
    check the status is not the upstream 404/501 (which the plain
    APIHandler do_POST would produce for an unknown path)."""

    def _stack(self, order):
        import types
        from mlx_omarchy_serve._mlxlm_server import YieldGate

        class APIHandler:
            def __init__(self, response_generator=None, **kw):
                self.response_generator = response_generator

            def do_POST(self):
                self.status = 501

            def _set_completion_headers(self, status_code=200):
                pass

        captured = {}

        def inner(host, port, response_generator, server_class=None,
                  handler_class=None):
            captured["handler"] = handler_class
            return (host, port)

        server = types.SimpleNamespace(APIHandler=APIHandler,
                                       ThreadingHTTPServer=object,
                                       BatchGenerator=type('BG', (), {'next': lambda self: None}),
                                       _run_http_server=inner)
        captured = {}

        def inner(host, port, response_generator, server_class=None,
                  handler_class=None):
            captured["handler"] = handler_class
            return (host, port)

        gate = YieldGate()
        steps = {"gate": lambda s: shim.install_yield_gate(s, gate),
                 "route": shim.install_models_route}
        for step in order:
            steps[step](server)
        server._run_http_server(1, 2, None)
        return captured["handler"]

    def _post_yield(self, handler_cls):
        import io
        import json as jsonlib
        import types
        gen = types.SimpleNamespace(cli_args=types.SimpleNamespace(model="m"))
        h = handler_cls(gen)
        h.status = None
        h.headers_sent = []
        h.send_response = lambda code, *a: setattr(h, "status", code)
        h.send_header = lambda k, v: h.headers_sent.append((k, v))
        h.end_headers = lambda: None
        h.path = "/v1/internal/yield"
        class _H:
            def get(self, key, default=None):
                return default
        h.headers = _H()
        h.rfile = io.BytesIO(jsonlib.dumps(
            {"action": "hold", "secret": "s", "ack_timeout": 0.1}).encode())
        h.wfile = io.BytesIO()
        h.do_POST()
        return h.status

    def test_yield_route_survives_gate_then_route(self):
        handler = self._stack(["gate", "route"])
        self.assertIn(self._post_yield(handler), (200, 403, 409))

    def test_yield_route_survives_route_then_gate(self):
        handler = self._stack(["route", "gate"])
        self.assertIn(self._post_yield(handler), (200, 403, 409))


class ThinkingDefaultTests(unittest.TestCase):
    """The client that hit the break is our own serve profiler
    (receipts/2026-11-16-gpu-client-cap/lb-prof3g.py): its exact body
    carries no chat_template_kwargs - like tools/serve-bench - so an
    opt-in flag cannot help it. The serve default is thinking OFF;
    opt-in is chat_template_kwargs.enable_thinking. Requests are real
    mlx_lm.server.CompletionRequest objects, so an invented attribute
    cannot pass the test (w7X review finding). Skipped where mlx_lm is
    not importable."""

    FAILING_BODY = {
        "model": "mlx-community/gemma-4-e4b-it-4bit",
        "max_tokens": 120,
        "temperature": 0.2,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content":
                      "Explain, in about one hundred words, how a counterweight "
                      "elevator system keeps the motor load small, and name one "
                      "real failure mode."}],
    }

    def setUp(self):
        try:
            import mlx_lm.server  # noqa: F401
        except ImportError:
            self.skipTest("mlx_lm not importable on this host")

    def _kwargs(self, body, args_kwargs=None):
        import types
        import mlx_lm.server as srv
        import mlx_omarchy_serve._mlxlm_server as mod

        captured = {}
        request = srv.CompletionRequest(
            request_type="chat", prompt=None, messages=body["messages"],
            tools=None, role_mapping=None)
        # the real request carries no body and no chat_template_kwargs
        self.assertFalse(hasattr(request, "body"))
        args = types.SimpleNamespace(
            chat_template_kwargs=dict(args_kwargs) if args_kwargs else None)

        class FakeGenerator:
            pass

        def original(self_gen, tokenizer, req, a):
            captured["kwargs"] = a.chat_template_kwargs
            return [1], [[1]], ["assistant"], "normal"

        FakeGenerator._tokenize = original
        mod.install_thinking_switch(
            types.SimpleNamespace(ResponseGenerator=FakeGenerator))
        FakeGenerator()._tokenize(None, request, args)
        return captured["kwargs"]

    def test_exact_failing_body_gets_content_default(self):
        body = dict(self.FAILING_BODY)
        self.assertNotIn("chat_template_kwargs", body)
        kwargs = self._kwargs(body)
        self.assertEqual(kwargs, {"enable_thinking": False})

    def test_bench_client_body_gets_content_default(self):
        kwargs = self._kwargs({"model": "qwen3.8-27b",
                               "messages": [{"role": "user",
                                             "content": "Say hi in one word."}],
                               "max_tokens": 16})
        self.assertEqual(kwargs, {"enable_thinking": False})

    def test_explicit_args_kwargs_win(self):
        kwargs = self._kwargs(self.FAILING_BODY, {"enable_thinking": True})
        self.assertEqual(kwargs, {"enable_thinking": True})
class NagleTests(unittest.TestCase):
    """The 64 ms uniform per-chunk stall of the served stream is the
    Nagle/delayed-ACK interaction on the SSE socket (loopback included).
    Every handler class the shim builds must set
    disable_nagle_algorithm so http.server opens connections with
    TCP_NODELAY; otherwise decode at ~46 ms/token emits at ~64 ms."""

    def _captured(self, install):
        import types
        captured = {}

        class APIHandler:
            def __init__(self, response_generator=None, **kw):
                self.response_generator = response_generator

            def do_GET(self):
                pass

            def do_POST(self):
                pass

        def inner(host, port, response_generator, server_class=None,
                  handler_class=None):
            captured["handler"] = handler_class
            return (host, port)

        server = types.SimpleNamespace(
            APIHandler=APIHandler, ThreadingHTTPServer=object,
            BatchGenerator=type("BG", (), {"next": lambda self: None}),
            _run_http_server=inner)
        install(server)
        server._run_http_server(1, 2, None)
        return captured["handler"]

    def test_route_handler_disables_nagle(self):
        handler = self._captured(shim.install_models_route)
        self.assertTrue(getattr(handler, "disable_nagle_algorithm", False))

    def test_gated_handler_disables_nagle(self):
        from mlx_omarchy_serve._mlxlm_server import YieldGate
        gate = YieldGate()

        def install(server):
            shim.install_yield_gate(server, gate)

        handler = self._captured(install)
        self.assertTrue(getattr(handler, "disable_nagle_algorithm", False))


class PreEvalTests(unittest.TestCase):
    """The upstream batch loop reads r.logprobs[r.token].item() per
    token on the same thread that drives the steps, so delaying
    delivery cannot overlap that GPU readback (jw16 A/B: the delay
    left the per-chunk gap at 60-65 ms). Instead the wrapped step
    evaluates each response's logprobs array itself - the forward pass
    was already scheduled at the step's start, so the array is ready
    by the time the loop's .item() runs - making that read a host
    read. The wrapper is otherwise pass-through."""

    def _wrapped(self, batches, evals):
        import types
        import mlx_omarchy_serve._mlxlm_server as mod
        from mlx_omarchy_serve._mlxlm_server import YieldGate

        fake_mx = types.SimpleNamespace(eval=lambda *a: evals.append(a))
        fake_pkg = types.ModuleType("mlx")
        real_core = sys.modules.get("mlx.core")
        real_pkg = sys.modules.get("mlx")
        sys.modules["mlx"] = fake_pkg
        sys.modules["mlx.core"] = fake_mx
        self.addCleanup(sys.modules.pop, "mlx.core", None)
        self.addCleanup(sys.modules.pop, "mlx", None)
        for name, real in (("mlx.core", real_core), ("mlx", real_pkg)):
            if real is not None:
                self.addCleanup(sys.modules.__setitem__, name, real)

        class FakeGen:
            def __init__(self, script):
                self._script = list(script)

            def next(self):
                return self._script.pop(0) if self._script else ([], [])

        gate = YieldGate()
        mod._ttft_armed["first_chunk"] = False
        return mod.wrap_generator_with_gate(FakeGen(batches), gate)

    def test_evals_logprobs_arrays_of_every_response(self):
        import types
        from itertools import count
        ids = count()
        pr = types.SimpleNamespace(logprobs=f"lp{next(ids)}")
        gr = types.SimpleNamespace(logprobs=f"lp{next(ids)}")
        batches = [
            ([pr], [gr]),
            ([], []),
        ]
        evals = []
        wrapped = self._wrapped(batches, evals)
        wrapped.next()
        self.assertEqual(len(evals), 1)
        self.assertEqual(evals[0], ("lp0", "lp1"))

    def test_passes_batches_through_untouched(self):
        batches = [
            ([("p",)], [("g",)]),
            ([], []),
        ]
        evals = []
        wrapped = self._wrapped(batches, evals)
        self.assertEqual(wrapped.next(), ([("p",)], [("g",)]))
        self.assertEqual(wrapped.next(), ([], []))
        self.assertEqual(evals, [])

    def test_skips_import_when_no_arrays(self):
        batches = [([], [])]
        evals = []
        wrapped = self._wrapped(batches, evals)
        self.assertEqual(wrapped.next(), ([], []))
        self.assertEqual(evals, [])


class OutputPreservationTests(unittest.TestCase):
    """One-cycle-late delivery must not change output: the token
    sequence, each response's text, finish_reason and logprobs are
    delivered exactly as the underlying generator produced them, and
    the stream still ends on the same final token (EOS, stop string
    and max_tokens all end through finish_reason on the last batch)."""

    def _run(self, script, stop_after=None):
        """script: list of batches the fake generator returns.
        Returns the flattened deliveries the consumer sees."""
        from mlx_omarchy_serve._mlxlm_server import YieldGate
        import mlx_omarchy_serve._mlxlm_server as mod

        class FakeGen:
            def __init__(self, script):
                self._script = list(script)

            def next(self):
                return self._script.pop(0) if self._script else ([], [])

        gate = YieldGate()
        mod._ttft_armed["first_chunk"] = False
        wrapped = mod.wrap_generator_with_gate(FakeGen(script), gate)
        out = []
        for _ in range(len(script) + 4):
            prompt_r, gen_r = wrapped.next()
            for r in prompt_r:
                out.append(("p",) + tuple(r))
            for r in gen_r:
                out.append(("g",) + tuple(r))
            if stop_after is not None and len(out) >= stop_after:
                break
        return out

    def test_sequence_and_payloads_identical_after_delay(self):
        batches = [
            ([], [("tok", i, f"logprob-{i}", None) for i in range(3)]),
            ([], [("tok", 3, "logprob-3", "length")]),
        ]
        delivered = self._run(batches)
        self.assertEqual(delivered, [
            ("g", "tok", 0, "logprob-0", None),
            ("g", "tok", 1, "logprob-1", None),
            ("g", "tok", 2, "logprob-2", None),
            ("g", "tok", 3, "logprob-3", "length"),
        ])

    def test_final_batch_with_finish_reason_flushed(self):
        # EOS/stop/max_tokens all arrive as finish_reason on the last
        # generation batch; the delay must still deliver that batch.
        batches = [
            ([], [("eos", 9, "logprob-9", "stop")]),
            ([], []),
        ]
        delivered = self._run(batches)
        self.assertEqual(delivered, [
            ("g", "eos", 9, "logprob-9", "stop"),
        ])

    def test_prompt_responses_keep_order_before_generation(self):
        batches = [
            ([("pre", 0, None, None)], []),
            ([], [("tok", 1, "lp1", None)]),
            ([], []),
        ]
        delivered = self._run(batches)
        self.assertEqual([d[0] for d in delivered], ["p", "g"])
        self.assertEqual(delivered[0], ("p", "pre", 0, None, None))
        self.assertEqual(delivered[1], ("g", "tok", 1, "lp1", None))
