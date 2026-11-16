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
            h2 = captured["handler"](h.response_generator)
            h2.path = "/v1/models"
            h2.do_GET()
            fixed = json.loads(h2.wfile.getvalue())["data"]
            self.assertEqual([m["id"] for m in fixed], [loaded])
            self.assertIn(("Access-Control-Allow-Origin", "*"),
                          h2.headers_sent)


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
