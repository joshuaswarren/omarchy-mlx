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
                                     _run_http_server=lambda *a, **k: (a, k))

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
