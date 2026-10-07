"""Bonsai-2 server tests against the tiny fixture pack (CPU, real HTTP).

Starts the actual stdlib server over the tiny pack fixture and exercises
/health, /v1/models, non-streaming and streaming chat completions, the
hard context cap, request validation, the CPU-device guard, and error
paths. The model is served through the real loader and the real
generation loop; no part of the serving path is mocked.
"""

import json
import shutil
import socket
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "serve") not in sys.path:
    sys.path.insert(0, str(REPO / "serve"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

mx = pytest.importorskip("mlx.core")

import bonsai2_fixture
from bonsai2_fixture import build_tiny_pack
from mlx_omarchy_bonsai2.server import serve_main


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _request(base, path, payload=None, method=None):
    url = base + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method or ("POST" if data else "GET"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


class ServerTests(unittest.TestCase):
    httpd = None
    base = None

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        pack_dir, _ = build_tiny_pack(cls.tmp)
        port = _free_port()
        argv = [
            "--model", str(pack_dir),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--model-id", "bonsai2-fixture",
            "--max-context", "256",
            "--allow-cpu",
        ]
        from http.server import ThreadingHTTPServer
        from mlx_omarchy_bonsai2 import server as server_module

        args = server_module._parse_args(argv)
        state = server_module.Bonsai2State(args)
        cls.worker_state = state
        cls.worker_thread = threading.Thread(
            target=server_module._worker_loop, args=(state,), daemon=True
        )
        cls.worker_thread.start()
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", port), server_module._make_handler(state))
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = "http://127.0.0.1:%d" % cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        if cls.httpd:
            cls.httpd.shutdown()
            cls.httpd.server_close()
        if cls.worker_state is not None:
            cls.worker_state.job_queue.put((None, None))
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_health_reports_pack_facts(self):
        status, body = _request(self.base, "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["backend"], "mlx-omarchy-bonsai2")
        self.assertEqual(body["model_type"], "prism_hadamard_qwen35")
        self.assertEqual(body["quantization"], {"bits": 2, "group_size": 128, "mode": "affine"})
        expected_device = "gpu" if mx.default_device() == mx.gpu else "cpu"
        self.assertEqual(body["device"], expected_device)
        self.assertGreater(body["resident_bytes"], 0)
        self.assertIn("visual", body["excluded_bytes"])
        self.assertTrue(body["license_files"]["LICENSE"])

    def test_models_listing(self):
        status, body = _request(self.base, "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(body["data"][0]["id"], "bonsai2-fixture")
        self.assertEqual(body["data"][0]["max_context"], 256)

    def test_chat_completion_non_stream(self):
        status, body = _request(
            self.base,
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 4},
        )
        self.assertEqual(status, 200)
        choice = body["choices"][0]
        self.assertEqual(choice["message"]["role"], "assistant")
        self.assertTrue(choice["message"]["content"])
        self.assertIn(choice["finish_reason"], ("stop", "length"))
        timings = body["timings"]
        self.assertEqual(
            sorted(timings),
            sorted(
                [
                    "prompt_n", "cached_n", "prompt_ms", "prompt_per_second",
                    "predicted_n", "predicted_ms", "predicted_per_second",
                ]
            ),
        )
        self.assertEqual(timings["cached_n"], 0)
        self.assertGreaterEqual(timings["prompt_n"], 1)
        self.assertLessEqual(timings["prompt_n"], len("Hello ".split()))  # tiny template
        self.assertEqual(timings["predicted_n"], 4)
        usage = body["usage"]
        self.assertEqual(usage["total_tokens"], timings["prompt_n"] + timings["predicted_n"])

    def test_chat_completion_stream(self):
        req = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=json.dumps(
                {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 3, "stream": True}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            self.assertEqual(resp.headers.get("Content-Type"), "text/event-stream")
            events = []
            for raw in resp.read().decode().split("\n\n"):
                line = raw.strip()
                if line.startswith("data: "):
                    events.append(line[6:])
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(e) for e in events[:-1]]
        self.assertTrue(all(c["object"] == "chat.completion.chunk" for c in chunks))
        content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        self.assertTrue(content)
        self.assertTrue(any("timings" in c for c in chunks))
        self.assertTrue(any(c["choices"][0]["finish_reason"] for c in chunks))

    def test_context_cap_is_enforced(self):
        status, body = _request(
            self.base,
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 400},
        )
        self.assertEqual(status, 400)
        self.assertIn("context cap 256", body["error"])

    def test_invalid_requests_rejected(self):
        status, body = _request(self.base, "/v1/chat/completions", {"messages": []})
        self.assertEqual(status, 400)
        status, body = _request(
            self.base, "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 0},
        )
        self.assertEqual(status, 400)
        status, body = _request(self.base, "/nope")
        self.assertEqual(status, 404)

    def test_cpu_device_guard_refuses_without_flag(self):
        from mlx_omarchy_bonsai2 import server as server_module

        if mx.default_device() == mx.gpu:
            self.skipTest("default device is a GPU; CPU-guard exit only applies off-GPU")
        argv = ["--model", str(self.tmp), "--host", "127.0.0.1", "--port", "0"]
        with self.assertRaises(SystemExit) as ctx:
            server_module.serve_main(argv)
        self.assertEqual(ctx.exception.code, 2)


    def test_managed_mode_fails_closed_before_load(self):
        """--managed must refuse BEFORE touching the model, not after."""
        from mlx_omarchy_bonsai2 import server as server_module

        try:
            import mlx_omarchy_serve  # noqa: F401

            self.skipTest("mlx_omarchy_serve importable; budget may legitimately admit")
        except ImportError:
            pass
        empty = Path(tempfile.mkdtemp())  # no config.json: a post-load check would raise PackError
        self.addCleanup(shutil.rmtree, empty, ignore_errors=True)
        argv = [
            "--model", str(empty), "--host", "127.0.0.1", "--port", "0",
            "--allow-cpu", "--managed",
        ]
        with self.assertRaises(SystemExit) as ctx:
            server_module.serve_main(argv)
        self.assertEqual(ctx.exception.code, 3)


class RequestValidationTests(unittest.TestCase):
    """Malformed-body contracts over real HTTP (Main review blockers)."""

    httpd = None
    base = None

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        pack_dir, _ = build_tiny_pack(cls.tmp)
        port = _free_port()
        from http.server import ThreadingHTTPServer
        from mlx_omarchy_bonsai2 import server as server_module

        args = server_module._parse_args(
            ["--model", str(pack_dir), "--host", "127.0.0.1", "--port", str(port), "--allow-cpu"]
        )
        state = server_module.Bonsai2State(args)
        cls.worker_state = state
        cls.worker_thread = threading.Thread(
            target=server_module._worker_loop, args=(state,), daemon=True
        )
        cls.worker_thread.start()
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", port), server_module._make_handler(state))
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.base = "http://127.0.0.1:%d" % cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        if cls.httpd:
            cls.httpd.shutdown()
            cls.httpd.server_close()
        if cls.worker_state is not None:
            cls.worker_state.job_queue.put((None, None))
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _raw(self, payload: bytes):
        req = urllib.request.Request(
            self.base + "/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status
        except urllib.error.HTTPError as exc:
            return exc.code

    def test_top_level_array_rejected(self):
        self.assertEqual(self._raw(b"[1, 2, 3]"), 400)

    def test_top_level_null_rejected(self):
        self.assertEqual(self._raw(b"null"), 400)

    def test_invalid_utf8_rejected(self):
        self.assertEqual(self._raw(b'{"messages": "\xff\xfe"}'), 400)

    def test_nan_temperature_rejected(self):
        body = json.dumps(
            {"messages": [{"role": "user", "content": "x"}], "temperature": float("nan")}
        ).encode()
        self.assertEqual(self._raw(body), 400)

    def test_infinite_temperature_rejected(self):
        body = json.dumps(
            {"messages": [{"role": "user", "content": "x"}], "temperature": float("inf")}
        ).encode()
        self.assertEqual(self._raw(body), 400)

    def test_boolean_temperature_rejected(self):
        body = json.dumps(
            {"messages": [{"role": "user", "content": "x"}], "temperature": True}
        ).encode()
        self.assertEqual(self._raw(body), 400)

    def test_negative_temperature_rejected(self):
        status, _ = _request(
            self.base, "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "x"}], "temperature": -0.5},
        )
        self.assertEqual(status, 400)

    def test_top_p_out_of_range_rejected(self):
        status, _ = _request(
            self.base, "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "x"}], "top_p": 1.5},
        )
        self.assertEqual(status, 400)

    def test_boolean_top_p_rejected(self):
        status, _ = _request(
            self.base, "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "x"}], "top_p": True},
        )
        self.assertEqual(status, 400)

    def test_non_string_content_rejected(self):
        status, _ = _request(
            self.base, "/v1/chat/completions", {"messages": [{"role": "user", "content": 123}]}
        )
        self.assertEqual(status, 400)

    def test_non_object_message_rejected(self):
        status, _ = _request(self.base, "/v1/chat/completions", {"messages": ["hello"]})
        self.assertEqual(status, 400)

    def test_missing_role_rejected(self):
        status, _ = _request(
            self.base, "/v1/chat/completions", {"messages": [{"content": "hello"}]}
        )
        self.assertEqual(status, 400)

    def test_non_bool_stream_rejected(self):
        status, _ = _request(
            self.base, "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "x"}], "stream": "yes"},
        )
        self.assertEqual(status, 400)

    def test_max_context_zero_rejected_at_parse(self):
        from mlx_omarchy_bonsai2 import server as server_module

        with self.assertRaises(SystemExit):
            server_module._parse_args(
                ["--model", "x", "--max-context", "0"]
            )


    def test_managed_fails_closed_without_atomic_api(self):
        """Budget without admit_and_reserve must exit 3, never race."""
        import types

        from mlx_omarchy_bonsai2 import server as server_module

        fake_budget = types.SimpleNamespace(admit=lambda required, home=None: None)
        fake_pkg = types.SimpleNamespace(budget=fake_budget)
        saved = sys.modules.get("mlx_omarchy_serve")
        sys.modules["mlx_omarchy_serve"] = fake_pkg
        try:
            argv = [
                "--model", str(self.tmp / "pack"), "--host", "127.0.0.1", "--port", "0",
                "--allow-cpu", "--managed",
            ]
            with self.assertRaises(SystemExit) as ctx:
                server_module.serve_main(argv)
            self.assertEqual(ctx.exception.code, 3)
        finally:
            if saved is None:
                sys.modules.pop("mlx_omarchy_serve", None)
            else:
                sys.modules["mlx_omarchy_serve"] = saved

    def test_managed_fails_closed_when_atomic_admission_refuses(self):
        """An atomic admit_and_reserve that refuses must exit 3 pre-load."""
        import types

        from mlx_omarchy_bonsai2 import server as server_module

        def refuse(name, byte_count, *, note="", owner=None, state="pending"):
            raise ValueError("required bytes exceed available headroom")

        fake_budget = types.SimpleNamespace(
            estimate_required=lambda memory, context_tokens: types.SimpleNamespace(total=1),
            admit_and_reserve=refuse,
        )
        fake_pkg = types.SimpleNamespace(budget=fake_budget)
        saved = sys.modules.get("mlx_omarchy_serve")
        sys.modules["mlx_omarchy_serve"] = fake_pkg
        try:
            argv = [
                "--model", str(self.tmp / "pack"), "--host", "127.0.0.1", "--port", "0",
                "--allow-cpu", "--managed",
            ]
            with self.assertRaises(SystemExit) as ctx:
                server_module.serve_main(argv)
            self.assertEqual(ctx.exception.code, 3)
        finally:
            if saved is None:
                sys.modules.pop("mlx_omarchy_serve", None)
            else:
                sys.modules["mlx_omarchy_serve"] = saved

    def test_reservation_name_is_unique_per_process(self):
        from mlx_omarchy_bonsai2 import server as server_module

        names = {server_module._reservation_name("bonsai-2-27b-mlx-2bit") for _ in range(4)}
        self.assertEqual(len(names), 4)
        self.assertTrue(all(n.startswith("bonsai-2-27b-mlx-2bit-") for n in names))


    def test_sigterm_during_load_releases_reservation(self):
        """BaseException cleanup: an interrupt after admit but before
        steady serve still clears the reservation (Main round-6)."""
        import types

        from mlx_omarchy_bonsai2 import server as server_module

        events = []

        def fake_atomic(name, byte_count, *, note="", owner=None, state="pending"):
            events.append("reserve")
            return types.SimpleNamespace(fits=True, lines=[])

        fake_budget = types.SimpleNamespace(
            estimate_required=lambda memory, context_tokens: types.SimpleNamespace(total=1024),
            admit_and_reserve=fake_atomic,
            set_reservation_state=lambda *a, **k: events.append("resident"),
            clear_reservation=lambda name, **k: events.append("clear"),
        )
        fake_pkg = types.SimpleNamespace(budget=fake_budget)
        saved = sys.modules.get("mlx_omarchy_serve")
        sys.modules["mlx_omarchy_serve"] = fake_pkg
        original_state = server_module.Bonsai2State

        def boom(args):
            raise KeyboardInterrupt  # simulates SIGTERM mid-load

        server_module.Bonsai2State = boom
        try:
            argv = ["--model", str(self.tmp / "pack"), "--host", "127.0.0.1",
                    "--port", "0", "--allow-cpu", "--managed"]
            # The interrupt is handled: clean return, reservation released.
            server_module.serve_main(argv)
        finally:
            if saved is None:
                sys.modules.pop("mlx_omarchy_serve", None)
            else:
                sys.modules["mlx_omarchy_serve"] = saved
            server_module.Bonsai2State = original_state
        self.assertIn("reserve", events)
        self.assertIn("clear", events)

    def test_concurrent_requests_serialize_and_succeed(self):
        """Two overlapping chat requests both complete via the mx worker."""
        import concurrent.futures

        def one(i):
            return _request(
                self.base,
                "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 2},
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(one, range(2)))
        for status, body in results:
            self.assertEqual(status, 200)
            self.assertTrue(body["choices"][0]["message"]["content"])
            self.assertEqual(body["timings"]["predicted_n"], 2)


    def test_managed_preflight_happy_path_admits_and_reserves(self):
        """Full preflight with a working atomic budget: pack facts parsed,
        estimate asked, admit_and_reserve called with owner + pending."""
        import types

        from mlx_omarchy_bonsai2 import server as server_module

        calls = {}

        def fake_atomic(name, byte_count, *, note="", owner=None, state="pending"):
            calls.update(name=name, byte_count=byte_count, note=note, owner=owner, state=state)
            return types.SimpleNamespace(fits=True, lines=["ok"])

        fake_budget = types.SimpleNamespace(
            estimate_required=lambda memory, context_tokens: types.SimpleNamespace(
                total=memory["weights_bytes"] + 1024
            ),
            admit_and_reserve=fake_atomic,
        )
        fake_pkg = types.SimpleNamespace(budget=fake_budget)
        saved = sys.modules.get("mlx_omarchy_serve")
        sys.modules["mlx_omarchy_serve"] = fake_pkg
        argv = types.SimpleNamespace(
            model=str(self.tmp / "pack"), max_context=64, model_id="bonsai-2-27b-mlx-2bit",
            owner="pid0-abc",
        )
        try:
            total = server_module._preflight_managed(argv, "res-name")
        finally:
            if saved is None:
                sys.modules.pop("mlx_omarchy_serve", None)
            else:
                sys.modules["mlx_omarchy_serve"] = saved
        self.assertGreater(total, 0)
        # The preflight estimate is the CONSERVATIVE bound: the whole
        # safetensors payload (LM tensors + excluded tensors), not just the
        # live LM weights.
        from mlx_omarchy_bonsai2 import loader as _loader

        facts = _loader.pack_footprint(self.tmp / "pack")
        self.assertGreaterEqual(calls["byte_count"], facts["total_header_bytes"] + 1024)
        self.assertGreater(calls["byte_count"], facts["live_weights_bytes"])
        self.assertEqual(calls["owner"], "pid0-abc")
        self.assertEqual(calls["state"], "pending")

    def test_bind_failure_releases_reservation(self):
        """F2: an HTTP bind failure after load must clear the reservation."""
        import types

        from mlx_omarchy_bonsai2 import server as server_module

        events = []

        def fake_atomic(name, byte_count, *, note="", owner=None, state="pending"):
            events.append("reserve")
            return types.SimpleNamespace(fits=True, lines=[])

        fake_budget = types.SimpleNamespace(
            estimate_required=lambda memory, context_tokens: types.SimpleNamespace(total=1024),
            admit_and_reserve=fake_atomic,
            set_reservation_state=lambda *a, **k: events.append("resident"),
            clear_reservation=lambda name, **k: events.append("clear"),
        )
        fake_pkg = types.SimpleNamespace(budget=fake_budget)
        saved = sys.modules.get("mlx_omarchy_serve")
        sys.modules["mlx_omarchy_serve"] = fake_pkg
        # Occupy the port so ThreadingHTTPServer's bind fails after load.
        blocker = socket.socket()
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        occupied = blocker.getsockname()[1]
        argv = [
            "--model", str(self.tmp / "pack"), "--host", "127.0.0.1",
            "--port", str(occupied), "--allow-cpu", "--managed",
        ]
        try:
            with self.assertRaises(OSError):
                server_module.serve_main(argv)
        finally:
            if saved is None:
                sys.modules.pop("mlx_omarchy_serve", None)
            else:
                sys.modules["mlx_omarchy_serve"] = saved
            blocker.close()
        self.assertIn("reserve", events)
        self.assertIn("clear", events)

    def test_rejects_max_context_over_trained_positions(self):
        """F1: --max-context above the pack's trained positions is rejected."""
        from mlx_omarchy_bonsai2 import server as server_module

        pack_dir = str(self.tmp / "pack")
        argv = ["--model", pack_dir, "--host", "127.0.0.1", "--port", "0",
                "--max-context", str(10 ** 9), "--allow-cpu"]
        with self.assertRaises(SystemExit) as ctx:
            server_module.serve_main(argv)
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    print("provenance: mlx %s on %s (CPU reference)" % (mx.__version__, mx.default_device()))
    unittest.main()
