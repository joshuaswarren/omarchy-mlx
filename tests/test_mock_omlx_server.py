"""Dev-box tests for packaging/omlx-linux/mock_omlx_server.py (issue 206 item 4).

The mock is the oMLX v0.7.0 wire-format stub that api_parity.sh sections run
against when no mlx hardware is present (receipt 2026-10-05-omlx-api-parity).
These tests pin the routes parity.sh curls and the STATE discipline
(alias_set / profile_created / usage_rows move only through their endpoints).

The server starts in-process on an ephemeral port (the CLI takes a literal
port, so the tests bind port 0 through ThreadingHTTPServer directly and read
the assigned port back). Stdlib only, under pytest and unittest.
"""

import http.client
import json
import sys
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUBS = ROOT / "packaging" / "omlx-linux"
sys.path.insert(0, str(STUBS))

import mock_omlx_server as mock_server  # noqa: E402


def _wait_until_ready(server: ThreadingHTTPServer) -> None:
    host, port = server.server_address[:2]
    for _ in range(100):
        try:
            conn = http.client.HTTPConnection(host, port, timeout=2)
            conn.request("GET", "/health")
            conn.getresponse().read()
            conn.close()
            return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("mock server never answered /health")


class MockServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), mock_server.Handler)
        cls.server.daemon_threads = True
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        _wait_until_ready(cls.server)
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        mock_server.STATE.clear()
        mock_server.STATE.update(alias_set=False, profile_created=False, usage_rows=0)

    # --- helpers -----------------------------------------------------------
    def req(self, method, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        payload = json.dumps(body).encode() if body is not None else None
        headers = {"content-type": "application/json"} if payload else {}
        conn.request(method, path, payload, headers)
        resp = conn.getresponse()
        data = resp.read()
        status, ctype = resp.status, resp.getheader("content-type") or ""
        conn.close()
        return status, ctype, data

    def get(self, path):
        status, ctype, data = self.req("GET", path)
        return status, (json.loads(data) if "json" in ctype else data)

    def sse_events(self, body: dict):
        status, ctype, data = self.req("POST", "/v1/chat/completions", body)
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", ctype)
        chunks = []
        for block in data.decode().split("\n\n"):
            for line in block.splitlines():
                if line.startswith("data: "):
                    chunks.append(line[len("data: "):])
        self.assertGreaterEqual(len(chunks), 1)
        return chunks

    # --- wire formats api_parity.sh depends on -----------------------------
    def test_health_ok(self):
        status, body = self.get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"status": "ok", "mock": True})

    def test_models_lists_models(self):
        status, body = self.get("/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(body["object"], "list")
        self.assertEqual([m["id"] for m in body["data"]], mock_server.MODELS)
        for m in body["data"]:
            self.assertEqual(m["object"], "model")
            self.assertEqual(m["max_model_len"], mock_server.REAL_CONTEXT)

    def test_chat_stream_ends_with_done_and_carries_usage_chunk(self):
        chunks = self.sse_events({
            "model": "m", "stream": True,
            "stream_options": {"include_usage": True},
        })
        self.assertEqual(chunks[-1], "[DONE]")
        parsed = [json.loads(c) for c in chunks[:-1]]
        for d in parsed:
            self.assertEqual(d["object"], "chat.completion.chunk")
        usage_chunks = [d for d in parsed if "usage" in d]
        self.assertEqual(len(usage_chunks), 1)
        self.assertEqual(usage_chunks[0]["usage"],
                         {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20})
        self.assertEqual(usage_chunks[0]["choices"], [])
        text = "".join(d["choices"][0]["delta"].get("content", "")
                       for d in parsed if d["choices"] and "content" in d["choices"][0]["delta"])
        self.assertEqual(text, "Red, blue, yellow — the primary colors.")
        # keepalive wire form (A3): empty assistant content on a keepalive model
        keepalives = [d for d in parsed if d["model"] == "keepalive"]
        self.assertGreater(len(keepalives), 0)
        for d in keepalives:
            self.assertEqual(d["choices"][0]["delta"],
                             {"role": "assistant", "content": ""})

    def test_chat_stream_without_include_usage_has_no_usage_chunk(self):
        chunks = self.sse_events({"model": "m", "stream": True})
        self.assertEqual(chunks[-1], "[DONE]")
        self.assertTrue(all("usage" not in json.loads(c) for c in chunks[:-1]))

    def test_chat_stream_with_tools_is_one_tool_call_then_done(self):
        chunks = self.sse_events({
            "model": "m", "stream": True,
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        })
        self.assertEqual(chunks[-1], "[DONE]")
        parsed = [json.loads(c) for c in chunks[:-1]]
        self.assertEqual(len(parsed), 1)
        tc = parsed[0]["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(tc["function"]["name"], "get_weather")
        self.assertEqual(json.loads(tc["function"]["arguments"]), {"city": "Paris"})

    def test_chat_nonstream_carries_usage_and_stop(self):
        status, _, data = self.req("POST", "/v1/chat/completions", {"model": "m"})
        body = json.loads(data)
        self.assertEqual(status, 200)
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(body["usage"], {"prompt_tokens": 12, "completion_tokens": 8,
                                         "total_tokens": 20})
        self.assertEqual(body["choices"][0]["message"]["content"], "Red, blue, and yellow.")

    def test_completion_messages_embeddings_rerank_search_routes(self):
        status, _, data = self.req("POST", "/v1/completions", {"model": "m", "prompt": "colors"})
        body = json.loads(data)
        self.assertEqual(status, 200)
        self.assertEqual(body["object"], "text_completion")
        self.assertEqual(body["choices"][0]["text"], " red, blue, and yellow.")
        self.assertEqual(body["usage"]["prompt_tokens"], 6)

        status, _, data = self.req("POST", "/v1/messages/count_tokens", {"model": "m"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data), {"input_tokens": 12})

        status, _, data = self.req("POST", "/v1/embeddings", {"model": "embed", "input": ["a", "b", "c"]})
        body = json.loads(data)
        self.assertEqual(status, 200)
        self.assertEqual([d["index"] for d in body["data"]], [0, 1, 2])
        self.assertEqual(len(body["data"][0]["embedding"]), 24)
        self.assertEqual(body["usage"], {"prompt_tokens": 9})

        status, _, data = self.req("POST", "/v1/rerank", {"model": "m", "query": "q", "documents": ["a", "b"]})
        body = json.loads(data)
        self.assertEqual(status, 200)
        self.assertEqual([r["relevance_score"] for r in body["results"]], [0.93, 0.02])

        status, _, data = self.req("POST", "/v1/web/search", {"query": "einstein"})
        body = json.loads(data)
        self.assertEqual((status, body["ok"]), (200, True))
        self.assertIn("Einstein", body["results"][0]["title"])
        for path in ("/api/web-search/test", "/admin/api/web-search/test"):
            status, _, data = self.req("POST", path, {})
            self.assertEqual((status, json.loads(data)["ok"]), (200, True))

    def test_messages_nonstream_and_stream_wire_shapes(self):
        status, _, data = self.req("POST", "/v1/messages",
                                   {"model": "m", "max_tokens": 16})
        body = json.loads(data)
        self.assertEqual(status, 200)
        self.assertEqual(body["type"], "message")
        self.assertEqual(body["stop_reason"], "end_turn")
        self.assertEqual(body["content"], [{"type": "text", "text": "Red, blue, and yellow."}])
        self.assertEqual(body["usage"], {"input_tokens": 11, "output_tokens": 9})

        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("POST", "/v1/messages", json.dumps(
            {"model": "m", "max_tokens": 16, "stream": True}).encode(),
            {"content-type": "application/json"})
        resp = conn.getresponse()
        self.assertIn("text/event-stream", resp.getheader("content-type"))
        text = resp.read().decode()
        conn.close()
        events = [e.split("\n")[0][len("event: "):]
                  for e in text.split("\n\n") if e.startswith("event: ")]
        self.assertEqual(events, ["message_start", "content_block_start",
                                  "content_block_delta", "content_block_stop",
                                  "message_delta", "message_stop"])

    def test_messages_stream_thinking_delta_carries_the_text(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("POST", "/v1/messages", json.dumps(
            {"model": "m", "max_tokens": 16, "stream": True,
             "thinking": {"type": "enabled"}}).encode(),
            {"content-type": "application/json"})
        resp = conn.getresponse()
        text = resp.read().decode()
        conn.close()
        events = []
        for block in text.split("\n\n"):
            if not block.startswith("event: "):
                continue
            lines = block.splitlines()
            name = lines[0][len("event: "):]
            data = json.loads(lines[1][len("data: "):])
            events.append((name, data))
        names = [n for n, _ in events]
        self.assertEqual(names, ["message_start", "content_block_start",
                                 "content_block_delta", "content_block_stop",
                                 "content_block_start", "content_block_delta",
                                 "content_block_stop", "message_delta",
                                 "message_stop"])
        deltas = {d["delta"]["type"]: d["delta"] for n, d in events
                  if n == "content_block_delta"}
        self.assertEqual(deltas["thinking_delta"]["thinking"], "17*23=391")
        self.assertEqual(deltas["text_delta"]["text"], "391")

    def test_messages_thinking_block_when_enabled(self):
        status, _, data = self.req("POST", "/v1/messages",
                                   {"model": "m", "max_tokens": 16,
                                    "thinking": {"type": "enabled"}})
        body = json.loads(data)
        self.assertEqual([b["type"] for b in body["content"]], ["thinking", "text"])

    # --- STATE discipline --------------------------------------------------
    def test_state_flags_move_only_through_their_endpoints(self):
        model = mock_server.MODELS[0]
        # reads move nothing
        for path in ("/health", "/v1/models", "/api/usage"):
            self.get(path)
        self.assertEqual(mock_server.STATE,
                         {"alias_set": False, "profile_created": False, "usage_rows": 0})

        # alias_set: only PUT .../settings with a model_alias key
        status, _, _ = self.req("PUT", f"/api/models/{model}/settings",
                                {"temperature": 0.7})
        self.assertEqual(status, 200)
        self.assertFalse(mock_server.STATE["alias_set"])
        status, _, _ = self.req("PUT", f"/api/models/{model}/settings",
                                {"model_alias": "parity-alias-model"})
        self.assertEqual(status, 200)
        self.assertTrue(mock_server.STATE["alias_set"])
        _, body = self.get("/v1/models")
        self.assertEqual(body["data"][0]["id"], mock_server.ALIAS)
        self.assertNotIn(model, [m["id"] for m in body["data"]])
        status, _, _ = self.req("PUT", f"/admin/api/models/{model}/settings",
                                {"model_alias": None})
        self.assertFalse(mock_server.STATE["alias_set"])

        # profile_created: only POST .../profiles with expose_as_model, cleared by DELETE
        status, _, _ = self.req("POST", f"/api/models/{model}/profiles", {"name": "parity"})
        self.assertEqual(status, 200)
        self.assertFalse(mock_server.STATE["profile_created"])
        status, _, _ = self.req("POST", f"/api/models/{model}/profiles",
                                {"name": "parity", "expose_as_model": "parity:parity"})
        self.assertEqual(status, 200)
        self.assertTrue(mock_server.STATE["profile_created"])
        _, body = self.get("/v1/models")
        self.assertIn(mock_server.EXPOSED, [m["id"] for m in body["data"]])
        status, _, _ = self.req("DELETE", f"/admin/api/models/{model}/profiles/parity")
        self.assertEqual(status, 200)
        self.assertFalse(mock_server.STATE["profile_created"])
        _, body = self.get("/v1/models")
        self.assertNotIn(mock_server.EXPOSED, [m["id"] for m in body["data"]])

        # usage_rows: only chat completions bump it; reads and other posts do not
        _, before = self.get("/admin/api/usage?range=today&include_details=true")
        self.req("POST", "/v1/completions", {"model": "m"})
        self.req("POST", "/v1/messages", {"model": "m", "max_tokens": 8})
        _, after_one = self.get("/admin/api/usage?range=today")
        self.assertEqual(after_one["totals"], before["totals"])
        self.req("POST", "/v1/chat/completions", {"model": "m"})
        self.req("POST", "/v1/chat/completions", {"model": "m", "stream": True})
        _, after = self.get("/admin/api/usage?range=today")
        self.assertEqual(after["totals"][0], before["totals"][0] + 2)
        self.assertEqual(after["models"][model][0], before["totals"][0] + 2)

    def test_profile_apply_route(self):
        status, _, data = self.req("POST", "/api/models/x/profiles/parity/apply", {})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["settings"], {"temperature": 0.5})

    # --- unknown paths -----------------------------------------------------
    def test_unknown_paths_are_non_200_on_every_method(self):
        for method in ("GET", "POST", "PUT", "DELETE"):
            status, _, data = self.req(method, "/definitely-not-a-route", {} if method != "GET" else None)
            self.assertEqual(status, 404)
            self.assertEqual(json.loads(data), {"detail": "not found"})
        # parity.sh curls /api/status but the stub deliberately has no such
        # route; pinned so a silent 200 can never appear here.
        status, _, _ = self.req("GET", "/api/status")
        self.assertEqual(status, 404)

    def test_chat_response_format_and_websearch_trigger_nonstream(self):
        status, _, data = self.req("POST", "/v1/chat/completions",
                                   {"model": "m", "response_format": {"type": "json_object"}})
        self.assertEqual(json.loads(data)["choices"][0]["message"]["content"],
                         '{"name": "Ada", "age": 36}')
        status, _, data = self.req("POST", "/v1/chat/completions",
                                   {"model": "m", "messages": [
                                       {"role": "user", "content": "use the web_search tool"}]})
        self.assertEqual(json.loads(data)["choices"][0]["message"]["content"],
                         "Albert Einstein was born in the year 1879.")
        status, _, data = self.req("POST", "/v1/chat/completions",
                                   {"model": "m", "tools": [{"type": "function"}]})
        msg = json.loads(data)["choices"][0]["message"]
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "get_weather")


if __name__ == "__main__":
    unittest.main()
