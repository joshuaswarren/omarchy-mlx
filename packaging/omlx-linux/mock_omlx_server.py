#!/usr/bin/env python3
"""Mock oMLX server for dev-box harness runs — HTTP/SSE protocol stub ONLY.

This emulates the oMLX v0.7.0 wire formats (learned from omlx/server.py at pin
4d4f5a28) so api_parity.sh sections can be exercised WITHOUT mlx hardware.
It is harness scaffolding: responses are canned and prove NOTHING about real
model behavior. Real receipts come from M2/jwm1 runs against `omlx serve`.

Usage: mock_omlx_server.py <port>
"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = [
    "mlx-community--Qwen3-4B-Instruct-2507-4bit",
    "mlx-community--all-MiniLM-L6-v2",
    "mlx-community--Qwen3-Reranker-0.6B-4bit",
]
REAL_CONTEXT = 262144
ALIAS = "parity-alias-model"
EXPOSED = "parity-alias-model:parity"

STATE = {"alias_set": False, "profile_created": False, "usage_rows": 0}


def sse_chunk(cid, model, delta=None, usage=None, last=False):
    d = {"id": cid, "object": "chat.completion.chunk", "created": 0, "model": model,
         "choices": [] if (last or delta is None) else [{"index": 0, "delta": delta, "finish_reason": None}]}
    if usage:
        d["usage"] = usage
    return "data: " + json.dumps(d) + "\n\n"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _body(self):
        n = int(self.headers.get("content-length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path == "/health":
            return self._json({"status": "ok", "mock": True})
        if self.path.startswith("/v1/models"):
            ids = [ALIAS if (m == MODELS[0] and STATE["alias_set"]) else m for m in MODELS]
            if STATE["profile_created"]:
                ids.append(EXPOSED)
            return self._json({"object": "list", "data": [
                {"id": i, "object": "model", "max_model_len": REAL_CONTEXT} for i in ids]})
        if "/api/usage" in self.path:
            return self._json({"range": "today", "enabled": True,
                               "models": {MODELS[0]: [STATE["usage_rows"], 10, 10]},
                               "totals": [STATE["usage_rows"], 10, 10]})
        if "/profiles" in self.path and self.path.endswith("/parity"):
            return self._json({"model_id": MODELS[0], "settings": {"temperature": 0.5}})
        return self._json({"detail": "not found"}, 404)

    def do_PUT(self):
        if "/settings" in self.path and "/models/" in self.path:
            b = self._body()
            if "model_alias" in b:
                STATE["alias_set"] = bool(b["model_alias"])
            return self._json({"model_id": MODELS[0], "settings": b})
        return self._json({"detail": "not found"}, 404)

    def do_POST(self):
        p = self.path.split("?")[0]
        if p == "/v1/chat/completions":
            return self._chat()
        if p == "/v1/completions":
            self._body()
            return self._json({"id": "cmpl-m", "object": "text_completion",
                               "choices": [{"text": " red, blue, and yellow."}],
                               "usage": {"prompt_tokens": 6, "completion_tokens": 6}})
        if p == "/v1/messages":
            return self._messages()
        if p == "/v1/messages/count_tokens":
            return self._json({"input_tokens": 12})
        if p == "/v1/embeddings":
            vecs = [[1.0, 0.2, 0.1] * 8, [0.9, 0.3, 0.1] * 8, [0.1, 0.0, 1.0] * 8]
            return self._json({"object": "list", "data": [
                {"object": "embedding", "index": i, "embedding": v}
                for i, v in enumerate(vecs)], "model": "embed", "usage": {"prompt_tokens": 9}})
        if p in ("/v1/web/search",):
            return self._json({"ok": True, "results": [
                {"title": "Albert Einstein - Wikipedia", "url": "https://example.org/einstein",
                 "snippet": "Albert Einstein was born on 14 March 1879."}]})
        if p == "/v1/rerank":
            return self._json({"results": [
                {"index": 0, "relevance_score": 0.93}, {"index": 1, "relevance_score": 0.02}]})
        if p in ("/api/web-search/test", "/admin/api/web-search/test"):
            return self._json({"ok": True, "results": [
                {"title": "Albert Einstein", "url": "https://example.org/einstein",
                 "snippet": "Born 1879"}]})
        if p.endswith("/profiles/parity/apply"):
            return self._json({"model_id": MODELS[0], "settings": {"temperature": 0.5}})
        if p.endswith("/profiles") and p.startswith(("/api/models/", "/admin/api/models/")):
            b = self._body()
            if b.get("expose_as_model"):
                STATE["profile_created"] = True
            return self._json({"name": b.get("name", "parity"), "exposed": True})
        return self._json({"detail": "not found"}, 404)

    def do_DELETE(self):
        if self.path.endswith("/profiles/parity"):
            STATE["profile_created"] = False
            return self._json({"deleted": True})
        return self._json({"detail": "not found"}, 404)

    # --- streaming endpoints (wire shapes copied from omlx/server.py) ---
    def _chat(self):
        STATE["usage_rows"] += 1
        b = self._body()
        cid = "chatcmpl-mock"
        if b.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            usage = {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}
            if b.get("tools"):
                tc = {"id": "c1", "type": "function", "function":
                      {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
                self.wfile.write(sse_chunk(cid, b["model"],
                                           {"role": "assistant", "tool_calls": [tc],
                                            "content": None}).encode())
                self.wfile.write(b"data: [DONE]\n\n")
                return
            for w in ["Red,", " blue,", " yellow", " —", " the", " primary", " colors."]:
                self.wfile.write(sse_chunk(cid, "keepalive", {"role": "assistant", "content": ""}).encode())
                self.wfile.write(sse_chunk(cid, b["model"], {"content": w}).encode())
                self.wfile.flush()
                time.sleep(0.05)
            if (b.get("stream_options") or {}).get("include_usage"):
                self.wfile.write(sse_chunk(cid, b["model"], usage=usage).encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            msg = {"role": "assistant", "content": "Red, blue, and yellow."}
            if b.get("response_format"):
                msg = {"role": "assistant", "content": '{"name": "Ada", "age": 36}'}
            elif "web_search tool" in json.dumps(b.get("messages", [])):
                msg = {"role": "assistant", "content": "Albert Einstein was born in the year 1879."}
            if b.get("tools"):
                msg["tool_calls"] = [{"id": "c1", "type": "function", "function":
                                      {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]
            self._json({"id": cid, "object": "chat.completion",
                        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}})

    def _messages(self):
        b = self._body()
        blocks = [{"type": "text", "text": "Red, blue, and yellow."}]
        if b.get("thinking", {}).get("type") == "enabled":
            blocks = [{"type": "thinking", "thinking": "17*23=391"}, {"type": "text", "text": "391"}]
        if not b.get("stream"):
            return self._json({"id": "msg-m", "type": "message", "role": "assistant",
                               "content": blocks, "stop_reason": "end_turn",
                               "usage": {"input_tokens": 11, "output_tokens": 9}})
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()

        def ev(name, data):
            self.wfile.write(f"event: {name}\ndata: ".encode() +
                             json.dumps(data).encode() + b"\n\n")

        ev("message_start", {"type": "message_start", "message": {
            "id": "msg-m", "type": "message", "role": "assistant",
            "content": [], "usage": {"input_tokens": 11, "output_tokens": 1}}})
        for i, blk in enumerate(blocks):
            key = "text" if blk["type"] == "text" else "thinking"
            ev("content_block_start", {"type": "content_block_start", "index": i,
                                       "content_block": {"type": blk["type"]}})
            ev("content_block_delta", {"type": "content_block_delta", "index": i,
                                       "delta": {"type": key + "_delta", key: blk.get(key)}})
            ev("content_block_stop", {"type": "content_block_stop", "index": i})
        ev("message_delta", {"type": "message_delta",
                             "delta": {"stop_reason": "end_turn"},
                             "usage": {"output_tokens": 9}})
        ev("message_stop", {"type": "message_stop"})


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
