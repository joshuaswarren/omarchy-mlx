"""Failing-first tests for the SSE TTFT probe (scripts-local/ttft_probe.py).

Mock SSE server replays a scripted chunk timeline with real delays:
  - chunk A at ~0.05s: reasoning-only delta (content empty)  -> must NOT start TTFT
  - chunk B at ~0.30s: first non-empty content delta         -> TTFT lands HERE
  - chunk C at ~0.40s: another content delta
  - [DONE]

Asserts:
  1. ttft_s is within the chunk-B window (>= 0.25, <= 0.45) — a buffered
     reader (v3's 4096-byte defect) would report ~0.45+ for everything.
  2. ttft_s < total_s (strictly).
  3. first_token_event_index == 2 (chunk B is the 2nd data event).
  4. A response with NO content deltas returns an error, not a fake TTFT.
"""

import json
import sys
import threading
import time
import types
import unittest
from unittest import mock
import http.server
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts-local"))

import ttft_probe  # noqa: E402
from ttft_probe import measure_ttft  # noqa: E402


def sse_bytes(script):
    out = b""
    for delay, payload in script:
        time.sleep(delay)
        out += b"data: " + json.dumps(payload).encode() + b"\n\n"
    out += b"data: [DONE]\n\n"
    return out


class _MockSSE(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for chunk in self.server.script_chunks:
            time.sleep(chunk["delay"])
            payload = (b"data: " + json.dumps(chunk["payload"]).encode()
                       + b"\n\n")
            self.wfile.write(f"{len(payload):x}\r\n".encode() + payload
                             + b"\r\n")
            self.wfile.flush()
        tail = b"data: [DONE]\n\n"
        self.wfile.write(f"{len(tail):x}\r\n".encode() + tail + b"\r\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def log_message(self, *_a):
        pass


def _delta(content, reasoning=None):
    delta = {}
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    if content is not None:
        delta["content"] = content
    return {"choices": [{"index": 0, "delta": delta}],
            "object": "chat.completion.chunk"}


class TtftProbeTests(unittest.TestCase):
    def setUp(self):
        self.server = http.server.HTTPServer(("127.0.0.1", 0), _MockSSE)
        self.port = self.server.server_port
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.body = {"model": "m", "messages": [{"role": "user", "content": "x"}],
                     "max_tokens": 8, "stream": True}

    def test_ttft_lands_on_first_nonempty_content_token(self):
        self.server.script_chunks = [
            {"delay": 0.05, "payload": _delta(None, reasoning="thinking...")},
            {"delay": 0.25, "payload": _delta("Paris")},
            {"delay": 0.10, "payload": _delta(" is the city.")},
        ]
        result = measure_ttft("127.0.0.1", self.port, self.body, timeout=10)
        self.assertIsNone(result.get("error"), result)
        # first_event_s: the reasoning delta at ~0.05s starts the clock.
        self.assertLessEqual(result["first_event_s"], 0.20,
                             f"first_event_s missed the reasoning token: {result}")
        # first_content_s: chunk B lands at ~0.30s; a buffered reader or a
        # first-byte reader would report ~0.05s or ~0.45s instead.
        self.assertGreaterEqual(result["first_content_s"], 0.20,
                                f"first_content_s too early: {result}")
        self.assertLessEqual(result["first_content_s"], 0.50,
                             f"first_content_s too late (buffered read?): {result}")
        # The reasoning lead is reported separately, not folded into content.
        self.assertGreater(result["reasoning_lead_s"], 0.15)
        self.assertLess(result["first_content_s"], result["total_s"])
        self.assertEqual(result["first_content_event_index"], 2)
        self.assertEqual(result["events"], 3)

    def test_content_first_reports_equal_clocks(self):
        """No reasoning deltas: first_event_s == first_content_s."""
        self.server.script_chunks = [
            {"delay": 0.10, "payload": _delta("direct answer")},
            {"delay": 0.05, "payload": _delta(" more")},
        ]
        # A fake clock that advances on every read: the two figures can only
        # match if the probe reads the clock once for the event.
        ticks = iter(range(1, 1000))
        fake_time = types.SimpleNamespace(
            perf_counter=lambda: next(ticks) * 0.001)
        with mock.patch.object(ttft_probe, "time", fake_time):
            result = measure_ttft("127.0.0.1", self.port, self.body, timeout=10)
        self.assertIsNone(result.get("error"), result)
        self.assertEqual(result["first_event_s"], result["first_content_s"])
        self.assertEqual(result["reasoning_lead_s"], 0)

    def test_no_content_deltas_is_an_error_not_fake_ttft(self):
        self.server.script_chunks = [
            {"delay": 0.05, "payload": _delta(None, reasoning="only reasoning")},
            {"delay": 0.05, "payload": _delta("")},  # empty content does not count
        ]
        result = measure_ttft("127.0.0.1", self.port, self.body, timeout=10)
        self.assertIn("error", result)
        self.assertNotIn("first_content_s", result)
        # But the reasoning-start clock IS recorded for the record.
        self.assertIsNotNone(result.get("first_event_s"))


if __name__ == "__main__":
    unittest.main()
