"""Issue: speech turns collide with HTTP 409 "Another speech operation is
active" mid-dictation (bc-voice-followup-1903, third run 6:51 PM).

Contract under test:
1. A second speech turn arriving while one is active QUEUES for a bounded
   window instead of failing instantly; when the active turn finishes inside
   the window, the queued turn proceeds (200, not 409).
2. A queued turn barges in on an in-flight TTS: the active preview's loop
   observes its cancel event and ends early (via the same cancel path as
   /api/voice/cancel, i.e. VoiceError "preview produced no audio") instead of
   playing its full stream, so the queued turn takes the slot fast.
3. When the active turn never yields within the bound, the queued turn still
   gets the honest bounded BusyError (409) — not an unbounded hang.

Runs against the real HTTP server with stubbed recognition/synthesis so no
model or audio hardware is needed.
"""
import http.client
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from mlx_omarchy_assistant.server import AssistantServer


class SpeechTurnTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.server = AssistantServer(('127.0.0.1', 0), Path(self.temp.name))
        self.server._manager = self.server.manager  # materialize the real one
        self.server.manager.status = lambda: {"voice": {"requested": True}}
        self.server.speech_turns.wait_s = 5.0
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_port
        self.origin = f'http://127.0.0.1:{self.port}'
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.temp.cleanup()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=15)
        self.addCleanup(conn.close)
        data = json.dumps(body) if body is not None else None
        conn.request(method, path, data, headers or {})
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()

    def login(self):
        status, headers, data = self.request('POST', '/api/session',
            {'token': self.server.bootstrap},
            {'Origin': self.origin, 'Content-Type': 'application/json'})
        self.assertEqual(status, 200)
        return {'Cookie': headers['Set-Cookie'].split(';')[0], 'Origin': self.origin,
                'X-Assistant-CSRF': json.loads(data)['csrf'], 'Content-Type': 'application/json'}

    def post_speech(self, path, headers, body=None):
        send = dict(headers)
        if path == '/api/transcribe':
            send['Content-Type'] = 'audio/wav'
            conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=15)
            self.addCleanup(conn.close)
            conn.request('POST', path, body or b'wave-bytes', send)
            response = conn.getresponse()
            return response.status, response.read()
        return self.request('POST', path, body, send)

    def stub_transcribe(self, release):
        started = threading.Event()
        def transcribe(raw, cancel):
            started.set()
            release.wait(10)
            return "hello"
        self.server._recognition = SimpleNamespace(transcribe=transcribe, close=lambda: None)
        return started

    def test_second_turn_queues_and_completes_when_first_finishes(self):
        release = threading.Event()
        started = self.stub_transcribe(release)
        headers = self.login()
        first_status, first_body = [None], [None]

        def run_first():
            first_status[0], first_body[0] = self.post_speech('/api/transcribe', headers)

        first = threading.Thread(target=run_first, daemon=True)
        first.start()
        self.assertTrue(started.wait(5), "first turn never reached recognition")
        second_result = [None, None]

        def run_second():
            second_result[0], second_result[1] = self.post_speech('/api/transcribe', headers)

        second = threading.Thread(target=run_second, daemon=True)
        second.start()
        time.sleep(0.5)  # let the second turn reach the gate and start queueing
        release.set()
        second.join(15)
        first.join(15)
        self.assertEqual(first_status[0], 200)
        self.assertEqual(second_result[0], 200,
            f"queued second turn must proceed, got {second_result[0]} {second_result[1]!r}")

    def test_queued_turn_barges_in_on_active_tts_preview(self):
        pulls = {"first": 0, "second": 0}
        who = {"current": None}

        class FakeChunks:
            def __init__(self, cancel):
                self.cancel = cancel
                self.owner = who["current"]  # bound at creation, on the owner's thread
                self.left = 40  # ~2s at 50ms/chunk: barge-in must cut it far short
            def __iter__(self):
                return self
            def __next__(self):
                if self.left <= 0:
                    raise StopIteration
                self.left -= 1
                pulls[self.owner] += 1
                time.sleep(0.05)
                return SimpleNamespace(sample_rate=24000, data=b'\x00\x00')
            def close(self):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False

        def make_chunks(sentence, cancel):
            return FakeChunks(cancel)

        self.server._synthesis = SimpleNamespace(
            set_voice=lambda voice: {"voice": voice},
            synthesize_chunks=make_chunks,
            shutdown=lambda: None)
        grant = SimpleNamespace(release=lambda: None)
        self.server._coordinator = SimpleNamespace(
            speech=SimpleNamespace(enter=lambda cancel: grant), close=lambda: None)
        headers = self.login()
        first_status, first_body = [None], [None]

        def run_first():
            first_status[0], _h, first_body[0] = self.post_speech(
                '/api/voice/preview', headers, {})

        first = threading.Thread(target=run_first, daemon=True)
        who["current"] = "first"
        first.start()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and self.server.audio_kind != "tts":
            time.sleep(0.01)
        self.assertEqual(self.server.audio_kind, "tts", "first preview never became active")
        who["current"] = "second"
        second_status, _headers_b, second_body = self.post_speech(
            '/api/voice/preview', headers, {})
        who["current"] = None
        first.join(10)
        self.assertEqual(second_status, 200,
            f"queued preview must proceed after barge-in, got {second_status} {second_body!r}")
        self.assertLessEqual(pulls["first"], 3,
            f"barge-in must cut the active stream short, first pulled {pulls['first']} chunks")
        self.assertGreater(pulls["second"], pulls["first"],
            "the queued turn must run to its normal completion")
        self.assertEqual(first_status[0], 503,
            f"a barged-out preview ends via the existing cancel path (VoiceError), got {first_status[0]} {first_body!r}")

    def test_turn_still_reports_busy_after_the_bound(self):
        started = threading.Event()
        def stuck_transcribe(raw, cancel):
            started.set()
            time.sleep(3)
            return "hello"
        self.server._recognition = SimpleNamespace(transcribe=stuck_transcribe, close=lambda: None)
        self.server.speech_turns.wait_s = 0.3
        headers = self.login()
        first = threading.Thread(
            target=lambda: self.post_speech('/api/transcribe', headers), daemon=True)
        first.start()
        self.assertTrue(started.wait(5))
        t0 = time.monotonic()
        status, body = self.post_speech('/api/transcribe', headers)
        waited = time.monotonic() - t0
        first.join(10)
        self.assertEqual(status, 409, f"expected bounded busy, got {status} {body!r}")
        self.assertGreaterEqual(waited, 0.25,
            "busy must come only after the bounded wait, not instantly")


if __name__ == '__main__':
    unittest.main()
