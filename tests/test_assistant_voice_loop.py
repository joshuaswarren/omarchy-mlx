"""Offline voice loop: wake + VAD endpointing -> STT -> streamed local reply ->
sentence-streamed TTS -> sink. A loopback WAV round trip with stub models and a
local SSE server; real Parakeet, Kokoro and openWakeWord runs are receipts, not CI.
"""

import io
import json
import sys
import tempfile
import threading
import unittest
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_assistant import voice_loop  # noqa: E402
from mlx_omarchy_assistant.synthesis import PcmChunk  # noqa: E402

RATE = 16000


def tone(seconds, amp=8000):
    t = np.arange(int(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.int16).tobytes()


def silence(seconds):
    return bytes(int(seconds * RATE) * 2)


class PhraseEndDetector:
    """Stands in for WakeWordDetector: like openWakeWord, it fires when the
    wake phrase ends (the first quiet chunk after loud audio), once."""

    def __init__(self):
        self.loud = self.fired = False

    def process(self, pcm):
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
        rms = np.sqrt(np.mean(x ** 2))
        if self.fired:
            return False
        if rms > 1000:
            self.loud = True
        elif self.loud:
            self.fired = True
            return True
        return False


def sse_server(chunks, gate=None):
    """Local OpenAI-style streaming server. If gate is set, the chunk after the
    first sentence waits for it, so the test can see TTS start mid-stream."""
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i, text in enumerate(chunks):
                if gate is not None and i == 1:
                    seen["tts_before_rest"] = gate.wait(2)
                event = {"choices": [{"delta": {"content": text}}]}
                self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


class LoopbackRoundTrip(unittest.TestCase):
    def test_wav_in_to_wav_out_streams_each_sentence(self):
        speech = tone(1.0, amp=6000)
        source = silence(0.5) + tone(0.4) + silence(0.2) + speech + silence(1.0)
        gate = threading.Event()
        server, seen = sse_server(["It is noon. ", "The sun", " is up!"], gate)
        self.addCleanup(server.shutdown)
        heard, spoken = {}, []

        def transcribe(wav_bytes, cancel=None):
            with wave.open(io.BytesIO(wav_bytes)) as w:
                heard["rate"], heard["seconds"] = w.getframerate(), w.getnframes() / w.getframerate()
            return "what time is it"

        def synthesize(text, cancel):
            spoken.append(text)
            gate.set()
            yield PcmChunk(24000, bytes(480 * len(spoken)))

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "reply.wav"
            sink = voice_loop.WavSink(out, 24000)
            loop = voice_loop.VoiceLoop(
                voice_loop.Endpointer(PhraseEndDetector(), rms_floor=500, end_silence_s=0.5),
                transcribe, "http://127.0.0.1:%d/v1/chat/completions" % server.server_port,
                synthesize, sink, model="local-model")
            turns = loop.run(voice_loop.chunks(source))
            sink.close()
            with wave.open(str(out)) as w:
                frames, rate = w.getnframes(), w.getframerate()

        self.assertEqual(heard["rate"], RATE)
        self.assertGreaterEqual(heard["seconds"], 1.0)
        self.assertLessEqual(heard["seconds"], 1.0 + 0.5 + 0.2)
        self.assertEqual(seen["body"]["stream"], True)
        self.assertEqual(seen["body"]["messages"][-1], {"role": "user", "content": "what time is it"})
        self.assertEqual(spoken, ["It is noon.", "The sun is up!"])
        self.assertTrue(seen["tts_before_rest"], "the first sentence was not spoken while the reply was still streaming")
        self.assertEqual((rate, frames), (24000, 240 + 480))
        ms = turns[0].pop("ms")
        self.assertEqual(turns, [{"heard": "what time is it", "reply": "It is noon. The sun is up!", "sentences": 2}])
        self.assertEqual(set(ms), {"endpoint", "stt", "first_token", "first_audio", "total"})
        self.assertGreaterEqual(ms["endpoint"], 500)  # the configured end silence, in whole 80 ms chunks
        self.assertLessEqual(ms["endpoint"], 580)
        self.assertTrue(0 <= ms["stt"] <= ms["first_token"] <= ms["first_audio"] <= ms["total"], ms)


class Offline(unittest.TestCase):
    def test_reply_url_must_be_loopback(self):
        for url in ("http://example.com/v1/chat/completions", "http://192.168.1.5:8080/v1/chat/completions"):
            with self.assertRaises(ValueError):
                list(voice_loop.stream_reply(url, [], model="m"))

    def test_sentences_split_on_terminal_punctuation_and_flush_the_rest(self):
        self.assertEqual(list(voice_loop.sentences(["Hi", " there. How", " are you? Fine"])),
                         ["Hi there.", "How are you?", "Fine"])

    def test_pipewire_sink_plays_through_the_default_sink(self):
        calls = {}

        class FakeProc:
            def __init__(self, argv, stdin):
                calls["argv"], self.stdin = argv, io.BytesIO()
                self.stdin.close = lambda: None

            def wait(self, timeout=None):
                return 0

        sink = voice_loop.PipeWireSink(24000, popen=FakeProc)
        sink.write(b"\x01\x02")
        sink.close()
        self.assertEqual(calls["argv"][0], "pw-play")
        self.assertIn("24000", calls["argv"])
        self.assertFalse(any(a.startswith(("hw:", "plughw:")) for a in calls["argv"]))


if __name__ == "__main__":
    unittest.main()
