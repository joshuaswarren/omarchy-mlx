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
import unittest.mock
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_assistant import voice_loop  # noqa: E402
from mlx_omarchy_assistant.synthesis import PcmChunk, VoiceOutputLimitError  # noqa: E402

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
            yield PcmChunk(24000, bytes(480 * len(spoken)))

        class GateSink(voice_loop.WavSink):
            def write(self, pcm):
                super().write(pcm)
                gate.set()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "reply.wav"
            sink = GateSink(out, 24000)
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
        self.assertEqual(seen["body"]["seed"], 0)  # mlx_lm.server serves a seeded request on its single-request path
        self.assertEqual(seen["body"]["temperature"], 0)  # so the seed can never pin a sampled reply
        self.assertEqual(seen["body"]["messages"][0], {"role": "system", "content": voice_loop.VOICE_SYSTEM_PROMPT})
        self.assertEqual(seen["body"]["max_tokens"], voice_loop.REPLY_MAX_TOKENS)
        self.assertEqual(seen["body"]["messages"][-1], {"role": "user", "content": "what time is it"})
        self.assertEqual(spoken, ["It is noon.", "The sun is up!"])
        self.assertTrue(seen["tts_before_rest"], "the first sentence was not spoken while the reply was still streaming")
        self.assertEqual((rate, frames), (24000, 240 + 480))
        ms = turns[0].pop("ms")
        self.assertEqual(turns, [{"heard": "what time is it", "reply": "It is noon. The sun is up!", "sentences": 2, "truncated": False}])
        self.assertEqual(set(ms), {"endpoint", "stt", "first_token", "first_audio", "total"})
        self.assertGreaterEqual(ms["endpoint"], 500)  # the configured end silence, in whole 80 ms chunks
        self.assertLessEqual(ms["endpoint"], 580)
        self.assertTrue(0 <= ms["stt"] <= ms["first_token"] <= ms["first_audio"] <= ms["total"], ms)


class OutputBound(unittest.TestCase):
    def test_the_output_bound_truncates_the_reply_instead_of_aborting(self):
        server, _seen = sse_server(["A very long first sentence. ", "A second one."])
        self.addCleanup(server.shutdown)
        spoken = []

        def synthesize(text, cancel):
            spoken.append(text)
            yield PcmChunk(24000, bytes(480))
            raise VoiceOutputLimitError("synthesis exceeded the 30s output bound")

        with tempfile.TemporaryDirectory() as tmp:
            sink = voice_loop.WavSink(Path(tmp) / "r.wav", 24000)
            loop = voice_loop.VoiceLoop(
                voice_loop.Endpointer(PhraseEndDetector(), rms_floor=500), lambda wav, c=None: "tell me everything",
                "http://127.0.0.1:%d/v1/chat/completions" % server.server_port, synthesize, sink, model="m")
            result = loop.turn(bytes(3200))
            sink.close()
            with wave.open(str(Path(tmp) / "r.wav")) as w:
                frames = w.getnframes()
        self.assertEqual(spoken, ["A very long first sentence."])
        self.assertEqual(frames, 240)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["reply"], "A very long first sentence.")


class Offline(unittest.TestCase):
    def test_reply_url_must_be_loopback(self):
        for url in ("http://example.com/v1/chat/completions", "http://192.168.1.5:8080/v1/chat/completions"):
            with self.assertRaises(ValueError):
                list(voice_loop.stream_reply(url, [], model="m"))

    def test_the_first_clause_goes_to_tts_once_it_has_four_words(self):
        self.assertEqual(list(voice_loop.sentences(["The capital", " of France,", " as you know, is Paris. Next."])),
                         ["The capital of France,", "as you know, is Paris.", "Next."])

    def test_a_short_first_clause_waits_for_the_sentence(self):
        self.assertEqual(list(voice_loop.sentences(["Yes, it is.", " Done."])), ["Yes, it is.", "Done."])

    def test_numbers_times_and_decimals_never_cut_a_clause(self):
        for text, first in (
                ("The total cost was 1,000 dollars today. Ok.", "The total cost was 1,000 dollars today."),
                ("We will all meet at 10:30 tomorrow morning. Bye.", "We will all meet at 10:30 tomorrow morning."),
                ("The average test score was 3.5 points overall. Done.", "The average test score was 3.5 points overall.")):
            self.assertEqual(list(voice_loop.sentences([text])), [first, text.rsplit(" ", 1)[1]])

    def test_semicolon_and_colon_end_the_first_clause(self):
        self.assertEqual(list(voice_loop.sentences(["Here is the plan for today: we rest. Then work."])),
                         ["Here is the plan for today:", "we rest.", "Then work."])
        self.assertEqual(list(voice_loop.sentences(["I like the idea a lot; it works. Ok."])),
                         ["I like the idea a lot;", "it works.", "Ok."])

    def test_abbreviations_never_end_a_piece(self):
        self.assertEqual(list(voice_loop.sentences(["Pick small pets e.g., a cat or dog. Ok."])),
                         ["Pick small pets e.g., a cat or dog.", "Ok."])
        self.assertEqual(list(voice_loop.sentences(["Use one tool, i.e., a saw. Ok."])),
                         ["Use one tool, i.e., a saw.", "Ok."])
        self.assertEqual(
            list(voice_loop.sentences(["Dr. Smith lives", " on Main St. in the U.S. today. Mr. Lee",
                                       " starts at 9 a.m. tomorrow. Bye."])),
            ["Dr. Smith lives on Main St. in the U.S. today.", "Mr. Lee starts at 9 a.m. tomorrow.", "Bye."])

    def test_a_pause_between_two_spoken_sentences_stays_one_turn(self):
        # Kokoro leaves up to 0.72 s between the sentences of one question
        # (measured on the hardware input); the default end silence must outlast it.
        ep = voice_loop.Endpointer(PhraseEndDetector(), rms_floor=500)
        source = tone(0.4) + silence(0.24) + tone(0.8, 6000) + silence(0.72) + tone(0.8, 6000) + silence(1.5)  # whole 80 ms chunks
        turns = [u for u in map(ep.feed, voice_loop.chunks(source)) if u is not None]
        self.assertEqual(len(turns), 1)
        self.assertGreaterEqual(len(turns[0]) / 2 / RATE, 0.8 + 0.72 + 0.8)

    def test_line_breaks_end_a_piece_and_markdown_is_not_spoken(self):
        self.assertEqual(
            list(voice_loop.sentences(["Here are", " the options\n\n- **Sum**: 20\n", "### Next\n1. `Product` is 30"])),
            ["Here are the options", "Sum: 20", "Next", "Product is 30"])

    def test_sentences_split_on_terminal_punctuation_and_flush_the_rest(self):
        self.assertEqual(list(voice_loop.sentences(["Hi", " there. How", " are you? Fine"])),
                         ["Hi there.", "How are you?", "Fine"])

    def test_pipewire_sink_plays_raw_pcm_and_flushes_each_chunk(self):
        calls = {"flushes": 0}

        class Stdin(io.BytesIO):
            def flush(self):
                calls["flushes"] += 1

            def close(self):
                pass

        class FakeProc:
            def __init__(self, argv, stdin):
                calls["argv"], self.stdin = argv, Stdin()

            def wait(self, timeout=None):
                return 0

        sink = voice_loop.PipeWireSink(24000, popen=FakeProc)
        sink.write(b"\x01\x02")
        sink.write(b"\x03\x04")
        sink.close()
        argv = calls["argv"]
        self.assertEqual(argv[0], "pw-play")
        for flag in ("--raw", "--rate=24000", "--channels=1", "--format=s16"):
            self.assertIn(flag, argv)
        self.assertEqual(calls["flushes"], 2)
        self.assertFalse(any(a.startswith(("hw:", "plughw:")) for a in argv))

    def test_microphone_records_raw_pcm(self):
        calls = {}

        class FakeProc:
            def __init__(self, argv, stdout):
                calls["argv"], self.stdout = argv, io.BytesIO(bytes(2560 * 2))

            def terminate(self):
                calls["terminated"] = True

        got = list(voice_loop.microphone(popen=FakeProc))
        self.assertEqual(len(got), 2)
        self.assertEqual(calls["argv"][0], "pw-record")
        for flag in ("--raw", "--rate=16000", "--channels=1", "--format=s16"):
            self.assertIn(flag, calls["argv"])
        self.assertTrue(calls["terminated"])


class ReplyTransport(unittest.TestCase):
    def serve(self, handler_body):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                handler_body(self)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        return "http://127.0.0.1:%d/v1/chat/completions" % server.server_port

    def test_reply_url_must_be_http_or_https(self):
        for url in ("ftp://127.0.0.1/x", "file:///etc/passwd", "127.0.0.1:8080/v1/chat/completions"):
            with self.assertRaises(ValueError):
                list(voice_loop.stream_reply(url, [], model="m"))

    def test_a_redirect_is_not_followed(self):
        def redirect(h):
            h.send_response(302)
            h.send_header("Location", "http://example.com/v1/chat/completions")
            h.end_headers()

        with self.assertRaises(Exception) as caught:
            list(voice_loop.stream_reply(self.serve(redirect), [], model="m"))
        self.assertIn("302", str(caught.exception))

    def test_environment_proxies_are_ignored(self):
        def ok(h):
            h.send_response(200)
            h.end_headers()
            h.wfile.write(b'data: {"choices": [{"delta": {"content": "hi"}}]}\n\ndata: [DONE]\n\n')

        url = self.serve(ok)
        with unittest.mock.patch.dict("os.environ", {"http_proxy": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9", "no_proxy": "", "NO_PROXY": ""}), \
                unittest.mock.patch("urllib.request._opener", None):  # urllib caches its default opener
            self.assertEqual(list(voice_loop.stream_reply(url, [], model="m")), ["hi"])

    def test_an_empty_choices_list_fails_clearly(self):
        def empty(h):
            h.send_response(200)
            h.end_headers()
            h.wfile.write(b'data: {"choices": []}\n\n')

        with self.assertRaisesRegex(RuntimeError, "no choices"):
            list(voice_loop.stream_reply(self.serve(empty), [], model="m"))


class MainClosesWorkers(unittest.TestCase):
    def test_recognition_and_synthesis_are_closed_when_the_loop_ends(self):
        closed = []

        class Closer:
            def __init__(self, name):
                self.name = name

            def close(self):
                closed.append(self.name)

            def transcribe(self, *a):
                return ""

            def synthesize_chunks(self, *a):
                return iter(())

        from unittest import mock
        with mock.patch("mlx_omarchy_assistant.recognition.Recognition", lambda home: Closer("recognition")), \
                mock.patch("mlx_omarchy_assistant.synthesis.Synthesis", lambda home: Closer("synthesis")), \
                mock.patch("mlx_omarchy_assistant.wake_word.prepare", lambda home, model: "model.onnx"), \
                mock.patch("mlx_omarchy_assistant.wake_word.WakeWordDetector", lambda path, threshold: LoudNever()), \
                mock.patch.object(voice_loop, "PipeWireSink", lambda rate: mock.Mock(rate=rate)), \
                mock.patch.object(voice_loop, "microphone", lambda: iter([bytes(2560)])):
            self.assertEqual(voice_loop.main(["--home", "/nonexistent"]), 0)
        self.assertEqual(sorted(closed), ["recognition", "synthesis"])


class LoudNever:
    def process(self, pcm):
        return False


class PrepareModels(unittest.TestCase):
    """--prepare: the one networked step. Fetch and verify every model, then exit."""

    def run_prepare(self, synthesis_result=None):
        from unittest import mock
        calls = []

        class Model:
            def __init__(self, name, home, result):
                self.name, self.home, self.result = name, home, result

            def prepare(self, approve_download):
                calls.append((self.name, approve_download))
                return self.result

            def close(self):
                calls.append((self.name, "close"))

        def refuse(*a, **k):
            raise AssertionError("--prepare must not open the microphone, speaker or detector")

        ok = {"verified": True, "path": "/models"}
        with mock.patch("mlx_omarchy_assistant.recognition.Recognition",
                        lambda home: Model("recognition", home, {"verified": True, "backend": "gpu"})), \
                mock.patch("mlx_omarchy_assistant.synthesis.Synthesis",
                           lambda home: Model("synthesis", home, synthesis_result or ok)), \
                mock.patch("mlx_omarchy_assistant.wake_word.prepare",
                           lambda home, model: calls.append(("wake", str(home), model)) or Path("w.onnx")), \
                mock.patch("mlx_omarchy_assistant.vad.prepare",
                           lambda home: calls.append(("vad", str(home))) or Path("v.onnx")), \
                mock.patch("mlx_omarchy_assistant.wake_word.WakeWordDetector", refuse), \
                mock.patch.object(voice_loop, "PipeWireSink", refuse), \
                mock.patch.object(voice_loop, "microphone", refuse):
            err = io.StringIO()
            with unittest.mock.patch("sys.stderr", err), unittest.mock.patch("sys.stdout", io.StringIO()):
                code = voice_loop.main(["--home", "/models-home", "--wake-model", "hey_jarvis", "--prepare"])
        return code, calls, err.getvalue()

    def test_fetches_wake_vad_parakeet_and_kokoro_then_exits(self):
        code, calls, _ = self.run_prepare()
        self.assertEqual(code, 0)
        self.assertIn(("wake", "/models-home", "hey_jarvis"), calls)
        self.assertIn(("vad", "/models-home"), calls)
        self.assertIn(("recognition", True), calls)
        self.assertIn(("synthesis", True), calls)
        self.assertIn(("recognition", "close"), calls)
        self.assertIn(("synthesis", "close"), calls)

    def test_an_unverified_model_fails_the_step_by_name(self):
        code, _, err = self.run_prepare({"verified": False, "reason": "kokoro sha256 mismatch"})
        self.assertEqual(code, 1)
        self.assertIn("kokoro sha256 mismatch", err)


class ModelFilesAreVerifiedNotTrusted(unittest.TestCase):
    """Running a fetch twice is safe, and a half-written file is fetched again."""

    def test_vad_refetches_a_truncated_file_and_skips_a_verified_one(self):
        import hashlib
        from mlx_omarchy_assistant import vad
        good = b"silero" * 1000
        digest = hashlib.sha256(good).hexdigest()
        with tempfile.TemporaryDirectory() as home:
            target = Path(home) / "vad" / "silero_vad.onnx"
            target.parent.mkdir(parents=True)
            target.write_bytes(good[:100])  # an interrupted earlier run
            fetched = []
            vad.prepare(home, fetch=lambda url: fetched.append(url) or good, digest=digest)
            self.assertEqual(target.read_bytes(), good)
            self.assertEqual(len(fetched), 1)

            def no_network(url):
                raise AssertionError("a verified file must not be fetched again")
            self.assertEqual(vad.prepare(home, fetch=no_network, digest=digest), target)

    def test_vad_refuses_a_truncated_download_and_keeps_nothing(self):
        import hashlib
        from mlx_omarchy_assistant import vad
        good = b"silero" * 1000
        with tempfile.TemporaryDirectory() as home:
            with self.assertRaises(vad.VadModelError):
                vad.prepare(home, fetch=lambda url: good[:10],
                            digest=hashlib.sha256(good).hexdigest())
            self.assertEqual(sorted(p.name for p in (Path(home) / "vad").iterdir()), [])


class LoopSurvivesReplyFailure(unittest.TestCase):
    def test_a_failed_reply_keeps_the_session_listening(self):
        """One failed reply transport must not end the hands-free session (the
        #642 review finding): the loop drops the turn and keeps listening."""
        codes = iter([500, 200])

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                code = next(codes, 200)
                if code != 200:
                    self.send_response(code)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b"data: " + json.dumps(
                    {"choices": [{"delta": {"content": "Two. "}}]}).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.addCleanup(server.shutdown)
        threading.Thread(target=server.serve_forever, daemon=True).start()

        class TwoPhrases:
            """Fires on the first quiet chunk after each loud phrase, twice."""

            def __init__(self):
                self.loud = 0

            def process(self, pcm):
                x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
                rms = np.sqrt(np.mean(x ** 2))
                if rms > 1000:
                    self.loud = 1
                elif self.loud == 1:
                    self.loud = 2
                    return True
                return False

        source = (tone(0.4) + silence(0.16) + tone(0.8) + silence(1.3)
                  + tone(0.4) + silence(0.16) + tone(0.8) + silence(1.5))
        with tempfile.TemporaryDirectory() as tmp:
            sink = voice_loop.WavSink(Path(tmp) / "r.wav", 24000)
            loop = voice_loop.VoiceLoop(
                voice_loop.Endpointer(TwoPhrases(), rms_floor=500),
                lambda wav, c=None: "count",
                "http://127.0.0.1:%d/v1/chat/completions" % server.server_port,
                lambda text, c=None: iter([PcmChunk(24000, bytes(480))]),
                sink, model="m")
            turns = loop.run(voice_loop.chunks(source))
            sink.close()
        roles = [message["role"] for message in loop._history if message["role"] != "system"]
        self.assertEqual(len(turns), 1, "the second turn must still be served")
        self.assertEqual(turns[0]["reply"], "Two.")
        self.assertNotIn(["user", "user"], [roles[i:i + 2] for i in range(len(roles))],
                         "a failed turn must not leave a dangling user exchange")


if __name__ == "__main__":
    unittest.main()
