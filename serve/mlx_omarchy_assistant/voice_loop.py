"""Offline voice assistant loop: wake + VAD endpointing -> resident STT ->
streamed local chat reply -> sentence-streamed TTS -> audio sink.

Every stage is injected, so the loop runs the same with the shipped models
(WakeWordDetector, Recognition.transcribe, Synthesis.synthesize_chunks,
PipeWireSink) and with test stubs. The reply comes only from a loopback
OpenAI-compatible server (mlx_lm.server or oMLX): stream_reply refuses any
other host, so the loop makes no network calls. Audio goes to PipeWire's
default sink, never to a raw ALSA device, so the speaker safety daemon
(speakerguardd) stays in the path.
"""

import argparse
import io
import ipaddress
import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path
from typing import Iterator

import numpy as np

SAMPLE_RATE = 16000
CHUNK = 1280  # 80 ms, the wake detector's streaming step
_ANY_END = re.compile(r"(?<=[.!?,;:])\s+")
FIRST_CLAUSE_WORDS = 4  # the first clause goes to TTS early once it has this many words
# A period after one of these, or after dotted letters (U.S., a.m., e.g., J.), ends no piece.
_ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "st", "jr", "sr", "prof", "vs", "etc", "no", "mt", "ave", "approx"}
_DOTTED = re.compile(r"(?:[A-Za-z]\.)+")


def chunks(pcm: bytes, size: int = CHUNK) -> Iterator[bytes]:
    """Split 16 kHz mono int16 PCM into detector-sized chunks."""
    step = size * 2
    for off in range(0, len(pcm), step):
        yield pcm[off:off + step]


def _rms(pcm: bytes) -> float:
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


def wav_bytes(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


class Endpointer:
    """Wake word opens a turn; voice activity (chunk RMS against a floor)
    closes it after end_silence_s of quiet. Silence between the wake and the
    first voiced chunk is dropped; a turn with no speech within
    listen_timeout_s, or longer than max_utterance_s, ends early."""

    # 1.0 s outlasts the pause between two sentences of one question (Kokoro:
    # up to 0.72 s measured; 0.7 s cut those turns in half).
    def __init__(self, detector, *, rms_floor: float, end_silence_s: float = 1.0,
                 listen_timeout_s: float = 5.0, max_utterance_s: float = 30.0,
                 vad=None, prefilter_floor: float | None = None):
        self._detector = detector
        self._floor = rms_floor
        self._end = end_silence_s
        self._timeout = listen_timeout_s
        self._max = max_utterance_s
        self._vad = vad
        self._prefilter = prefilter_floor
        self._reset()

    def _voiced(self, pcm: bytes) -> bool:
        """Speech decision: the VAD when one is set, else the RMS gate.

        The pre-filter only rejects early (a silent chunk never reaches the
        model); it never overrides the VAD."""
        if self._vad is None:
            return _rms(pcm) >= self._floor
        if self._prefilter is not None and _rms(pcm) < self._prefilter:
            return False
        return self._vad.voiced(pcm)

    def _reset(self):
        self._listening = False
        self.last_quiet_s = 0.0
        self._buf = []
        self._waited = self._quiet = self._spoken = 0.0
        if self._vad is not None:
            self._vad.reset()  # RNN state and buffered PCM must not leak across turns

    def feed(self, pcm: bytes) -> bytes | None:
        """Feed one chunk; returns the utterance PCM when a turn ends."""
        seconds = len(pcm) / 2 / SAMPLE_RATE
        if not self._listening:
            self._listening = bool(self._detector.process(pcm))
            return None
        voiced = self._voiced(pcm)
        if not self._buf and not voiced:
            self._waited += seconds
            if self._waited >= self._timeout:
                self._reset()
            return None
        self._buf.append(pcm)
        self._spoken += seconds
        self._quiet = 0.0 if voiced else self._quiet + seconds
        if self._quiet >= self._end or self._spoken >= self._max:
            utterance, quiet = b"".join(self._buf), self._quiet
            self._reset()
            self.last_quiet_s = quiet
            return utterance
        return None


def _require_loopback(url: str) -> None:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"reply server URL must be http or https, got {url!r}")
    host = parts.hostname or ""
    try:
        ok = ipaddress.ip_address(host).is_loopback
    except ValueError:
        ok = host == "localhost"
    if not ok:
        raise ValueError(f"reply server must be loopback, got {host!r}")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, f"reply server redirected ({code}) to {newurl}; not followed", headers, fp)


# No environment proxies and no redirects: the request stays on the loopback host.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def stream_reply(url: str, messages: list, *, model: str,
                 timeout: float = 120.0) -> Iterator[str]:
    """Yield content deltas from a loopback OpenAI-compatible chat server."""
    _require_loopback(url)
    body = json.dumps({"model": model, "messages": messages, "stream": True}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            choices = json.loads(data).get("choices") or []
            if not choices:
                raise RuntimeError(f"reply server sent an event with no choices: {data[:200]}")
            delta = choices[0].get("delta", {}).get("content")
            if delta:
                yield delta


def _abbreviation(token: str) -> bool:
    word = token.rstrip(",;:")
    return bool(_DOTTED.fullmatch(word)) or word.rstrip(".").lower() in _ABBREVIATIONS


def _next_cut(text: str, clause_words: int | None):
    """The first place to cut: after . ! or ? that does not follow an
    abbreviation, or (when clause_words is set) after , ; or : once the piece
    has that many words and the mark does not follow an abbreviation."""
    for m in _ANY_END.finditer(text):
        head = text[:m.start()]
        token = head.split()[-1] if head.split() else ""
        if _abbreviation(token):
            continue
        if head.endswith((".", "!", "?")) or (
                clause_words is not None and len(head.split()) >= clause_words):
            return head, text[m.end():]
    return None


def sentences(deltas, first_clause_words: int = FIRST_CLAUSE_WORDS) -> Iterator[str]:
    """Group streamed text into sentences as soon as each one ends. The first
    piece may be a clause, so speech starts before the first sentence ends."""
    pending, first = "", True
    for delta in deltas:
        pending += delta
        while (cut := _next_cut(pending, first_clause_words if first else None)) is not None:
            head, pending = cut
            first = False
            if head.strip():
                yield head.strip()
    if pending.strip():
        yield pending.strip()


class WavSink:
    """Writes the spoken reply to a WAV file (tests, receipts)."""

    def __init__(self, path: Path, rate: int):
        self.rate = rate
        self._w = wave.open(str(path), "wb")
        self._w.setnchannels(1)
        self._w.setsampwidth(2)
        self._w.setframerate(rate)

    def write(self, pcm: bytes) -> None:
        self._w.writeframes(pcm)

    def close(self) -> None:
        self._w.close()


class PipeWireSink:
    """Plays mono PCM16 through PipeWire's default sink with pw-play."""

    def __init__(self, rate: int, popen=subprocess.Popen):
        self.rate = rate
        self._proc = popen(["pw-play", "--raw", f"--rate={rate}", "--channels=1",
                            "--format=s16", "-"], stdin=subprocess.PIPE)

    def write(self, pcm: bytes) -> None:
        self._proc.stdin.write(pcm)
        self._proc.stdin.flush()

    def close(self) -> None:
        self._proc.stdin.close()
        self._proc.wait(timeout=30)


class VoiceLoop:
    def __init__(self, endpointer: Endpointer, transcribe, reply_url: str,
                 synthesize, sink, *, model: str, system_prompt: str | None = None):
        _require_loopback(reply_url)
        self._endpointer = endpointer
        self._transcribe = transcribe
        self._url = reply_url
        self._synthesize = synthesize
        self._sink = sink
        self._model = model
        self._history = [{"role": "system", "content": system_prompt}] if system_prompt else []
        self.cancel = threading.Event()

    def turn(self, utterance: bytes, endpoint_s: float = 0.0) -> dict | None:
        """One turn. ms: endpoint is the silence waited before the turn closed
        (audio time); the rest are wall time from the turn's close."""
        t0 = time.monotonic()
        ms = {"endpoint": round(endpoint_s * 1000)}
        since = lambda: round((time.monotonic() - t0) * 1000, 1)  # noqa: E731
        heard = self._transcribe(wav_bytes(utterance), self.cancel).strip()
        ms["stt"] = since()
        if not heard:
            return None
        self._history.append({"role": "user", "content": heard})

        def timed(deltas):
            for delta in deltas:
                ms.setdefault("first_token", since())
                yield delta

        spoken = []
        for sentence in sentences(timed(stream_reply(self._url, self._history, model=self._model))):
            spoken.append(sentence)
            for chunk in self._synthesize(sentence, self.cancel):
                if chunk.sample_rate != self._sink.rate:
                    raise RuntimeError(f"TTS rate {chunk.sample_rate} != sink rate {self._sink.rate}")
                self._sink.write(chunk.data)
                ms.setdefault("first_audio", since())
        ms["total"] = since()
        reply = " ".join(spoken)
        self._history.append({"role": "assistant", "content": reply})
        return {"heard": heard, "reply": reply, "sentences": len(spoken), "ms": ms}

    def run(self, source) -> list[dict]:
        """Run every turn found in an iterable of 16 kHz PCM chunks."""
        turns = []
        for pcm in source:
            if self.cancel.is_set():
                break
            utterance = self._endpointer.feed(pcm)
            if utterance is not None:
                result = self.turn(utterance, self._endpointer.last_quiet_s)
                if result:
                    turns.append(result)
        return turns


def microphone(popen=subprocess.Popen) -> Iterator[bytes]:
    """16 kHz mono int16 chunks from PipeWire's default source."""
    proc = popen(["pw-record", "--raw", f"--rate={SAMPLE_RATE}", "--channels=1",
                  "--format=s16", "-"], stdout=subprocess.PIPE)
    try:
        while chunk := proc.stdout.read(CHUNK * 2):
            yield chunk
    finally:
        proc.terminate()


def build_arg_parser() -> argparse.ArgumentParser:
    from .wake_word import WAKE_THRESHOLD_DEFAULT

    parser = argparse.ArgumentParser(
        prog="mlx-omarchy-voice-loop",
        description="Offline wake -> STT -> local reply -> TTS loop")
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--reply-url", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--model", default="default")
    parser.add_argument("--wake-model", default="hey_jarvis")
    parser.add_argument("--wake-threshold", type=float, default=WAKE_THRESHOLD_DEFAULT)
    parser.add_argument("--rms-floor", type=float, default=300.0)
    parser.add_argument("--vad", choices=("energy", "silero"), default=None,
                        help="speech decision: silero if prepared (default; the "
                             "installer stages it), the RMS gate otherwise; "
                             "--vad silero explicitly allows the model download")
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--vad-prefilter-floor", type=float, default=None,
                        help="energy pre-filter: chunks under this RMS never "
                             "reach the model (energy gate keeps this role)")
    return parser


def build_endpointer(args, home: Path, session=None, detector=None):
    """Endpointer per args plus a one-line fallback notice when silero's
    model is absent. Unset --vad means: silero if prepared, else the
    energy gate; an explicit --vad silero allows the download."""
    from . import vad as vad_mod

    notice = None
    if args.vad == "energy":
        return Endpointer(detector, rms_floor=args.rms_floor), notice
    if args.vad == "silero":
        model_path = vad_mod.prepare(home)
    else:
        model_path = vad_mod.prepared(home)
    if model_path is None:
        notice = ("silero VAD model not prepared; using the energy gate "
                  "(re-run install.sh, or pass --vad silero to download it)")
        return Endpointer(detector, rms_floor=args.rms_floor), notice
    model = vad_mod.SileroVad(model_path, threshold=args.vad_threshold, session=session)
    return Endpointer(detector, rms_floor=args.rms_floor, vad=model,
                      prefilter_floor=args.vad_prefilter_floor), notice


def main(argv=None) -> int:
    import sys

    from . import wake_word
    from .recognition import Recognition
    from .synthesis import Synthesis

    args = build_arg_parser().parse_args(argv)
    detector = wake_word.WakeWordDetector(
        wake_word.prepare(args.home, args.wake_model), threshold=args.wake_threshold)
    endpointer, notice = build_endpointer(args, args.home, detector=detector)
    if notice:
        print(notice, file=sys.stderr)
    recognition, synthesis = Recognition(args.home), Synthesis(args.home)
    try:
        sink = PipeWireSink(24000)
        loop = VoiceLoop(endpointer,
                         recognition.transcribe, args.reply_url,
                         synthesis.synthesize_chunks, sink, model=args.model)
        try:
            loop.run(microphone())
        except KeyboardInterrupt:
            loop.cancel.set()
        finally:
            sink.close()
    finally:
        recognition.close()
        synthesis.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
