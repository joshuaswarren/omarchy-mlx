#!/usr/bin/env python3
"""Stub backend for the voice screen-reader pass.

Serves the real static UI plus a fake /api surface whose voice states
mirror what the real coordinator emits (recognition.state in
ready/usable/unqualified/missing, synthesis.state in ready/unqualified/
missing; see serve/mlx_omarchy_assistant/recognition.py and server.py).
Scenario state switches via /api/__state?set=<name>.

stdlib only: SSE is written by hand; transcribe latency is a sleep.
"""
import base64
import json
import math
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC = Path("/srv/static")

CONV_ID = "conv-stub-1"
TURN_ID = "turn-stub-1"
TRANSCRIBE_DELAY_S = 1.5

RECOGNITION_ENCODING = "mono PCM WAV (s16/s32/f32), float32 to 16 kHz"

def recognition_ready():
    return {"state": "ready", "ready": True, "usable": True, "qualified": True,
            "reasons": [], "qualification": {"model": "parakeet-tdt-0.6b-v3"},
            "model": "parakeet-tdt-0.6b-v3", "memory": {"asset_bytes": 2_500_000_000},
            "max_duration_seconds": 30, "sample_rate": 16000, "encoding": RECOGNITION_ENCODING}

def recognition_usable():
    status = recognition_ready()
    status.update({"state": "usable", "ready": False, "qualified": False,
                   "reasons": ["runtime is loadable and its pipeline may run, but no "
                               "hardware acceptance receipt is recorded yet"]})
    return status

def recognition_unqualified():
    return {"state": "unqualified", "ready": False, "usable": False, "qualified": False,
            "reasons": ["platform: accelerator is not present"],
            "qualification": {}, "model": None, "memory": {"asset_bytes": 0},
            "max_duration_seconds": 30, "sample_rate": 16000, "encoding": RECOGNITION_ENCODING}

def recognition_missing():
    status = recognition_unqualified()
    status.update({"state": "missing",
                   "reasons": ["missing: voice runtime is not installed"]})
    return status

VOICE_OPTIONS = [
    {"id": "af_heart", "label": "Heart", "accent": "American English",
     "engine": "kokoro", "engine_label": "Kokoro"},
    {"id": "af_bella", "label": "Bella", "accent": "American English",
     "engine": "kokoro", "engine_label": "Kokoro"},
    {"id": "aiden", "label": "Aiden", "accent": "American English",
     "engine": "qwen3-tts", "engine_label": "Qwen3-TTS"},
]

def synthesis_ready():
    return {"state": "ready", "usable": True, "detail": "",
            "qualification": {"qualified": True},
            "engines": [{"id": "kokoro", "usable": True},
                        {"id": "qwen3-tts", "usable": True}],
            "pack": {"voice": "af_heart", "voice_default": "af_heart",
                     "voice_options": VOICE_OPTIONS},
            "memory": {"asset_bytes": 400_000_000}}

def synthesis_unqualified():
    status = synthesis_ready()
    status.update({"state": "unqualified", "usable": False,
                   "qualification": {"qualified": False}, "detail": "not qualified"})
    return status

def synthesis_missing():
    status = synthesis_ready()
    status.update({"state": "missing", "usable": False,
                   "qualification": {"qualified": False},
                   "detail": "assets unverified", "pack": {}})
    return status

def base_status(voice):
    return {
        "state": "ready",
        "pairs": [{"id": "everyday", "label": "Everyday (fast)",
                   "chat_model": "qwen3.8-2b-4bit", "decision_model": "laya-1",
                   "context_tokens": 8192, "qualification": {"status": "ready"}}],
        "recommended_pair": "everyday",
        "active_pair": {"id": "everyday", "label": "Everyday (fast)",
                        "chat_model": "qwen3.8-2b-4bit",
                        "context_tokens": 8192, "ready_offline": True},
        "context": {"max_tokens": 8192, "used_tokens": 128, "reason": "seeded"},
        "voice": voice,
        "theme": {"name": "stub", "source": "built-in", "status": "applied"},
        "error": None,
    }

SEED_MESSAGE = (
    "Captured reply for the screen-reader pass. The read-aloud action "
    "queues every sentence of this bubble to the speech queue. A second "
    "sentence makes the streamed playback observable."
)

def conversation_record():
    return {
        "id": CONV_ID, "title": "Stub conversation", "sequence": 4,
        "messages": [
            {"id": "m1", "role": "user", "turn_id": "t0",
             "content": "Reply with one short sentence.", "status": "complete"},
            {"id": "m2", "role": "assistant", "turn_id": "t0",
             "content": SEED_MESSAGE, "status": "complete"},
        ],
        "context": {},
    }

STATES = {
    "voice-ready": lambda: base_status({
        "state": "ready",
        "recognition": recognition_ready(),
        "synthesis": synthesis_ready(),
        "download_bytes": 0}),
    "voice-usable": lambda: base_status({
        "state": "ready",
        "recognition": recognition_usable(),
        "synthesis": synthesis_ready(),
        "download_bytes": 0}),
    "voice-unqualified": lambda: base_status({
        "state": "ready",
        "recognition": recognition_unqualified(),
        "synthesis": synthesis_unqualified(),
        "download_bytes": 0}),
    "voice-missing": lambda: base_status({
        "state": "ready",
        "recognition": recognition_missing(),
        "synthesis": synthesis_missing(),
        "download_bytes": 0}),
    "voice-transcribe-error": lambda: base_status({
        "state": "ready",
        "recognition": recognition_ready(),
        "synthesis": synthesis_ready(),
        "download_bytes": 0}),
    "voice-transcribe-empty": lambda: base_status({
        "state": "ready",
        "recognition": recognition_ready(),
        "synthesis": synthesis_ready(),
        "download_bytes": 0}),
    "tts-truncate": lambda: base_status({
        "state": "ready",
        "recognition": recognition_ready(),
        "synthesis": synthesis_ready(),
        "download_bytes": 0}),
}

lock = threading.Lock()
current = {"state": "voice-ready"}


def tone_pcm16le(seconds, rate=16000, freq=440.0, amp=0.25):
    count = int(seconds * rate)
    frames = bytearray()
    for i in range(count):
        sample = int(max(-1.0, min(1.0, amp * math.sin(2 * math.pi * freq * i / rate))) * 32767)
        frames += struct.pack("<h", sample)
    return bytes(frames)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        with open("/srv/logs/stub.log", "a") as f:
            f.write("%s %s\n" % (self.command if hasattr(self, "command") else "-",
                                 fmt % args))

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _state(self):
        with lock:
            return current["state"]

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            body = (STATIC / "index.html").read_bytes()
            return self._send(200, body, "text/html")
        if path.startswith("/js/") or path.startswith("/css/"):
            member = path.lstrip("/")
            target = STATIC / member
            if not target.is_file():
                return self._send(404, {"error": "no such static file"})
            ctype = "application/javascript" if member.endswith(".js") else "text/css"
            return self._send(200, target.read_bytes(), ctype)
        if path == "/js/worklet/capture-worklet.js":
            return self._send(200, (STATIC / "js/worklet/capture-worklet.js").read_bytes(),
                              "application/javascript")
        if path == "/api/__state":
            query = dict(pair.split("=", 1) for pair in self.path.split("?")[1].split("&"))
            with lock:
                current["state"] = query.get("set", current["state"])
            return self._send(200, {"state": current["state"]})
        if path == "/api/session":
            return self._send(200, {"csrf": "stub-csrf", "session_id": "stub-session"})
        if path == "/api/status":
            return self._send(200, STATES[self._state()]())
        if path == "/api/theme":
            return self._send(200, {"name": "stub", "source": "built-in", "status": "applied"})
        if path == "/api/conversations":
            return self._send(200, {"conversations": [
                {"id": CONV_ID, "title": "Stub conversation", "message_count": 2,
                 "created_at": 0, "updated_at": 0}]})
        if path == f"/api/conversations/{CONV_ID}":
            return self._send(200, conversation_record())
        if path == f"/api/conversations/{CONV_ID}/events":
            # Live event stream: stay open and silent (the seeded
            # conversation is already complete; nothing streams).
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    time.sleep(15)
            except (BrokenPipeError, ConnectionResetError):
                return
        if path == "/api/conversations/x":
            return self._send(404, {"error": "no such conversation"})
        return self._send(404, {"error": f"no route {path}"})

    def do_POST(self):
        path = self.path.split("?")[0]
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if path == "/api/conversations":
            return self._send(200, conversation_record())
        if path == f"/api/conversations/{CONV_ID}/turns":
            return self._send(200, {"turn_id": TURN_ID})
        if path == f"/api/conversations/{CONV_ID}/heartbeat":
            return self._send(200, {})
        if path == f"/api/conversations/{CONV_ID}/cancel":
            return self._send(200, {})
        if path == f"/api/conversations/{CONV_ID}/context":
            return self._send(200, conversation_record())
        if path == f"/api/conversations/{CONV_ID}/actions":
            return self._send(200, {"ok": True})
        if path == "/api/transcribe":
            state = self._state()
            if state == "voice-transcribe-error":
                return self._send(500, {"error": "stub recognizer down"})
            time.sleep(TRANSCRIBE_DELAY_S)
            if state == "voice-transcribe-empty":
                return self._send(200, {"text": ""})
            return self._send(200, {"text": "Transcribe the stub sentence, please."})
        if path == "/api/speak":
            state = self._state()
            # The real server closes the SSE response after the stream
            # ends; the client reader terminates on EOF, so the stub must
            # too or the queue waits forever for the stream to resolve.
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            chunk = tone_pcm16le(0.5)
            payload = base64.b64encode(chunk).decode()
            # tts-truncate streams 24 chunks (12 s) so the 10 s playback
            # cap truncates mid-stream; every other state streams 6 (3 s).
            chunks = 24 if state == "tts-truncate" else 6
            try:
                for i in range(chunks):
                    event = {"event": "audio",
                             "data": {"data": payload, "sample_rate": 16000,
                                      "encoding": "pcm16le"}}
                    self.wfile.write(f"id: {i}\nevent: audio\ndata: ".encode())
                    self.wfile.write(json.dumps(event["data"]).encode())
                    self.wfile.write(b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b'event: done\ndata: {}\n\n')
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        if path == "/api/voice/cancel":
            return self._send(200, {})
        if path == "/api/voice":
            return self._send(200, {"voice": "af_heart"})
        if path == "/api/voice/preview":
            payload = base64.b64encode(tone_pcm16le(1.0)).decode()
            return self._send(200, {"sample_rate": 16000, "encoding": "pcm16le",
                                    "data": payload})
        return self._send(404, {"error": f"no route {path}"})


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    server.serve_forever()
