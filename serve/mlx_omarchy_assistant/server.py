"""Authenticated loopback application; model endpoints never reach the browser."""

from __future__ import annotations

import array
import base64
import contextlib
import hmac
import json
import mimetypes
import os
import secrets
import threading
import time
from dataclasses import asdict
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from .history import BusyError, ConversationStore, EventGap
from . import synthesis

CSP = "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'; media-src 'self' blob:; worker-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"

# Fixed preview sentence: short, all-ASCII so the worker asks for the English
# language token, and never longer than 8 seconds at the slowest voice.
PREVIEW_SENTENCE = "Hello. This is a short preview of the selected voice."
# Seconds between pre-warm grant retries while the pair worker start holds
# the GPU (an idle resident worker never reaches a yield point).
PREWARM_RETRY_SECONDS = 5.0



def voice_ready(recognition, synthesis_state):
    """Speech is ready only when recognition has a live acceptance and synthesis is ready.

    A stored receipt alone is not enough: the runtime probe must also be usable.
    """
    return bool(recognition.get("ready")) and synthesis_state == "ready"


class AssistantServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = True

    def __init__(self, address, home):
        if address[0] != "127.0.0.1":
            raise ValueError("The assistant binds only to 127.0.0.1")
        self.home = Path(home)
        self.bootstrap = secrets.token_urlsafe(32)
        self.session = secrets.token_urlsafe(32)
        self.csrf = secrets.token_urlsafe(32)
        self.mutex = threading.RLock()
        self.store = ConversationStore(self.home)
        self._manager = None
        self._coordinator = None
        self._recognition = None
        self._synthesis = None
        self.setup_state = None
        self.setup_thread = None
        self.transfer_thread = None
        self.transfer_state = {"state": "idle"}
        self._prewarm_started = False
        self._prewarm_cancel = threading.Event()
        self.audio_kind: str | None = None
        self.audio_identity: tuple[str, str] | None = None
        self.audio_cancel = threading.Event()
        self.audio_lock = threading.Lock()
        super().__init__(address, Handler)
        self.origin = f"http://127.0.0.1:{self.server_port}"
        self.cookie_name = f"mlx_assistant_{self.server_port}"

    @property
    def manager(self):
        with self.mutex:
            if self._manager is None:
                from .pairs import PairManager
                self._manager = PairManager(self.home)
            return self._manager

    @property
    def coordinator(self):
        with self.mutex:
            if self._coordinator is None:
                from .coordinator import Coordinator
                self._coordinator = Coordinator(self.home, self.manager, self.store)
            return self._coordinator

    @property
    def recognition(self):
        with self.mutex:
            if self._recognition is None:
                from .recognition import Recognition
                self._recognition = Recognition(self.home)
            return self._recognition

    @property
    def synthesis(self):
        with self.mutex:
            if self._synthesis is None:
                from .synthesis import Synthesis
                self._synthesis = Synthesis(self.home)
            return self._synthesis

    def status(self):
        from .theme import read_theme
        with self.mutex:
            source = self.manager.status()
            keys = ("state", "pairs", "active_pair", "context", "error", "ready_offline",
                    "recommended_pair", "total_download_bytes", "download_components", "download_has_unknown")
            result = {key: source[key] for key in keys if key in source}
            active_id = source.get("active_pair")
            result["active_pair"] = next((dict(pair, ready_offline=source.get("ready_offline", False))
                                          for pair in source.get("pairs", []) if pair["id"] == active_id), None)
            recommendation = source.get("recommendation") or {}
            result["recommended_pair"] = recommendation.get("pair_id")
            result["recommendation_error"] = recommendation.get("error")
            context = source.get("context") or {}
            result["context"] = dict(context, max_tokens=context.get("context_tokens"),
                                     limit_tokens=source.get("context_tokens"), step_tokens=1)
            downloads = source.get("downloads") or []
            result["download_components"] = [
                {"id": item["model_id"], "label": item["model_id"], "bytes": item.get("bytes"),
                 "state": "required" if item["needed"] else "cached"} for item in downloads]
            result["download_has_unknown"] = not downloads or any(item.get("bytes") is None for item in downloads)
            result["total_download_bytes"] = sum(item.get("bytes") or 0 for item in downloads if item["needed"])
            result["qualification"] = source.get("qualification")
            result["theme"] = asdict(read_theme())
            recognition = self.recognition.status()
            synthesis = self.synthesis.status()
            qualified = synthesis.get("qualification", {}).get("qualified", False)
            synthesis["state"] = "ready" if synthesis.get("usable") and qualified else "unqualified"
            if not synthesis.get("assets", {}).get("verified"):
                synthesis["state"] = "missing"
            ready = voice_ready(recognition, synthesis["state"])
            result["voice"] = {"recognition": recognition, "synthesis": synthesis,
                               "state": "ready" if ready else "unqualified",
                               "detail": "Speech readiness is verified separately for input and output.",
                               "download_bytes": synthesis.get("memory", {}).get("asset_bytes")}
            self._maybe_prewarm(synthesis)
            if self.setup_state:
                result["setup"] = dict(self.setup_state)
                if self.setup_state["state"] == "preparing":
                    result["state"] = "preparing"
                    result["progress"] = [{"label": self.setup_state.get("stage", "Prepare models"),
                                           "state": "active", "detail": "Local model setup is in progress"}]
                elif self.setup_state["state"] == "error":
                    result["error"] = self.setup_state["message"]
            return result

    def _maybe_prewarm(self, synthesis) -> None:
        """One-shot voice pre-warm: once synthesis is usable, start the
        Kokoro worker and run a tiny synthesis in the background so the
        first real read-aloud does not pay model load + first-infer
        warmup. Never blocks the caller; MLX_OMARCHY_VOICE_PREWARM=0
        disables; respects the speech yield rules (one speak-class GPU
        grant, refused when generation holds the GPU)."""
        if os.environ.get("MLX_OMARCHY_VOICE_PREWARM", "1") == "0":
            return
        if self._prewarm_started or not synthesis.get("usable"):
            return
        self._prewarm_started = True
        print("PREWARM kicked", flush=True)
        threading.Thread(target=self._prewarm_run, daemon=True).start()

    def _prewarm_run(self) -> None:
        """Bounded retry loop: right after setup the pair worker start
        holds the GPU (an idle resident worker never yields), so the first
        enter() is refused; once the load releases the lock, the next
        enter succeeds. Bounded by deadline and the cancel event."""
        cancel = self._prewarm_cancel
        deadline = time.monotonic() + 300.0
        attempt = 0
        while time.monotonic() < deadline and not cancel.is_set():
            attempt += 1
            try:
                grant = self.coordinator.speech.enter(cancel)
            except Exception as exc:
                print(f"PREWARM enter failed: {exc!r}", flush=True)
                return
            if grant is None:
                time.sleep(PREWARM_RETRY_SECONDS)
                continue
            try:
                ok = self.synthesis.prime(cancel)
            except Exception as exc:
                print(f"PREWARM prime raised: {exc!r}", flush=True)
                ok = False
            finally:
                grant.release()
            print(f"PREWARM_RESULT ok={ok} attempts={attempt}", flush=True)
            return
        print(f"PREWARM gave up after {attempt} attempts", flush=True)

    def setup(self, body):
        if set(body) - {"pair_id", "approve_download", "preference", "context_tokens", "voice"}:
            raise ValueError("Unknown setup field")
        if not isinstance(body.get("pair_id"), str):
            raise ValueError("Choose a model pair")
        for key in ("approve_download", "voice"):
            if key in body and not isinstance(body[key], bool):
                raise ValueError(f"{key} must be a boolean")
        with self.mutex:
            if self.setup_thread and self.setup_thread.is_alive():
                raise BusyError("Model setup is already active")
            if not self.coordinator.gpu.acquire(blocking=False):
                raise BusyError("Stop the current model operation before changing the model pair")
            self.setup_state = {"state": "preparing"}

            def progress(stage, **detail):
                with self.mutex:
                    self.setup_state = {"state": "preparing", "stage": str(stage), "detail": detail}

            def run():
                state = None
                try:
                    if self._synthesis:
                        self._synthesis.close()
                    if self._recognition:
                        self._recognition.close()
                    stopped = self.manager.stop()
                    if not stopped.get("stopped"):
                        raise RuntimeError("Previous model workers have not stopped")
                    result = self.manager.setup(progress=progress, **body)
                    if result.get("state") == "planned":
                        raise ValueError("Review and approve the required downloads before setup")
                    if body.get("voice"):
                        voice = self.synthesis.prepare(approve_download=body.get("approve_download", False))
                        if not voice.get("verified"):
                            raise ValueError("Approve the voice download before preparing speech")
                        speech = self.recognition.prepare(approve_download=body.get("approve_download", False))
                        if not speech.get("verified"):
                            raise ValueError(speech.get("reason") or "Approve the speech recognition download")
                    result = self.manager.start()
                    state = {"state": "complete", "ready_offline": result.get("ready_offline", False)}
                except Exception as error:
                    state = {"state": "error", "message": str(error)[:1000]}
                finally:
                    self.coordinator.gpu.release()
                    # Publish the outcome only once the GPU lock is released:
                    # a client that sees "complete" must never race the
                    # release and take a spurious 409 on its first turn.
                    with self.mutex:
                        self.setup_state = state

            self.setup_thread = threading.Thread(target=run, daemon=True)
            self.setup_thread.start()

    def resume(self):
        """Start the saved pair without a download. No saved pair is absent, not an error."""
        with self.mutex:
            if self.setup_thread and self.setup_thread.is_alive():
                raise BusyError("Model setup is already active")
            if not self.coordinator.gpu.acquire(blocking=False):
                raise BusyError("Stop the current model operation before changing the model pair")
            self.setup_state = {"state": "preparing", "stage": "Resume saved pair"}

            def run():
                state = None
                try:
                    self.manager.adopt_saved()
                    result = self.manager.start()
                    state = {"state": "complete",
                             "ready_offline": result.get("ready_offline", False)}
                except Exception as error:
                    message = str(error)[:1000]
                    absent = "no saved pair lock" in message
                    state = {"state": "absent" if absent else "error", "message": message}
                finally:
                    self.coordinator.gpu.release()
                    # Publish the outcome only once the GPU lock is released:
                    # a client that sees "complete" must never race the
                    # release and take a spurious 409 on its first turn.
                    with self.mutex:
                        self.setup_state = state

            self.setup_thread = threading.Thread(target=run, daemon=True)
            self.setup_thread.start()
    def transfer(self, body):
        from .transfer import run_transfer_request
        allowed = {"action", "bundle", "output", "pair_ids", "approved_licenses", "voice", "wheel_caches"}
        if set(body) - allowed or body.get("action") not in ("plan", "inspect", "prepare", "install"):
            raise ValueError("Unknown transfer action or field")
        if "voice" in body and not isinstance(body["voice"], bool):
            raise ValueError("voice must be a boolean")
        for key in ("pair_ids", "approved_licenses", "wheel_caches"):
            values = body.get(key, [])
            if not isinstance(values, list) or len(values) > 32 or any(not isinstance(v, str) or not v for v in values):
                raise ValueError(f"{key} must be a list of names or paths")
        if any(pair not in ("everyday", "quality", "compact") for pair in body.get("pair_ids", [])):
            raise ValueError("Choose Everyday, Quality, or Compact for transfer")
        for key in ("bundle", "output"):
            if key in body and (not isinstance(body[key], str) or not Path(body[key]).is_absolute()):
                raise ValueError(f"{key} must be an absolute local path")
        with self.mutex:
            if self.transfer_thread and self.transfer_thread.is_alive():
                raise BusyError("A transfer is already active")
            if not self.coordinator.gpu.acquire(blocking=False):
                raise BusyError("Stop the current model operation before transfer")
            self.transfer_state = {"state": "running"}

            def run():
                try:
                    if body["action"] == "install":
                        if self._synthesis:
                            self._synthesis.close()
                        if self._recognition:
                            self._recognition.close()
                        if not self.manager.stop().get("stopped"):
                            raise RuntimeError("Model workers must stop before installation")
                    result = run_transfer_request(self.home, body)
                    with self.mutex:
                        self.transfer_state = {"state": "complete", "result": result}
                        if body["action"] == "install":
                            self.coordinator.failure = "Restart MLX Chat to use the imported runtime"
                except Exception as error:
                    with self.mutex:
                        self.transfer_state = {"state": "error", "error": str(error)[:1000]}
                finally:
                    self.coordinator.gpu.release()

            self.transfer_thread = threading.Thread(target=run, daemon=False)
            self.transfer_thread.start()

    def server_close(self):
        self.audio_cancel.set()
        self._prewarm_cancel.set()
        if self.transfer_thread and self.transfer_thread.is_alive():
            self.transfer_thread.join(timeout=10)
            if self.transfer_thread.is_alive():
                raise RuntimeError("Transfer is still active; keep its application process open until it stops")
        if self._manager:
            self._manager.cancel()
        if self.setup_thread and self.setup_thread.is_alive():
            self.setup_thread.join(timeout=10)
            if self.setup_thread.is_alive():
                raise RuntimeError("Model setup has not stopped; its process must remain owned")
        if self._recognition:
            self._recognition.close()
        if self._synthesis:
            self._synthesis.close()
        if self._coordinator:
            self._coordinator.close()
        elif self._manager:
            self._manager.stop()
        super().server_close()


class Handler(BaseHTTPRequestHandler):
    server: AssistantServer
    server_version = "MLXChat"

    def log_message(self, _format, *_args):
        pass

    def _headers(self, status, content_type, length=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "microphone=(self), camera=(), geolocation=()")
        if length is not None:
            self.send_header("Content-Length", str(length))

    def _json(self, status, data, cookie=False):
        raw = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
        self._headers(status, "application/json; charset=utf-8", len(raw))
        if cookie:
            self.send_header("Set-Cookie", f"{self.server.cookie_name}={self.server.session}; HttpOnly; SameSite=Strict; Path=/")
        self.end_headers()
        self.wfile.write(raw)

    def _authenticated(self):
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
        except Exception:
            return False
        value = cookie.get(self.server.cookie_name)
        return value is not None and hmac.compare_digest(value.value, self.server.session)

    def _guard(self, mutation=False):
        if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
            self._json(403, {"error": "Untrusted Host"})
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin != self.server.origin:
            self._json(403, {"error": "Cross-origin requests are not allowed"})
            return False
        if self.headers.get("Sec-Fetch-Site") not in (None, "same-origin", "none"):
            self._json(403, {"error": "Cross-site requests are not allowed"})
            return False
        path = unquote(urlsplit(self.path).path)
        if path == "/api/session" and self.command == "POST":
            if origin != self.server.origin:
                self._json(403, {"error": "Session bootstrap needs a same-origin request"})
                return False
            return True
        if path.startswith("/api/"):
            if not self._authenticated():
                self._json(401, {"error": "Open MLX Chat from its launcher to sign in locally"})
                return False
            if mutation and not hmac.compare_digest(self.headers.get("X-Assistant-CSRF", ""), self.server.csrf):
                self._json(403, {"error": "Invalid request token"})
                return False
        return True

    def _body(self, binary=False):
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("Chunked uploads are not supported")
        length = int(self.headers.get("Content-Length", "0"))
        maximum = 12 * 1024 * 1024 if binary else 2 * 1024 * 1024
        if not 0 <= length <= maximum:
            raise ValueError("Request exceeds the upload limit")
        self.connection.settimeout(15)
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise ValueError("Incomplete request body")
        if binary:
            if self.headers.get_content_type() not in ("audio/wav", "audio/wave", "audio/x-wav"):
                raise ValueError("Upload PCM WAV audio")
            return raw
        if self.headers.get_content_type() != "application/json":
            raise ValueError("Use application/json")
        data = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON value")))
        if not isinstance(data, dict):
            raise ValueError("Request must be a JSON object")
        return data

    def _dispatch(self):
        """Route by (verb, path). Exact endpoints live in _ROUTES, the
        conversation subtree in _conversation, static files in _static.
        Unmatched verbs fall through to the same KeyError -> 404 as before."""
        parsed = urlsplit(self.path)
        path = unquote(parsed.path)
        parts = path.strip("/").split("/")
        handler = self._ROUTES.get((self.command, path))
        if handler is not None:
            return handler(self, parsed)
        if path == "/api/transfer":
            response = self._transfer()
            if response is not None:
                return response
        elif path == "/api/conversations":
            response = self._conversations()
            if response is not None:
                return response
        elif len(parts) >= 3 and parts[:2] == ["api", "conversations"]:
            return self._conversation(parts[2], parts[3:] if len(parts) > 3 else [], parsed)
        elif self.command == "GET" and not path.startswith("/api/"):
            return self._static(path)
        raise KeyError("Endpoint not found")

    def _session_post(self, parsed):
        body = self._body()
        token = body.get("token")
        app = self.server
        with app.mutex:
            if not isinstance(token, str) or not app.bootstrap or not hmac.compare_digest(token, app.bootstrap):
                return self._json(401, {"error": "This launch link expired. Reopen MLX Chat"})
            app.bootstrap = None
        return self._json(200, {"csrf": app.csrf, "session_id": app.cookie_name}, cookie=True)

    def _session_get(self, parsed):
        app = self.server
        return self._json(200, {"csrf": app.csrf, "session_id": app.cookie_name})

    def _status(self, parsed):
        return self._json(200, self.server.status())

    def _launch(self, parsed):
        app = self.server
        self._body()
        with app.mutex:
            app.bootstrap = secrets.token_urlsafe(32)
            token = app.bootstrap
        return self._json(200, {"token": token})

    def _theme(self, parsed):
        from .theme import read_theme
        return self._json(200, asdict(read_theme()))

    def _resume(self, parsed):
        self._body()
        self.server.resume()
        return self._json(202, {"state": "preparing"})

    def _setup(self, parsed):
        self.server.setup(self._body())
        return self._json(202, {"state": "preparing"})

    def _transfer(self):
        """/api/transfer for GET and POST; None on other verbs (falls
        through to the 404 exactly as the old cascade did)."""
        app = self.server
        if self.command == "GET":
            with app.mutex:
                return self._json(200, dict(app.transfer_state))
        if self.command == "POST":
            app.transfer(self._body())
            return self._json(202, {"state": "running"})
        return None

    def _conversations(self):
        """/api/conversations for GET and POST; None on other verbs."""
        app = self.server
        if self.command == "GET":
            return self._json(200, {"conversations": app.store.list()})
        if self.command == "POST":
            body = self._body()
            return self._json(201, app.store.create(save=body.get("save", False)))
        return None

    def _conversation(self, cid, tail, parsed):
        """/api/conversations/{cid}[...] subtree: record read/delete/patch,
        export, context selection, turns, cancel/heartbeat, actions, SSE."""
        app = self.server
        record = app.store.get(cid)
        if not tail:
            if self.command == "GET":
                return self._json(200, record)
            if self.command == "DELETE":
                app.store.delete(cid)
                return self._json(200, {"deleted": True})
            if self.command == "PATCH":
                return self._json(200, app.store.set_saved(cid, self._body().get("save")))
        if self.command == "GET" and tail == ["export"]:
            return self._json(200, record)
        if self.command == "POST" and tail == ["context"]:
            body = self._body()
            if set(body) != {"selected_turn_ids", "pinned_constraints"}:
                raise ValueError("Provide selected turn IDs and pinned constraints only")
            return self._json(200, app.store.set_context(
                cid, body["selected_turn_ids"], body["pinned_constraints"]))
        if self.command == "POST" and tail == ["turns"]:
            return self._json(202, {"turn_id": app.coordinator.submit(cid, self._body())})
        if self.command == "POST" and tail in (["cancel"], ["heartbeat"]):
            turn = self._body().get("turn_id")
            if not isinstance(turn, str):
                raise ValueError("Missing turn ID")
            if tail == ["cancel"]:
                app.coordinator.cancel(cid, turn)
                if app.audio_kind == "tts" and app.audio_identity == (cid, turn):
                    app.audio_cancel.set()
            else:
                app.coordinator.heartbeat(cid, turn)
            return self._json(200, {"ok": True})
        if self.command == "POST" and tail == ["actions"]:
            return self._json(202, {"turn_id": app.coordinator.action(cid, self._body())})
        if self.command == "GET" and tail == ["events"]:
            after = int(parse_qs(parsed.query).get("after", ["0"])[0])
            app.store.events(cid, after)
            self._headers(200, "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                try:
                    events = app.store.events(cid, after)
                except (EventGap, KeyError):
                    break
                for event in events:
                    raw = json.dumps(event, ensure_ascii=False, allow_nan=False)
                    self.wfile.write(f"id: {event['sequence']}\nevent: {event['type']}\ndata: {raw}\n\n".encode())
                    after = event["sequence"]
                if events:
                    self.wfile.flush()
                if not app.store.get(cid)["active_turn"]:
                    break
                with app.store.changed:
                    app.store.changed.wait(timeout=1)
            self.close_connection = True
            return
        raise KeyError("Endpoint not found")

    def _voice_cancel(self, parsed):
        app = self.server
        kind = self._body().get("kind")
        if kind not in ("tts", "stt"):
            raise ValueError("Choose tts or stt cancellation")
        if app.audio_kind == kind:
            app.audio_cancel.set()
        return self._json(200, {"stopped": True})

    def _voice(self, parsed):
        """POST /api/voice (voice selection) and /api/voice/preview (one
        synthesised sentence handed back as PCM16LE in one JSON envelope)."""
        app = self.server
        path = unquote(urlsplit(self.path).path)
        body = self._body()
        if path == "/api/voice":
            requested = body.get("voice")
            if not isinstance(requested, str):
                raise ValueError("voice must be one of the pack speakers")
            try:
                return self._json(200, app.synthesis.set_voice(requested))
            except synthesis.VoiceError as exc:
                raise ValueError(str(exc)) from exc
        # Preview: synthesise one fixed sentence in the current voice and
        # hand the PCM16LE back to the client (single JSON envelope, not
        # SSE) so a one-shot playback does not need a conversation.
        if not app.audio_lock.acquire(blocking=False):
            raise BusyError("Another speech operation is active")
        try:
            if not app.manager.status().get("voice", {}).get("requested"):
                raise ValueError(
                    "Enable voice in model setup before using speech")
            app.audio_cancel = threading.Event()
            app.audio_kind = "tts"
            grant = app.coordinator.speech.enter(app.audio_cancel)
            if grant is None:
                raise BusyError(
                    "Speech is waiting for the current model operation")
            rate = None
            pcm = array.array("h")
            try:
                chunks = app.synthesis.synthesize_chunks(
                    PREVIEW_SENTENCE, app.audio_cancel)
                with contextlib.closing(chunks) as stream:
                    for chunk in stream:
                        if app.audio_cancel.is_set():
                            break
                        if rate is None:
                            rate = chunk.sample_rate
                        pcm.frombytes(chunk.data)
            finally:
                grant.release()
                app.audio_kind = None
                app.audio_identity = None
            if not pcm or rate is None:
                raise synthesis.VoiceError(
                    "preview produced no audio")
            encoded = base64.b64encode(pcm.tobytes()).decode("ascii")
            return self._json(200, {"sample_rate": int(rate),
                                    "encoding": "pcm16le",
                                    "data": encoded})
        finally:
            app.audio_lock.release()

    def _speech(self, parsed):
        """POST /api/transcribe (recognition) and /api/speak (read aloud
        as an SSE audio stream)."""
        app = self.server
        path = unquote(urlsplit(self.path).path)
        raw = self._body(binary=path == "/api/transcribe")
        if not app.audio_lock.acquire(blocking=False):
            raise BusyError("Another speech operation is active")
        grant = None
        try:
            if not app.manager.status().get("voice", {}).get("requested"):
                raise ValueError("Enable voice in model setup before using speech")
            app.audio_cancel = threading.Event()
            app.audio_kind = "stt" if path == "/api/transcribe" else "tts"
            if path == "/api/transcribe":
                # Recognition runs on its own qualified runtime and does
                # not interleave with generation; keep the plain busy
                # rejection while a model operation holds the GPU.
                if not app.coordinator.gpu.acquire(blocking=False):
                    raise BusyError("Speech is waiting for the current model operation")
                try:
                    text = app.recognition.transcribe(raw, app.audio_cancel)
                finally:
                    app.coordinator.gpu.release()
                return self._json(200, {"text": text})
            return self._read_aloud(app, raw)
        finally:
            if grant is not None:
                grant.release()
            app.audio_identity = None
            app.audio_kind = None
            app.audio_lock.release()

    def _read_aloud(self, app, raw):
        """The /api/speak half of _speech: validate visible answer text,
        enter the speech scheduler, and stream synthesised audio as SSE."""
        cid, turn = raw.get("conversation_id"), raw.get("turn_id")
        if not isinstance(cid, str) or not isinstance(turn, str):
            raise ValueError("Speech requires a conversation and turn ID")
        record = app.store.get(cid)
        text = raw.get("text")
        message = next((m for m in record["messages"] if m["turn_id"] == turn and m["role"] == "assistant"), None)
        if not isinstance(text, str) or not text.strip() or len(text) > 4000 or not message or text not in message["content"]:
            raise ValueError("Read aloud only visible answer text")
        sequence = raw.get("sentence_sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("Invalid speech sentence sequence")
        grant = app.coordinator.speech.enter(app.audio_cancel)
        if grant is None:
            raise BusyError("Speech is waiting for the current model operation")
        app.audio_identity = (cid, turn)
        identity = {"conversation_id": cid, "turn_id": turn, "sentence_sequence": sequence}
        try:
            self._headers(200, "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()

            def send_audio_event(kind, data):
                payload = json.dumps(dict(identity, **data), allow_nan=False)
                self.wfile.write(f"event: {kind}\ndata: {payload}\n\n".encode())
                self.wfile.flush()

            try:
                with contextlib.closing(app.synthesis.synthesize_chunks(text, app.audio_cancel)) as chunks:
                    for chunk in chunks:
                        if app.audio_cancel.is_set():
                            break
                        send_audio_event("audio", {"sample_rate": chunk.sample_rate, "encoding": "pcm16le",
                                        "data": base64.b64encode(chunk.data).decode("ascii")})
                send_audio_event("done", {"stopped": app.audio_cancel.is_set()})
            except (BrokenPipeError, ConnectionResetError):
                app.audio_cancel.set()
            except Exception as error:
                send_audio_event("error", {"message": str(error)[:1000]})
            self.close_connection = True
        finally:
            grant.release()

    def _static(self, path):
        """Static files under the packaged static/ directory, index.html
        at the root; anything escaping the tree or missing raises 404."""
        static = Path(__file__).with_name("static").resolve()
        target = (static / ("index.html" if path == "/" else path.lstrip("/"))).resolve()
        if not target.is_relative_to(static) or not target.is_file():
            raise KeyError("Page not found")
        content = target.read_bytes()
        mime = mimetypes.guess_type(target)[0] or "application/octet-stream"
        self._headers(200, mime, len(content))
        self.end_headers()
        self.wfile.write(content)
        return

    def _handle(self):
        try:
            if self._guard(self.command != "GET"):
                self._dispatch()
        except (BrokenPipeError, ConnectionResetError):
            return
        except BusyError as error:
            self._json(409, {"error": str(error)})
        except EventGap as error:
            self._json(409, {"error": str(error), "reload": True})
        except KeyError as error:
            self._json(404, {"error": str(error)})
        except (ValueError, TypeError) as error:
            self._json(400, {"error": str(error)})
        except Exception as error:
            self._json(503, {"error": str(error)[:1000], "code": type(error).__name__})

    do_GET = _handle
    do_POST = _handle
    do_PATCH = _handle
    do_DELETE = _handle

    _ROUTES = {
        ("POST", "/api/session"): _session_post,
        ("GET", "/api/session"): _session_get,
        ("GET", "/api/status"): _status,
        ("POST", "/api/launch"): _launch,
        ("GET", "/api/theme"): _theme,
        ("POST", "/api/resume"): _resume,
        ("POST", "/api/setup"): _setup,
        ("POST", "/api/voice/cancel"): _voice_cancel,
        ("POST", "/api/voice"): _voice,
        ("POST", "/api/voice/preview"): _voice,
        ("POST", "/api/transcribe"): _speech,
        ("POST", "/api/speak"): _speech,
    }
