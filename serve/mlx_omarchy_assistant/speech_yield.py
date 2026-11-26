"""Bounded speech-synthesis scheduling between generation chunks.

Half of the cooperative TTS design (offline assistant design, "Recognition
and synthesis"): during one admitted generation the coordinator holds the GPU
lock for the whole chat stream, which used to force every ``/api/speak`` to a
busy rejection until the whole text response existed. Instead, at genuine
chunk boundaries the generation thread parks the chat worker inside the pinned
mlx-lm decode loop (``mlx_omarchy_serve._mlxlm_server.YieldGate`` — a real
GPU-idle boundary with KV state intact, never SIGSTOP and never TCP
backpressure), releases the GPU lock so exactly one queued speak request can
synthesize, then re-takes the lock and resumes the stream.

Invariants enforced here, by construction:

- No overlapping GPU generation and synthesis: the lock handoff happens only
  after the worker acknowledged the park ("held" means no decode step can run
  until release), and the generation thread re-takes the lock before it
  un-parks the worker.
- Cancellation is bounded: every wait is time-bounded. When a pause cannot be
  proven (worker without the gate, park-ack timeout), the speak request gets
  today's honest busy rejection instead of a fake pause.
- The queue of pending speak requests is bounded (two, mirroring the design's
  TTS queue); further requests are rejected immediately.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
from urllib.parse import urlsplit

from mlx_omarchy_serve._mlxlm_server import (
    YIELD_ACK_MAX_SECONDS,
    YIELD_CONTROL_PATH,
    YIELD_SECRET_HEADER,
)

SPEAK_GRANT_WAIT_SECONDS = 20.0
SPEAK_PARK_ACK_SECONDS = 10.0
SPEAK_TAKE_TIMEOUT_SECONDS = 10.0
MAX_PENDING_SPEAKS = 2
PROBE_TIMEOUT_SECONDS = 3.0


class YieldClient:
    """Control-plane client for one chat worker's yield gate."""

    def __init__(self, base_url: str):
        parsed = urlsplit(base_url)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "::1",
                                                              "localhost"):
            raise ValueError("Model workers must use local loopback HTTP")
        self._host = parsed.hostname
        self._port = parsed.port

    def _post(self, payload: dict, timeout: float) -> dict | None:
        try:
            connection = http.client.HTTPConnection(self._host, self._port,
                                                    timeout=timeout)
            try:
                connection.request("POST", YIELD_CONTROL_PATH, json.dumps(payload),
                                   {"Content-Type": "application/json"})
                response = connection.getresponse()
                body = response.read(4096)
                if response.status != 200:
                    return None
                result = json.loads(body)
            finally:
                connection.close()
        except (OSError, ValueError):
            return None
        return result if isinstance(result, dict) else None

    def probe(self, secret: str) -> bool:
        """True only when the worker runs the gate and bound this secret."""
        result = self._post({"action": "probe", "secret": secret},
                            PROBE_TIMEOUT_SECONDS)
        return bool(result and result.get("gate") and result.get("secret_ok"))

    def hold(self, secret: str, ack_timeout: float) -> bool:
        """Park the worker at a genuine chunk boundary; True when proven."""
        result = self._post({"action": "hold", "secret": secret,
                             "ack_timeout": min(ack_timeout,
                                                YIELD_ACK_MAX_SECONDS)},
                            ack_timeout + PROBE_TIMEOUT_SECONDS)
        return bool(result and result.get("held"))

    def release(self, secret: str) -> None:
        # Best effort: a lost release is recovered by the worker's bounded
        # hold net, and the next turn re-binds and re-probes anyway.
        self._post({"action": "release", "secret": secret},
                   PROBE_TIMEOUT_SECONDS)


class SpeakGrant:
    """A granted GPU slot for one speak request; always pass to ``release``."""

    def __init__(self, scheduler: "SpeechYieldScheduler", waiter):
        self._scheduler = scheduler
        self._waiter = waiter

    def release(self) -> None:
        self._scheduler._exit(self)


class SpeechYieldScheduler:
    """GPU-lock handoff between the generation thread and speak requests."""

    def __init__(self, gpu: threading.Lock):
        self._gpu = gpu
        self._cond = threading.Condition()
        self._pending: list[_Waiter] = []

    # ---- speak-request side (server handler threads)

    def enter(self, cancel: threading.Event,
              wait: float = SPEAK_GRANT_WAIT_SECONDS) -> SpeakGrant | None:
        """Acquire the GPU for synthesis, waiting at most ``wait`` seconds
        for the active generation to reach a yield point. None means busy."""
        if self._gpu.acquire(blocking=False):
            return SpeakGrant(self, None)
        waiter = _Waiter()
        with self._cond:
            if len(self._pending) >= MAX_PENDING_SPEAKS:
                return None
            self._pending.append(waiter)
        try:
            if not _waiter_wait(waiter, wait, cancel):
                return None
        finally:
            with self._cond:
                if waiter in self._pending:
                    self._pending.remove(waiter)
        if not self._gpu.acquire(timeout=SPEAK_TAKE_TIMEOUT_SECONDS):
            return None
        waiter.taken.set()
        return SpeakGrant(self, waiter)

    def _exit(self, grant: SpeakGrant) -> None:
        self._gpu.release()
        with self._cond:
            self._cond.notify_all()

    # ---- generation-thread side (coordinator turn)

    def yield_point(self, client: YieldClient, secret: str,
                    cancel: threading.Event) -> None:
        """Serve pending speak requests, one queue entry at a time.

        Called between generation chunks with the GPU lock held. Returns with
        the lock held, the worker un-parked, and the queue empty (or with
        ``cancel`` set after the current handoff finished). Bounded
        throughout; a park that cannot be proven refuses the pending speak
        instead of pausing.
        """
        while True:
            with self._cond:
                waiter = self._pending.pop(0) if self._pending else None
            if waiter is None:
                return
            if not client.hold(secret, SPEAK_PARK_ACK_SECONDS):
                waiter.refuse.set()
                return
            # Parked (or quiescent) in the worker: hand the lock to speak.
            self._gpu.release()
            waiter.granted.set()
            if not waiter.taken.wait(SPEAK_TAKE_TIMEOUT_SECONDS):
                # The speak request gave up before taking the lock (cancel or
                # timeout race). Re-take; its side either never took the lock
                # or releases it in SpeakGrant.release.
                self._gpu.acquire(timeout=SPEAK_TAKE_TIMEOUT_SECONDS)
            self._gpu.acquire()
            client.release(secret)
            if cancel.is_set():
                return

    def refuse_pending(self) -> None:
        """Wake queued speak requests that can no longer be served."""
        with self._cond:
            waiters = list(self._pending)
        for waiter in waiters:
            waiter.refuse.set()


class _Waiter:
    def __init__(self):
        self.granted = threading.Event()
        self.taken = threading.Event()
        self.refuse = threading.Event()


def _waiter_wait(waiter: _Waiter, wait: float,
                 cancel: threading.Event) -> bool:
    """Bounded wait for a grant; False on timeout, refuse, or cancel."""
    end = time.monotonic() + wait
    while not waiter.granted.is_set():
        if waiter.refuse.is_set() or cancel.is_set():
            return False
        remaining = end - time.monotonic()
        if remaining <= 0:
            return False
        if waiter.refuse.wait(min(remaining, 0.5)):
            return False
    return True
