#!/usr/bin/env python3
"""Gate 12 driver: the STREAMED Kokoro default path end to end.

Voice-enabled setup (as g10), primer wait, then a warm-up read-aloud and a
measured read-aloud of a multi-sentence answer over the real /api/speak
SSE stream:
  - warm TTFA (first `event: audio`) <= TTFA_LIMIT_S (1.5 s);
  - gap-free streaming: at every chunk arrival the audio already buffered
    must cover playback up to that moment (zero starved arrivals, 50 ms
    tolerance) — the shipped stream keeps the playout buffer non-empty;
  - zero error events, a done event, more than one chunk, and the
    accumulated audio must match the request (>= MIN_AUDIO_S).

Usage: g12-kokoro-stream-driver.py ASSISTANT_HOME LOG
"""
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

HOME, LOG = sys.argv[1], sys.argv[2]
RUNTIME = os.path.join(HOME, "assistant", "application.json")
PRIMED_TIMEOUT_S = 60.0
TTFA_LIMIT_S = 1.5
STARVE_TOLERANCE_S = 0.05
MIN_AUDIO_S = 3.0
LONG_PROMPT = ("Reply with exactly three short sentences about today's "
               "weather, then stop.")


def say(text):
    print(text, flush=True)
    with open(LOG, "a") as fh:
        fh.write(text + "\n")


def wait_runtime(deadline_s=1200.0):
    started = time.monotonic()
    while True:
        try:
            return json.load(open(RUNTIME))
        except FileNotFoundError:
            if time.monotonic() - started > deadline_s:
                raise
            time.sleep(1)


def call(method, path, body=None, timeout=120):
    rt = wait_runtime()
    base = f"http://127.0.0.1:{rt['port']}"
    req = urllib.request.Request(
        base + path, data=None if body is None else json.dumps(body).encode(),
        method=method, headers={"Cookie": rt["cookie"], "X-Assistant-CSRF": rt["csrf"],
                                "Origin": base, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_setup():
    t0 = time.monotonic()
    while True:
        s = call("GET", "/api/status")
        state = (s.get("setup") or {}).get("state")
        if state == "complete":
            return time.monotonic() - t0
        if state == "error":
            say("SETUP_ERROR " + str(s["setup"].get("message"))[:400])
            raise SystemExit(1)
        if time.monotonic() - t0 > 1200:
            say("SETUP_TIMEOUT")
            raise SystemExit(1)
        time.sleep(1)


def speak(text, cid, tid):
    """Stream /api/speak; returns (ttfa, chunks, audio_s, errors, done)."""
    rt = wait_runtime()
    req = urllib.request.Request(
        f"http://127.0.0.1:{rt['port']}/api/speak",
        data=json.dumps({"conversation_id": cid, "turn_id": tid,
                         "text": text, "sentence_sequence": 0}).encode(),
        method="POST", headers={"Cookie": rt["cookie"], "X-Assistant-CSRF": rt["csrf"],
                                "Origin": f"http://127.0.0.1:{rt['port']}",
                                "Content-Type": "application/json"})
    t0 = time.monotonic()
    ttfa = None
    arrivals = []
    errors = 0
    done = False
    with urllib.request.urlopen(req, timeout=300) as r:
        event = None
        for raw in r:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
                continue
            if not line.startswith("data:"):
                continue
            payload = json.loads(line[5:])
            now = time.monotonic() - t0
            if event == "audio":
                if ttfa is None:
                    ttfa = now
                pcm = base64.b64decode(payload.get("data", ""))
                dur = len(pcm) / 2 / payload.get("sample_rate", 24000)
                arrivals.append((now, dur))
            elif event == "error":
                errors += 1
                say(f"SPEAK_ERROR {str(payload)[:200]}")
            elif event == "done":
                done = True
    audio_s = sum(d for _, d in arrivals)
    return ttfa, arrivals, audio_s, errors, done


def starved_count(arrivals):
    """Arrivals where the buffered audio ran out before this chunk came."""
    starved = 0
    buffered = 0.0
    first = arrivals[0][0] if arrivals else 0.0
    for arrival, dur in arrivals:
        played = arrival - first
        if buffered + STARVE_TOLERANCE_S < played:
            starved += 1
        buffered += dur
    return starved


def main() -> int:
    setup_s = wait_setup()
    say(f"SETUP_COMPLETE {setup_s:.1f}s")
    call("POST", "/api/setup", {"pair_id": "everyday",
                                "approve_download": True, "voice": True})
    setup_s = wait_setup()
    say(f"VOICE_SETUP_COMPLETE {setup_s:.1f}s")

    primed = False
    t0 = time.monotonic()
    while time.monotonic() - t0 < PRIMED_TIMEOUT_S:
        voice = call("GET", "/api/status").get("voice", {}).get("synthesis", {})
        if voice.get("primed") is True:
            primed = True
            break
        time.sleep(2)
    say(f"PRIMED {'true' if primed else 'false'} after {time.monotonic() - t0:.1f}s")

    cid = call("POST", "/api/conversations", {"save": False})["id"]
    tid = call("POST", f"/api/conversations/{cid}/turns",
               {"text": LONG_PROMPT, "mode": "chat", "max_tokens": 400})["turn_id"]
    text = None
    t0 = time.monotonic()
    while time.monotonic() - t0 < 600:
        rec = call("GET", f"/api/conversations/{cid}")
        m = next((m for m in rec.get("messages") or []
                  if m.get("turn_id") == tid and m.get("role") == "assistant"), None)
        if m and m.get("status") in ("complete", "error", "stopped"):
            text = str(m.get("content") or "")
            break
        time.sleep(1)
    if not text or not text.strip():
        say("CHAT_FAILED")
        return 1
    say(f"CHAT status={m.get('status')} len={len(text)}")
    text = text[:4000]

    # Warm-up: primes segment caches; timing discarded.
    wttaf, wch, waud, werr, wdone = speak(text, cid, tid)
    say(f"WARM ttfa={wttaf} chunks={len(wch)} audio={waud:.2f}s "
        f"errors={werr} done={wdone}")

    ttfa, arrivals, audio_s, errors, done = speak(text, cid, tid)
    starved = starved_count(arrivals) if arrivals else -1
    say(f"TTFA {ttfa:.3f}s (limit {TTFA_LIMIT_S:.2f}s)")
    say(f"STREAM chunks={len(arrivals)} audio={audio_s:.2f}s "
        f"(min {MIN_AUDIO_S}) starved={starved} errors={errors} done={done}")

    ok = (primed and ttfa is not None and ttfa <= TTFA_LIMIT_S
          and starved == 0 and errors == 0 and done
          and len(arrivals) >= 2 and audio_s >= MIN_AUDIO_S)
    say("GATE12 " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
