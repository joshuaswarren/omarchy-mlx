#!/usr/bin/env python3
"""Gate 9 driver: read-aloud playout in a real browser, no GPU.

Loads the INSTALLED package's static/js/voice.js into headless Chromium and
reads four sentences against a fake /api/speak that behaves like the real
server: one speech stream at a time (409 while another streams), two PCM
chunks per sentence, 1.8 s of compute per 2.6 s of audio. The page records
every AudioBufferSourceNode.start() on the real Web Audio clock and posts
the schedule back. PASS needs, on that schedule: sentences in order, no
chunk starting while an earlier one plays, no gap between chunks, no 409,
and one audio-done event after the last chunk ends.

Usage: g9-speak-queue-driver.py <chromium-binary>
       (the package is imported from PYTHONPATH; G9_STATIC_JS overrides the
       static/js directory for a source-tree run)
"""
import base64
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RATE = 24000
CHUNKS = ((1.0, 1.4), (0.8, 1.2))  # (compute delay s, audio s) per sentence
SENTENCES = 4

PAGE = b"""<!doctype html><html><body><script type=module>
const starts = [];
const done = [];
const errors = [];
const orig = AudioBufferSourceNode.prototype.start;
AudioBufferSourceNode.prototype.start = function (when) {
  starts.push({ at: when, dur: this.buffer.duration, seq: window.__seq });
  return orig.call(this, when);
};
const report = () => fetch('/result', { method: 'POST',
  body: JSON.stringify({ starts, done, errors }) });
const { SpeakQueue } = await import('/js/voice.js');
const q = new SpeakQueue();
q.setConversationId('c');
q.attachHooks({
  onAudioDone: (d) => { done.push({ truncated: d.truncated, t: q._ctx.currentTime }); setTimeout(report, 500); },
  onError: (e) => errors.push(String(e && e.message)),
});
for (let s = 1; s <= %d; s++) q.enqueue('t', s, 'Sentence ' + s + '.');
setTimeout(report, 30000);
</script></body></html>""" % SENTENCES


def tone(seconds, seq):
    n = int(seconds * RATE)
    freq = 180 + 40 * seq
    pcm = b"".join(struct.pack("<h", int(3000 * math.sin(2 * math.pi * freq * i / RATE)))
                   for i in range(n))
    return base64.b64encode(pcm).decode()


def main():
    chromium = sys.argv[1]
    js = os.environ.get("G9_STATIC_JS")
    if js:
        js_dir = Path(js)
    else:
        import mlx_omarchy_assistant
        js_dir = Path(mlx_omarchy_assistant.__file__).with_name("static") / "js"
    if not (js_dir / "voice.js").is_file():
        print(f"SPEAK_QUEUE_SMOKE FAIL no voice.js under {js_dir}")
        return 1
    print(f"STATIC_JS {js_dir}")

    speech = threading.Lock()
    requests = []
    result = {}
    got = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                return self._send(200, PAGE, "text/html")
            name = self.path.removeprefix("/js/")
            target = (js_dir / name).resolve()
            if not self.path.startswith("/js/") or not target.is_relative_to(js_dir.resolve()) \
                    or not target.is_file():
                return self._send(404, b"", "text/plain")
            return self._send(200, target.read_bytes(), "text/javascript")

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.path == "/result":
                if not got.is_set():
                    result.update(json.loads(body))
                    got.set()
                return self._send(204, b"", "text/plain")
            seq = json.loads(body)["sentence_sequence"]
            if not speech.acquire(blocking=False):
                requests.append((seq, 409))
                return self._send(409, b'{"error":"busy"}', "application/json")
            try:
                requests.append((seq, 200))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                for delay, seconds in CHUNKS:
                    time.sleep(delay)
                    data = json.dumps({"sample_rate": RATE, "encoding": "pcm16le",
                                       "data": tone(seconds, seq)})
                    self.wfile.write(f"event: audio\ndata: {data}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"event: done\ndata: {}\n\n")
                self.wfile.flush()
                self.close_connection = True
            finally:
                speech.release()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    profile = tempfile.mkdtemp(prefix="g9-chromium-")
    args = [chromium, "--headless=new", "--disable-gpu", "--no-first-run",
            "--no-default-browser-check", "--autoplay-policy=no-user-gesture-required",
            f"--user-data-dir={profile}"]
    if os.environ.get("G9_NO_SANDBOX") == "1":
        args.append("--no-sandbox")
    browser = subprocess.Popen(args + [url], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        if not got.wait(60):
            err = browser.stderr.read1(4000).decode(errors="replace") if browser.poll() is not None else ""
            print(f"SPEAK_QUEUE_SMOKE FAIL no result from the page within 60 s {err.strip()[:400]}")
            return 1
    finally:
        browser.terminate()
        try:
            browser.wait(10)
        except subprocess.TimeoutExpired:
            browser.kill()
        server.shutdown()
        shutil.rmtree(profile, ignore_errors=True)

    starts, done, errors = result["starts"], result["done"], result["errors"]
    overlaps, gaps, playhead = 0, [], None
    for s in starts:
        if playhead is not None:
            if s["at"] < playhead - 1e-3:
                overlaps += 1
            elif s["at"] > playhead + 0.05:
                gaps.append(round(s["at"] - playhead, 3))
        playhead = max(playhead or 0.0, s["at"] + s["dur"])
    busy = sum(1 for _, code in requests if code == 409)
    order = [seq for seq, code in requests if code == 200]
    summary = (f"chunks={len(starts)} overlaps={overlaps} gaps={gaps} busy={busy} "
               f"order={order} done={done} errors={errors}")
    ok = (len(starts) >= 2 and overlaps == 0 and not gaps and busy == 0
          and order == sorted(order) and len(done) == 1 and not errors
          and done[0]["t"] >= playhead - 0.05)
    print(f"SPEAK_QUEUE_SMOKE {'PASS' if ok else 'FAIL'} {summary}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
