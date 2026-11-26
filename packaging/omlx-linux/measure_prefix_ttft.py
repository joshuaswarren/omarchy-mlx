#!/usr/bin/env python3
"""Sequential shared-prefix TTFT measurement (inline streaming).

req1 streams to completion, THEN req2 streams. Each request's
time-to-first-content-chunk is measured from its own request start,
inline, via urllib streaming — no post-transfer artifacts.
"""
import json, time, urllib.request

BASE = "http://127.0.0.1:8900"
MODEL = "mlx-community--Qwen3-4B-Instruct-2507-4bit"
pre = "Reference passage: " + ("Water evaporates from the surface, condenses into clouds, and returns as precipitation in a closed loop. " * 30)

def stream_ttft(payload):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"content-type": "application/json"})
    t0 = time.monotonic()
    first = None
    nch = 0
    with urllib.request.urlopen(req, timeout=90) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            try:
                d = json.loads(p)
            except Exception:
                continue
            if (d.get("choices") or [{}])[0].get("delta", {}).get("content"):
                if first is None:
                    first = time.monotonic() - t0
                nch += 1
    total = time.monotonic() - t0
    return first, nch, total

r1 = {"model": MODEL, "stream": True,
      "messages": [{"role": "user", "content": pre + "\nQuestion: summarize the passage in one sentence."}],
      "max_tokens": 48, "temperature": 0}
r2 = {"model": MODEL, "stream": True,
      "messages": [{"role": "user", "content": pre + "\nQuestion: list the three stages mentioned."}],
      "max_tokens": 48, "temperature": 0}

f1, n1, t1 = stream_ttft(r1)
print(f"[prefix-seq] req1 (cold prefix):  ttft={f1:.3f}s total={t1:.3f}s chunks={n1}")
f2, n2, t2 = stream_ttft(r2)
print(f"[prefix-seq] req2 (warm prefix):  ttft={f2:.3f}s total={t2:.3f}s chunks={n2}")
if f1 and f2:
    print(f"[prefix-seq] TTFT drop: {f1 - f2:+.3f}s ({(f2 / f1 * 100):.1f}% of cold)")
