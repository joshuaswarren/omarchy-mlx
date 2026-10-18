#!/usr/bin/env python3
"""SSE TTFT probe: timestamp of the FIRST generated token and the FIRST
non-empty content token, reported SEPARATELY (Main correction: a generic
TTFT that counts only content hides reasoning-start latency).

Two clocks, both from request send:
  first_event_s  arrival of the first `data:` event carrying any delta
                 (reasoning-only deltas count — this is when visible
                 generation actually starts)
  first_content_s arrival of the first delta with non-empty
                 `choices[0].delta.content`
A buffered reader that waits for N bytes measures transfer throughput,
not token latency — that was the v3 defect.
"""
import json
import sys
import time
import urllib.request


def measure_ttft(host, port, body, timeout=300):
    url = f"http://{host}:{port}/v1/chat/completions"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first_event = None
    first_content = None
    first_idx = None
    events = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                events += 1
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                now = time.perf_counter() - t0  # one read per event
                if first_event is None and any(
                        v for k, v in delta.items()
                        if k in ("content", "reasoning_content", "reasoning")
                        and isinstance(v, str) and v):
                    first_event = now
                content = delta.get("content")
                if content and first_content is None:
                    first_content = now
                    first_idx = events
        total = time.perf_counter() - t0
        if first_content is None:
            return {"error": "no non-empty content token observed",
                    "events": events, "total_s": round(total, 4),
                    "first_event_s": (round(first_event, 4)
                                      if first_event else None)}
        return {"first_event_s": round(first_event, 4) if first_event else None,
                "first_content_s": round(first_content, 4),
                "reasoning_lead_s": (round(first_content - first_event, 4)
                                     if first_event else None),
                "total_s": round(total, 4),
                "first_content_event_index": first_idx, "events": events}
    except Exception as exc:  # noqa: BLE001 - probe records, never raises
        return {"error": f"{type(exc).__name__}: {exc}",
                "events": events, "ttft_s": round(time.perf_counter() - t0, 4)}


if __name__ == "__main__":
    host, port, path = sys.argv[1], sys.argv[2], sys.argv[3]
    body = json.load(open(path))
    print(json.dumps(measure_ttft(host, int(port), body), indent=2))
