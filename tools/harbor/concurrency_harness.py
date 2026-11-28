#!/usr/bin/env python3
"""Q-104/#62 offline sub-step: the multi-node concurrency harness for the
harbor-gap receipt (30 batches x 4 simultaneous OpenAI-compatible requests).

Run against an endpoint once the two-node serve exists:
    concurrency_harness.py --endpoint http://node:8020/v1 --model <id> \
        --batches 30 --parallel 4 --out receipt.json

Offline now: --endpoint http://127.0.0.1:0 stub-mode runs against a local
aiohttp stub so the harness's own accounting (aggregate tok/s, p50/p95,
HTTP 200 rate, empty-completion rate) is validated without any node.
"""
import argparse
import asyncio
import json
import statistics
import sys
import time

import aiohttp


async def one_request(session, url, model, payload):
    started = time.perf_counter()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=600)) as r:
            body = await r.json()
            elapsed = time.perf_counter() - started
            text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = body.get("usage", {})
            return {"rc": r.status, "elapsed": elapsed, "chars": len(text),
                    "tokens": usage.get("completion_tokens", 0)}
    except Exception as exc:
        return {"rc": None, "elapsed": time.perf_counter() - started,
                "chars": 0, "tokens": 0, "error": repr(exc)[:120]}


async def run_batches(session, url, model, batches, parallel, prompt):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 64}
    rows = []
    for b in range(batches):
        t0 = time.perf_counter()
        results = await asyncio.gather(*(one_request(session, url, model, payload)
                                         for _ in range(parallel)))
        batch_s = time.perf_counter() - t0
        for i, r in enumerate(results):
            rows.append({**r, "batch": b, "slot": i, "batch_s": round(batch_s, 2)})
        print(f"batch {b}: {batch_s:.2f}s rc={[r['rc'] for r in results]}")
    return rows


def summarize(rows, batches, parallel):
    ok = [r for r in rows if r["rc"] == 200 and r["chars"] > 0]
    lat = sorted(r["elapsed"] for r in ok)
    toks = sum(r["tokens"] for r in ok)
    wall = sum(r["batch_s"] for r in rows[:parallel]) if rows else 0
    return {"requests": len(rows), "ok": len(ok),
            "http200_pct": round(100 * len(ok) / max(len(rows), 1), 1),
            "empty_completion_pct": round(100 * sum(1 for r in ok if r["chars"] == 0)
                                          / max(len(ok), 1), 1),
            "p50_s": round(statistics.median(lat), 2) if lat else None,
            "p95_s": round(lat[int(len(lat) * 0.95) - 1], 2) if lat else None,
            "aggregate_tokens": toks,
            "aggregate_tokens_per_s": round(toks / sum(r["batch_s"] for r in rows), 1) if rows else 0}


async def stub_mode(batches, parallel):
    """A local stub so the harness's own accounting is validated offline."""
    from aiohttp import web

    async def handler(request):
        await asyncio.sleep(0.05)
        return web.json_response({"choices": [{"message": {"content": "stub"}}],
                                  "usage": {"completion_tokens": 7}})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[0].sockport if False else site._server.sockets[0].getsockname()[1]}/v1/chat/completions"
    async with aiohttp.ClientSession() as session:
        rows = await run_batches(session, url, "stub", batches, parallel, "stub prompt")
    await runner.cleanup()
    return rows


async def main_async(args):
    if args.endpoint.endswith(":0"):
        rows = await stub_mode(args.batches, args.parallel)
    else:
        async with aiohttp.ClientSession() as session:
            rows = await run_batches(session, args.endpoint + "/chat/completions",
                                     args.model, args.batches, args.parallel, args.prompt)
    summary = summarize(rows, args.batches, args.parallel)
    print(json.dumps(summary, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "rows": rows}, f, indent=1)
    bad = [r for r in rows if r["rc"] != 200 or r["chars"] == 0]
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:0")
    ap.add_argument("--model", default="stub")
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--prompt", default="Summarize the meeting notes in three bullets.")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    sys.exit(asyncio.run(main_async(a)))


if __name__ == "__main__":
    main()
