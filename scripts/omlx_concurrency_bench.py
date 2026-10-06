#!/usr/bin/env python3
"""Measure streamed OpenAI-compatible generation at concurrent request counts."""

import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PROMPT = (
    "Apple Silicon runs local language models on Linux through Vulkan and a "
    "reverse engineered GPU driver. " * 28
)


def read_stream(response, started_at=None):
    started_at = time.monotonic() if started_at is None else started_at
    first_content_at = None
    pieces = []
    usage = None

    for raw_line in response:
        line = raw_line.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        event = json.loads(payload)
        if event.get("error"):
            raise RuntimeError(f"server returned error: {event['error']}")
        for choice in event.get("choices", []):
            piece = (choice.get("delta") or {}).get("content")
            if piece:
                if not isinstance(piece, str):
                    raise RuntimeError("server returned non-text content")
                if first_content_at is None:
                    first_content_at = time.monotonic()
                pieces.append(piece)
        if event.get("usage") is not None:
            usage = event["usage"]

    if first_content_at is None:
        raise RuntimeError("stream contained no text content")
    if not isinstance(usage, dict):
        raise RuntimeError("stream omitted usage; exact token throughput is unavailable")
    try:
        prompt_tokens = int(usage["prompt_tokens"])
        completion_tokens = int(usage["completion_tokens"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("stream usage omitted prompt_tokens or completion_tokens") from exc
    if prompt_tokens < 0 or completion_tokens <= 0:
        raise RuntimeError("server returned invalid token counts")

    text = "".join(pieces)
    finished_at = time.monotonic()
    return {
        "ttft_ms": round((first_content_at - started_at) * 1000, 2),
        "elapsed_s": round(finished_at - started_at, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "completion_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def request_once(endpoint, model, prompt, max_tokens, timeout):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "enable_thinking": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_at = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = read_stream(response, started_at)
    except urllib.error.HTTPError as exc:
        detail = exc.read(4000).decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    result["model"] = model
    return result


def run_level(endpoint, model, prompt, concurrency, repeat, max_tokens, timeout):
    started_at = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(
            pool.map(
                lambda _: request_once(endpoint, model, prompt, max_tokens, timeout),
                range(concurrency),
            )
        )
    wall_s = time.monotonic() - started_at
    completion_tokens = sum(result["completion_tokens"] for result in results)
    return {
        "type": "result",
        "repeat": repeat,
        "concurrency": concurrency,
        "requests": concurrency,
        "wall_s": round(wall_s, 3),
        "completion_tokens": completion_tokens,
        "aggregate_tok_s": round(completion_tokens / wall_s, 2),
        "request_results": results,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible chat completions URL")
    parser.add_argument("--model", required=True, help="Model ID accepted by the endpoint")
    parser.add_argument("--prompt-file", type=Path, help="UTF-8 file; otherwise use the built-in benchmark prompt")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    args = parser.parse_args(argv)
    if args.max_tokens <= 0 or args.timeout <= 0 or args.repeats <= 0:
        parser.error("max-tokens, timeout, and repeats must be positive")
    if not args.concurrency or any(level <= 0 for level in args.concurrency):
        parser.error("concurrency levels must be positive")
    if len(set(args.concurrency)) != len(args.concurrency):
        parser.error("concurrency levels must not repeat")
    args.prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else DEFAULT_PROMPT
    return args


def main(argv=None):
    args = parse_args(argv)
    print(
        json.dumps(
            {
                "type": "configuration",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "endpoint": args.endpoint,
                "model": args.model,
                "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
                "prompt_chars": len(args.prompt),
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "repeats": args.repeats,
                "concurrency": args.concurrency,
            }
        ),
        flush=True,
    )
    try:
        for repeat in range(1, args.repeats + 1):
            for level in args.concurrency:
                result = run_level(
                    args.endpoint,
                    args.model,
                    args.prompt,
                    level,
                    repeat,
                    args.max_tokens,
                    args.timeout,
                )
                print(json.dumps(result), flush=True)
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
