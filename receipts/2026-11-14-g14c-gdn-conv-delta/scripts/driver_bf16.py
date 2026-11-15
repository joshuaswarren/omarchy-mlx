#!/usr/bin/env python3
"""Dispatch-profile arm driver: literal prompt IDs, greedy decode, shared phases.

One invocation is one arm (shared contract
<HOME>/scratch/w9c-profile-prereq-contract-20261114.md lines 8-17):
untimed full fresh-cache warmup, fresh-cache measured pass, fresh-cache
untimed correctness pass that must reproduce the measured IDs and captures
lp/top5 (in the patched greedy-prune path logprobs stay lazy and only a
consumer evaluates them, so capturing them here never touches the measured
window). Writes markers.jsonl (buffered, one flush after the final drain) and
result.json into the fresh --output directory. Result is written only when
every gate passes; a failed arm leaves markers through check_start, no
check_done, no result.json.

Phases (exact shared set): load_start load_done warmup_start warmup_done
prefill_start prefill_done decode_start tok* decode_done drain_done
check_start check_done. Marker writes are buffered during the measured loop;
no file I/O and no stderr echo inside it. No per-op synchronization inside
the measured loop: each yield already blocks in y.item() (mlx-lm 0.32.2
generate.py:478), and drains happen only at phase boundaries.

Drain (source-derived, not guessed): mlx-lm generates on
generation_stream = mx.new_thread_local_stream(mx.default_device())
(generate.py:234) and `load` materializes weights on the default stream
(utils.py:602 mx.eval(model.parameters)). scheduler.cpp offers
synchronize(stream) overloads only - there is no all-streams API in the
retained source - so the driver drains exactly the streams in use:
mx.synchronize() at load_done (default stream) and
mx.synchronize(generation_stream) at every later phase boundary.

Environment is inherited untouched: the wrapper owns
MLX_OMARCHY_BATCH_WORK=100000, MLX_OMARCHY_GPU_PROFILE* and
MLX_DISABLE_COMPILE; the driver sets, reads and hides none of them, and
runtime errors propagate. No downloads. EOS is ignored: the raw generate_step
path has no stop check, so exactly --tokens greedy tokens are produced.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

CTX_CHOICES = (1, 512, 2048)
TOKEN_CHOICES = (8, 16, 128)  # 8/16 only for an explicitly requested pilot
PHASES = (
    "load_start", "load_done", "warmup_start", "warmup_done",
    "prefill_start", "prefill_done", "decode_start", "tok", "decode_done",
    "drain_done", "check_start", "check_done",
)


def load_prompt_ids(path, ctx):
    """Literal prompt IDs: JSON array of exactly ctx non-negative ints."""
    with open(path, "rb") as fh:
        raw = fh.read()
    ids = json.loads(raw)
    if not isinstance(ids, list) or len(ids) != ctx:
        raise ValueError(f"prompt ids must be a JSON array of exactly {ctx} ints")
    for i in ids:
        if type(i) is not int or i < 0:
            raise ValueError("prompt ids must be non-negative integers")
    return ids


def ids_sha(ids):
    """First 16 hex of SHA256 over the comma-joined decimal IDs."""
    return hashlib.sha256(
        ",".join(str(int(i)) for i in ids).encode()).hexdigest()[:16]


def build_result(ctx, mx_version, ids, lp, top5, token_times_ns):
    """Assemble result.json; raises ValueError when a gate fails."""
    tokens = len(ids)
    if not (len(lp) == len(top5) == len(token_times_ns) == tokens):
        raise ValueError("per-token capture lengths do not match ids")
    if tokens < 2:
        raise ValueError("need at least two tokens for an interval")
    for v in lp:
        if not math.isfinite(v):
            raise ValueError("non-finite logprob captured")
    for row in top5:
        if len(row) != 5 or not all(type(t) is int for t in row):
            raise ValueError("top5 rows must be five ints")
    elapsed = token_times_ns[-1] - token_times_ns[0]
    if elapsed <= 0:
        raise ValueError("decode_elapsed_ns is not positive")
    return {
        "ctx": ctx,
        "tokens": tokens,
        "mx_version": str(mx_version),
        "ids": [int(i) for i in ids],
        "ids_sha": ids_sha(ids),
        "lp": [round(float(v), 4) for v in lp],
        "top5": top5,
        "token_times_ns": [int(t) for t in token_times_ns],
        "decode_elapsed_ns": int(elapsed),
        "decode_tps": (tokens - 1) * 1e9 / elapsed,  # == 127e9/elapsed at 128
    }


def expected_phases(tokens):
    head = ["load_start", "load_done", "warmup_start", "warmup_done",
            "prefill_start"]
    mid = ["prefill_done", "decode_start"]
    tail = ["decode_done", "drain_done", "check_start", "check_done"]
    return head + mid + ["tok"] * tokens + tail


def check_markers(marks, tokens):
    """Driver-side gate over the buffered marker list; returns failures."""
    fails = []
    for m in marks:
        if m["p"] not in PHASES:
            fails.append(f"phase {m['p']!r} outside the shared set")
        if type(m["t"]) is not int:
            fails.append(f"phase {m['p']!r} timestamp not an int")
    last = -1
    for m in marks:
        if m["t"] < last:
            fails.append("marker clock went backwards")
            break
        last = m["t"]
    if [m["p"] for m in marks] != expected_phases(tokens):
        fails.append("marker phases do not match the shared contract")
    return fails


def ensure_output_dir(path):
    """--output must not exist; create it fresh or fail."""
    if os.path.exists(path):
        raise ValueError(f"--output directory already exists: {path}")
    os.mkdir(path)


def capture_tokens(iterator, tokens, markers=None, capture_scores=None):
    ids, times, lps, fives = [], [], [], []
    try:
        for index in range(tokens):
            token, scores = next(iterator)
            timestamp = time.monotonic_ns()
            ids.append(int(token))
            times.append(timestamp)
            if markers is not None:
                if index == 0:
                    markers.extend({"p": phase, "t": timestamp}
                                   for phase in ("prefill_done", "decode_start"))
                markers.append({"p": "tok", "t": timestamp})
            if capture_scores is not None:
                lp, top5 = capture_scores(token, scores)
                lps.append(lp)
                fives.append(top5)
    finally:
        iterator.close()
    return ids, times, lps, fives


def main(argv=None):
    ap = argparse.ArgumentParser(description="One fixed decode-profile arm")
    ap.add_argument("--model", required=True, help="literal model snapshot path")
    ap.add_argument("--prompt-ids", required=True,
                    help="JSON array of exactly --ctx non-negative ints")
    ap.add_argument("--ctx", type=int, choices=CTX_CHOICES, default=512)
    ap.add_argument("--output", required=True,
                    help="result directory; must not exist")
    ap.add_argument("--tokens", type=int, default=128, choices=TOKEN_CHOICES,
                    help="128 for main arms; 8/16 only for an explicit pilot")
    args = ap.parse_args(argv)

    prompt_ids = load_prompt_ids(args.prompt_ids, args.ctx)
    if not os.path.isabs(args.model) or not os.path.isdir(args.model):
        ap.error("--model must name an existing absolute local snapshot directory")


    ensure_output_dir(args.output)

    marks = []

    def mark(phase, t=None):
        marks.append({"t": time.monotonic_ns() if t is None else t,
                      "p": phase})

    def run_pass(tokens, want_logprobs=False, markers=None):
        cache = make_prompt_cache(model)
        iterator = generate_step(mx.array(prompt_ids), model, prompt_cache=cache,
                                 max_tokens=tokens)
        score_capture = (
            (lambda token, scores: (scores[token], mx.argsort(-scores)[:5]))
            if want_logprobs else None
        )
        return capture_tokens(iterator, tokens, markers, score_capture)

    try:
        mark("load_start")
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.generate import generate_step, generation_stream
        from mlx_lm.models.cache import make_prompt_cache

        model, _tok = load(args.model)
        mx.synchronize()
        mark("load_done")

        mark("warmup_start")
        run_pass(args.tokens)
        mx.synchronize(generation_stream)
        mx.synchronize()
        mark("warmup_done")

        mark("prefill_start")
        gen_ids, token_times_ns, _l, _f = run_pass(args.tokens, markers=marks)
        mark("decode_done")
        mx.synchronize(generation_stream)
        mx.synchronize()
        mark("drain_done")

        mark("check_start")
        check_ids, _times, lp_vals, top5_vals = run_pass(
            args.tokens, want_logprobs=True)
        flat = lp_vals + [v for row in top5_vals for v in row]
        mx.eval(*flat)
        lp = [round(float(v.item()), 4) for v in lp_vals]
        top5 = [[int(v.item()) for v in row] for row in top5_vals]
        mx.synchronize(generation_stream)
        mx.synchronize()

        fails = []
        result = None
        if check_ids != gen_ids:
            fails.append("correctness pass did not reproduce measured ids")
        try:
            result = build_result(args.ctx, mx.__version__, gen_ids, lp, top5,
                                  token_times_ns)
            result["check_ids"] = check_ids
        except ValueError as exc:
            fails.append(str(exc))
        if not fails:
            mark("check_done")
            fails.extend(check_markers(marks, args.tokens))

        if fails or result is None:
            if marks[-1]["p"] == "check_done":
                marks.pop()
            for failure in fails:
                print(f"CHECK-FAIL: {failure}", file=sys.stderr)
            return 2

        with open(os.path.join(args.output, "result.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(result, fh, allow_nan=False)
            fh.write("\n")
        print(f"[profile_driver] arm ok tokens={args.tokens} "
              f"ids_sha={result['ids_sha']} "
              f"decode_tps={result['decode_tps']:.2f}")
        return 0
    finally:
        _write_markers(args.output, marks)


def _write_markers(output, marks):
    with open(os.path.join(output, "markers.jsonl"), "w",
              encoding="utf-8") as fh:
        fh.writelines(json.dumps(marker) + "\n" for marker in marks)
        fh.flush()
        os.fsync(fh.fileno())


if __name__ == "__main__":
    sys.exit(main())
