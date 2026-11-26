#!/usr/bin/env python3
"""Steady-state decode benchmark at a pinned generation length.

Every earlier decode tok/s row in this project was an EOS-truncated
short-burst rate: with a greedy prompt like "What is the capital of
France?" the model stops after 2-10 tokens, so --max-tokens 32 is only a
cap and never a measurement length. Those rates mix load, prompt
processing, startup, and a couple of decode steps, and are not
comparable across machines or wheels (see
receipts/2026-09-03-decode-metric-fix.md).

This script fixes the metric:
  - EOS is suppressed for the run, so generation produces exactly
    --tokens tokens.
  - The produced token count is asserted against the request. A short
    burst fails loudly with a nonzero exit instead of printing a rate.
  - Load is excluded. Prompt processing (prefill) is timed separately.
  - The decode rate is always printed with the token count it was
    measured over, e.g. "decode 1.97 tok/s over 64 tokens". A rate
    without its token count is not comparable and must not be quoted.
  - The EXACT generated token IDs travel with every run (the
    "generated_ids sha256:..." line and one JSON result line), so
    before/after greedy identity is decided on the IDs themselves.
    Never re-derive IDs via tokenizer.decode(ids) then encode(): the
    text round trip truncates and renormalizes and can both hide and
    fabricate divergences.

Run it with MLX_DISABLE_COMPILE=1 to match the receipt baseline.

Self-test (no GPU, no mlx needed): python3 bench_decode.py --self-test
proves the pinned length is honored, that the assertion fires when
generation stops early, and that the ID digest covers the exact IDs.
Self-test timings are meaningless; only the assertion behavior is
verified.

Exit codes: 0 measured; 1 pinned-length violation; 2 bad invocation or
unusable environment; 3 provenance refusal.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def ids_digest(ids):
    """Stable digest over the EXACT generated token IDs, in order.

    Greedy identity between two wheels is decided by comparing these
    digests. Never decide identity by tokenizer.decode(ids) followed by
    encode(): the text round trip truncates and renormalizes (special
    tokens, multi-byte characters, EOS), so it can hide a real ID
    divergence and can fabricate one. Compare the IDs themselves.
    """
    h = hashlib.sha256()
    h.update(",".join(str(int(i)) for i in ids).encode("ascii"))
    return h.hexdigest()[:16]


def run_generation(gen_iter, ids=None, stats=None):
    """Consume a token iterator, returning (token_times_ns, n_tokens).

    gen_iter yields one object per generated token. Timing wraps only
    the iteration, so it excludes load and prompt setup. When `ids` is
    a list, the exact generated token IDs are appended to it: each item
    is either a GenerationResponse (.token is the id) or a bare id
    (self-test stubs). Identity work never re-derives ids from decoded
    text. When `stats` is a dict, the first response's prompt_tokens
    (the count mlx_lm actually processed) is stored in it - never
    retokenized, never guessed.
    """
    times = []
    for item in gen_iter:
        times.append(time.monotonic_ns())
        if ids is not None:
            ids.append(getattr(item, "token", item))
        if stats is not None and "prompt_tokens" not in stats:
            pt = getattr(item, "prompt_tokens", None)
            stats["prompt_tokens"] = int(pt) if pt is not None else None
    return times, len(times)


def result_line(decode_tps, ids, prefill_ns, prompt_tokens=None,
                device=None):
    """One machine-readable JSON line per run (parse with json.loads)."""
    return json.dumps({
        "engine": "bench_decode",
        "decode_tps": round(decode_tps, 4),
        "generated": len(ids),
        "ids_sha256_16": ids_digest(ids),
        "ids_first": [int(i) for i in ids[:3]],
        "ids_last": [int(i) for i in ids[-3:]],
        "prefill_s": round(prefill_ns / 1e9, 6),
        "prompt_tokens": prompt_tokens,
        "prefill_tps": (round(prompt_tokens / (prefill_ns / 1e9), 4)
                        if prompt_tokens and prefill_ns else None),
        "device": device,
    }, sort_keys=True)


def report(prefill_ns, token_times, requested, ids=None,
           prompt_tokens=None, device=None):
    """Print prefill and pinned-length decode rates, or fail loudly."""
    n = len(token_times)
    if n != requested:
        print(
            f"FAILED: requested {requested} tokens, generated {n}. "
            "This is the EOS-truncated short-burst defect; the decode "
            "rate below would be invalid, so none is emitted.",
            file=sys.stderr)
        sys.exit(1)

    decode_gaps = [b - a for a, b in zip(token_times, token_times[1:])]
    decode_ns = sum(decode_gaps)
    decode_tps = (n - 1) / (decode_ns / 1e9) if decode_ns else float("nan")
    print(f"decode {decode_tps:.2f} tok/s over {n - 1} tokens "
          f"({n} requested, EOS suppressed)")
    if prefill_ns:
        print(f"prefill {prefill_ns / 1e9:.3f}s (reported separately, "
              "excluded from decode)")
    # Contract (bench_matrix PROMPT_TOKENS_RE): exactly one line
    # "prompt_tokens <int>" and nothing else on it. When the response
    # carried no count, no line is emitted at all - consumers treat a
    # missing line as null; a guessed or padded line would be worse.
    if prompt_tokens is not None:
        print(f"prompt_tokens {prompt_tokens}")
    per_tok = decode_ns / max(1, n - 1) / 1e6
    print(f"decode mean per-token {per_tok:.1f} ms")
    if ids is not None:
        print(f"generated_ids sha256:{ids_digest(ids)} n={len(ids)} "
              f"first={','.join(str(int(i)) for i in ids[:3])} "
              f"last={','.join(str(int(i)) for i in ids[-3:])}")
        print(result_line(decode_tps, ids, prefill_ns,
                          prompt_tokens=prompt_tokens, device=device))
    return decode_tps


def self_test():
    """Prove pinning, the assertion, and the ID digest with stubs."""

    class StubTokenizer:
        eos_token_ids = {0}

    def stub_iter(n):
        for i in range(n):
            yield i

    # 1. Pinned length honored: 64 requested, 64 produced, no failure.
    times, n = run_generation(stub_iter(64))
    assert n == 64
    print("self-test: pinned length 64 honored "
          f"({len(times)} tokens produced)")

    # 2. Early stop fires the assertion (run_generation itself counts;
    # the real generator's stream_generate stops at EOS the same way).
    _, n = run_generation(stub_iter(3))
    assert n == 3
    try:
        report(prefill_ns=0, token_times=times[:3], requested=32)
    except SystemExit as e:
        assert e.code != 0
        print("self-test: short-burst assertion fired "
              f"(exit {e.code}) for 3 tokens vs 32 requested")
    else:
        raise AssertionError("short generation did not fail")

    # 3. Exactly-pinned run passes report().
    times, n = run_generation(stub_iter(32))
    report(prefill_ns=1_000_000, token_times=times, requested=32)

    # 4. The digest covers the EXACT ids in order. The stub iterable has
    # no encode/decode at all, so a decode->encode round trip would
    # crash here rather than pass: the identity path cannot be the text
    # one.
    exact = [5, 9214, 0, 151643, 7]
    got = []
    run_generation(iter(exact), got)
    assert got == exact, got
    want = hashlib.sha256(",".join(map(str, exact)).encode("ascii"))
    assert ids_digest(got) == want.hexdigest()[:16]
    assert ids_digest(exact[:-1] + [8]) != ids_digest(exact), (
        "digest did not notice a changed id")
    parsed = json.loads(result_line(1.0, exact, 0))
    assert parsed["ids_sha256_16"] == ids_digest(exact)
    assert parsed["generated"] == 5 and parsed["ids_first"] == [5, 9214, 0]
    print("self-test: exact generated-ID digest verified "
          "(no text round trip possible)")

    # 5. The prompt token count comes from the first response's
    # prompt_tokens - the field mlx_lm's GenerationResponse carries -
    # and is never guessed by retokenizing. A response without the
    # field reports unknown, not zero.
    class FakeResp:
        def __init__(self, token, prompt_tokens):
            self.token = token
            self.prompt_tokens = prompt_tokens

    stats = {}
    got = []
    run_generation((FakeResp(7, 11), FakeResp(8, 11)), got, stats)
    assert stats["prompt_tokens"] == 11 and got == [7, 8], stats
    bare_stats = {}
    run_generation([5, 6], [], bare_stats)
    assert bare_stats["prompt_tokens"] is None, bare_stats
    parsed = json.loads(result_line(1.0, exact, 1_000_000,
                                    prompt_tokens=11, device="probe"))
    assert parsed["prompt_tokens"] == 11 and parsed["device"] == "probe"
    assert parsed["prefill_tps"] == 11000.0, parsed
    missing = json.loads(result_line(1.0, exact, 1_000_000))
    assert missing["prompt_tokens"] is None and missing["prefill_tps"] is None
    print("self-test: prompt_tokens captured from the first response "
          "(unknown when absent, never guessed)")
    print("self-test: OK")


def main():
    # Self-test is stub-only and stays runnable anywhere; the guard
    # sits behind it so only real measurement refuses under simulation.
    if not ({"--self-test"} & set(sys.argv[1:])):
        import caps_sim_guard
        caps_sim_guard.refuse_if_simulated("bench_decode")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--tokens", type=int, default=64,
                    help="pinned number of generated tokens (not a cap)")
    ap.add_argument("--wheel", default=None,
                    help="wheel file this run claims to be testing; the "
                         "loaded libmlx.so and core extension are "
                         "hash-compared against it and any mismatch "
                         "refuses to emit a rate")
    ap.add_argument("--temp", type=float, default=0.0,
                    help="sampling temperature; 0.0 is greedy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup-tokens", type=int, default=4,
                    help="untimed tokens generated first")
    ap.add_argument("--raw-prompt", action="store_true",
                    help="skip the chat template")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.model:
        ap.error("--model is required (path to the mlx_lm model)")
    if not args.prompt:
        ap.error("--prompt is required")
    if args.tokens < 2:
        ap.error("--tokens must be >= 2: a decode rate needs at least "
                 "one inter-token gap")
    if args.wheel and not Path(args.wheel).is_file():
        print(f"ERROR: --wheel {args.wheel} does not exist", file=sys.stderr)
        sys.exit(2)
    # Provenance gate: no number leaves this process without the identity
    # of the binary that produced it. Runs BEFORE any mlx import so a
    # lying environment refuses instead of measuring.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mlx_provenance import (ProvenanceRefusal, installed_provenance,
                                provenance_line)
    try:
        prov = installed_provenance(
            expect_wheel=Path(args.wheel) if args.wheel else None)
        if prov["verified"] == "mismatch":
            raise ProvenanceRefusal(prov["mismatch"])
    except ProvenanceRefusal as exc:
        print(f"REFUSING TO EMIT NUMBERS: {exc}", file=sys.stderr)
        sys.exit(3)
    print(provenance_line(prov))
    if prov["verified"] == "no-mlx":
        print("mlx is not importable in this interpreter; cannot benchmark.",
              file=sys.stderr)
        sys.exit(2)
    from mlx_lm.utils import load
    from mlx_lm.generate import stream_generate

    model, tokenizer = load(args.model)
    import mlx.core as mx
    device = "unknown"
    try:
        di = mx.metal.device_info()
        device = (str(di.get("device_name", di)) if hasattr(di, "get")
                  else str(di))
    except Exception:
        pass  # hardware identity is best-effort; provenance stays exact
    from mlx_lm.sample_utils import make_sampler
    mx.random.seed(args.seed)
    sampler = make_sampler(temp=args.temp)

    prompt = args.prompt
    if not args.raw_prompt and hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True)
    # Build the 150k-entry streaming detokenizer and tokenize once, outside the
    # measured window. stream_generate otherwise repeats both for warmup and
    # measurement even though this benchmark only records token IDs.
    if isinstance(prompt, str):
        add_special_tokens = (getattr(tokenizer, "bos_token", None) is None or
                              not prompt.startswith(tokenizer.bos_token))
        prompt = tokenizer.encode(
            prompt, add_special_tokens=add_special_tokens)
    prompt_ids = mx.array(prompt)
    tokenizer.eos_token_ids = set()
    detokenizer = tokenizer.detokenizer
    tokenizer._detokenizer_class = lambda _: detokenizer

    def generate(n):
        detokenizer.reset()
        return stream_generate(
            model, tokenizer, prompt_ids, max_tokens=n, sampler=sampler)

    if args.warmup_tokens:
        for _ in generate(args.warmup_tokens):
            pass

    t0 = time.monotonic_ns()
    generated_ids = []
    run_stats = {"prompt_tokens": int(prompt_ids.size)}
    times, n = run_generation(generate(args.tokens), generated_ids,
                              run_stats)
    prefill_ns = times[0] - t0 if times else 0
    report(prefill_ns, times, args.tokens, generated_ids,
           prompt_tokens=run_stats.get("prompt_tokens"), device=device)


if __name__ == "__main__":
    main()
