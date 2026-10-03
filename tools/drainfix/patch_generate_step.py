#!/usr/bin/env python3
"""DrainFix: apply the env-gated depth-2 decode lookahead to a venv's mlx_lm/generate.py.

MLX_OMARCHY_DECODE_LOOKAHEAD (unset/0 = original 1-deep async structure; >=2 = enqueue
chain k+2 before the per-token item() sync). The patch preserves the original loop's
enqueue/break/cache semantics exactly when the gate is off.

Refuses to touch anything but a byte-identical copy of the upstream 0.31.3 block.
"""
import hashlib
import sys

OLD = """    if greedy:
        mx.async_eval(y)
    else:
        mx.async_eval(y, logprobs)
    n = 0
    while True:
        if n != max_tokens:
            next_y, next_logprobs = _step(y)
            if greedy:
                mx.async_eval(next_y)
            else:
                mx.async_eval(next_y, next_logprobs)
        if n == 0:
            mx.eval(y)
            prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
        if n == max_tokens:
            break
        yield y.item(), logprobs
        if n % 256 == 0:
            mx.clear_cache()
        y, logprobs = next_y, next_logprobs
        n += 1
"""

NEW = """    if greedy:
        mx.async_eval(y)
    else:
        mx.async_eval(y, logprobs)
    # DrainFix: env-gated lookahead depth. Off = the original 1-deep structure
    # (next chain enqueued after the previous item() sync wakes the generator).
    # Depth >= 2 additionally enqueues chain k+2 BEFORE the yield, so the host
    # encode+submit path no longer has to fit in the margin between chains.
    lookahead = 0
    _la_raw = __import__("os").environ.get("MLX_OMARCHY_DECODE_LOOKAHEAD")
    if _la_raw not in (None, "", "0"):
        try:
            lookahead = int(_la_raw)
        except ValueError:
            lookahead = 0
    n = 0
    next_y = next_logprobs = None
    la_y = la_logprobs = None
    while True:
        if n != max_tokens and next_y is None:
            next_y, next_logprobs = _step(y)
            if greedy:
                mx.async_eval(next_y)
            else:
                mx.async_eval(next_y, next_logprobs)
        if lookahead >= 2 and n + 1 != max_tokens and next_y is not None and la_y is None:
            la_y, la_logprobs = _step(next_y)
            if greedy:
                mx.async_eval(la_y)
            else:
                mx.async_eval(la_y, la_logprobs)
        if n == 0:
            mx.eval(y)
            prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
        if n == max_tokens:
            break
        yield y.item(), logprobs
        if n % 256 == 0:
            mx.clear_cache()
        if lookahead >= 2:
            y, logprobs = next_y, next_logprobs
            next_y, next_logprobs = la_y, la_logprobs
            la_y = la_logprobs = None
        else:
            y, logprobs = next_y, next_logprobs
            next_y = next_logprobs = None
        n += 1
"""


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_generate_step.py <venv>/lib/python3.*/site-packages/mlx_lm/generate.py")
        return 2
    path = sys.argv[1]
    src = open(path).read()
    if OLD not in src:
        if NEW in src:
            print(f"ALREADY-PATCHED {path}")
            return 0
        print(f"REFUSE: expected upstream block not found in {path}")
        return 1
    sha_before = hashlib.sha256(src.encode()).hexdigest()
    patched = src.replace(OLD, NEW, 1)
    compile(patched, path, "exec")  # syntax gate before writing
    with open(path, "w") as f:
        f.write(patched)
    sha_after = hashlib.sha256(patched.encode()).hexdigest()
    print(f"PATCHED {path}")
    print(f"generate.py sha256 before: {sha_before}")
    print(f"generate.py sha256 after:  {sha_after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
