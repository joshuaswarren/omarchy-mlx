"""Total-context-capped launcher for mlx_lm.server (pinned 0.31.3).

The upstream server treats --max-tokens as a per-request DEFAULT: a client
may send any prompt plus any max_tokens of its own, and nothing bounds
prompt + output. This shim adds the missing cap at the tokenization trust
boundary — the one place where the full prompt token list exists — and
rejects any request whose prompt + max_tokens exceeds the admitted context
budget before a single token is generated.

Aggregate bound: the serve CLI launches this server with decode and prompt
concurrency 1 and the prompt cache disabled, so exactly ONE request's
tokens are ever resident and the aggregate equals the admitted context.
A larger --prompt-cache-size re-enables prefix caching at a documented
cost: worst-case resident KV becomes (cache + 1) x the admitted context.

Fail-closed pinning: the shim supports exactly the pinned mlx-lm version.
A different mlx_lm.__version__ refuses to launch unless
MLX_OMARCHY_ALLOW_UNPINNED_MLXLM=1 is set explicitly (loud warning); if
the hooked internals moved anyway, the launch still fails loudly instead
of silently dropping the cap.
"""

from __future__ import annotations

import inspect
import json
import os
import sys

LIMIT_ENV = "MLX_OMARCHY_SERVE_CONTEXT_LIMIT"
UNPINNED_ENV = "MLX_OMARCHY_ALLOW_UNPINNED_MLXLM"
SUPPORTED_MLXLM_VERSIONS = ("0.31.3",)


def _mlxlm_version(server_module) -> str:
    for holder in (server_module, getattr(server_module, "_version", None)):
        version = getattr(holder, "__version__", None)
        if version:
            return str(version)
    return "unknown"


def check_pinned(server_module) -> str:
    version = _mlxlm_version(server_module)
    if version in SUPPORTED_MLXLM_VERSIONS:
        return version
    if os.environ.get(UNPINNED_ENV) == "1":
        print(
            f"warning: {UNPINNED_ENV}=1 — mlx-lm {version} is not the pinned "
            f"{SUPPORTED_MLXLM_VERSIONS}; the cap hooks may not hold",
            file=sys.stderr,
        )
        return version
    raise RuntimeError(
        f"mlx-lm {version} is not the pinned version "
        f"{SUPPORTED_MLXLM_VERSIONS}; refusing to launch without the "
        f"verified total-context cap (export {UNPINNED_ENV}=1 to override "
        "explicitly)"
    )


EXPECTED_TOKENIZE_SIGNATURE = ("self", "tokenizer", "request", "args")


def install(server_module, limit: int) -> None:
    generator = server_module.ResponseGenerator
    original = getattr(generator, "_tokenize", None)
    if not callable(original):
        raise RuntimeError(
            "mlx_lm.server.ResponseGenerator._tokenize not found; this shim "
            "does not know this mlx-lm version and refuses to launch without "
            "the total-context cap"
        )
    # Enforced for the pinned version AND any pin bypass: if the internals
    # moved, the cap cannot be proven to hold and the launch refuses.
    try:
        params = tuple(inspect.signature(original).parameters)
    except (TypeError, ValueError):
        params = ()
    if params != EXPECTED_TOKENIZE_SIGNATURE:
        raise RuntimeError(
            "unexpected _tokenize signature "
            f"{params} != {EXPECTED_TOKENIZE_SIGNATURE}; this shim does not "
            "know this mlx-lm build and refuses to launch without the "
            "verified total-context cap"
        )

    def capped(self, tokenizer, request, args):
        result = original(self, tokenizer, request, args)
        prompt = result[0]
        # Upstream semantics (verified in the pinned source): the HTTP
        # handler resolves max_tokens from the body or the CLI default
        # (an int, default 512) and validates int >= 0 BEFORE queuing, so
        # None or negatives never legitimately reach this boundary.
        # 0 generates nothing (bounded); -1 (infinite) is handler-blocked.
        max_tokens = getattr(args, "max_tokens", None)
        if max_tokens is None or isinstance(max_tokens, bool) \
                or not isinstance(max_tokens, int) or max_tokens < 0:
            raise ValueError(
                "request rejected: max_tokens must be a non-negative int at "
                f"the cap boundary, got {max_tokens!r}"
            )
        total = len(prompt) + max_tokens
        if total > limit:
            raise ValueError(
                f"request rejected: prompt ({len(prompt)} tokens) + max_tokens "
                f"({max_tokens}) exceeds the admitted context budget of {limit}; "
                "lower the prompt or max_tokens"
            )
        return result

    generator._tokenize = capped


PROBE_ENV = "MLX_OMARCHY_SERVE_IDS_PROBE"


def install_ids_probe(server_module, emit=None) -> None:
    """Bench-only ids capture for the managed route (default OFF; enabled
    only when MLX_OMARCHY_SERVE_IDS_PROBE=1). Wraps the streaming
    detokenizer classes so every raw sampled token id is recorded at the
    add_token boundary and the completed per-request id list is emitted as
    one `PROBE: {...}` JSON line on stderr at reset(). Nothing on disk is
    touched; with the env unset this function is never called."""
    if emit is None:
        def emit(event):
            print("PROBE:", json.dumps(event), file=sys.stderr, flush=True)
    state = {"ids": []}
    from mlx_lm.tokenizer_utils import (
        NaiveStreamingDetokenizer,
        BPEStreamingDetokenizer,
        SPMStreamingDetokenizer,
        StreamingDetokenizer,
    )

    def wrap(cls):
        if cls is None or getattr(cls, "_ids_probe_installed", False):
            return
        orig_add = getattr(cls, "add_token", None)
        if orig_add is None:
            return

        def add_token_probe(self, token, _orig=orig_add):
            state["ids"].append(int(token))
            _orig(self, token)

        cls.add_token = add_token_probe
        cls._ids_probe_installed = True

        orig_reset = getattr(cls, "reset", None)
        if orig_reset is not None:

            def reset_probe(self, _orig=orig_reset):
                if state["ids"]:
                    import hashlib
                    emit({"event": "generation", "n": len(state["ids"]),
                          "ids_sha16": hashlib.sha256(
                              json.dumps(state["ids"]).encode()).hexdigest()[:16],
                          "ids": state["ids"]})
                    state["ids"] = []
                return _orig(self)

            cls.reset = reset_probe

        # mlx_lm's stream_generate finalizes (never resets) at request end —
        # the server path's actual flush boundary (generate.py:741).
        orig_finalize = getattr(cls, "finalize", None)
        if orig_finalize is not None:

            def finalize_probe(self, _orig=orig_finalize):
                if state["ids"]:
                    import hashlib
                    emit({"event": "generation", "n": len(state["ids"]),
                          "ids_sha16": hashlib.sha256(
                              json.dumps(state["ids"]).encode()).hexdigest()[:16],
                          "ids": state["ids"]})
                    state["ids"] = []
                return _orig(self)

            cls.finalize = finalize_probe

    for cls in (StreamingDetokenizer, NaiveStreamingDetokenizer,
                BPEStreamingDetokenizer, SPMStreamingDetokenizer):
        wrap(cls)

    # Third capture site: the BATCHED branch (mlx_lm/server.py:884) feeds
    # generated tokens to the detokenizer via add_token but NEVER calls
    # reset/finalize — the batch flush boundary is BatchGenerator.remove
    # (finished uids) and close. With the managed CLI's decode_concurrency
    # of 1, the accumulated state at remove time is exactly the finished
    # request's token array.
    batch_cls = getattr(server_module, "BatchGenerator", None)

    def _flush_uids(uids=None):
        import hashlib
        store = state.get("by_uid", {})
        done = list(store) if not uids else [u for u in uids if u in store]
        for uid in done:
            ids = store.pop(uid)
            emit({"event": "generation", "n": len(ids),
                  "ids_sha16": hashlib.sha256(
                      json.dumps(ids).encode()).hexdigest()[:16],
                  "ids": ids})

    if batch_cls is not None and not getattr(batch_cls, "_ids_probe_flush",
                                             False):
        orig_next = getattr(batch_cls, "next", None)
        if orig_next is not None:
            # Primary batched capture: gen_responses carry (.uid, .token) —
            # accumulate per uid and emit each uid's array when the server
            # removes that uid (request completion).
            def next_probe(self, *_a, _orig=orig_next, **_k):
                prompt_responses, gen_responses = _orig(self, *_a, **_k)
                for r in gen_responses or []:
                    token = getattr(r, "token", None)
                    uid = getattr(r, "uid", None)
                    if token is None or uid is None:
                        continue
                    state.setdefault("by_uid", {}).setdefault(
                        uid, []).append(int(token))
                    # The server never resets the detokenizer on this route;
                    # flush a uid's array the moment its generation finishes.
                    if getattr(r, "finish_reason", None) is not None:
                        _flush_uids([uid])
                return prompt_responses, gen_responses
            batch_cls.next = next_probe

        orig_remove = getattr(batch_cls, "remove", None)
        if orig_remove is not None:
            def remove_probe(self, uids, _orig=orig_remove):
                _flush_uids(list(uids) if uids else None)
                return _orig(self, uids)
            batch_cls.remove = remove_probe
        else:
            orig_remove = None
        orig_close = getattr(batch_cls, "close", None)
        if orig_close is not None:
            def close_probe(self, _orig=orig_close):
                _flush_uids(None)
                return _orig(self)
            batch_cls.close = close_probe
        batch_cls._ids_probe_flush = True

    # Second, independent capture site: mlx_lm.server calls stream_generate
    # on the single-stream path (the managed CLI pins decode_concurrency=1,
    # so this is THE generation route). Collect each yielded response's
    # token and flush the per-request array when the generator exhausts —
    # independent of any detokenizer reset/finalize behavior.
    orig_stream = getattr(server_module, "stream_generate", None)
    if orig_stream is not None:
        def stream_probe(*args, **kwargs):
            gen = orig_stream(*args, **kwargs)
            print("shim: stream_probe ENTERED", file=sys.stderr, flush=True)
            count = 0
            while True:
                try:
                    item = next(gen)
                except StopIteration:
                    break
                token = getattr(item, "token", None)
                if token is not None:
                    state["ids"].append(int(token))
                    count += 1
                try:
                    yield item
                except GeneratorExit:
                    # Client disconnected mid-generation: flush what the
                    # request produced so partial arrays are recorded.
                    if count:
                        import hashlib
                        emit({"event": "generation_client_disconnected",
                              "n": len(state["ids"]),
                              "ids": state["ids"]})
                        state["ids"] = []
                    raise
            if count:
                import hashlib
                emit({"event": "generation", "n": len(state["ids"]),
                      "ids_sha16": hashlib.sha256(
                          json.dumps(state["ids"]).encode()).hexdigest()[:16],
                      "ids": state["ids"]})
                state["ids"] = []

        server_module.stream_generate = stream_probe


def main() -> None:
    raw = os.environ.get(LIMIT_ENV, "")
    try:
        limit = int(raw)
    except ValueError:
        print(f"error: {LIMIT_ENV} must be an integer, got {raw!r}", file=sys.stderr)
        raise SystemExit(3)
    if limit <= 0:
        print(f"error: {LIMIT_ENV} must be positive", file=sys.stderr)
        raise SystemExit(3)
    try:
        import mlx_lm.server as server_module
    except ImportError as exc:
        print(f"error: mlx_lm is not installed ({exc})", file=sys.stderr)
        raise SystemExit(3)
    check_pinned(server_module)
    install(server_module, limit)
    probe_wanted = os.environ.get(PROBE_ENV) == "1"
    print(f"shim: PROBE_ENV={os.environ.get(PROBE_ENV)!r} "
          f"installing={probe_wanted}", file=sys.stderr, flush=True)
    if probe_wanted:
        install_ids_probe(server_module)
        print("shim: ids probe installed", file=sys.stderr, flush=True)
    server_module.main()


if __name__ == "__main__":
    main()
