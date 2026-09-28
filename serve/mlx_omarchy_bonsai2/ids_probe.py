"""Env-gated ids-probe hook for the bonsai2 managed shim (bench-only).

Default OFF: no behavior change. With MLX_OMARCHY_SERVE_IDS_PROBE=1, the
hook captures the raw sampled token ids at the per-token boundary and
emits one PROBE: JSON line per request. The bonsai2 server's iteration
loop (for response in stream_generate(...): break on finish_reason) never
calls detok.finalize explicitly, so the hook wraps stream_generate and
force-finalizes the detokenizer on generator close.

mlx_lm 0.31.3 split mlx_lm.generate (non-streaming wrapper returning
str) from mlx_lm.stream_generate (the generator). The bonsai2 server
reads sys.modules['mlx_lm'].stream_generate on each call; this hook
replaces that package attribute with the wrap."""
from __future__ import annotations
import hashlib
import json
import os
import sys

PROBE_ENV = "MLX_OMARCHY_SERVE_IDS_PROBE"


def install_ids_probe(emit=None):
    if os.environ.get(PROBE_ENV) != "1":
        return None
    if emit is None:
        def emit(event):
            print("PROBE:", json.dumps(event), file=sys.stderr, flush=True)
    try:
        from mlx_lm.tokenizer_utils import (
            BPEStreamingDetokenizer,
            NaiveStreamingDetokenizer,
            SPMStreamingDetokenizer,
            StreamingDetokenizer,
        )
        import mlx_lm as _mlx_lm_pkg
    except Exception as exc:
        print(f"shim: ids probe install failed: {exc}", file=sys.stderr,
              flush=True)
        return None
    state: dict = {"ids": [], "detok": None}

    def _flush():
        if state["ids"]:
            emit({"event": "generation",
                  "n": len(state["ids"]),
                  "ids_sha16": hashlib.sha256(
                      json.dumps(state["ids"]).encode()).hexdigest()[:16],
                  "ids": state["ids"]})
            state["ids"] = []

    def wrap(cls):
        # Use cls.__dict__ (not getattr, which walks MRO) to avoid the
        # base-class install flag leaking down to subclasses that override
        # add_token/finalize. If a subclass has its own add_token, we want
        # to patch THAT one; the base-class patch alone does not cover the
        # override.
        if cls is None or "_ids_probe_installed" in cls.__dict__:
            return
        orig_add = cls.__dict__.get("add_token")
        if orig_add is None:
            return

        def add_token_probe(self, token, _orig=orig_add):
            state["ids"].append(int(token))
            _orig(self, token)

        cls.add_token = add_token_probe
        cls._ids_probe_installed = True

        orig_reset = cls.__dict__.get("reset")
        if orig_reset is not None:
            def reset_probe(self, _orig=orig_reset):
                _flush()
                return _orig(self)
            cls.reset = reset_probe

        orig_finalize = cls.__dict__.get("finalize")
        if orig_finalize is not None:
            def finalize_probe(self, _orig=orig_finalize):
                _flush()
                return _orig(self)
            cls.finalize = finalize_probe

    # Iterate most-derived first so a subclass override of add_token/
    # finalize is patched with its OWN unpatched bound method as the
    # saved original. If we patched the base class first, the subclass
    # would never get a wrap.
    for cls in (BPEStreamingDetokenizer, SPMStreamingDetokenizer,
                NaiveStreamingDetokenizer, StreamingDetokenizer):
        wrap(cls)

    # CRITICAL: In mlx_lm 0.31.3, `mlx_lm.stream_generate` is the
    # generator and `mlx_lm.generate` is the non-streaming wrapper
    # that returns str. The bonsai2 server reads stream_generate from
    # sys.modules['mlx_lm'].stream_generate on each call, so rebinding
    # the package attribute is what the wrap must do.
    _original = getattr(_mlx_lm_pkg, "stream_generate", None)
    if _original is None or not callable(_original):
        print("shim: mlx_lm.stream_generate missing or non-callable "
              "(unsupported mlx_lm shape; ids-probe wrap not installed)",
              file=sys.stderr, flush=True)
        return None

    def _resolve_detok(*args, **kwargs):
        # In mlx_lm 0.31.3 stream_generate, the caller passes the raw
        # tokenizer (PreTrainedTokenizer or TokenizerWrapper); the
        # stream_generate body wraps it: `if not isinstance(tokenizer,
        # TokenizerWrapper): tokenizer = TokenizerWrapper(tokenizer)` and
        # then reads `tokenizer.detokenizer`. Match that here so the
        # finalize-side hook fires on the same detokenizer instance.
        try:
            from mlx_lm.tokenizer_utils import TokenizerWrapper
            tok = args[1] if len(args) > 1 else None
            if tok is None:
                return None
            if not isinstance(tok, TokenizerWrapper):
                tok = TokenizerWrapper(tok)
            return getattr(tok, "detokenizer", None)
        except Exception:
            return None

    def stream_probe(*args, **kwargs):
        state["detok"] = _resolve_detok(*args, **kwargs)
        state["_finalized"] = False
        gen = _original(*args, **kwargs)
        try:
            yielded = next(gen)
        except StopIteration:
            _probe_finalize(state)
            return
        try:
            while True:
                try:
                    yield yielded
                except GeneratorExit:
                    _probe_finalize(state)
                    raise
                yielded = next(gen)
        except StopIteration:
            _probe_finalize(state)

    def _probe_finalize(state):
        if state.get("_finalized"):
            return
        state["_finalized"] = True
        # First try the detok-side finalize (the patched StreamingDetokenizer
        # method will _flush() ids via the emit callback). Then fall back
        # to a direct _flush() in case the detok was not resolved or the
        # patched finalize was never called by the underlying generator.
        detok = state.get("detok")
        if detok is not None:
            try:
                detok.finalize()
            except Exception:
                pass
        _flush()

    # Rebind the package's stream_generate attr (the bonsai2 server reads
    # sys.modules['mlx_lm'].stream_generate on each call) and the
    # bonsai2 server module's local name. The non-streaming
    # mlx_lm.generate is left alone.
    _mlx_lm_pkg.stream_generate = stream_probe
    try:
        import mlx_omarchy_bonsai2.server as _bonsai_server
        _bonsai_server.stream_generate = stream_probe
    except Exception as exc:
        print(f"shim: bonsai2 server rebind failed: {exc}",
              file=sys.stderr, flush=True)
    print("shim: ids probe installed (bonsai2)", file=sys.stderr, flush=True)
    return state
