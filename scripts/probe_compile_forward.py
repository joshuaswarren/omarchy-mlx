#!/usr/bin/env python3
"""Probe: what happens if the decode forward is wrapped in mx.compile?

mlx_lm 0.31.3 compiles only the swiglu activation, so on this backend a
decode token runs ~96% of its GPU dispatches eager. This script tests
the structural fix in isolation: it re-implements the Qwen2 decode
layer functionally (explicit cache arrays in and out, offset as a
scalar array, full-slot cache with an additive -inf mask) and wraps one
whole decoder layer in mx.compile. It then compares an eager greedy
decode against the compiled-layer greedy decode:

- median inter-token time
- greedy token parity between the two paths
- GPU dispatches per token, per leg (via MLX_OMARCHY_GPU_PROFILE +
  markers consumed by scripts/chain_census.py)

Failure modes this probe reports rather than works around:
- mx.compile refusing any op in the layer (named error propagates)
- shapeless recompilation per step (visible as repeated tracing)
- the KV-cache state mutation contract (handled functionally here BY
  DESIGN: full-slot where-select + masked softmax, no in-place writes)

Usage:
  MLX_OMARCHY_ALLOW_NON_APPLE=1 MLX_OMARCHY_GPU_PROFILE=/tmp/p.jsonl \
    python3 scripts/probe_compile_forward.py --model PATH \
      --markers /tmp/p.markers.jsonl --max-tokens 16
"""

import argparse
import json
import time

import mlx.core as mx


def mark(mf, phase):
    mf.write(json.dumps({"t": time.monotonic_ns(), "p": phase}) + "\n")
    mf.flush()


def load_model(path):
    from mlx_lm.utils import load

    return load(path)


def build_compiled_layer(layer, dtype):
    """Return fn(x, k_cache, v_cache, offset) -> (x', k_cache', v_cache').

    Functional re-expression of TransformerBlock.__call__ for one-token
    decode: no in-place cache writes (full-slot where-select), offset as
    a scalar array, attention masked additively over the whole slot.
    """

    attn = layer.self_attn
    mlp = layer.mlp
    n_heads = attn.n_heads
    n_kv = attn.n_kv_heads
    head_dim = int(round(attn.scale ** -2))
    scale = attn.scale
    rope = attn.rope
    neg = mx.finfo(dtype).min if dtype == mx.float16 else -1e30

    def fn(x, k_cache, v_cache, offset):
        B = x.shape[0]
        h = layer.input_layernorm(x)
        q = attn.q_proj(h)
        k = attn.k_proj(h)
        v = attn.v_proj(h)
        q = q.reshape(B, 1, n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, 1, n_kv, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, 1, n_kv, head_dim).transpose(0, 2, 1, 3)
        q = rope(q, offset=offset)
        k = rope(k, offset=offset)

        # Functional cache update: select the slot row at offset.
        positions = mx.arange(k_cache.shape[2])
        sel = (positions == offset).reshape(1, 1, -1, 1)
        k_full = mx.where(sel, k, k_cache)
        v_full = mx.where(sel, v, v_cache)

        # Additive mask: the query at `offset` attends positions <= offset.
        attn_mask = mx.where(positions <= offset, 0.0, neg).reshape(1, 1, 1, -1)
        attn_mask = attn_mask.astype(x.dtype)

        out = mx.fast.scaled_dot_product_attention(
            q, k_full, v_full, scale=scale, mask=attn_mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, 1, -1)
        r = attn.o_proj(out)
        h2 = x + r
        m = mlp(layer.post_attention_layernorm(h2))
        return h2 + m, k_full, v_full

    return mx.compile(fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--markers")
    args = ap.parse_args()

    mf = open(args.markers, "w") if args.markers else None
    from mlx_lm.models import cache as cache_mod

    model, tokenizer = load_model(args.model)
    mx.random.seed(0)

    tokens = tokenizer.encode(args.prompt)
    y = mx.array(tokens)
    prompt_len = len(tokens)
    n_layers = model.args.num_hidden_layers

    # ---------- EAGER LEG ----------
    caches = [cache_mod.KVCache() for _ in range(n_layers)]
    logits = model(y[None], cache=caches)
    tok = mx.argmax(logits[:, -1, :], axis=-1)
    eager_out = []
    eager_times = []
    if mf:
        mark(mf, "decode_start")
    for _ in range(args.max_tokens):
        if mf:
            mark(mf, "tok")
        t_tok = time.perf_counter()
        eager_out.append(int(tok.item()))
        tok = mx.argmax(
            model(tok[None], cache=caches)[:, -1, :], axis=-1)
        mx.eval(tok)
        eager_times.append(time.perf_counter() - t_tok)
    eager_ms = sorted(t * 1000 for t in eager_times)
    print(f"[eager ] text: {tokenizer.decode(eager_out)!r}")
    print(f"[eager ] median inter-token: {eager_ms[len(eager_ms) // 2]:.2f} ms"
          f" (prompt {prompt_len} tokens, {args.max_tokens} decode steps)")

    # ---------- COMPILED LEG ----------
    dtype = model.model.embed_tokens.weight.dtype
    compiled_layers = [
        build_compiled_layer(layer, dtype) for layer in model.model.layers]

    caches_c = [cache_mod.KVCache() for _ in range(n_layers)]
    logits = model(y[None], cache=caches_c)
    tok = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(tok)

    k_states = [c.keys for c in caches_c]
    v_states = [c.values for c in caches_c]
    offsets = [c.offset for c in caches_c]

    compiled_out = []
    compiled_times = []
    for _ in range(args.max_tokens):
        if mf:
            mark(mf, "tok")
        t_tok = time.perf_counter()
        compiled_out.append(int(tok.item()))
        x = model.model.embed_tokens(tok[None])
        new_k = list(k_states)
        new_v = list(v_states)
        for i, cfn in enumerate(compiled_layers):
            x, new_k[i], new_v[i] = cfn(
                x,
                k_states[i],
                v_states[i],
                mx.array([offsets[i]], mx.int32))
        mx.eval(x, *new_k[:1])
        k_states = new_k
        v_states = new_v
        offsets = [o + 1 for o in offsets]
        logits = model.model.embed_tokens.as_linear(model.model.norm(x))
        tok = mx.argmax(logits[:, -1, :], axis=-1)
        compiled_times.append(time.perf_counter() - t_tok)
    if mf:
        mark(mf, "decode_done")
        mf.close()

    compiled_ms = sorted(t * 1000 for t in compiled_times)
    print(f"[compd ] text: {tokenizer.decode(compiled_out)!r}")
    print(f"[compd ] median inter-token: "
          f"{compiled_ms[len(compiled_ms) // 2]:.2f} ms")

    parity = eager_out == compiled_out
    print(f"[parity] eager == compiled tokens: {parity}")
    if not parity:
        print(f"[parity] eager={eager_out}")
        print(f"[parity] cmpd ={compiled_out}")


if __name__ == "__main__":
    main()
