#!/usr/bin/env python3
"""PT-Y: stock HF models through torch.compile(backend='mlx_vulkan').

Modes (each writes one JSON line to stdout):
  torch-compile  model on the GPU through the mlx_vulkan backend
  cpu            eager torch on CPU: the reference ids
  mlx-lm         native mlx_lm greedy on the same wheel's mlx

128 greedy tokens (steps), prompt pinned by --prompt-seed.
tok/s = generated/steps over the loop wall; ids digest = sha256[:16].
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time


def ids_digest(ids):
    return hashlib.sha256(bytes(int(i) for i in ids)).hexdigest()[:16]


def load_hf(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    model.eval()
    return tok, model


def hf_greedy(tok, model, prompt_ids, steps, backend=None, device="cpu"):
    import torch
    inputs = torch.tensor([prompt_ids], device=device)
    mask = torch.ones_like(inputs)
    generated = []
    t0 = time.monotonic()
    for _ in range(steps):
        if backend is None:
            out = model(inputs, attention_mask=mask)
        else:
            out = torch.compile(model, backend=backend, fullgraph=False)(
                inputs, attention_mask=mask)
        nxt = int(out[0, -1].argmax())
        generated.append(nxt)
        inputs = torch.cat([inputs, torch.tensor([[nxt]], device=device)], dim=1)
        mask = torch.ones_like(inputs)
    wall = time.monotonic() - t0
    return generated, wall


def mlx_lm_greedy(tok, model, prompt_ids, steps):
    from mlx_lm.generate import generate_step
    import mlx.core as mx
    tokens = []
    t0 = time.monotonic()
    cache = None
    try:
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(model)
    except Exception:
        cache = None
    inputs = mx.array([prompt_ids])
    for token, _w in generate_step(inputs, model, prompt_cache=cache,
                                   max_tokens=steps):
        tokens.append(int(token))
    wall = time.monotonic() - t0
    return tokens, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("torch-compile", "cpu", "mlx-lm"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--prompt-seed", type=int, default=292)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.mode == "mlx-lm":
        from mlx_lm.utils import load
        tok, model = load(args.model)
        model.eval()
    else:
        tok, model = load_hf(args.model)
        model.eval()

    import torch
    g = torch.Generator().manual_seed(args.prompt_seed)
    vocab = len(tok)
    prompt_ids = torch.randint(0, vocab, (32,), generator=g).tolist()

    if args.mode == "mlx-lm":
        generated, wall = mlx_lm_greedy(tok, model, prompt_ids, args.steps)
        tps = len(generated) / wall if wall else 0
    else:
        backend = "mlx_vulkan" if args.mode == "torch-compile" else None
        device = "cpu"
        generated, wall = hf_greedy(tok, model, prompt_ids, args.steps,
                                    backend=backend, device=device)
        tps = args.steps / wall if wall else 0

    result = {
        "mode": args.mode, "model": args.model,
        "steps": len(generated), "wall_s": round(wall, 3),
        "tok_s": round(tps, 2),
        "ids_sha256_16": ids_digest(generated),
    }
    print(json.dumps(result), flush=True)
    if args.out:
        with open(args.out, "a") as fh:
            fh.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
