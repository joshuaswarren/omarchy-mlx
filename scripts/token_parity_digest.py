#!/usr/bin/env python3
"""Token-parity digest: greedy 64 ids for one (model, ctx) on this machine.

The m1d3 recipe: prompt ids from numpy rng 292, greedy sampler 0.0, 64 tokens,
digest = sha256(str(ids)). Run with the machine's production venv python:

    python3 scripts/token_parity_digest.py <model_dir> <ctx> [out_json]

Prints the digest and writes {"ctx", "model", "ids_sha256", "ids"} to out_json.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main():
    model_dir, context = sys.argv[1], int(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    assert Path(model_dir, "config.json").is_file(), f"no config in {model_dir}"

    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load
    import mlx

    mx.set_default_device(mx.gpu)
    ids = np.random.default_rng(292).integers(0, 100000, size=context).tolist()
    sampler = make_sampler(0.0)
    model, _ = load(model_dir)
    tokens = []
    for (token, _logprobs), _ in zip(
            generate_step(mx.array(np.array(ids, dtype=np.int32)), model,
                          max_tokens=64, sampler=sampler), range(64)):
        tokens.append(int(token))
    assert len(tokens) == 64, len(tokens)
    result = {
        "context": context,
        "model": Path(model_dir).name or model_dir,
        "ids": tokens,
        "ids_sha256": hashlib.sha256(str(tokens).encode()).hexdigest(),
    }
    print(result["ids_sha256"])
    if out:
        Path(out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
