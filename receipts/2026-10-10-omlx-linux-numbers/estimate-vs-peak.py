#!/usr/bin/env python3
"""Measure how far oMLX's offload admission estimate is from the memory the offloaded model really uses.

One fraction per process (fresh allocator). Prints one json line:
  estimate_bytes   estimate_offload_admission_bytes(model dir, full size, fraction)
  after_load       mx active memory right after apply_moe_expert_offload + materialize_offload_state
  peak_prefill     peak memory while prefilling PROMPT tokens in chunks of CHUNK (fresh peak counter)
  peak_decode      peak memory after DECODE more tokens
Usage: estimate-vs-peak.py FRACTION [PROMPT_TOKENS=128] [CHUNK=64] [DECODE=8]. Env: OMLX_TREE (oMLX tree), SPLIT_MODEL (repo id).
"""
import json
import os
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, os.environ["OMLX_TREE"])
from huggingface_hub import snapshot_download
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load
from omlx.patches import moe_expert_offload as mo

MODEL = os.environ.get("SPLIT_MODEL", "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit")
FRACTION = float(sys.argv[1])
PROMPT = int(sys.argv[2]) if len(sys.argv) > 2 else 128
CHUNK = int(sys.argv[3]) if len(sys.argv) > 3 else 64
DECODE = int(sys.argv[4]) if len(sys.argv) > 4 else 8
TEXT = "The committee reviewed the quarterly logistics report and noted that regional warehouses reduced delivery times. "


def active():
    return int(mx.get_active_memory()) if hasattr(mx, "get_active_memory") else int(mx.metal.get_active_memory())


def peak():
    return int(mx.get_peak_memory()) if hasattr(mx, "get_peak_memory") else int(mx.metal.get_peak_memory())


def reset_peak():
    (mx.reset_peak_memory if hasattr(mx, "reset_peak_memory") else mx.metal.reset_peak_memory)()


def main():
    path = Path(snapshot_download(MODEL, local_files_only=True))
    full = sum(f.stat().st_size for f in path.glob("*.safetensors"))
    est = int(mo.estimate_offload_admission_bytes(path, full, FRACTION))
    model, tok = load(MODEL, lazy=True)
    n = mo.apply_moe_expert_offload(model, MODEL, FRACTION)
    mo.materialize_offload_state(model)
    mx.eval(model.parameters())
    after_load = active()
    ids = mx.array(tok.encode(TEXT * 40)[:PROMPT])
    cache = make_prompt_cache(model)
    reset_peak()
    last = None
    for i in range(0, ids.shape[0], CHUNK):
        last = model(ids[None, i:i + CHUNK], cache=cache)
        mx.eval(last)
    peak_prefill = peak()
    nxt = mx.argmax(last[0, -1])
    for _ in range(DECODE):
        nxt = mx.argmax(model(nxt.reshape(1, 1), cache=cache)[0, -1])
        mx.eval(nxt)
    print("EVPJSON " + json.dumps({"fraction": FRACTION, "wrapped_layers": n, "full_checkpoint_bytes": full, "estimate_bytes": est,
                                  "after_load": after_load, "peak_prefill": peak_prefill, "peak_decode": peak(),
                                  "prompt_tokens": int(ids.shape[0]), "chunk": CHUNK}))


main()
