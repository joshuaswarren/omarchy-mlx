import argparse
import hashlib
import json
import os
import time

import mlx.core as mx

from mlx_lm import stream_generate
from mlx_lm.utils import sharded_load

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-host pipeline generation receipt")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Write a haiku about shared memory.")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    # Optional CPU pinning for a stock-MLX rank whose collectives must stay on
    # the CPU stream while its model runs on CPU. Leave unset for GPU compute.
    if os.environ.get("OD_FORCE_CPU") == "1":
        mx.set_default_device(mx.cpu)
    group = mx.distributed.init(backend="ring")
    rank = group.rank()
    print(f"PIPELINE_START rank={rank} version={mx.__version__} device={mx.default_device()} model={args.model}", flush=True)
    model, tokenizer = sharded_load(args.model, group, None)
    messages = [{"role": "user", "content": args.prompt}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    wall0 = time.perf_counter()
    response = None
    for response in stream_generate(model, tokenizer, prompt, max_tokens=args.max_tokens):
        pass
    wall1 = time.perf_counter()
    record = {
        "rank": rank,
        "device": str(mx.default_device()),
        "text": response.text,
        "text_sha256": hashlib.sha256(response.text.encode()).hexdigest(),
        "prompt_tokens": response.prompt_tokens,
        "prompt_tps": response.prompt_tps,
        "generation_tokens": response.generation_tokens,
        "generation_tps": response.generation_tps,
        "peak_memory_gb": response.peak_memory,
        "wall_s": wall1 - wall0,
    }
    print("PIPELINE_RECORD " + json.dumps(record), flush=True)
    print(f"PIPELINE_PASS rank={rank}", flush=True)
