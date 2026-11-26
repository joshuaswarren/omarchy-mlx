"""llama-bench-style K2 throughput on MLX/Honeykrisp (GPU). Not an ANE test.

pp N: prefill throughput for N random prompt tokens (fresh KV cache).
tg 128: decode throughput for 128 greedy tokens after a 1-token prompt
(llama-bench's tg definition). Repetitions report each run and the mean.
Loading is chunked with file-scoped cache advice (8.5 GiB cgroup).
"""
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model_path = Path(sys.argv[1]).resolve()
prompt_sizes = [int(x) for x in sys.argv[2].split(",")]
reps = int(sys.argv[3])
for name, expected in {
    "k2_horizon.py": "80b501f2ea7364842bd25659928d3af7692238e7c6c2a50bad2f35f61f5483bc",
    "config.json": "5e523194f1e6a64b5a3abd9b60406d4512f16fdd218a69ff67017a8d1b98bbca",
}.items():
    assert hashlib.sha256((model_path / name).read_bytes()).hexdigest() == expected, name
assert mx.default_device() == mx.gpu
mx.set_cache_limit(0)

fds = [os.open(p, os.O_RDONLY | os.O_CLOEXEC) for p in sorted(model_path.glob("*.safetensors"))]
try:
    for fd in fds:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    t0 = time.perf_counter()
    model, tokenizer = load(str(model_path), lazy=True)
    weights = tree_flatten(model.parameters())
    for start in range(0, len(weights), 8):
        mx.eval(*[a for _, a in weights[start:start + 8]])
        for fd in fds:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    load_s = time.perf_counter() - t0
finally:
    for fd in fds:
        os.close(fd)

vocab = model.args.vocab_size
# deterministic tokens per length (this MLX has no random.fold_in)

def prefill(tokens, cache, step=512):
    for i in range(0, tokens.size, step):
        logits = model(tokens[None, i:i + step], cache=cache)
        mx.eval(logits, [c.state for c in cache])
    return logits

def pp(n):
    tokens = mx.random.randint(0, vocab, (n,), key=mx.random.key(17 * 1000003 + n))
    mx.eval(tokens)
    cache = make_prompt_cache(model)
    t = time.perf_counter()
    prefill(tokens, cache)
    return n / (time.perf_counter() - t)

def tg(n=128):
    cache = make_prompt_cache(model)
    y = mx.array([1])
    logits = prefill(y, cache)
    y = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(y)
    t = time.perf_counter()
    # Pipelined like mlx_lm.generate_step: queue step k+1 before reading k.
    nxt = mx.argmax(model(y[None], cache=cache)[:, -1, :], axis=-1)
    mx.async_eval(nxt)
    for _ in range(n - 1):
        cur = nxt
        nxt = mx.argmax(model(cur[None], cache=cache)[:, -1, :], axis=-1)
        mx.async_eval(nxt)
        cur.item()
    nxt.item()
    return n / (time.perf_counter() - t)

pp(32)  # warm-up: shader/pipeline creation, not reported
tg(8)
results = {"backend": "mlx-omarchy / Honeykrisp Vulkan (GPU)", "load_seconds": load_s, "tests": []}
for n in prompt_sizes:
    runs = [pp(n) for _ in range(reps)]
    results["tests"].append({"test": f"pp{n}", "tokens_per_second": runs, "mean": statistics.mean(runs)})
    print(json.dumps(results["tests"][-1]), flush=True)
runs = [tg(128) for _ in range(reps)]
results["tests"].append({"test": "tg128", "tokens_per_second": runs, "mean": statistics.mean(runs)})
results["peak_memory_bytes"] = mx.get_peak_memory()
print(json.dumps(results), flush=True)
