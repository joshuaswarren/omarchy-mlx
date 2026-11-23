import argparse, sys, time, json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen3_mlx import Qwen3Fast
import mlx.core as mx

_p = argparse.ArgumentParser(description="Qwen3 greedy decode: CPU torch vs fused omarchy-mlx")
_p.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF id or local snapshot dir")
_p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
_p.add_argument("--smax", type=int, default=160)
_a = _p.parse_args()
SNAP = _a.model
PROMPT = "The capital of France is"
N_TOK = 32

tok = AutoTokenizer.from_pretrained(SNAP)
model = AutoModelForCausalLM.from_pretrained(SNAP, dtype=torch.float32,
                                           attn_implementation="eager").eval()
ids_t = tok(PROMPT, return_tensors="pt").input_ids
print("prompt ids:", ids_t.tolist(), "S =", ids_t.shape[1])

# ---- CPU fp32 reference (identical to M2 harness) ----
def eager_step(ids_, pos, cache):
    with torch.no_grad():
        o = model(input_ids=ids_, position_ids=pos, past_key_values=cache, use_cache=True)
    return o.logits

cache = DynamicCache()
ref = []
pos = torch.arange(ids_t.shape[1]).unsqueeze(0)
t0 = time.perf_counter()
logits = eager_step(ids_t, pos, cache)
tp_cpu = time.perf_counter() - t0
steps_cpu = []
t0 = time.perf_counter()
for i in range(N_TOK - 1):
    s = time.perf_counter()
    nid = logits[:, -1].argmax(-1)
    ref.append(nid.item())
    pos = pos[:, -1:] + 1
    logits = eager_step(nid.unsqueeze(0), pos, cache)
    steps_cpu.append(time.perf_counter() - s)
ref.append(logits[:, -1].argmax(-1).item())
print("CPU:", json.dumps({"prefill_s": round(tp_cpu, 3),
                          "decode_tps": round((N_TOK - 1) / sum(steps_cpu), 2)}))
print("CPU ids:", ref)
print("CPU step tok/s:", [round(1 / s, 2) for s in steps_cpu])

# ---- fused MLX path, bf16 ----
_dt = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}[_a.dtype]
qm = Qwen3Fast(SNAP, smax=_a.smax, dtype=_dt)
print("dtype:", _dt)
print("Qwen3Fast loaded")
out = []
ids_np = ids_t.numpy()
t0 = time.perf_counter()
logits, K, V = qm.prefill(ids_np)
tp_mx = time.perf_counter() - t0
steps_mx = []
pos_n = ids_np.shape[1]
t0 = time.perf_counter()
for i in range(N_TOK - 1):
    s = time.perf_counter()
    nid = int(mx.argmax(logits[0]).item())
    out.append(nid)
    logits, K, V = qm.decode(nid, pos_n + i, K, V)
    steps_mx.append(time.perf_counter() - s)
out.append(int(mx.argmax(logits[0]).item()))
print("MLX:", json.dumps({"prefill_s": round(tp_mx, 3),
                          "decode_tps": round((N_TOK - 1) / sum(steps_mx), 2)}))
print("MLX ids:", out)
print("MLX text:", tok.decode(out))
print("MLX step tok/s:", [round(1 / s, 2) for s in steps_mx])
print("MATCH:", ref == out,
      "| first diff at:", next((i for i, (a, b) in enumerate(zip(ref, out)) if a != b), -1))
