import sys, time, json, logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, StaticCache

logdir = "/tmp"
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s",
                    filename=f"{logdir}/mlx_vulkan_lowering.log", filemode="w")
import mlx_vulkan  # registers backend mlx_vulkan

SNAP = "/home/joshuawarren/scratch/w7Y-mlxvk/qwen3-0.6b"
PROMPT = "The capital of France is"
N_TOK = 32
SMAX = 64

tok = AutoTokenizer.from_pretrained(SNAP)
model = AutoModelForCausalLM.from_pretrained(SNAP, dtype=torch.float32,
                                           attn_implementation="sdpa").eval()
print("attn:", model.config._attn_implementation, "dtype:", model.dtype)

ids = tok(PROMPT, return_tensors="pt").input_ids

def greedy(step_fn, n, mk_cache):
    cache = mk_cache()
    out = []
    pos = torch.arange(ids.shape[1]).unsqueeze(0)
    with torch.no_grad():
        t0 = time.perf_counter()
        # eager prefill populates StaticCache buffers before the compiled loop
        logits = model(input_ids=ids, position_ids=pos, past_key_values=cache,
                       use_cache=True).logits
        t_prefill = time.perf_counter() - t0
        # first decode step is the warmup: it pays dynamo trace + mx.compile,
        # so it is a real generated token but outside the timed window
        nid = logits.reshape(1, -1, logits.shape[-1])[:, -1].argmax(-1)
        out.append(nid.item())
        pos = pos[:, -1:] + 1
        logits = step_fn(nid.unsqueeze(0), pos, cache)
        t0 = time.perf_counter()
        for i in range(n - 2):
            nid = logits.reshape(1, -1, logits.shape[-1])[:, -1].argmax(-1)
            out.append(nid.item())
            pos = pos[:, -1:] + 1
            logits = step_fn(nid.unsqueeze(0), pos, cache)
        out.append(logits.reshape(-1, logits.shape[-1])[-1].argmax(-1).item())
    dt = time.perf_counter() - t0
    return out, t_prefill, dt / (n - 2)

def eager_step(ids_, pos, cache):
    o = model(input_ids=ids_, position_ids=pos, past_key_values=cache, use_cache=True)
    return o.logits

ref_ids, tp_e, td_e = greedy(eager_step, N_TOK, DynamicCache)
print("CPU:", json.dumps({"prefill_s": round(tp_e, 3), "decode_s_per_tok": round(td_e, 4),
                          "decode_tps": round(1 / td_e, 2)}))
print("REF ids:", ref_ids)
print("REF text:", tok.decode(ref_ids))

@torch.compile(backend="mlx_vulkan", dynamic=False)
def comp_step(ids_, pos, cache):
    o = model(input_ids=ids_, position_ids=pos, past_key_values=cache, use_cache=True)
    return o.logits

comp_ids, tp_c, td_c = greedy(comp_step, N_TOK,
                              lambda: StaticCache(config=model.config, max_cache_len=SMAX))
print("MLX:", json.dumps({"prefill_s": round(tp_c, 3), "decode_s_per_tok": round(td_c, 4),
                        "decode_tps": round(1 / td_c, 2)}))
print("MLX ids:", comp_ids)
print("MLX text:", tok.decode(comp_ids))
print("MATCH:", ref_ids == comp_ids,
      "| first diff at:", next((i for i, (a, b) in enumerate(zip(ref_ids, comp_ids)) if a != b), -1))
