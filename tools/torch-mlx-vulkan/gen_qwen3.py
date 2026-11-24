import sys, time, json, logging, os
import torch
torch.backends.mkldnn.enabled = False  # aarch64 mkldnn bf16 GEMM is broken
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, StaticCache

logdir = "/tmp"
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s",
                    filename=os.environ.get("MV_LOG", f"{logdir}/mlx_vulkan_lowering.log"),
                    filemode="w")
import mlx_vulkan  # registers backend mlx_vulkan
from greedy_match import step_row, guided_token, match_report, TIE_TOL

SNAP = os.environ.get("SNAP", "Qwen/Qwen3-0.6B")
DT = {"bf16": torch.bfloat16, "fp32": torch.float32}[os.environ.get("MDT", "fp32")]
PROMPT = "The capital of France is"
N_TOK = int(os.environ.get("NTOK", "32"))

tok = AutoTokenizer.from_pretrained(SNAP)
model = AutoModelForCausalLM.from_pretrained(SNAP, attn_implementation="sdpa", dtype=DT).eval()
print("attn:", model.config._attn_implementation, "dtype:", model.dtype)

ids = tok(PROMPT, return_tensors="pt").input_ids
PLEN = int(os.environ.get("PLEN", "0"))
SMAX = max(64, PLEN + N_TOK + 8)
if PLEN:
    ids = ids.repeat(1, -(-PLEN // ids.shape[1]))[:, :PLEN]
CPREF = os.environ.get("CPREF", "") == "1"
WARM = os.environ.get("WARM", "") == "1"

def greedy(step_fn, n, mk_cache, guide=None, cpref=False):
    cache = mk_cache()
    out, marks, rows = [], [], []

    def pick(logits):
        row = step_row(logits.reshape(-1, logits.shape[-1])[-1])
        rows.append(row)
        out.append(row["tok"])  # the device's own argmax, always recorded
        feed = row["tok"]
        if guide is not None:
            # guide is indexed by the step just appended
            feed, mark = guided_token(row, guide[len(out) - 1])
            marks.append(mark)
        return torch.tensor([[feed]])

    pos = torch.arange(ids.shape[1]).unsqueeze(0)
    extra_t = {}
    with torch.no_grad():
        if cpref and CPREF:
            t0 = time.perf_counter()
            logits = step_fn(ids, pos, cache)
            extra_t["prefill_cold_s"] = time.perf_counter() - t0
            if WARM:
                cache = mk_cache()
                t0 = time.perf_counter()
                logits = step_fn(ids, pos, cache)
                extra_t["prefill_warm_s"] = time.perf_counter() - t0
            t_prefill = extra_t.get("prefill_warm_s", extra_t["prefill_cold_s"])
        else:
            t0 = time.perf_counter()
            # eager prefill populates StaticCache buffers before the compiled loop
            logits = model(input_ids=ids, position_ids=pos, past_key_values=cache,
                           use_cache=True).logits
            t_prefill = time.perf_counter() - t0
        # first decode step is the warmup: it pays dynamo trace + mx.compile,
        # so it is a real generated token but outside the timed window
        nid = pick(logits)
        pos = pos[:, -1:] + 1
        logits = step_fn(nid, pos, cache)
        t0 = time.perf_counter()
        for i in range(n - 2):
            nid = pick(logits)
            pos = pos[:, -1:] + 1
            logits = step_fn(nid, pos, cache)
        pick(logits)
    dt = time.perf_counter() - t0
    return out, t_prefill, dt / (n - 2), marks, rows, extra_t

def eager_step(ids_, pos, cache):
    o = model(input_ids=ids_, position_ids=pos, past_key_values=cache, use_cache=True)
    return o.logits

import os, sys
PHASE = os.environ.get("PHASE", "both")
REFJ = os.environ.get("REFJ", "")
ref_rows = None
if PHASE != "mlx":
    ref_ids, tp_e, td_e, _, ref_rows, _xt = greedy(eager_step, N_TOK, DynamicCache)
    print("CPU:", json.dumps({"prefill_s": round(tp_e, 3), "decode_s_per_tok": round(td_e, 4),
                              "decode_tps": round(1 / td_e, 2)}))
    print("REF ids:", ref_ids)
    print("REF text:", tok.decode(ref_ids))
    if REFJ:
        json.dump({"ids": ref_ids, "rows": ref_rows, "prefill_s": tp_e, "td": td_e},
                  open(REFJ, "w"))
if PHASE == "cpu":
    if not REFJ:
        json.dump({"ids": ref_ids, "rows": ref_rows, "prefill_s": tp_e, "td": td_e},
                  open("/tmp/ref_ids.json", "w"))
    sys.exit(0)
if PHASE == "mlx":
    _refj = REFJ or "/tmp/ref_ids.json"
    if os.path.exists(_refj):
        saved = json.load(open(_refj))
        ref_ids = saved["ids"]
        ref_rows = saved.get("rows")  # old snapshots lack rows; then MATCH is exact-equality
    else:
        ref_ids = ref_rows = None

@torch.compile(backend="mlx_vulkan", dynamic=False)
def comp_step(ids_, pos, cache):
    o = model(input_ids=ids_, position_ids=pos, past_key_values=cache, use_cache=True)
    return o.logits

comp_ids, tp_c, td_c, marks, _rows, xt = greedy(comp_step, N_TOK,
                                        lambda: StaticCache(config=model.config, max_cache_len=SMAX),
                                        guide=ref_rows, cpref=True)
print("MLX:", json.dumps(dict({"prefill_s": round(tp_c, 3), "decode_s_per_tok": round(td_c, 4),
                        "decode_tps": round(1 / td_c, 2)},
                        **{k: round(v, 3) for k, v in xt.items()})))
print("MLX ids:", comp_ids)
print("MLX text:", tok.decode(comp_ids))
if ref_ids is None:
    _outj = _refj + ".mlx"
    json.dump({"ids": comp_ids, "prefill_s": tp_c, "td": td_c}, open(_outj, "w"))
    print("NO CPU REFERENCE: saved own ids to", _outj, "(not a comparison)")
elif ref_rows is None:
    print("MATCH:", ref_ids == comp_ids,
          "| first diff at:", next((i for i, (a, b) in enumerate(zip(ref_ids, comp_ids)) if a != b), -1))
else:
    rep = match_report(marks)
    print("MATCH:", rep["match"], "| tie_tol:", TIE_TOL,
          "| ties tolerated at:", rep["ties"], "| first hard diff at:", rep["first_diff"])
