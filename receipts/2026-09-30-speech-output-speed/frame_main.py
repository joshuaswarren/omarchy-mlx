#!/usr/bin/env python3
"""Same-session baseline + hd128 default-route frame on current main.

Reads current commit and wheel version, runs one decode frame with the same
seed and same noise on the stock kernel M2. Reports dispatches (line count),
ms, and the dominant kernels. The hd128 fused SDPA should engage at
q_len=1, k>=12 on the new wheel; if it doesn't, the bench falls back to
the composition.
"""
import json, os, statistics, subprocess, sys, time
from pathlib import Path
import mlx.core as mx

sys.path.insert(0, "<voice-site>")
HOME, OUT = Path(sys.argv[1]), Path(sys.argv[2])
R = {"uname": subprocess.run("uname -r", shell=True, capture_output=True, text=True).stdout.strip(),
     "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
     "loadavg_start": open("/proc/loadavg").read().strip(),
     "mlx": mx.__version__,
     "trace": os.environ.get("MLX_OMARCHY_TRACE_DISPATCH", "0")}
def mark(n): sys.stderr.write(f"MARK {n}\n"); sys.stderr.flush()

from mlx_audio.tts.utils import load_model
model = load_model(HOME / "voice" / "qwen3-tts-0.6b-customvoice-4bit")
mod = sys.modules[type(model).__module__]
talker = model.talker; cp = talker.code_predictor; emb = talker.get_input_embeddings()
cfg = model.config.talker_config
sup = [i for i in range(cfg.vocab_size - 1024, cfg.vocab_size) if i != cfg.codec_eos_token_id]
V = cp.lm_head[0].weight.shape[0]


def cp_loop(ch, t0, noise):
    caches = cp.make_cache(); toks = [t0]; row = [0]
    def draw(logits, temp):
        u = noise[row[0], : logits.shape[-1]]; row[0] += 1
        return mx.argmax(logits.astype(mx.float32) * (1.0 / temp) - mx.log(-mx.log(u)), axis=-1)
    saved = mod.categorical_sampling; mod.categorical_sampling = draw
    try:
        for k in range(15):
            inp = mx.concatenate([ch, emb(t0)], axis=1) if k == 0 else cp.codec_embedding[k - 1](toks[-1])
            lo, caches, _ = cp(inp, cache=caches, generation_step=k)
            toks.append(model._sample_token(lo, temperature=0.9, top_k=50, top_p=1.0))
    finally:
        mod.categorical_sampling = saved
    return mx.concatenate(toks[1:], axis=1)
cp_compiled = mx.compile(cp_loop)


def run_variant(name, env):
    for k in ("MLX_OMARCHY_SDPA_BF16_FAST", "MLX_OMARCHY_SDPA_DECODE_NATIVE"):
        os.environ.pop(k, None)
    os.environ.update(env)
    ie, trailing, _ = model._prepare_generation_inputs(
        "The local assistant is ready to help.", language="english", speaker="aiden")
    cache = talker.make_cache()
    lo, hid = talker(ie, cache=cache); mx.eval(lo, hid)
    state = {"x": ie[:, -1:, :]}
    def frame():
        lo, hid = talker(state["x"], cache=cache)
        noise = mx.random.uniform(shape=(16, 3072))
        saved = mod.categorical_sampling
        mod.categorical_sampling = lambda l, t: mx.argmax(
            l.astype(mx.float32) / t - mx.log(-mx.log(noise[15, : l.shape[-1]])), axis=-1)
        try:
            t0 = model._sample_token(lo, temperature=0.9, top_k=50, top_p=1.0,
                                     repetition_penalty=1.05, generated_tokens=[5, 9],
                                     suppress_tokens=sup)
        finally:
            mod.categorical_sampling = saved
        codes = cp_compiled(hid[:, -1:, :], t0, noise[:15, :V])
        e = emb(t0)
        for i in range(15):
            e = e + cp.codec_embedding[i](codes[:, i:i + 1])
        state["x"] = trailing[:, :1, :] + e
        return state["x"]
    for _ in range(3): mx.eval(frame())
    mark(name + "_START")
    xs = []
    for _ in range(10):
        s = time.perf_counter(); mx.eval(frame()); xs.append((time.perf_counter() - s) * 1000)
    mark(name + "_END")
    R[f"{name}_frame_ms_p50"] = round(statistics.median(xs), 2)


run_variant("default", {})
OUT.write_text(json.dumps(R, indent=2)); print(json.dumps(R), flush=True)
