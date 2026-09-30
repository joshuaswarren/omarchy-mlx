#!/usr/bin/env python3
"""Fix C: compile the whole 15-pass code-predictor loop.

cp_loop(code_hidden, tok0, noise) is a pure function: fresh KV caches are
created inside it, offsets are trace-time ints, and the sampler reads a
per-frame uniform array (Gumbel-max) passed in, so mx.compile sees no
hidden state. Variants, each 10 frames after warmup:
  upstream  - 15 x (cp forward + model._sample_token), Gumbel draw per call
  loop      - cp_loop uncompiled (same math, noise from the input array)
  compiled  - mx.compile(cp_loop)
Writes MARK lines to stderr so a traced run can be split per variant.
"""
import json, os, statistics, sys, time
from pathlib import Path
import mlx.core as mx

sys.path.insert(0, "<voice-site>")
from mlx_audio.tts.utils import load_model

HOME, OUT = Path(sys.argv[1]), Path(sys.argv[2])
R = {"loadavg": open("/proc/loadavg").read().strip(), "mlx": mx.__version__,
     "trace": os.environ.get("MLX_OMARCHY_TRACE_DISPATCH", "0")}
def mark(n): sys.stderr.write(f"MARK {n}\n"); sys.stderr.flush()

model = load_model(HOME / "voice" / "qwen3-tts-0.6b-customvoice-4bit")
mod = sys.modules[type(model).__module__]
cp = model.talker.code_predictor
emb = model.talker.get_input_embeddings()
ie, _, _ = model._prepare_generation_inputs("The local assistant is ready to help.",
                                            language="english", speaker="aiden")
cache = model.talker.make_cache()
logits, hidden = model.talker(ie, cache=cache)
code_hidden = hidden[:, -1:, :]
tok0 = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True).astype(mx.int32)
mx.eval(code_hidden, tok0)


def gumbel(logits, temp):
    u = mx.random.uniform(shape=logits.shape)
    return mx.argmax(logits.astype(mx.float32) * (1.0 / temp) - mx.log(-mx.log(u)), axis=-1)


def upstream():
    mod.categorical_sampling = gumbel
    caches = cp.make_cache(); toks = [tok0]
    for k in range(15):
        inp = mx.concatenate([code_hidden, emb(tok0)], axis=1) if k == 0 else cp.codec_embedding[k - 1](toks[-1])
        lo, caches, _ = cp(inp, cache=caches, generation_step=k)
        toks.append(model._sample_token(lo, temperature=0.9, top_k=50, top_p=1.0))
    return mx.concatenate(toks[1:], axis=1)


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


compiled = mx.compile(cp_loop)
V = cp.lm_head[0].weight.shape[0]
def noise(): return mx.random.uniform(shape=(15, V))

# Equality: compiled vs uncompiled on identical noise
same = 0; diffs = []
for s in range(10):
    n = noise(); a = cp_loop(code_hidden, tok0, n); b = compiled(code_hidden, tok0, n); mx.eval(a, b)
    eq = int(mx.sum(a == b).item()); same += int(eq == 15); diffs.append(15 - eq)
R["compiled_vs_loop_frames_identical_of_10"] = same
R["compiled_vs_loop_token_mismatches"] = diffs


def timed(name, fn):
    for _ in range(3): mx.eval(fn())
    mark(name + "_START")
    xs = []
    for _ in range(10):
        s = time.perf_counter(); mx.eval(fn()); xs.append((time.perf_counter() - s) * 1000)
    mark(name + "_END")
    R[f"{name}_ms_p50"] = round(statistics.median(xs), 2)

timed("upstream", upstream)
timed("loop", lambda: cp_loop(code_hidden, tok0, noise()))
timed("compiled", lambda: compiled(code_hidden, tok0, noise()))
OUT.write_text(json.dumps(R, indent=2)); print(json.dumps(R), flush=True)
