#!/usr/bin/env python3
"""Perrow divergence bisect (forced legacy path on jw16):
prompt 1 diverges at token 23. Capture the OFF-run (composed) per-step GDU
operands + cache states for the first ~30 steps; then step the ON-run
(fused perrow) from the same prompt; after each step compare the GDN cache
state to the OFF-run's. At the first mismatch: single-step both routes on
the PREVIOUS captured operands and report per-op diffs + the top-2 logit
gap of the next-token distribution in both arms.
usage: perrow_bisect.py <model-dir> <out-dir> [nsteps=30] [prompt_index=1]
"""
import glob
import json
import os
import sys

import mlx.core as mx
import numpy as np

model_dir = sys.argv[1]
out_dir = sys.argv[2]
NSTEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 30
PIDX = int(sys.argv[4]) if len(sys.argv) > 4 else 1
os.makedirs(out_dir, exist_ok=True)

prompts_path = "/var/tmp/Jw16NumericsGpu/qwen38-2b-prompts-10.jsonl"
texts = [json.loads(l)["text"] for l in open(prompts_path) if l.strip()]
TEXT = texts[PIDX]

captured = []
CAPTURING = {"on": False}


def set_route(on):
    if on:
        os.environ.pop("MLX_OMARCHY_GDN_RAW_REPEAT", None)
    else:
        os.environ["MLX_OMARCHY_GDN_RAW_REPEAT"] = "0"


import importlib
import mlx_lm.models.gated_delta as gd

orig_gdu = gd.gated_delta_update


def spy(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
    if CAPTURING["on"] and len(captured) < NSTEPS + 2 and q.shape[1] == 1:
        captured.append({k2: v2 for k2, v2 in
                         (("q", q), ("k", k), ("v", v), ("a", a), ("b", b),
                          ("A_log", A_log), ("dt", dt_bias), ("state", state),
                          ("mask", mask)) if v2 is not None})
    return orig_gdu(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel)


gd.gated_delta_update = spy

from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

model, tokenizer = load(model_dir)
ids = tokenizer.encode(TEXT)
cache = make_prompt_cache(model)
logits = model(mx.array([ids], dtype=mx.int32), cache=cache)

# Phase 1: route OFF, capture operands + per-step states + token ids.
set_route(False)
CAPTURING["on"] = True
states_off = []
tokens = []
for t in range(NSTEPS):
    tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    tokens.append(tok)
    st = cache[PIDX * 0 + 1][1] if isinstance(cache[1], list) else None
    # GDN layers keep per-layer caches; capture the FIRST GDN layer state
    try:
        states_off.append(np.array(cache[1][0][1]))
    except Exception:
        states_off.append(None)
    logits = model(mx.array([[tok]], dtype=mx.int32), cache=cache)
mx.eval(logits)
CAPTURING["on"] = False
print(f"phase1: captured {len(captured)} operand sets, {len(states_off)} states", flush=True)

# Phase 2: route ON (fused perrow), same prompt; compare states per step.
set_route(True)
cache2 = make_prompt_cache(model)
logits2 = model(mx.array([ids], dtype=mx.int32), cache=cache2)
first_bad = None
for t in range(NSTEPS):
    tok = tokens[t]
    try:
        st = np.array(cache2[1][0][1])
        ref = states_off[t]
        if ref is not None:
            d = float(np.abs(st.astype(np.float64) - ref.astype(np.float64)).max())
            if d > 0 and first_bad is None:
                first_bad = t
                print(f"first state mismatch at step {t}: max_abs={d}", flush=True)
    except Exception:
        pass
    logits2 = model(mx.array([[tok]], dtype=mx.int32), cache=cache2)
mx.eval(logits2)
print(f"phase2 done; first_bad={first_bad}", flush=True)

# Phase 3: token-level: logits top-2 gap at the first diverging token.
# The free-run OFF tokens vs fused tokens for the first NSTEPS steps.
set_route(True)
cache3 = make_prompt_cache(model)
logits3 = model(mx.array([ids], dtype=mx.int32), cache=cache3)
tok_div = None
for t in range(NSTEPS):
    last = logits3[:, -1, :].astype(mx.float32)
    a_f = int(mx.argmax(last, axis=-1).item())
    a_c = tokens[t]
    top = mx.sort(mx.flatten(last))[-2:]
    gap = float(top[1] - top[0])
    if a_f != a_c:
        tok_div = {"step": t, "fused_argmax": a_f, "composed_argmax": a_c,
                   "gap": gap}
        print(json.dumps(tok_div), flush=True)
        break
    logits3 = model(mx.array([[a_c]], dtype=mx.int32), cache=cache3)
res = {"first_state_mismatch_step": first_bad, "token_divergence": tok_div,
       "captured_sets": len(captured), "mx": mx.__version__}
json.dump(res, open(f"{out_dir}/bisect.json", "w"), indent=1)

# Phase 4: single-step both routes on EVERY captured set; report the first
# mismatching step and save its operands.
set_route(True)
first_diff = None
for idx, ops in enumerate(captured):
    set_route(False)
    out_c, st_c = orig_gdu(ops["q"], ops["k"], ops["v"], ops["a"], ops["b"],
                           ops["A_log"], ops["dt"], ops["state"],
                           ops.get("mask"), True)
    mx.eval(out_c, st_c)
    set_route(True)
    Hk, Hv = ops["q"].shape[-2], ops["v"].shape[-2]
    qr = mx.repeat(ops["q"], Hv // Hk, -2)
    kr = mx.repeat(ops["k"], Hv // Hk, -2)
    out_f, st_f = mx.fast.gated_delta_update_raw(qr, kr, ops["v"], ops["a"],
                                                 ops["b"], ops["A_log"],
                                                 ops["dt"], ops["state"],
                                                 ops.get("mask"))
    mx.eval(out_f, st_f)
    same_out = (np.array(out_c.view(mx.uint16)) == np.array(out_f.view(mx.uint16))).all()
    same_st = (np.array(st_c.view(mx.uint32)) == np.array(st_f.view(mx.uint32))).all()
    if not (same_out and same_st):
        first_diff = idx
        np.savez(f"{out_dir}/diverging-operands-{idx}.npz",
                 **{k: np.array(v.astype(mx.float32)) for k, v in ops.items()})
        oc = np.array(out_c.astype(mx.float32))
        of = np.array(out_f.astype(mx.float32))
        sc = np.array(st_c.astype(mx.float32))
        sf = np.array(st_f.astype(mx.float32))
        d_out = float(np.abs(oc.astype(np.float64) - of.astype(np.float64)).max())
        d_st = float(np.abs(sc.astype(np.float64) - sf.astype(np.float64)).max())
        print(f"first single-step mismatch at captured[{idx}]: "
              f"out_max_abs={d_out} state_max_abs={d_st}", flush=True)
        break
res["first_single_step_mismatch"] = first_diff
json.dump(res, open(f"{out_dir}/bisect.json", "w"), indent=1)
print("bisect done, first_diff =", first_diff, flush=True)
