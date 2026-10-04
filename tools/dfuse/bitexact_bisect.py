#!/usr/bin/env python3
"""Bit-exact chase, phase 1 (tiled path): state-injection bisect.

Route OFF free-run of the earliest-diverging prompt (prompt 1, fd 8):
capture EVERY GDU call's operands + composed out/state. Then route ON with
the composed state INJECTED into every call (teacher-forced trajectory) and
compare fused out/state per call. The first call where injected-fused !=
composed on identical operands is the minimal failing case; its operands
are saved for stage-level analysis.

usage: bitexact_bisect.py <model-dir> <out-dir> [steps=12] [prompt_index=1]
"""
import glob
import json
import os
import sys

import mlx.core as mx
import numpy as np

model_dir = sys.argv[1]
out_dir = sys.argv[2]
NSTEP = int(sys.argv[3]) if len(sys.argv) > 3 else 12
PIDX = int(sys.argv[4]) if len(sys.argv) > 4 else 1
os.makedirs(out_dir, exist_ok=True)

prompts_path = "/var/tmp/Jw16NumericsGpu/qwen38-2b-prompts-10.jsonl"
texts = [json.loads(l)["text"] for l in open(prompts_path) if l.strip()]
TEXT = texts[PIDX]

import mlx_lm.models.gated_delta as gd  # noqa: E402

orig_gdu = gd.gated_delta_update
captures = []
inject = {"on": False, "idx": 0}
inject_results = []
PHASE = {"capture": False, "inject": False}


def spy_capture(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
    out, st = orig_gdu(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel)
    mx.eval(out, st)
    captures.append({
        "op_q": q, "op_k": k, "op_v": v, "op_a": a, "op_b": b,
        "op_A_log": A_log, "op_dt": dt_bias, "op_mask": mask,
        "state_in_bits": np.array(state.view(mx.uint32)),
        "out_f32": np.array(out.astype(mx.float32)),
        "out_bits": np.array(out.view(mx.uint16)),
        "state_f32": np.array(st.astype(mx.float32)),
        "state_bits": np.array(st.view(mx.uint32)),
    })
    return out, st


def spy_inject(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
    cap = captures[inject["idx"]]
    state_in = state
    out, st = orig_gdu(q, k, v, a, b, A_log, dt_bias, state_in, mask, use_kernel)
    mx.eval(out, st)
    same_out = (np.array(out.view(mx.uint16)) == cap["out_bits"]).all()
    same_st = (np.array(st.view(mx.uint32)) == cap["state_bits"]).all()
    oc = np.array(out.astype(mx.float32)).astype(np.float64)
    of = cap["out_f32"].astype(np.float64)
    sc = np.array(st.astype(mx.float32)).astype(np.float64)
    sf = cap["state_f32"].astype(np.float64)
    r = {"call": inject["idx"],
         "out_max_abs": float(np.abs(oc - of).max()),
         "state_max_abs": float(np.abs(sc - sf).max()),
         "out_bits_identical": bool(same_out),
         "state_bits_identical": bool(same_st)}
    inject_results.append(r)
    tag = "OK  " if (same_out and same_st) else "DIFF"
    print(f"{tag} call {r['call']}: out {r['out_max_abs']:.3g} state "
          f"{r['state_max_abs']:.3g} bits(out={same_out},state={same_st})", flush=True)
    return out, st


def spy_dispatch(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
    if PHASE["capture"] and q.shape[1] == 1:
        return spy_capture(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel)
    if PHASE["inject"]:
        return spy_inject(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel)
    return orig_gdu(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel)


# Patch BEFORE load(): qwen3_5 binds the helper at its first import.
gd.gated_delta_update = spy_dispatch

from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

model, tokenizer = load(model_dir)
ids = tokenizer.encode(TEXT)

# Phase 1: route OFF free-run, capture composed per-call results.
os.environ["MLX_OMARCHY_GDN_RAW_REPEAT"] = "0"
PHASE["capture"] = True
PHASE["inject"] = False
inject["on"] = False
inject["idx"] = 0
cache = make_prompt_cache(model)
logits = model(mx.array([ids], dtype=mx.int32), cache=cache)
tokens = []
for t in range(NSTEP):
    tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    tokens.append(tok)
    logits = model(mx.array([[tok]], dtype=mx.int32), cache=cache)
mx.eval(logits)
print(f"phase1: {len(captures)} calls captured, tokens={tokens}", flush=True)

# Control: route OFF replay of the first 24 calls must be bit-identical
# (replay determinism); its rows are tagged and dropped afterwards.
for idx in range(min(24, len(captures))):
    inject["idx"] = idx
    cap = captures[idx]
    st_in = mx.array(cap["state_in_bits"]).view(mx.float32)
    spy_inject(cap["op_q"], cap["op_k"], cap["op_v"], cap["op_a"], cap["op_b"],
               cap["op_A_log"], cap["op_dt"], st_in, cap["op_mask"])
control = inject_results[:]
inject_results.clear()
ctrl_ok = all(r["out_bits_identical"] and r["state_bits_identical"] for r in control)
print(f"CONTROL route-off replay identical: {ctrl_ok} ({len(control)} calls)", flush=True)

# Phase 2: route ON (default): replay every captured composed call - its
# exact live operands and input state - through the fused route.
os.environ.pop("MLX_OMARCHY_GDN_RAW_REPEAT", None)
PHASE["capture"] = False
for idx, cap in enumerate(captures):
    inject["idx"] = idx
    st_in = mx.array(cap["state_in_bits"]).view(mx.float32)
    spy_inject(cap["op_q"], cap["op_k"], cap["op_v"], cap["op_a"], cap["op_b"],
               cap["op_A_log"], cap["op_dt"], st_in, cap["op_mask"])

# Phase 3: route ON free-run, same prompt: does the trajectory diverge?
PHASE["inject"] = False
cache3 = make_prompt_cache(model)
logits3 = model(mx.array([ids], dtype=mx.int32), cache=cache3)
tokens3 = []
for t in range(NSTEP):
    tok = int(mx.argmax(logits3[:, -1, :], axis=-1).item())
    tokens3.append(tok)
    logits3 = model(mx.array([[tok]], dtype=mx.int32), cache=cache3)
print(f"phase3 route-on tokens={tokens3} identical={tokens3 == tokens}", flush=True)

mismatches = [r for r in inject_results if not (r["out_bits_identical"] and r["state_bits_identical"])]
summary = {"calls": len(inject_results), "mismatches": len(mismatches),
           "first_mismatch": mismatches[0] if mismatches else None,
           "control_identical": ctrl_ok, "tokens": tokens, "tokens_route_on": tokens3, "prompt_index": PIDX}
print(json.dumps(summary), flush=True)
json.dump({"summary": summary, "per_call": inject_results},
          open(f"{out_dir}/injected-bisect.json", "w"), indent=1)

if mismatches:
    idx = mismatches[0]["call"]
    cap = captures[idx]
    def bits(x):
        return np.array(x.view(mx.uint16)) if x.dtype == mx.bfloat16 else np.array(x)
    np.savez(f"{out_dir}/failing-operands.npz",
             q_bf16bits=bits(cap["op_q"]), k_bf16bits=bits(cap["op_k"]),
             v_bf16bits=bits(cap["op_v"]), a_bf16bits=bits(cap["op_a"]),
             b_bf16bits=bits(cap["op_b"]), A_log=bits(cap["op_A_log"]),
             dt_bf16bits=bits(cap["op_dt"]), state_in_bits=cap["state_in_bits"],
             out_bits=cap["out_bits"], state_bits=cap["state_bits"])
    print(f"saved failing-operands.npz for call {idx} "
          f"(token {idx // 24}, gdn layer {idx % 24})", flush=True)
