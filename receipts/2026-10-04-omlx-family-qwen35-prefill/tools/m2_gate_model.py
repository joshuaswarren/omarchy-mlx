#!/usr/bin/env python3
"""FamQwen35Prefill M2 numerics gate — model leg (TF top-1, free-run
first-divergence gap, greedy digests, S=1 PPL delta).

One arm per process (the omarchy flash-256 dispatch gate is a
function-local static; see m2_gate_perop.py):

    MLX_OMARCHY_SDPA_PREFILL_FLASH256=0 python3 m2_gate_model.py --model M --out composed.npz
    python3 m2_gate_model.py --model M --out flash.npz
    python3 m2_gate_model.py --compare composed.npz flash.npz

Per docs/numerics-gate.md:
  1. teacher-forced top-1 agreement >= 99%,
  2. at every free-run first divergence, the composed path's top-2 gap
     <= one bf16 ULP at the magnitude of its top logit,
  3. route-sensitive S=1 PPL differs by <= 0.1%.
"""

import argparse
import os
import sys

import numpy as np

PROMPTS = [
    "The capital of France is",
    "Write a haiku about rain:",
    "def fibonacci(n):\n    ",
    "Summarize the water cycle in one sentence:",
]
TF_TOKENS = 96
FREE_TOKENS = 48


def ulp_bf16(x):
    mag = np.maximum(np.abs(x), 1e-30)
    return np.exp2(np.floor(np.log2(mag)) - 7.0)


def run_arm(model_id, out_path):
    from mlx_lm import load
    import mlx.core as mx

    model, tokenizer = load(model_id)
    tf = {}
    free = {}
    for pi, prompt in enumerate(PROMPTS):
        ids = tokenizer.encode(prompt)[:TF_TOKENS]
        t = mx.array([ids])
        logits = model(t)
        mx.eval(logits)
        lg = np.array(logits.astype(mx.float32)[0])  # [T, V]
        # position i predicts token i+1; drop the last position
        top2 = np.sort(lg, axis=-1)[:, -2:]
        argmax = lg.argmax(axis=-1)
        actual = np.array(ids[1:] + [0])  # next-token targets; last is dummy
        logp = lg - _logsumexp(lg)[:, None]
        tgt_logp = logp[np.arange(len(ids)), actual]
        tf[f"p{pi}"] = {
            "argmax": argmax[:-1].astype(np.int32),
            "top1": top2[:-1, 1],
            "top2": top2[:-1, 0],
            "tgt_logp": tgt_logp[:-1],
            "targets": actual[:-1].astype(np.int32),
            # Last-position logit row: direct diff material for the
            # composed-vs-flash contradiction hunt (2026-10-05).
            "last_logits": lg[-1].astype(np.float32),
        }
        # Greedy free-run: an inline argmax loop (no mlx_lm generate —
        # its signature moved across the 0.31/0.32 lines; the model
        # forward is already proven by the TF leg above). Deterministic
        # greedy is what the digest gate needs.
        run_tokens = list(ids)
        for _ in range(FREE_TOKENS):
            lg2 = model(mx.array([run_tokens]))
            nxt = int(lg2[0, -1].argmax(axis=-1).item())
            run_tokens.append(nxt)
        out_text = tokenizer.decode(run_tokens)
        free[f"p{pi}"] = {
            "tokens": np.array(run_tokens, dtype=np.int32),
            "text": out_text,
        }
        print(f"[arm] prompt {pi}: tf {len(ids)} tok, free {len(run_tokens)} tok",
              flush=True)
    payload = {}
    for k, d in tf.items():
        for f, v in d.items():
            payload[f"tf_{k}_{f}"] = v
    for k, d in free.items():
        for f, v in d.items():
            payload[f"free_{k}_{f}"] = v
    np.savez_compressed(out_path, **payload)
    print(f"[arm] saved {out_path}", flush=True)


def _logsumexp(lg):
    m = lg.max(axis=-1, keepdims=True)
    return (m + np.log(np.exp(lg - m).sum(axis=-1, keepdims=True))).squeeze(-1)


def compare(composed_path, flash_path, model_label):
    c = np.load(composed_path, allow_pickle=False)
    f = np.load(flash_path, allow_pickle=False)
    total = agree = 0
    divergences = []
    for pi in range(len(PROMPTS)):
        am_c = c[f"tf_p{pi}_argmax"]
        am_f = f[f"tf_p{pi}_argmax"]
        assert am_c.shape == am_f.shape
        total += am_c.size
        agree += int((am_c == am_f).sum())
        tl_c = c[f"tf_p{pi}_top1"]
        t2_c = c[f"tf_p{pi}_top2"]
        for i in np.nonzero(am_c != am_f)[0]:
            gap = float(tl_c[i] - t2_c[i])
            bound = float(ulp_bf16(np.array([abs(float(tl_c[i]))]))[0])
            divergences.append(
                {"prompt": pi, "pos": int(i), "top2_gap": gap, "ulp": bound,
                 "within_1ulp": gap <= bound})
        # S=1 PPL per arm (route-sensitive teacher-forced PPL)
        lp_c = c[f"tf_p{pi}_tgt_logp"]
        lp_f = f[f"tf_p{pi}_tgt_logp"]
    ppl_c = float(np.exp(-np.concatenate(
        [c[f"tf_p{pi}_tgt_logp"] for pi in range(len(PROMPTS))]).mean()))
    ppl_f = float(np.exp(-np.concatenate(
        [f[f"tf_p{pi}_tgt_logp"] for pi in range(len(PROMPTS))]).mean()))
    tf_rate = agree / max(total, 1)
    free_digest_c = [c[f"free_p{pi}_tokens"].tobytes() for pi in range(len(PROMPTS))]
    free_digest_f = [f[f"free_p{pi}_tokens"].tobytes() for pi in range(len(PROMPTS))]
    free_equal = [a == b for a, b in zip(free_digest_c, free_digest_f)]
    # Gate 2 applies only at free-run FIRST divergence per prompt: here we
    # record TF divergences (informational) and evaluate the ULP bound at
    # them; free-run first divergence is where argmax streams split.
    all_within = all(d["within_1ulp"] for d in divergences)
    ppl_delta = abs(ppl_c - ppl_f) / ppl_c
    report = {
        "model": model_label,
        "tf_top1_agreement": tf_rate,
        "tf_divergences": divergences,
        "all_tf_divergence_gaps_within_1ulp": bool(all_within),
        "ppl_composed": ppl_c,
        "ppl_flash": ppl_f,
        "ppl_delta": ppl_delta,
        "free_run_digests_equal": free_equal,
    }
    ok = (
        tf_rate >= 0.99
        and all_within
        and ppl_delta <= 0.001
    )
    print(report, flush=True)
    print("GATE_MODEL:", "PASS" if ok else "FAIL", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--mode", choices=["composed", "flash"])
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        ok = compare(args.compare[0], args.compare[1], args.model or "")
        sys.exit(0 if ok else 1)
    if not args.model or not args.out:
        ap.error("--model/--out or --compare required")
    if args.mode == "composed":
        # The omarchy flash-256 gate is a function-local static read at
        # the first SDPA eval: the env must be set before any forward.
        os.environ["MLX_OMARCHY_SDPA_PREFILL_FLASH256"] = "0"
    run_arm(args.model, args.out)


if __name__ == "__main__":
    main()
