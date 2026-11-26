#!/usr/bin/env python3
"""BF16 prefill component attribution at all three legs (30/262/1053).

Extension of receipts/2026-09-11-bf16-prefill-attention/attribution_components.py
adds m=30 to every component loop and the host launch floor context.
Wall-clock medians around mx.eval + mx.synchronize only.
"""
import argparse
import json
import os
import statistics
import time


def timed(fn, reps=9, discard=2):
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(samples[discard:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    os.environ.pop("MLX_OMARCHY_SDPA_BF16_FAST", None)
    os.environ.setdefault("MLX_DISABLE_COMPILE", "1")

    from mlx_lm.utils import load
    import mlx.core as mx

    def nn_silu(x):
        return x / (1.0 + mx.exp(-x))

    model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    model.eval()
    out = open(args.out, "w")

    def emit(row):
        row["wheel"] = args.tag
        out.write(json.dumps(row) + "\n")
        out.flush()

    prompts = {
        "short": "Hi",
        "long": ("You are reviewing a measurement protocol for a small "
                 "language-model benchmark. The benchmark runs the same "
                 "workload matrix on two operating systems and compares "
                 "prefill and decode throughput. Each leg loads one pinned "
                 "model snapshot, applies the model chat template to the "
                 "prompt, runs greedy decoding with a fixed random seed, "
                 "and records the prompt phase, the per-token phase, the "
                 "generated token count, a hash of the generated token "
                 "ids, and peak memory. A leg is valid only when the model "
                 "snapshot revision matches the manifest pin, the power "
                 "state is recorded, and no other model-serving process is "
                 "holding the accelerator."),
    }
    short_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompts["short"]}],
        add_generation_prompt=True)
    long_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompts["long"]}],
        add_generation_prompt=True)
    filler = (long_ids * 8)[:1053]
    legs = {"short": short_ids, "long": long_ids, "ctx1024": filler}

    # ---- whole-model prefill, shipped composition ----
    evals = {}
    for name, ids in legs.items():
        x = mx.array(ids)[None]
        logits = model(x)
        mx.eval(logits)
        mx.synchronize()
        evals[name] = x
        us = timed(lambda: (mx.eval(model(x)), mx.synchronize()),
                   reps=args.reps)
        emit({"k": "whole_prefill", "leg": name,
              "tokens": int(x.shape[1]), "us_median": round(us, 1),
              "tok_s": round(x.shape[1] / (us / 1e6), 1)})

    h = 896
    for m in (30, 262, 1053):
        q = mx.random.normal((1, 14, m, 64)).astype(mx.bfloat16)
        kk = mx.random.normal((1, 14, m, 64)).astype(mx.bfloat16)
        v = mx.random.normal((1, 14, m, 64)).astype(mx.bfloat16)
        scale = 0.125

        # ---- f32 composition components (the shipped path) ----
        q32 = q.astype(mx.float32)
        k32 = kk.astype(mx.float32)
        v32 = v.astype(mx.float32)
        mx.eval(q32, k32, v32)
        mx.synchronize()
        us = timed(lambda: (mx.eval(q.astype(mx.float32)), mx.synchronize()),
                   reps=args.reps)
        emit({"k": "cast_q_f32", "m": m, "us_median": round(us, 1)})
        us = timed(
            lambda: (mx.eval(kk.astype(mx.float32),
                             v.astype(mx.float32)), mx.synchronize()),
            reps=args.reps)
        emit({"k": "cast_kv_f32", "m": m, "us_median": round(us, 1)})
        us = timed(lambda: (mx.eval(q32 * scale), mx.synchronize()),
                   reps=args.reps)
        emit({"k": "scale_mul_f32", "m": m, "us_median": round(us, 1)})
        qs = q32 * scale
        scores = mx.matmul(qs, k32.swapaxes(-1, -2))
        probs = mx.softmax(scores, axis=-1)
        pv = mx.matmul(probs, v32)
        cast = pv.astype(mx.bfloat16)
        mx.eval(qs, scores, probs, pv, cast)
        mx.synchronize()
        us = timed(lambda: (mx.eval(
            mx.matmul(qs, k32.swapaxes(-1, -2))), mx.synchronize()),
            reps=args.reps)
        emit({"k": "scores_f32", "m": m, "us_median": round(us, 1),
              "tflops": round(2 * 14 * m * m * 64 / (us * 1e6), 3)})
        us = timed(lambda: (mx.eval(mx.softmax(scores, axis=-1)),
                            mx.synchronize()), reps=args.reps)
        emit({"k": "softmax_f32", "m": m, "us_median": round(us, 1)})
        us = timed(lambda: (mx.eval(mx.matmul(probs, v32)), mx.synchronize()),
                   reps=args.reps)
        emit({"k": "pv_f32", "m": m, "us_median": round(us, 1),
              "tflops": round(2 * 14 * m * m * 64 / (us * 1e6), 3)})
        us = timed(lambda: (mx.eval(pv.astype(mx.bfloat16)),
                            mx.synchronize()), reps=args.reps)
        emit({"k": "cast_out_bf16", "m": m, "us_median": round(us, 1)})

        # ---- per-layer projection + mlp rates, lm_head, norms, rope ----
        chain_us = 0.0
        for (k, n, tag) in ((896, 896, "q_o"), (896, 128, "kv"),
                            (896, 4864, "gate_up"), (4864, 896, "down")):
            xb = mx.random.normal((m, k)).astype(mx.bfloat16)
            w = mx.random.normal((n, k)).astype(mx.bfloat16)
            mx.eval(xb, w)
            mx.synchronize()
            us = timed(
                lambda: (mx.eval(xb @ w.T), mx.synchronize()),
                reps=args.reps)
            chain_us += us
            emit({"k": "matmul_bf16", "m": m, "kn": f"{k}x{n}", "tag": tag,
                  "us_median": round(us, 1),
                  "tflops": round(2 * m * k * n / (us * 1e6), 3)})
        emit({"k": "proj_chain_sum", "m": m, "us_median": round(chain_us, 1)})
        lw = mx.random.normal((151936, 896)).astype(mx.bfloat16)
        lt = mx.random.normal((m, 896)).astype(mx.bfloat16)
        mx.eval(lw, lt)
        mx.synchronize()
        us = timed(lambda: (mx.eval(lt @ lw.T), mx.synchronize()),
                   reps=args.reps)
        emit({"k": "lm_head_bf16", "m": m, "us_median": round(us, 1),
              "tflops": round(2 * m * 896 * 151936 / (us * 1e6), 3)})
        t = mx.random.normal((1, m, h)).astype(mx.bfloat16)
        g = mx.random.normal((m, 4864)).astype(mx.bfloat16)
        r = mx.random.normal((m, 896)).astype(mx.bfloat16)
        mx.eval(t, g, r)
        mx.synchronize()
        us = timed(lambda: (mx.eval(mx.fast.rope(
            t, 64, traditional=False, base=10000.0, scale=1.0,
            offset=0)), mx.synchronize()), reps=args.reps)
        emit({"k": "rope", "m": m, "us_median": round(us, 1)})
        us = timed(lambda: (mx.eval(mx.fast.rms_norm(
            t, mx.ones((h,)).astype(mx.bfloat16), 1e-6)),
            mx.synchronize()), reps=args.reps)
        emit({"k": "rms_norm", "m": m, "us_median": round(us, 1)})
        us = timed(lambda: (mx.eval(nn_silu(g)), mx.synchronize()),
                   reps=args.reps)
        emit({"k": "silu", "m": m, "us_median": round(us, 1)})
        us = timed(lambda: (mx.eval(r + r), mx.synchronize()),
                   reps=args.reps)
        emit({"k": "residual_add", "m": m, "us_median": round(us, 1)})

    a = mx.array([1.0])
    mx.eval(a)
    mx.synchronize()
    us = timed(lambda: (mx.eval(a + a), mx.synchronize()), reps=51,
               discard=10)
    emit({"k": "host_launch_floor", "us_median": round(us, 1)})
    out.close()


if __name__ == "__main__":
    main()
