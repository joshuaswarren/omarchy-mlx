#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("prompts")
    ap.add_argument("teacher_tokens")
    ap.add_argument("output")
    args = ap.parse_args()
    prompts = [json.loads(line)["text"] for line in open(args.prompts, encoding="utf-8") if line.strip()]
    teacher = json.load(open(args.teacher_tokens, encoding="utf-8"))
    if len(prompts) != 10 or len(teacher["prompts"]) != 10:
        raise RuntimeError("expected exactly ten prompts and token sequences")
    if teacher.get("max_tokens") != 512 or len(teacher.get("prompts", [{}])[0].get("tokens", [])) != 512:
        raise RuntimeError("teacher data must contain 512 generated tokens per prompt")
    model, tokenizer = load(args.model)
    results = []
    for prompt_index, text in enumerate(prompts):
        prompt_tokens = tokenizer.encode(text)
        teacher_row = teacher["prompts"][prompt_index]
        if teacher_row["prompt_index"] != prompt_index or teacher_row["prompt"] != text:
            raise RuntimeError("teacher prompts do not match the fixed prompt order")
        tokens = teacher_row["tokens"]
        cache = make_prompt_cache(model)
        logits = model(mx.array([prompt_tokens], dtype=mx.int32), cache=cache)
        predicted = []
        gaps = []
        top_magnitudes = []
        chosen_logprobs = []
        for step, forced_token in enumerate(tokens):
            last = logits[:, -1, :].astype(mx.float32)
            top = mx.topk(last, k=2, axis=-1)
            argmax = mx.argmax(last, axis=-1)
            gap = mx.max(top, axis=-1) - mx.min(top, axis=-1)
            top_magnitude = mx.max(mx.abs(top), axis=-1)
            chosen_logprob = last[0, forced_token] - mx.logsumexp(last, axis=-1)[0]
            mx.eval(argmax, gap, top_magnitude, chosen_logprob)
            predicted.append(int(argmax.item()))
            gaps.append(float(gap.item()))
            top_magnitudes.append(float(top_magnitude.item()))
            chosen_logprobs.append(float(chosen_logprob.item()))
            if step + 1 < len(tokens):
                logits = model(mx.array([[forced_token]], dtype=mx.int32), cache=cache)
        results.append({"prompt_index": prompt_index, "tokens": tokens,
                        "predicted": predicted, "top2_gaps": gaps,
                        "top2_magnitudes": top_magnitudes,
                        "chosen_logprobs": chosen_logprobs})
        print(f"teacher-forced prompt {prompt_index + 1}/{len(prompts)}", flush=True)
    payload = {"model": args.model, "mx_version": mx.__version__,
               "backend": str(mx.default_device()),
               "prompt_sha256": hashlib.sha256(json.dumps(prompts, ensure_ascii=False).encode()).hexdigest(),
               "gdn_raw_repeat": os.environ.get("MLX_OMARCHY_GDN_RAW_REPEAT", "0"),
               "prompts": results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(json.dumps({"output": args.output, "positions": sum(len(row["tokens"]) for row in results)}))


if __name__ == "__main__":
    main()
