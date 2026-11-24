#!/usr/bin/env python3
"""PT-R follow-on: cold vs warm torch.compile first-call latency, mlx_vulkan backend.

One process per arm. Warm arm relies on the dynamo DiskDynamoCache populated by
the cold arm's atexit save_cache_artifacts (MLX_VK_DISK_CACHE=1).
"""
import argparse, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "bin"))
import mlx_vulkan  # noqa: F401  registers backend + MLX_VK_DISK_CACHE opt-in
import torch


def build_model(name):
    if name == "qwen3":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.float32).eval()
        ids = tok("The quick brown fox", return_tensors="pt").input_ids
        def call(model, _):
            return model(ids).logits
        return m, ids, call, "logits"
    if name == "resnet50":
        import torchvision.models as M
        m = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2).eval()
        x = torch.randn(1, 3, 224, 224)
        def call(model, _x):
            return model(x)
        return m, x, call, "logits"
    raise SystemExit(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen3", "resnet50"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t_load0 = time.time()
    model, inp, call, _ = build_model(args.model)
    t_load = time.time() - t_load0

    with torch.no_grad():
        t0 = time.time()
        ref = call(model, inp)
        ref_s = time.time() - t0

    compiled = torch.compile(model, backend="mlx_vulkan")
    with torch.no_grad():
        t0 = time.time()
        out = call(compiled, inp)
        first_call_s = time.time() - t0
        # second call: steady-state sanity (guards hit, no trace either way)
        t0 = time.time()
        out2 = call(compiled, inp)
        second_call_s = time.time() - t0

    from torch._dynamo.utils import counters
    dc = dict(counters.get("dynamo_cache", {}))
    maxdiff = (out - ref).abs().max().item()
    argmax_match = bool((out.argmax(-1) == ref.argmax(-1)).all().item())

    result = {
        "model": args.model,
        "model_load_s": round(t_load, 3),
        "eager_ref_s": round(ref_s, 3),
        "first_call_s": round(first_call_s, 3),
        "second_call_s": round(second_call_s, 3),
        "maxdiff": maxdiff,
        "argmax_match": argmax_match,
        "dynamo_cache": dc,
        "env": {k: os.environ.get(k) for k in ("MLX_VK_DISK_CACHE", "TORCHINDUCTOR_CACHE_DIR")},
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
