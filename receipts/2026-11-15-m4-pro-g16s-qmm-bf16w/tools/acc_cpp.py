# Perplexity and logit agreement, C++ two-pass route (default on G16S) vs the single-kernel f32-W route (MLX_OMARCHY_QMM_BF16W=0).
# Same process, env toggled per forward (the flag is read live). Qwen3-8B-4bit, wikitext-2 test, 8 x 2048-token windows.
import os, sys, math, subprocess, time
import mlx.core as mx
from mlx_lm import load
D = os.path.dirname(os.path.abspath(__file__))
G = os.environ.get("GPU_TEMP_CMD", "true")
def cool(lim=80):
    while int(subprocess.run([G], capture_output=True, text=True).stdout.strip() or 0) >= lim: time.sleep(3)
model, tok = load("mlx-community/Qwen3-8B-4bit")
ids = tok.encode(open(os.environ.get("WIKITEXT", os.path.join(D, "wikitext-2-test.txt"))).read())
print("mlx", mx.__version__, "tokens", len(ids), flush=True)
CTX = 2048
tot = {"f32": 0.0, "bf16w": 0.0}; n = 0; agree = 0; kl_sum = 0.0; maxabs = 0.0
for w in range(8):
    seg = ids[w*CTX:(w+1)*CTX]
    x = mx.array(seg)[None]; tgt = mx.array(seg[1:])
    lps = {}
    for arm in ("f32", "bf16w"):
        cool()
        if arm == "f32": os.environ["MLX_OMARCHY_QMM_BF16W"] = "0"
        else: os.environ.pop("MLX_OMARCHY_QMM_BF16W", None)
        lg = model(x).astype(mx.float32)[0, :-1]
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        mx.eval(lp); lps[arm] = (lg, lp)
        tot[arm] += -mx.sum(mx.take_along_axis(lp, tgt[:, None], axis=-1)).item()
    (la, pa), (lb, pb) = lps["f32"], lps["bf16w"]
    agree += mx.sum(mx.argmax(la, -1) == mx.argmax(lb, -1)).item()
    kl_sum += mx.sum(mx.exp(pa) * (pa - pb)).item()
    maxabs = max(maxabs, mx.max(mx.abs(la - lb)).item())
    n += len(seg) - 1
    print(f"WIN {w} f32_nll={tot['f32']:.2f} bf16w_nll={tot['bf16w']:.2f}", flush=True)
print(f"PPL f32={math.exp(tot['f32']/n):.4f} bf16w={math.exp(tot['bf16w']/n):.4f} tokens={n}", flush=True)
print(f"LOGITS argmax_agree={agree}/{n} ({agree/n*100:.2f}%) mean_KL(f32||bf16w)={kl_sum/n:.3e} max_abs_logit_diff={maxabs:.3f}", flush=True)
print("ACC_DONE", flush=True)
