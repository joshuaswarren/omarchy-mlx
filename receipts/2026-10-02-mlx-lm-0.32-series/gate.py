"""Gate: greedy tokens of mlx-lm 94cdcae with and without the omarchy-mlx 0.32 series, plus prefill/decode speed.
Run once per configuration in a fresh process: python3 cg-series-gate.py MODEL LABEL."""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/.local/share/coreglass/omarchy-mlx-src/scripts"))
import mlx_provenance  # noqa: E402
import mlx_lm  # noqa: E402
from mlx_lm import load, stream_generate  # noqa: E402

model_dir, label = sys.argv[1], sys.argv[2]
prov = mlx_provenance.installed_provenance()
model, tok = load(model_dir)
text = "Apple Silicon runs local language models on Linux through Vulkan and a reverse engineered GPU driver. "
for _ in stream_generate(model, tok, tok.encode("Warm up the GPU."), max_tokens=8):
    pass
for reps in (28, 114):
    ids = tok.encode(text * reps)
    toks, last, t = [], None, time.time()
    for last in stream_generate(model, tok, ids, max_tokens=256):
        toks.append(last.token)
    print(json.dumps({
        "label": label, "mlx_lm": os.path.dirname(mlx_lm.__file__), "prompt_tokens": len(ids),
        "hash64": hashlib.sha256(str(toks[:64]).encode()).hexdigest()[:12],
        "hash128": hashlib.sha256(str(toks[:128]).encode()).hexdigest()[:12],
        "hash256": hashlib.sha256(str(toks[:256]).encode()).hexdigest()[:12],
        "prefill_tok_s": round(last.prompt_tps, 1), "decode_tok_s": round(last.generation_tps, 1),
        "mlx_version": prov.get("version"), "mlx_verified": prov.get("verified"),
        "env_qknorm": os.environ.get("MLX_OMARCHY_GDN_QKNORM_FUSE", "default")}), flush=True)
