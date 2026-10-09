"""pub.py <linux_dir> <macos_dir> <section_template.md> <out.md>: public README section (model names only) from the sealed score outputs."""
import json
import os
import sys

lin, mac, tpl, out = sys.argv[1:5]
MLX = [("qwen35-9b", "Qwen3.5-9B"), ("clef-flash", "clef-flash"), ("gemma-4-e2b", "gemma-4-e2b"),
       ("gpt-oss-20b", "gpt-oss-20b"), ("qwen38-27b", "Qwen3.8-27B"), ("qwen38-flash-next", "Qwen3.8-Flash-Next")]
GG = [("gguf-gemma-4-12b", "gemma-4-12b (Q4_K_M)"), ("gguf-qwen38-27b", "Qwen3.8-27B (Q4_K_M)")]


def mlx(d, k):
    p = os.path.join(d, "out", k + ".json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        j = json.load(f)
    return j["decode_tok_s"]["median"], j["prefill_tok_s"]["median"]


def llama(d, k):
    p = os.path.join(d, "llama-" + k + ".json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        j = json.load(f)
    r = {}
    for e in j:
        r["pp" if e.get("n_prompt") and not e.get("n_gen") else "tg"] = e["avg_ts"]
    return r["pp"], r["tg"]


def cell(a, b):
    if b is None:
        return "skipped (does not fit in memory)" if a is None else "n/a"
    if a is None:
        return "fails to run (0%)"
    return "{:.1f} / {:.1f} = **{:.0f}%**".format(a, b, 100.0 * a / b)


rows = ["| Model | Decode, tok/s (Linux / macOS) | Prefill 512, tok/s (Linux / macOS) |", "|---|---|---|"]
for k, name in MLX:
    a, b = mlx(lin, k), mlx(mac, k)
    rows.append("| {} | {} | {} |".format(name, cell(a and a[0], b and b[0]), cell(a and a[1], b and b[1])))
rows += ["", "llama.cpp, Vulkan on Linux against Metal on macOS (same llama.cpp commit):", "",
         "| Model | Decode tg128, tok/s (Linux / macOS) | Prefill pp512, tok/s (Linux / macOS) |", "|---|---|---|"]
for k, name in GG:
    a, b = llama(lin, k), llama(mac, k)
    rows.append("| {} | {} | {} |".format(name, cell(a and a[1], b and b[1]), cell(a and a[0], b and b[0])))
with open(tpl) as f:
    t = f.read()
with open(out, "w") as f:
    f.write(t.replace("@@TABLE@@", "\n".join(rows)))
print("\n".join(rows))
