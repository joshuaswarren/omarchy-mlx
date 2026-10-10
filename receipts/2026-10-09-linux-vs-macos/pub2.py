"""pub2.py <linux_dir> <macos_dir> <template.md> <out.md>: the public v2 section: both columns (Linux, macOS) and the ratio per cell.
Cell states: value / 'fails to run' (Linux failed, counts 0%) / 'not measured' / 'skipped (does not fit)'."""
import json
import os
import sys

lin, mac, tpl, out = sys.argv[1:5]
FAILS = {"gemma4-26b-a4b": 0, "gemma4-e4b": 0, "gpt-oss-20b": 0}  # replaced below by reading the files
MODELS = [("qwen38-27b", "Qwen3.8-27B", "dense 27B"), ("gemma4-26b-a4b", "gemma-4-26B-A4B", "MoE, 4B active"), ("gemma4-31b", "gemma-4-31B", "dense 31B"),
          ("qwen35-9b", "Qwen3.5-9B", "dense 9B, gated-delta"), ("qwen36-35b-a3b", "Qwen3.6-35B-A3B", "MoE, 3B active"), ("gemma4-e4b", "gemma-4-E4B", "dense 4B"),
          ("gpt-oss-20b", "gpt-oss-20b", "MoE, attention sinks")]
LOG = {"lin": os.path.join(lin, "score-full.log"), "mac": os.path.join(mac, "score.log")}


def mlx(d, k):
    p = os.path.join(d, "out", k + ".json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        j = json.load(f)
    return j["decode_tok_s"]["median"], j["prefill_tok_s"]["median"]


def llama(d, k):
    p = os.path.join(d, "llama-gguf-" + k + ".json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        j = json.load(f)
    r = {}
    for e in j:
        r["pp" if e.get("n_prompt") and not e.get("n_gen") else "tg"] = e["avg_ts"]
    return r["pp"], r["tg"]


def failed(logpath, key):
    if not os.path.exists(logpath):
        return False
    with open(logpath) as f:
        return any(line.startswith(f"SCORE|{key}|") and "|FAILED|" in line for line in f)


def cell(a, b, failed_lin):
    if b is None:
        return "not measured"
    if a is None:
        return "fails to run (0%)" if failed_lin else "not measured"
    return f"{a:.1f} / {b:.1f} = **{100.0 * a / b:.0f}%**"


rows = ["MLX 4-bit through the installer's mlx-lm patches, tokens per second, Linux / macOS:", "",
        "| Model | Decode | Prefill 512 |", "|---|---|---|"]
for k, name, note in MODELS:
    a, b = mlx(lin, k), mlx(mac, k)
    f = failed(LOG["lin"], k)
    rows.append(f"| {name} ({note}) | {cell(a and a[0], b and b[0], f)} | {cell(a and a[1], b and b[1], f)} |")
rows += ["", "llama.cpp Q4_K_M, Vulkan on Linux against Metal on macOS (same llama.cpp commit), Linux / macOS:", "",
         "| Model | Decode tg128 | Prefill pp512 |", "|---|---|---|"]
for k, name, note in MODELS:
    if k == "gpt-oss-20b":
        continue
    a, b = llama(lin, k), llama(mac, k)
    rows.append(f"| {name} | {cell(a and a[1], b and b[1], False)} | {cell(a and a[0], b and b[0], False)} |")
rows += ["", "Skipped, too large for this machine: GLM-5.3-Flash, Kimi-K3, DeepSeek-V4-Flash, Qwen3.8-Flash-Next."]
with open(tpl) as f:
    t = f.read()
with open(out, "w") as f:
    f.write(t.replace("@@TABLE@@", "\n".join(rows)))
print("\n".join(rows))
