#!/usr/bin/env python3
"""t7b: decode and prefill runner for mlx-omarchy (MLX on Vulkan, Omarchy Linux). Output JSON has the shape that
receipts/w9C/site-ingest/ingest_runner_json.py reads from t7a-llamacpp-vulkan.py: {env, commands, method, rawRuns, errors}.

Run it with the interpreter of the packaged install:
  /usr/lib/omarchy-mlx/venv/bin/python t7b-mlx-omarchy.py --model REPO_OR_DIR --host-label "MacBook Pro 14-inch (M2 Max, 96 GB)" \
      --out result.json [--sizes 8192,32768] [--rounds 3] [--gen 128] [--prefill-step 512] [--download-only]

One run = a prompt of N random token ids (seeded, the same ids every round, like llama-bench -p N) through mlx_lm generate_step with
max_tokens=gen (generate_step does not stop on EOS, so every run decodes exactly gen tokens). The first yielded token marks the end of prefill:
  prefillTokS = N / (time to first token)           (includes one decode step)
  decodeTokS  = (gen - 1) / (time from first to last token), at depth N
Prefill and decode of one run therefore come from the same pass, not two passes as in llama-bench (-p N then -d N). Say so wherever
the numbers are compared. Per round the sizes run in the order given; a load average gate (default 1.0) and a pause run before each run.
A run that fails (out of memory, device lost) is recorded under errors; the file is written after every run.
"""
import argparse
import datetime
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import time


def sha256_file(path):
    """sha256 of a file; cached by (path, size, mtime) in ~/.cache/t7b-hashes.json so every ticket of one model hashes it once."""
    st = os.stat(path)
    key = f"{os.path.realpath(path)}:{st.st_size}:{st.st_mtime_ns}"
    cache_file = os.path.expanduser("~/.cache/t7b-hashes.json")
    try:
        with open(cache_file) as f:
            cache = json.load(f)
    except (OSError, ValueError):
        cache = {}
    if key in cache:
        return cache[key]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    cache[key] = h.hexdigest()
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(cache, f)
    return cache[key]


def info_lines():
    try:
        out = subprocess.run(["mlx-omarchy-info"], capture_output=True, text=True, timeout=60, check=False).stdout
    except OSError:
        return {}
    d = {}
    for ln in out.splitlines():
        if ":" in ln:
            k, v = ln.split(":", 1)
            d[k.strip()] = v.strip()
    return d


def render_node_users():
    users = set()
    me = os.getpid()
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == me:
            continue
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                if os.readlink(f"/proc/{pid}/fd/{fd}").startswith("/dev/dri/renderD"):
                    with open(f"/proc/{pid}/comm") as c:
                        users.add(c.read().strip())
        except OSError:
            continue
    return sorted(users) or ["none"]


def wait_for_load(limit, wait_s):
    t0 = time.time()
    while os.getloadavg()[0] >= limit:
        if time.time() - t0 > wait_s:
            return False
        time.sleep(5)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host-label", required=True, help="machine model for the public receipt, never a host name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="8192,32768")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--prefill-step", type=int, default=512)
    ap.add_argument("--seed", type=int, default=292)
    ap.add_argument("--pause", type=float, default=30.0)
    ap.add_argument("--load-limit", type=float, default=1.0)
    ap.add_argument("--load-wait", type=float, default=300.0)
    ap.add_argument("--download-only", action="store_true")
    a = ap.parse_args()

    from huggingface_hub import snapshot_download
    path = a.model if os.path.isdir(a.model) else snapshot_download(a.model)
    if a.download_only:
        print("downloaded", path)
        return 0
    weights = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    hashes = {f: sha256_file(os.path.join(path, f)) for f in weights}
    quant_sha = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(hashes.items())).encode()).hexdigest()

    import mlx.core as mx
    import mlx_lm
    from mlx_lm.generate import generate_step
    from mlx_lm.utils import load
    model, tok = load(path)
    vocab = getattr(getattr(model, "args", None), "vocab_size", None) or 30000
    sizes = [int(x) for x in a.sizes.split(",")]
    rng = random.Random(a.seed)
    prompts = {n: [rng.randrange(1000, vocab - 1) for _ in range(n)] for n in sizes}

    info = info_lines()
    version = mx.__version__
    commit = version.split("+")[-1] if "+" in version else version
    started = datetime.datetime.now(datetime.timezone.utc)
    result = {
        "env": {
            "date": started.strftime("%Y-%m-%d"), "host": a.host_label, "kernel": os.uname().release, "engineCommit": commit,
            "llamaBenchVersion": f"mlx_lm {mlx_lm.__version__} generate_step on mlx-omarchy {version}",
            "vulkanDevices": f"{info.get('device', '')}; {info.get('driver', '')}; api {info.get('api version', '')}",
            "mesa": info.get("driver info", ""), "driNodeUsers": render_node_users(),
            "modelFile": a.model if not os.path.isdir(a.model) else os.path.basename(a.model.rstrip("/")),
            "modelSnapshot": os.path.basename(path.rstrip("/")), "quantFileSha256": quant_sha, "weightFileSha256": hashes,
            "startedAt": started.isoformat(timespec="seconds"), "finishedAt": None,
        },
        "commands": [f"mlx_lm.generate.generate_step(prompt=<{n} random ids, seed {a.seed}>, model, max_tokens={a.gen}, "
                     f"prefill_step_size={a.prefill_step})" for n in sizes],
        "method": (f"{a.rounds} rounds; per round and size one run: a prompt of N random token ids, prefill in chunks of {a.prefill_step}, "
                   f"then exactly {a.gen} decode tokens at depth N, from one pass (prefill rate = N over time to first token; decode rate = "
                   f"{a.gen - 1} tokens over the time from first to last token). One repetition per round, so every run is a raw value. "
                   f"Load average gate under {a.load_limit} before each run, {a.pause:g} s pause between runs, stock clocks, "
                   "packaged mlx-omarchy and Honeykrisp driver."),
        "rawRuns": {str(n): [] for n in sizes}, "errors": [],
    }

    def save():
        result["env"]["finishedAt"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(result, f, indent=1)
        os.replace(tmp, a.out)

    exit_code = 0
    for r in range(a.rounds):
        for n in sizes:
            if not wait_for_load(a.load_limit, a.load_wait):
                result["errors"].append({"round": r, "size": n, "error": f"load average stayed at or above {a.load_limit} for {a.load_wait:g} s"})
                exit_code = 2
                save()
                continue
            try:
                gc.collect()
                mx.clear_cache()
                mx.reset_peak_memory()
                t0 = time.perf_counter()
                count, t_first, t_last = 0, None, None
                for _ in generate_step(mx.array(prompts[n]), model, max_tokens=a.gen, prefill_step_size=a.prefill_step):
                    count += 1
                    t_last = time.perf_counter()
                    if t_first is None:
                        t_first = t_last
                if count != a.gen:
                    raise RuntimeError(f"decoded {count} tokens, expected {a.gen}")
                result["rawRuns"][str(n)].append({
                    "prefillTokS": n / (t_first - t0), "decodeTokS": (count - 1) / (t_last - t_first),
                    "prefillSeconds": t_first - t0, "decodeSeconds": t_last - t_first,
                    "peakMemoryGiB": round(mx.get_peak_memory() / 2 ** 30, 2), "loadAverageBefore": os.getloadavg()[0], "round": r,
                })
                last = result["rawRuns"][str(n)][-1]
                print(f"run round {r} size {n}: prefill {last['prefillTokS']:.1f} tok/s ({last['prefillSeconds']:.0f} s), decode "
                      f"{last['decodeTokS']:.2f} tok/s, peak {last['peakMemoryGiB']} GiB", flush=True)
            except Exception as e:  # noqa: BLE001 - recorded, the other sizes still run
                result["errors"].append({"round": r, "size": n, "error": f"{type(e).__name__}: {e}"[:300]})
                exit_code = 2
            save()
            time.sleep(a.pause)
    print(json.dumps({"rawRuns": {k: [(round(x["prefillTokS"], 1), round(x["decodeTokS"], 2)) for x in v] for k, v in result["rawRuns"].items()},
                      "errors": result["errors"]}))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
