#!/usr/bin/env python3
"""Same-wheel ABBA: SdpaCausalFlashCoopmat16BF16 vs the fallback (PR #883).

Five pairs in A-B-B-A order. Each leg is a fresh subprocess with
MLX_OMARCHY_SDPA_CAUSAL_FLASH unset (A, the causal flash route engages
by capability) or set to 0 (B, the composed/8x8x8 fallback). Both legs
run the same wheel; the report asserts that and prints the stamp.

Per leg the driver records: wheel stamp, prompt token count (from the
response, never retokenized), prefill seconds (time to first streamed
token), pinned-length decode tok/s, the exact generated token IDs
(sha256-16 and first/last three), and a last-token logits row
(argmax + top-10 + the .npy the comparison reads). One leg per arm also
runs with MLX_OMARCHY_TRACE_DISPATCH=1 and its stderr is parsed for the
[rtmod] DISPATCH census by kernel id.

Done rules (DEC-16): identical greedy IDs across arms, and matched
tok/s within the run-to-run band. Any ID divergence fails the whole
capture - a kernel defect, not a slower one.

Usage (on the RX 9070 host, inside the GPU window):
  python3 scripts/sdpa16_ab_bench.py \
      --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
      --pairs 5 --prefill-tokens 512 --decode-tokens 64 \
      --out /tmp/sdpa16-abba

Requires: the wheel under test installed in this interpreter together
with mlx-lm. Prints provenance first (wheel stamp + provenance helper
when present).
"""

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

# One subprocess per leg. Prefill pass: stream_generate over the exact
# prompt with max_tokens=1; the first response carries prompt_tokens and
# its wall time is the prefill time. Decode pass: pinned length, greedy.
# The logits pass evaluates one forward over prompt+generated ids and
# dumps the final position's logits for the cross-arm comparison.
DRIVER = r"""
import argparse, hashlib, json, os, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--prompt-file", required=True)
ap.add_argument("--prefill-tokens", type=int, default=512)
ap.add_argument("--decode-tokens", type=int, default=64)
ap.add_argument("--logits-out", default="")
args = ap.parse_args()

model, tok = load(args.model)
# Exact-length prefill: the file is tokenized once, then truncated to
# the requested count and handed to the generator as ids. The response's
# prompt_tokens (recorded below) is the processed count - if it does not
# match the request the run is invalid, not merely mislabeled.
prompt_ids = tok.encode(open(args.prompt_file).read())[:args.prefill_tokens]
prompt = mx.array(prompt_ids)

def digest(ids):
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()[:16]

# prefill: one streamed token; time to first yield excludes load
t0 = time.monotonic_ns()
first = None
for r in stream_generate(model, tok, prompt, max_tokens=1):
    first = r
    break
prefill_s = (time.monotonic_ns() - t0) / 1e9
prompt_tokens = getattr(first, "prompt_tokens", None)

# decode: greedy pinned length; the clock starts on the first yielded
# token so prefill never counts toward decode tok/s
ids = []
t0 = None
last = None
for r in stream_generate(model, tok, prompt, max_tokens=args.decode_tokens):
    if t0 is None:
        t0 = time.monotonic_ns()
    ids.append(r.token)
    last = r
times_done = time.monotonic_ns() - t0 if t0 is not None else 0
decode_tps = (len(ids) - 1) / (times_done / 1e9) if len(ids) > 1 else None

# last-token logits over prompt + generated ids
row = {}
if args.logits_out:
    full = prompt_ids + ids
    logits = model(mx.array([full]))
    last_logits = logits[0, -1]
    mx.eval(last_logits)
    import numpy as np
    np.save(args.logits_out, np.asarray(last_logits.astype(mx.float32)))
    top = mx.argsort(-last_logits)[:10]
    mx.eval(top)
    row["argmax"] = int(mx.argmax(last_logits).item())
    row["top10"] = [int(t) for t in top]

print(json.dumps({
    "wheel": mx.__version__,
    "sdpa_causal_flash": os.environ.get("MLX_OMARCHY_SDPA_CAUSAL_FLASH"),
    "prompt_tokens": prompt_tokens,
    "prefill_s": round(prefill_s, 6),
    "generated": len(ids),
    "decode_tps": round(decode_tps, 4) if decode_tps else None,
    "ids_sha256_16": digest(ids),
    "ids_first": ids[:3],
    "ids_last": ids[-3:],
    "logits": row,
}, sort_keys=True))
"""


def target_kernel_id():
    """Numeric ComputeKernel id of SdpaCausalFlashCoopmat16BF16, resolved
    from this repo's compute.h so an enum reorder cannot silently retarget
    the census guard. Fails loudly when the name is not found."""
    header = (Path(__file__).resolve().parent.parent /
              "overlay/mlx/backend/omarchy/compute.h")
    text = header.read_text(encoding="utf-8")
    body = text.split("enum class ComputeKernel", 1)[1].split("};", 1)[0]
    kid = -1
    for raw in body.splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        name, eq, num = line.partition("=")
        name = name.strip().rstrip(",")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue
        if eq:
            kid = int(num.strip().rstrip(","))
        else:
            kid += 1
        if name == "SdpaCausalFlashCoopmat16BF16":
            return kid
    raise SystemExit(
        f"SdpaCausalFlashCoopmat16BF16 not found in {header}")


def target_dispatch_counts(results, target_kid):
    """(arm A target count, arm B target count) over the traced legs."""
    a = b = 0
    for r in results:
        count = r.get("dispatch_census", {}).get(str(target_kid), 0)
        if r["arm"] == "A":
            a += count
        else:
            b += count
    return a, b


def run_leg(driver, env_extra, args, logits_path, trace):
    env = dict(os.environ)
    # Arm A is "switch unset": an inherited MLX_OMARCHY_SDPA_CAUSAL_FLASH
    # from the caller's shell would leak into the child and silently turn
    # a fallback-vs-fallback run into a pass. Strip it before the per-arm
    # override so each leg's env is exactly what the arm declares.
    env.pop("MLX_OMARCHY_SDPA_CAUSAL_FLASH", None)
    env.update(env_extra)
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    if trace:
        env["MLX_OMARCHY_TRACE_DISPATCH"] = "1"
    cmd = [
        sys.executable, driver,
        "--model", args.model,
        "--prompt-file", args.prompt_file,
        "--prefill-tokens", str(args.prefill_tokens),
        "--decode-tokens", str(args.decode_tokens),
    ]
    if logits_path:
        cmd += ["--logits-out", logits_path]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        print(proc.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"leg failed rc={proc.returncode} (env {env_extra})")
    census = {}
    for line in proc.stderr.splitlines():
        if "DISPATCH kernel=" in line:
            kid = int(line.split("kernel=")[1].split()[0])
            census[kid] = census.get(kid, 0) + int(
                line.split("count=")[1].split()[0])
    row = json.loads(proc.stdout.strip().splitlines()[-1])
    row["dispatch_census"] = census
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-4B-Instruct-2507-4bit")
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--prefill-tokens", type=int, default=512)
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--prompt-file", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.prompt_file:
        # Deterministic 512+-token prompt: repeated distinct sentences
        # tok.encode is used only to size the file, never to count for
        # the measurement (prompt_tokens comes from the response).
        args.prompt_file = str(out / "prompt.txt")
        body = " ".join(
            f"Entry {i}: the prefill kernel dequantizes group-64 q4 "
            f"weights and multiplies the 16-row activation tile." for i in range(90))
        Path(args.prompt_file).write_text(body)

    with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False) as f:
        f.write(DRIVER)
        driver = f.name

    results = []
    order = ["A", "B", "B", "A"]
    target_kid = target_kernel_id()
    for pair in range(1, args.pairs + 1):
        for idx, arm in enumerate(order):
            env_extra = {} if arm == "A" else {"MLX_OMARCHY_SDPA_CAUSAL_FLASH": "0"}
            logits_path = str(out / f"logits_{arm}_{pair}_{idx}.npy")
            trace = pair == 1  # census on the first pair's legs
            row = run_leg(driver, env_extra, args, logits_path, trace)
            row["arm"] = arm
            row["pair"] = pair
            row["seq"] = f"{pair}{'ABBA'[idx]}"
            results.append(row)
            (out / "results.json").write_text(json.dumps(results, indent=1))
            print(f"[{row['seq']}] arm={arm} prefill={row['prefill_s']}s "
                  f"decode={row['decode_tps']} tok/s ids={row['ids_sha256_16']}")

    stamps = {r["wheel"] for r in results}
    assert len(stamps) == 1, f"same-wheel violated: {stamps}"
    for r in results:
        if r["prompt_tokens"] != args.prefill_tokens:
            raise SystemExit(
                f"leg {r['seq']} processed {r['prompt_tokens']} prompt "
                f"tokens, expected {args.prefill_tokens} — capture invalid")
    ids = {r["ids_sha256_16"] for r in results}
    a = [r for r in results if r["arm"] == "A"]
    b = [r for r in results if r["arm"] == "B"]

    # last-token logits comparison across the last pair's arms
    import numpy as np
    last_pair = [r for r in results if r["pair"] == args.pairs]
    la = np.load(str(out / f"logits_A_{args.pairs}_0.npy"))
    lb = np.load(str(out / f"logits_B_{args.pairs}_1.npy"))
    maxabs = float(np.max(np.abs(la - lb)))
    arg_same = int(np.argmax(la)) == int(np.argmax(lb))
    top10_same = sorted(np.argsort(-la)[:10].tolist()) == sorted(
        np.argsort(-lb)[:10].tolist())

    summary = {
        "wheel": sorted(stamps),
        "model": args.model,
        "ids_match": len(ids) == 1,
        "ids_sha256_16": sorted(ids),
        "prefill_s_A": sorted(r["prefill_s"] for r in a),
        "prefill_s_B": sorted(r["prefill_s"] for r in b),
        "prefill_median_A": statistics.median(r["prefill_s"] for r in a),
        "prefill_median_B": statistics.median(r["prefill_s"] for r in b),
        "decode_tps_A": sorted(r["decode_tps"] for r in a),
        "decode_tps_B": sorted(r["decode_tps"] for r in b),
        "decode_median_A": statistics.median(r["decode_tps"] for r in a),
        "decode_median_B": statistics.median(r["decode_tps"] for r in b),
        "last_logits": {"maxabs": maxabs, "argmax_same": arg_same,
                        "top10_same": top10_same},
        "census_A": a[0]["dispatch_census"],
        "census_B": b[0]["dispatch_census"],
    }
    print(json.dumps(summary, indent=1, sort_keys=True))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    if len(ids) != 1:
        raise SystemExit("FAIL: greedy token IDs diverged across arms")
    if not arg_same:
        raise SystemExit("FAIL: last-token argmax diverged across arms")
    a_count, b_count = target_dispatch_counts(results, target_kid)
    if a_count == 0:
        raise SystemExit(
            f"FAIL: target kernel {target_kid} never dispatched on arm A "
            "- the capture measured fallback vs fallback")
    if b_count != 0:
        raise SystemExit(
            f"FAIL: target kernel {target_kid} dispatched {b_count}x on "
            "arm B - the kill switch did not isolate the arms")


if __name__ == "__main__":
    main()
