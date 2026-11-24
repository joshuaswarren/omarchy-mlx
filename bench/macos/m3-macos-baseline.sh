#!/usr/bin/env bash
# m3-macos-baseline.sh — Apple-silicon macOS LLM+ANE baseline, one command.
#
# Produces a fixed set of measurement cells on stock macOS MLX:
#   arm A (pinned fixture): mlx-community/Qwen3-4B-Instruct-2507-4bit
#         rev 50d42775, transistor prompt + chat template, 64 greedy tokens,
#         temp 0, seed 0, warmup 4, 5 replicates, bench_decode.py sha 8220064c
#         -> decode tok/s + ms/tok + prefill s + ids_sha256_16 per rep
#         (identity digest acca26cfaf8e37d0 = the Linux reference)
#   arm B (card-row ctx2048): "history of France" prompt tiled to 2048 tokens,
#         prefill s + decode tok/s over 128 tokens + ids digest
#         (mirrors the Linux ctx-2048 row shape)
#   probe C: Parakeet encoder ms on a synthetic mel (best effort) +
#         CoreML Neural Engine smoke (qualification-kit probe)
#
# Usage (macOS arm64, internet required for the first run):
#   curl -fsSL <url>/m3-macos-baseline.sh | bash
#   or: bash m3-macos-baseline.sh [--work DIR]
set -euo pipefail

WORK="${M3BL_WORK:-$HOME/scratch/m3-macos-baseline}"
PIN_REF="197b068f3fd0cdcd4c5e713032fc3ce8a892396e"   # omarchy-mlx main pin
RAW="https://raw.githubusercontent.com/joshuaswarren/omarchy-mlx/$PIN_REF"
MODEL_REPO="mlx-community/Qwen3-4B-Instruct-2507-4bit"
MODEL_REV="50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
BENCH_SHA="8220064c5ea3aaafb7564bab704632083d4fab59605232abe3e135fab65b132c"
# The pinned fixture's identity digest on the Linux reference (the r4
# M4 Pro row). A macOS run that returns a different digest is a real
# divergence and must be flagged in the artifact, not only printed.
EXPECT_DIGEST="acca26cfaf8e37d0"
PROMPT="Explain how a transistor can function as a switch in a simple circuit."
REPS=5

[ "$(uname -s)" = "Darwin" ] || { echo "need macOS (Darwin)"; exit 2; }
[ "$(uname -m)" = "arm64" ] || { echo "need Apple silicon (arm64)"; exit 2; }
command -v python3 >/dev/null || { echo "need python3 (Xcode CLT: xcode-select --install)"; exit 2; }
# prefer the newest python3 on PATH (parakeet-mlx needs >=3.10)
PYBIN=""
for p in python3.13 python3.12 python3.11 python3.10 python3; do
  command -v "$p" >/dev/null 2>&1 && { PYBIN="$p"; break; }
done
while [ "${1:-}" = "--work" ]; do WORK="$2"; shift 2; done
mkdir -p "$WORK/harness" "$WORK/out"
cd "$WORK"

echo "== m3-macos-baseline =="
echo "host: $(sysctl -n machdep.cpu.brand_string 2>/dev/null) | $(uname -srm)"
echo "work: $WORK"

# --- 1. harness (pinned files, sha-verified) ---
for f in bench_decode.py caps_sim_guard.py mlx_provenance.py; do
  curl -fsSL "$RAW/scripts/$f" -o "harness/$f"
done
echo "$BENCH_SHA  harness/bench_decode.py" | shasum -a 256 -c --status \
  || { echo "bench_decode.py sha mismatch (want $BENCH_SHA)"; exit 3; }
echo "harness: bench_decode.py sha 8220064c verified"

# --- 2. venv ---
echo "python: $PYBIN ($($PYBIN --version 2>&1))"
[ -x venv/bin/python ] || "$PYBIN" -m venv venv
PY="$WORK/venv/bin/python"
"$PY" -m pip -q install --upgrade pip
"$PY" -m pip -q install mlx-lm huggingface_hub
PIPVER=$("$PY" -m pip freeze | grep -iE '^(mlx|mlx-lm|mlx_lm|huggingface|tokenizers|numpy)' | tr '\n' ' ')
"$PY" -m pip freeze > out/pip-freeze.txt 2>/dev/null || true
echo "pip: $PIPVER"

# --- 3. model snapshot (pinned revision) ---
SNAP=$("$PY" - "$MODEL_REPO" "$MODEL_REV" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2]))
PY
)
echo "model: $MODEL_REPO @ ${MODEL_REV:0:8} -> $SNAP"

# --- 4. arm A: pinned fixture, 5 replicates ---
echo "== arm A: pinned fixture (transistor prompt, 64 greedy tok, temp0 seed0, warm 4, $REPS reps) =="
: > out/armA.ndjson
for i in $(seq 1 "$REPS"); do
  echo "-- replicate $i"
  "$PY" harness/bench_decode.py --model "$SNAP" --prompt "$PROMPT" \
    --tokens 64 --warmup-tokens 4 --temp 0 --seed 0 \
    | tee "out/armA-rep$i.txt" \
    | grep -E '^\{"' >> out/armA.ndjson || true
done
"$PY" - out/armA.ndjson out/armA-summary.json <<'PY'
import json, sys, statistics
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
if not rows:
    print("arm A produced no result lines", file=sys.stderr); sys.exit(1)
dec = sorted(r["decode_tps"] for r in rows)
s = {"arm": "A-pinned-fixture", "reps": len(rows),
     "decode_tps_median": statistics.median(dec),
     "decode_tps_all": [round(x, 2) for x in dec],
     "prefill_s_all": [r.get("prefill_s") for r in rows],
     "prompt_tokens": rows[0].get("prompt_tokens"),
     "ids_sha256_16": sorted({r["ids_sha256_16"] for r in rows})}
json.dump(s, open(sys.argv[2], "w"), indent=1)
print("ARM_A " + json.dumps(s, sort_keys=True))
PY

# --- 5. arm B: card-row ctx2048 (France prompt tiled, prefill s + 128-tok decode) ---
echo "== arm B: card row ctx2048 =="
"$PY" - "$SNAP" out/armB.json <<'PY'
import hashlib, json, sys, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
snap, outp = sys.argv[1], sys.argv[2]
model, tok = load(snap)
ids = tok.encode("The history of France begins in antiquity. ")
ctx = 2048
ids = (ids * (ctx // len(ids) + 1))[:ctx]
prompt = mx.array(ids)
tokenizer = tok
tokenizer.eos_token_ids = set()
try:
    detok = tokenizer.detokenizer
    tokenizer._detokenizer_class = lambda _: detok
    detok.reset()
except Exception:
    pass
t0 = time.monotonic_ns()
gen_ids, times = [], []
for item in stream_generate(model, tokenizer, prompt, max_tokens=128):
    times.append(time.monotonic_ns())
    gen_ids.append(getattr(item, "token", item))
prefill_s = (times[0] - t0) / 1e9 if times else None
gaps = [b - a for a, b in zip(times, times[1:])]
dec_tps = (len(times) - 1) / (sum(gaps) / 1e9) if gaps else None
digest = hashlib.sha256(",".join(str(int(i)) for i in gen_ids).encode()).hexdigest()[:16]
row = {"arm": "B-card-ctx2048", "ctx_tokens": ctx, "decoded_tokens": len(gen_ids),
       "prefill_s": round(prefill_s, 4), "decode_tps": round(dec_tps, 2),
       "ms_per_tok": round(1000.0 / dec_tps, 1) if dec_tps else None,
       "ids_sha256_16": digest}
json.dump(row, open(outp, "w"), indent=1)
print("ARM_B " + json.dumps(row, sort_keys=True))
PY

# --- 6. probe C1: Parakeet encoder on synthetic mel (best effort) ---
echo "== probe C1: Parakeet encoder (best effort) =="
"$PY" -m pip -q install parakeet-mlx 2>/dev/null || true
"$PY" - out/parakeet.json <<'PY' || echo "PARAKEET encoder_ms=SKIP"
import json, sys, time
out = {"cell": "parakeet_encoder", "status": "skip"}
try:
    import mlx.core as mx
    from parakeet_mlx import from_pretrained
    from parakeet_mlx.audio import get_logmel
    m = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
    enc = m.get("encoder") if hasattr(m, "get") else getattr(m, "encoder", None)
    if enc is None:
        out["reason"] = "no encoder on %s" % type(m).__name__
    else:
        # 30 s synthetic audio @16k -> 3001 mel frames, matching the card's
        # 3000-frame window (the AN-18 per-clip encoder slot)
        audio = mx.random.normal((16000 * 30,))
        mel = get_logmel(audio, m.preprocessor_config)
        o, l = enc(mel, None); mx.eval(o, l)   # warm compile, eval both outputs
        ts = []
        for _ in range(11):
            t0 = time.perf_counter()
            o, l = enc(mel, None); mx.eval(o, l)
            ts.append((time.perf_counter() - t0) * 1000)
        ts = sorted(ts[1:])
        out.update(status="ok", mel_shape=list(mel.shape),
                   encoder_ms_median=round(ts[len(ts) // 2], 2), reps=10)
        out["reference_ms"] = {"m2_ane": 253.42, "m1max_ane": 199.0,
                               "m2_gpu": 259.0, "m1max_gpu": 13.29}
except Exception as e:
    out["status"] = "skip"; out["reason"] = ("%s: %s" % (type(e).__name__, e))[:160]
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("PARAKEET " + json.dumps(out, sort_keys=True))
PY

# --- 7. probe C2: CoreML Neural Engine smoke ---
echo "== probe C2: CoreML NE smoke =="
curl -fsSL "$RAW/qualification-kit/scripts/coreml_smoke.py" -o harness/coreml_smoke.py
"$PY" -m pip -q install coremltools 2>/dev/null || true
"$PY" harness/coreml_smoke.py | tee out/coreml.json || echo "COREML smoke=SKIP"

# --- 8. result bundle ---
"$PY" - out "$MODEL_REPO" "$MODEL_REV" "$PIN_REF" "$BENCH_SHA" "$EXPECT_DIGEST" <<'PY'
import json, os, platform, subprocess, sys
outd, model_repo, model_rev, pin_ref, bench_sha, expect_digest = sys.argv[1:7]
res = {"host": {"uname": platform.platform(),
                "chip": subprocess.run(["sysctl","-n","machdep.cpu.brand_string"],
                                       capture_output=True, text=True).stdout.strip()}}
for name in ("armA-summary.json", "armB.json", "parakeet.json", "coreml.json"):
    p = os.path.join(outd, name)
    if os.path.exists(p):
        try: res[name.split(".")[0]] = json.load(open(p))
        except Exception: res[name.split(".")[0]] = open(p).read().strip()
# Provenance travels with the artifact: the pins the run used, the
# package set it ran on, and the digest it was expected to reproduce.
pip_freeze = ""
pf = os.path.join(outd, "pip-freeze.txt")
if os.path.exists(pf):
    pip_freeze = open(pf).read().strip()
res["provenance"] = {"model_repo": model_repo, "model_rev": model_rev,
                     "pin_ref": pin_ref, "bench_sha256": bench_sha,
                     "pip_freeze": pip_freeze,
                     "expected_ids_sha256_16": expect_digest}
# The arm-A digest against the Linux reference, with the mismatch flagged.
flags = []
armA = res.get("armA-summary") or {}
got = armA.get("ids_sha256_16")
got_list = list(got) if isinstance(got, (list, tuple)) else ([got] if got else [])
match = bool(got_list) and all(d == expect_digest for d in got_list)
res["armA_digest"] = {"expected": expect_digest, "got": got_list, "match": match}
if got_list and not match:
    flags.append("armA_ids_digest_differs_from_linux_reference")
if not got_list:
    flags.append("armA_ids_digest_missing")
res["flags"] = flags
json.dump(res, open(os.path.join(outd, "m3-macos-result.json"), "w"), indent=1)
for flag in flags:
    print("FLAG " + flag + ": expected " + expect_digest + " got " + ",".join(got_list))
PY
echo "== done: $WORK/out/m3-macos-result.json =="
echo "CARD-CELLS: decode tok/s (armA median + armB), prefill s, ids_sha256_16, prompt_tokens, parakeet encoder_ms, coreml smoke"
