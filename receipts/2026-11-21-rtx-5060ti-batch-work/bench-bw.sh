#!/usr/bin/env bash
# RTX 5060 Ti batch-work A/B: Qwen3-4B 4bit greedy decode, release builds A and B interleaved
# (identical-token gate, build stamps must differ), then one buffer-mode GPU profile per side.
# usage: bench-bw.sh <tag> <release-A> <diag-A> no <release-B> <diag-B> <pairs> <tokens>
set -uo pipefail
BASE=${BASE:-$PWD}  # holds repo/ (the checkout) and busy_split.py
TAG=$1 OVR=$2 OVD=$3 LLAMA=${4:-yes} OVB=$5 OVDB=$6 PAIRS=${7:-5} TOKENS=${8:-256}
MODEL=$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b
GGUF=${GGUF:-}
LB=${LB:-}
PY=/usr/lib/omarchy-mlx/venv/bin/python
DRV=$BASE/repo/tools/profile_decode_driver.py
OUT=$BASE/runs/$(date -u +%Y%m%dT%H%M%SZ)-$TAG; mkdir -p "$OUT"
echo "RUN_START $(date -Is) out=$OUT"
d0=$(dmesg 2>/dev/null | wc -l)
mlx() { # name overlay extra-env...
  local name=$1 ov=$2; shift 2
  env -i HOME="$HOME" PATH=/usr/bin:/bin USER="$USER" bash -c '
    . /usr/lib/nvidia-egpu-615.78.08/vulkan-env.sh
    export MLX_DISABLE_COMPILE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MLX_OMARCHY_ALLOW_NON_APPLE=1
    export MLX_OMARCHY_VRAM_ACTIVATIONS=1 MLX_OMARCHY_VRAM_WEIGHTS="$1" PYTHONPATH="$2"
    shift 2; for kv in "$@"; do export "$kv"; done
    "'"$PY"'" "'"$BASE"'/repo/scripts/mlx_provenance.py" > "$PROVOUT" 2>&1
    exec timeout --signal=TERM --kill-after=15s 150s "'"$PY"'" -u "'"$DRV"'" --model "$MODEL_ARG" \
      --warmup-tokens 8 --tokens '"$TOKENS"' --out "$WIN"' _ "$MODEL" "$ov" "MODEL_ARG=$MODEL" "WIN=$OUT/$name.windows.jsonl" "PROVOUT=$OUT/$name.prov.json" "$@" \
    > "$OUT/$name.log" 2>&1
  local rc=$?
  "$PY" - "$OUT/$name.windows.jsonl" "$name" "$rc" <<'PY'
import json, sys, hashlib, statistics
rows = [json.loads(l) for l in open(sys.argv[1])][1:]
t1 = [r["t1"] for r in rows]; toks = [r["tok"] for r in rows]
d = [(b - a) / 1e6 for a, b in zip(t1, t1[1:])]
sha = hashlib.sha256(json.dumps(toks).encode()).hexdigest()[:16]
try:
    txt = open(sys.argv[1].replace(".windows.jsonl", ".prov.json")).read()
    prov = json.loads(txt[txt.index("{"):])
    print("PROV", sys.argv[2], {k: v for k, v in prov.items() if not isinstance(v, (dict, list))})
except Exception as e:
    print("PROV", sys.argv[2], "unreadable", e)
print(f"MLX {sys.argv[2]} rc={sys.argv[3]} tokens={len(toks)} tok_s={1000*len(d)/sum(d):.2f} median_ms={statistics.median(d):.2f} ids={sha}")
PY
}
# env A/B: ABABAB pairs, then one GPU profile of arm B.
for i in $(seq 1 "$PAIRS"); do mlx "a$i" "$OVR"; mlx "b$i" "$OVB"; done
mlx pa "$OVD" MLX_OMARCHY_GPU_PROFILE_DISPATCH=0 "MLX_OMARCHY_GPU_PROFILE=$OUT/pa.ndjson"
mlx pb "$OVDB" MLX_OMARCHY_GPU_PROFILE_DISPATCH=0 "MLX_OMARCHY_GPU_PROFILE=$OUT/pb.ndjson"
for a in pa pb; do echo "SPLIT $a $("$PY" "$BASE/busy_split.py" "$OUT/$a.ndjson" "$OUT/$a.windows.jsonl" 2>&1 | tail -1)"; done
"$PY" - "$OUT" "$PAIRS" <<'PY2'
import json, statistics, sys
from pathlib import Path
out, n = Path(sys.argv[1]), int(sys.argv[2])
def load(name):
    w = [json.loads(l) for l in open(out / f"{name}.windows.jsonl")][1:]
    t1 = [r["t1"] for r in w]; d = [(y - x) / 1e6 for x, y in zip(t1, t1[1:])]
    p = open(out / f"{name}.prov.json").read(); v = json.loads(p[p.index("{"):])["dist_version"]
    return d, tuple(r["tok"] for r in w), v
def tps(d): return 1000 * len(d) / sum(d)
A = [load(f"a{i}") for i in range(1, n + 1)]; B = [load(f"b{i}") for i in range(1, n + 1)]
va, vb = {x[2] for x in A}, {x[2] for x in B}
print("STAMPS A", va, "B", vb, "DIFFER" if va.isdisjoint(vb) and len(va) == len(vb) == 1 else "SAME-OR-MIXED")
print("IDS_IDENTICAL", "yes" if len({x[1] for x in A + B}) == 1 else "NO")
for lab, rows in (("A", A), ("B", B)):
    whole = [tps(x[0]) for x in rows]; steady = [tps(x[0][5:]) for x in rows]
    print(lab, "whole", [round(v, 2) for v in whole], "median", round(statistics.median(whole), 2),
          "| tokens 6+", [round(v, 2) for v in steady], "median", round(statistics.median(steady), 2),
          "| max token ms", [round(max(x[0]), 1) for x in rows])
print("PAIR_WINS_B whole", sum(tps(b[0]) > tps(a[0]) for a, b in zip(A, B)), "of", n)
for name in ("pa", "pb"):
    meta, gpu = 1.0, []
    for line in open(out / f"{name}.ndjson"):
        try: r = json.loads(line)
        except json.JSONDecodeError: continue
        if r.get("k") == "meta": meta = r["period_ns"]
        elif r.get("k") == "b" and "t0" in r: gpu.append((r["t1"] - r["t0"]) * meta / 1e6)
    gpu.sort(); q = lambda p: gpu[min(len(gpu) - 1, int(p * len(gpu)))]
    print("SUBMIT_GPU_MS", name, "n", len(gpu), "p50", round(q(0.5), 2), "p99", round(q(0.99), 2), "max", round(gpu[-1], 2))
PY2
if [[ $LLAMA == yes ]]; then
  env -i HOME="$HOME" PATH=/usr/bin:/bin bash -c ". /usr/lib/nvidia-egpu-615.78.08/vulkan-env.sh; exec timeout 120s $LB -m $GGUF -ngl 99 -p 0 -n 64 -r 3 -o json" > "$OUT/llama.json" 2> "$OUT/llama.err"
  echo "LLAMA rc=$? $("$PY" -c "import json,sys; [print(r['backends'],r['n_gen'],round(r['avg_ts'],2),'tok/s') for r in json.load(open('$OUT/llama.json'))]" 2>&1)"
  env -i HOME="$HOME" PATH=/usr/bin:/bin bash -c ". /usr/lib/nvidia-egpu-615.78.08/vulkan-env.sh; GGML_VK_PERF_LOGGER=1 exec timeout 120s $LB -m $GGUF -ngl 99 -p 0 -n 16 -r 1" > "$OUT/llama-perf.out" 2> "$OUT/llama-perf.err"
  echo "LLAMA_PERF rc=$? lines=$(wc -l < "$OUT/llama-perf.err")"
fi
dmesg 2>/dev/null | tail -n +$((d0 + 1)) | grep -iE 'NVRM|Xid|dart|iommu' > "$OUT/kmsg-new.txt"
echo "NVRM_LINES=$(wc -l < "$OUT/kmsg-new.txt")"
echo "RUN_DONE out=$OUT"
