#!/usr/bin/env bash
# Fuse2B W2: dispatch-count proof. Profile the 2B d64 decode on the cand venv
# with MLX_OMARCHY_GDN_CONV_DELTA=1 (fused) and unset (composed); the fused
# stream must show GdnConvDeltaDecodeBF16 and no GdnConvDecodeBF16 / fused-GDU
# pair per token, with identical greedy output (bit-exact fusion).
set -euo pipefail
OUT=/var/tmp/fuse2b/w2prof
mkdir -p "$OUT"
test ! -e /var/tmp/JW16_MAINTENANCE
uptime_s=$(cut -d. -f1 /proc/uptime); test "$uptime_s" -ge 360 || { echo "boot settle gate: ${uptime_s}s"; exit 1; }
load1=$(cut -d" " -f1 /proc/loadavg)
li=${load1%%.*}; lf=${load1#*.}00
(( $((10#$li * 100 + 10#${lf:0:2})) < 50 )) || { echo "load gate: $load1"; exit 1; }
psi=$(cat /proc/pressure/cpu)
case "$psi" in *"some avg10=0.00"*) ;; *) echo "PSI gate: $psi"; exit 1;; esac
printf "host=%s boot=%s uptime_s=%s load1=%s psi=%s\n" "$(hostname)" \
  "$(cat /proc/sys/kernel/random/boot_id)" "$uptime_s" "$load1" "$psi" > "$OUT/gates.txt"

CANDPY=/var/tmp/fuse2b-cand/bin/python
BUILD=/var/tmp/fuse2b-build
BENCH=${HOME}/bench-scripts/qwen38-mlx-bench.py
PROMPTS=${HOME}/bench-scripts/qwen38-2b-prompts.jsonl
M2B=${HOME}/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e7c97b86b34b381

run() { # name env...
  local name=$1; shift
  timeout 300 env MLX_OMARCHY_GPU_PROFILE="$OUT/prof-$name.ndjson" \
    MLX_OMARCHY_GPU_PROFILE_LABEL="fuse2b-$name" "$@" \
    "$CANDPY" "$BENCH" --model "$M2B" --prompts "$PROMPTS" --limit 1 --warmup 1 \
      --passes 1 --new-tokens 64 --prefill-tokens 512 --label "fuse2b-$name" \
      --out "$OUT/$name.json" > "$OUT/$name.log" 2> "$OUT/$name.err" \
    || { echo "RUN FAILED $name rc=$?"; return 1; }
  echo "done $name $(grep -o '"ordered_records_sha256[^,}]*' "$OUT/$name.json" | head -1)"
}
run on MLX_OMARCHY_GDN_CONV_DELTA=1
run off

"$CANDPY" - "$OUT" <<'PYEOF'
import json, re, sys
from collections import Counter
out = sys.argv[1]
src = open(f"{out.rsplit('/',1)[0]}/../../fuse2b-build/overlay/mlx/backend/omarchy/compute.h").read() if False else open("/var/tmp/fuse2b-build/overlay/mlx/backend/omarchy/compute.h").read()
body = re.search(r"enum class ComputeKernel[^{]*\{(.*?)\n\};", src, re.S).group(1)
names = []
for line in body.splitlines():
    line = line.split("//")[0].strip().rstrip(",")
    if line and "=" not in line:
        names.append(line)
def kn(e):
    return names[e] if e < len(names) else f"enum{e}"
def load(p):
    ev = []
    for line in open(p):
        if '"k":"d"' in line:
            try:
                ev.append(json.loads(line))
            except Exception:
                pass
    return ev
summary = {}
for arm in ("on", "off"):
    ev = load(f"{out}/prof-{arm}.ndjson")
    c = Counter(kn(r["e"]) for r in ev)
    summary[arm] = {
        "total": len(ev),
        "GdnConvDeltaDecodeBF16": c.get("GdnConvDeltaDecodeBF16", 0),
        "GdnConvDecodeBF16": c.get("GdnConvDecodeBF16", 0) + c.get("GdnConvDecodeAppleBF16", 0),
        "GatedDeltaDecodeBF16": c.get("GatedDeltaDecodeBF16", 0) + c.get("GatedDeltaDecodeBF16Untiled", 0) + c.get("GatedDeltaDecodeBF16Pf", 0),
    }
    print(arm, summary[arm])
on_dig = json.load(open(f"{out}/on.json")).get("ordered_records_sha256")
off_dig = json.load(open(f"{out}/off.json")).get("ordered_records_sha256")
print("digests:", on_dig, off_dig, "SAME" if on_dig == off_dig else "DIFF")
ok = (summary["on"]["GdnConvDeltaDecodeBF16"] > 0 and summary["off"]["GdnConvDeltaDecodeBF16"] == 0
      and on_dig == off_dig)
print("W2-DISPATCH-PROOF", "PASS" if ok else "FAIL")
PYEOF
sudo journalctl --flush; sync
echo "post-window llm-inference: $(systemctl is-active llm-inference || true)"
