#!/bin/bash
# DecodeFuse3 F3 gate window: rope-norm candidate (fence + env gate) vs serving.
# INVOCATION: bash /var/tmp/appbar/gpuwin.sh 'bash /var/tmp/fuse3/gate.sh'
set -u
OUT=/var/tmp/fuse3/gate; mkdir -p "$OUT"
SERVE=/var/tmp/v072-venv-fused/bin/python3
CAND=/var/tmp/fuse3-venv/bin/python3
BENCH=$HOME/bench-scripts/qwen38-mlx-bench.py
MODEL=$(echo ~/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98b*)
{ date -u +%FT%TZ; cat /proc/sys/kernel/random/boot_id
  for py in "$SERVE" "$CAND"; do "$py" -c 'import mlx.core as mx; print(mx.__version__)'; done
  "$CAND" -c 'import mlx.core as mx; print("rope_rms_norm:", hasattr(mx.fast, "rope_rms_norm"))'
} > "$OUT/identity.txt"

# 1) fused-primitive iso check v3 in the candidate venv: fuseable legs
#    bit-identical, fenced legs must raise.
env X=1 "$CAND" /var/tmp/fuse3/rope_norm_bitcheck3.py "$OUT/rope-norm-iso.json" \
  > "$OUT/rope-norm-iso.log" 2>&1
echo "iso rc=$? $(head -1 "$OUT/rope-norm-iso.log")"

# 2) shared-kernel battery (gdn/lse/arg/rms rows) serve vs candidate.
bitrun () { local tag=$1 py=$2 outjson=$3
  env X=1 "$py" /var/tmp/dg/dg_bitcheck.py "$outjson" > "${outjson%.json}.log" 2>&1
  echo "bit $tag rc=$?"; }
bitrun full-serve "$SERVE" "$OUT/bit-serve.json"
bitrun full-cand  "$CAND"  "$OUT/bit-cand.json"
python3 - "$OUT" > "$OUT/gate-full.txt" 2>&1 <<'PYEOF'
import json, sys
out = sys.argv[1]
serve = json.load(open(f"{out}/bit-serve.json"))
cand = json.load(open(f"{out}/bit-cand.json"))
def leaves(d):
    rows = d.get("rows", d)
    return {k: v for k, v in rows.items() if isinstance(v, str)}
s, c = leaves(serve), leaves(cand)
diff = {k for k in set(s) | set(c) if s.get(k) != c.get(k)}
print("GATE-FULL", "PASS" if not diff else "FAIL", f"rows={len(s)}")
for k in sorted(diff)[:10]:
    print(" ", k, s.get(k), "vs", c.get(k))
PYEOF
echo "gate-full: $(head -1 "$OUT/gate-full.txt")"

# 3) paired cells, interleaved, digest pins. cand arm runs with the fusion
#    flag ON; a d64 kill-switch run (flag unset) must be inert.
cell () { local N=$1 tag=$2 py=$3; shift 3
  env X=1 "$@" "$py" "$BENCH" --model "$MODEL"/ \
    --prompts "$HOME/bench-scripts/qwen38-2b-prompts.jsonl" \
    --limit 1 --new-tokens "$N" --warmup 1 --passes 5 --prefill-tokens 512 \
    --label "fuse3-$N-$tag" --out "$OUT/d$N-$tag.json" > "$OUT/d$N-$tag.log" 2>&1
  "$SERVE" -c "
import json
try:
  d=json.load(open('$OUT/d$N-$tag.json'))
  print('d$N-$tag', round(d['decode_tok_rate']['median'],2), d['ordered_records_sha256'][:12])
except Exception as e:
  print('d$N-$tag', 'PARSE-FAIL', repr(e))"; }
for N in 64 128 256 512; do
  cell $N ctl "$SERVE"
  cell $N cand "$CAND" MLX_OMARCHY_ROPE_NORM_FUSE=1
done
cell 64 ctl2 "$SERVE"
cell 64 ksoff "$CAND"   # kill switch: flag unset -> exact eager chain
( cd "$OUT" && sha256sum ./* > SHA256SUMS 2>/dev/null )

python3 - "$OUT" > "$OUT/gate-final.txt" 2>&1 <<'PYEOF'
import json, sys
out = sys.argv[1]
PINS = {"64": "c84b3e7a", "128": "07c515e0", "256": "c6aabbf0", "512": "5c120987"}
ok = True
med = {}
for N, pin in PINS.items():
    for tag in ("ctl", "cand", "ctl2", "ksoff"):
        try:
            d = json.load(open(f"{out}/d{N}-{tag}.json"))
        except FileNotFoundError:
            continue
        dig = d["ordered_records_sha256"]
        r = round(d["decode_tok_rate"]["median"], 2)
        med[(N, tag)] = r
        if dig[:8] != pin:
            print(f"DIGEST-FAIL d{N}-{tag} {dig[:12]} != {pin}"); ok = False
        else:
            print(f"d{N}-{tag} {r} {dig[:12]} PIN-OK")
g64 = (med[("64","cand")] / med[("64","ctl")] - 1) * 100
lo = min(med[("64","ctl")], med.get(("64","ctl2"), 1e9))
hi = max(med[("64","ctl")], med.get(("64","ctl2"), 0))
print(f"d64 gain {g64:+.2f}% (ctl {med[('64','ctl')]} ctl2 {med.get(('64','ctl2'))} cand {med[('64','cand')]} ctl-range [{lo}, {hi}]")
if g64 < 1.0 or med[("64","cand")] <= hi:
    print("GAIN-FAIL (<+1% or inside ctl range)"); ok = False
try:
    iso = json.load(open(f"{out}/rope-norm-iso.json"))
    if iso["meta"]["fails"]:
        print("ISO-FAIL", iso["meta"]["fails"]); ok = False
    else:
        print("ISO-PASS", iso["meta"]["n_cases"], "cases")
except Exception as e:
    print("ISO-FAIL (unreadable)", repr(e)); ok = False
full = open(f"{out}/gate-full.txt").read()
if not full.startswith("GATE-FULL PASS"):
    print("GATE-FULL-FAIL"); ok = False
print("GATE-FINAL", "PASS" if ok else "FAIL")
PYEOF
cat "$OUT/gate-final.txt"
