#!/bin/bash
# DecodeFuse4 F4 gate window: qk-norm epilogue candidate (env-gated) vs serving.
# INVOCATION: bash /var/tmp/appbar/gpuwin.sh 'bash /var/tmp/fuse4/gate.sh'
set -u
OUT=/var/tmp/fuse4/gate; mkdir -p "$OUT"
SERVE=/var/tmp/v072-venv-fused/bin/python3
CAND=/var/tmp/fuse4-venv/bin/python3
BENCH=$HOME/bench-scripts/qwen38-mlx-bench.py
MODEL=$(echo ~/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98b*)
{ date -u +%FT%TZ; cat /proc/sys/kernel/random/boot_id
  for py in "$SERVE" "$CAND"; do "$py" -c 'import mlx.core as mx; print(mx.__version__)'; done
  "$CAND" -c 'import mlx.core as mx; a=mx.fast.gdn_conv_update; print("qknorm-arg:", "qk_key_dim" in a.__doc__ if a.__doc__ else "probe")'
} > "$OUT/identity.txt"

# 1) fused-primitive iso check in the candidate venv: fuseable legs
#    bit-identical, fenced legs must raise, fallback leg exact.
env X=1 "$CAND" /var/tmp/fuse4/dg_bitcheck_qknorm.py "$OUT/qknorm-iso.json" \
  > "$OUT/qknorm-iso.log" 2>&1
echo "iso rc=$? $(head -1 "$OUT/qknorm-iso.log")"

# 2) shared-kernel battery (whole-model raw-output rows) serve vs candidate.
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

# 3) paired cells, interleaved. cand arm runs the F4 flag only (F3 stays
#    off here so the window isolates F4); a d64 kill-switch run (no flags)
#    must be inert, plus a combined-arm d64 preview (both flags).
cell () { local N=$1 tag=$2 py=$3; shift 3
  env X=1 "$@" "$py" "$BENCH" --model "$MODEL"/ \
    --prompts "$HOME/bench-scripts/qwen38-2b-prompts.jsonl" \
    --limit 1 --new-tokens "$N" --warmup 1 --passes 5 --prefill-tokens 512 \
    --label "fuse4-$N-$tag" --out "$OUT/d$N-$tag.json" > "$OUT/d$N-$tag.log" 2>&1
  "$SERVE" -c "
import json
try:
  d=json.load(open('$OUT/d$N-$tag.json'))
  print('d$N-$tag', round(d['decode_tok_rate']['median'],2), d['ordered_records_sha256'][:12])
except Exception as e:
  print('d$N-$tag', 'PARSE-FAIL', repr(e))"; }
for N in 64 128 256 512; do
  cell $N ctl "$SERVE"
  cell $N cand "$CAND" MLX_OMARCHY_GDN_QKNORM_FUSE=1
done
cell 64 ctl2 "$SERVE"
cell 64 ksoff "$CAND"   # kill switch: no flags -> exact eager chain
cell 64 combo "$CAND" MLX_OMARCHY_GDN_QKNORM_FUSE=1 MLX_OMARCHY_ROPE_NORM_FUSE=1
( cd "$OUT" && sha256sum ./* > SHA256SUMS 2>/dev/null )

python3 - "$OUT" > "$OUT/gate-final.txt" 2>&1 <<'PYEOF'
import json, sys
out = sys.argv[1]
PINS = {"64": "c84b3e7a", "128": "07c515e0", "256": "c6aabbf0", "512": "5c120987"}
ok = True
med = {}
for N, pin in PINS.items():
    for tag in ("ctl", "cand", "ctl2", "ksoff", "combo"):
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
print(f"d64 F4-only gain {g64:+.2f}% (ctl {med[('64','ctl')]} ctl2 {med.get(('64','ctl2'))} cand {med[('64','cand')]} ctl-range [{lo}, {hi}])")
if med[("64","cand")] <= hi:
    print("F4-GAIN-NOTE: inside ctl range at d64")
if med[("64","combo")] <= hi:
    print("COMBO-GAIN-NOTE: inside ctl range at d64")
try:
    iso = json.load(open(f"{out}/qknorm-iso.json"))
    if iso["meta"]["fails"]:
        print("ISO-FAIL", iso["meta"]["fails"]); ok = False
    else:
        print("ISO-PASS rows:", len(iso["rows"]))
except Exception as e:
    print("ISO-FAIL (unreadable)", repr(e)); ok = False
full = open(f"{out}/gate-full.txt").read()
if not full.startswith("GATE-FULL PASS"):
    print("GATE-FULL-FAIL"); ok = False
print("GATE-FINAL", "PASS" if ok else "FAIL")
PYEOF
cat "$OUT/gate-final.txt"
