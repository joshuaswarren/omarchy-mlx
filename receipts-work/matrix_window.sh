#!/usr/bin/env bash
# Final paired matrix: {fork,stock} x {base,cand} x 3 legs x 3 reps.
# Digest gates: base cells must reproduce the per-driver pins; cand
# cells must reproduce the SAME pins (bit-identical claim).
set -euo pipefail
cd ~/src/mlx-Bf16DecodeAttribution
OUT=receipts-work/2026-09-11-bf16-decode-attribution/matrix
mkdir -p "$OUT"
RUNPY=$PWD/.work/venv-run-bf16dec/bin/python
CANDPY=$PWD/.work/venv-cand/bin/python
BASE=$HOME/src/mlx-bf16dec-base/dist/mlx_omarchy-0.32.2.dev202609111524+a5b8c4a-cp314-cp314-linux_aarch64.whl
CAND=$1
STOCK_ICD=$HOME/stock-mesa/stock-icd.json
PIN=56d07e766edd7159fbe12ed12d9cf114bf38bf1e
export MLX_DISABLE_COMPILE=1
export HF_HUB_OFFLINE=1
ulimit -c 0

python3 - <<PYEOF
import json
m = json.load(open("scripts/bench_matrix.json"))
m["models"] = [x for x in m["models"] if x["id"] == "qwen25-0.5b-bf16"]
open("$OUT/manifest-bf16.json", "w").write(json.dumps(m, indent=2) + "\n")
PYEOF

leg_pin () { # driver workload  -> expected digest
  case "$1-$2" in
    fork-short-decode-32) echo f26175202f3dabe9 ;;
    fork-long-decode-128) echo 8690dc83246b39f8 ;;
    fork-longctx-1024-decode-32) echo ff502900d2a179a5 ;;
    stock-short-decode-32) echo 7fc0f968789b1882 ;;
    stock-long-decode-128) echo 46108ad71157cb4d ;;
    stock-longctx-1024-decode-32) echo ff502900d2a179a5 ;;
    *) echo UNKNOWN ;;
  esac
}

for rep in 1 2 3; do
  for driver in fork stock; do
    for side in base cand; do
      for wl in short-decode-32 long-decode-128 longctx-1024-decode-32; do
        stem="r$rep-$driver-$side-$wl"
        out="$OUT/$stem.json"
        if [ -e "$out" ]; then echo "$stem cached"; continue; fi
        py=$RUNPY; wheel=$BASE
        if [ "$side" = cand ]; then py=$CANDPY; wheel=$CAND; fi
        log="$OUT/$stem.log"
        if [ "$driver" = stock ]; then
          env VK_DRIVER_FILES=$STOCK_ICD \
            timeout 900 python3 scripts/bench_matrix.py --mode run \
            --manifest "$OUT/manifest-bf16.json" --select "$wl" \
            --python "$py" --wheel "$wheel" \
            --expect-pins "qwen25-0.5b-bf16=$PIN" \
            --host-label "m1-test-host-bf16dec-$stem" --timeout 840 \
            --out "$out" > "$log" 2>&1
        else
          env -u VK_DRIVER_FILES \
            timeout 900 python3 scripts/bench_matrix.py --mode run \
            --manifest "$OUT/manifest-bf16.json" --select "$wl" \
            --python "$py" --wheel "$wheel" \
            --expect-pins "qwen25-0.5b-bf16=$PIN" \
            --host-label "m1-test-host-bf16dec-$stem" --timeout 840 \
            --out "$out" > "$log" 2>&1
        fi
        python3 - "$stem" "$out" "$driver" "$wl" <<'PYEOF'
import json, sys
stem, out, driver, wl = sys.argv[1:5]
expect = {
 ("fork","short-decode-32"): "f26175202f3dabe9",
 ("fork","long-decode-128"): "8690dc83246b39f8",
 ("fork","longctx-1024-decode-32"): "ff502900d2a179a5",
 ("stock","short-decode-32"): "7fc0f968789b1882",
 ("stock","long-decode-128"): "46108ad71157cb4d",
 ("stock","longctx-1024-decode-32"): "ff502900d2a179a5",
}[(driver, wl)]
d = json.load(open(out))
leg = [l for l in d["legs"] if l["workload_id"] == wl][0]
m = leg["metrics"]
digest = m["generated_ids_sha256_16"]
assert leg["status"] == "measured", leg["status"]
assert d["clean_check"]["status"] == "clean", d["clean_check"]
assert d["binary_provenance"]["omarchy"]["verified"] == "match"
assert digest == expect, (stem, digest, expect)
print(stem, "ok", m["decode_tok_s"], digest)
PYEOF
      done
    done
  done
done
echo MATRIX_DONE
