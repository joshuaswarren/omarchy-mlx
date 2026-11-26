#!/usr/bin/env bash
# Engagement smoke: one baseline leg + two ablated legs (short workload).
set -euo pipefail
cd ~/src/mlx-Bf16DecodeAttribution
OUT=receipts-work/2026-09-11-bf16-decode-attribution/smoke
mkdir -p "$OUT"
PY=.work/venv-bf16dec/bin/python
RUNPY=.work/venv-run-bf16dec/bin/python
ABLPY=.work/venv-ablate-bf16dec/bin/python
BASE=~/src/mlx-bf16dec-base/dist/mlx_omarchy-0.32.2.dev202609111524+a5b8c4a-cp314-cp314-linux_aarch64.whl
ABLDIST=dist/mlx_omarchy-0.32.2.dev202609111520+d2ef0db-cp314-cp314-linux_aarch64.whl
export MLX_DISABLE_COMPILE=1
export HF_HUB_OFFLINE=1
ulimit -c 0

python3 - <<PYEOF
import json
m = json.load(open("scripts/bench_matrix.json"))
m["models"] = [x for x in m["models"] if x["id"] == "qwen25-0.5b-bf16"]
m["generation"]["engine_script"] = "bench_decode_identity.py"
open("receipts-work/2026-09-11-bf16-decode-attribution/smoke/manifest-bf16.json", "w").write(json.dumps(m, indent=2) + "\n")
PYEOF

run_leg () { # name python wheel ablate
  local name=$1 py=$2 wheel=$3 ablate=${4:-}
  env -u MLX_OMARCHY_ABLATE -u MLX_OMARCHY_ABLATE_DEBUG \
    $py -m pip show mlx-omarchy >/dev/null
  local -a envargs=()
  if [ -n "$ablate" ]; then
    MLX_OMARCHY_ABLATE=$ablate MLX_OMARCHY_ABLATE_DEBUG=1 \
    $RUNPY scripts/bench_matrix.py --mode run \
      --manifest receipts-work/2026-09-11-bf16-decode-attribution/smoke/manifest-bf16.json \
      --select short-decode-32 \
      --python "$py" --wheel "$wheel" \
      --expect-pins "qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e" \
      --host-label "m1-test-host-bf16-smoke-$name" --timeout 600 \
      --out "$OUT/$name.json" > "$OUT/$name.log" 2>&1
  else
    $RUNPY scripts/bench_matrix.py --mode run \
      --manifest receipts-work/2026-09-11-bf16-decode-attribution/smoke/manifest-bf16.json \
      --select short-decode-32 \
      --python "$py" --wheel "$wheel" \
      --expect-pins "qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e" \
      --host-label "m1-test-host-bf16-smoke-$name" --timeout 600 \
      --out "$OUT/$name.json" > "$OUT/$name.log" 2>&1
  fi
  $PY -c "
import json,sys
d=json.load(open('$OUT/$name.json'))
leg=[l for l in d['legs'] if l['workload_id']=='short-decode-32'][0]
m=leg['metrics']
print('$name', leg['status'], m['decode_tok_s'], m['generated_ids_sha256_16'])
"
}

run_leg base "$RUNPY" "$BASE"
run_leg abl_all "$ABLPY" "$ABLDIST" all
run_leg abl_attn "$ABLPY" "$ABLDIST" attn
run_leg abl_cast "$ABLPY" "$ABLDIST" cast
run_leg abl_gemv "$ABLPY" "$ABLDIST" gemv
echo SMOKE_DONE
