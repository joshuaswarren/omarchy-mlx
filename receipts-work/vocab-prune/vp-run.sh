#!/usr/bin/env bash
# VocabPrune jwm1 window: venvs (no lock), then under /tmp/m1-gpu.lock:
# qualification (3 modes), 3 interleaved contract reps ctl/cand, logits gate.
# usage: vp-run.sh <ctl-wheel> <cand-wheel> <outdir>
set -euo pipefail
CTL_WHL="$1"; CAND_WHL="$2"; OUT="$3"
BENCH="$HOME/bench-scripts/qwen38-mlx-bench.py"
PROMPTS="$HOME/bench-scripts/qwen38-2b-prompts.jsonl"
MODEL="$(echo ~/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/*/)"
GATE=/var/tmp/rmsprol-venv/logits-gate/logits_gate_v4.py
VP=/var/tmp/vp
mkdir -p "$OUT"

mkvenv() {  # mkvenv <venv> <wheel> [greedy]
  local v="$1" w="$2"
  if [[ ! -d "$v" ]]; then
    python3 -m venv --system-site-packages "$v"
    "$v/bin/pip" -q install "$w" mlx-lm==0.31.3
    python3 /var/tmp/dg/scripts/patch-mlx-lm-gdn.py "$v"
    python3 /var/tmp/dg/scripts/patch-mlx-lm-gdn-raw.py "$v"
    if [[ "${3:-}" == greedy ]]; then
      site="$(dirname "$(ls -d "$v"/lib/python3.*/site-packages/mlx_lm)")"
      patch --directory="$site" --strip=1 --forward --fuzz=0 < "$VP/mlx-lm-greedy-prune.patch"
    fi
  fi
  "$v/bin/python" -c "import importlib.metadata as m; print(m.version('mlx-omarchy'), m.version('mlx-lm'))"
}
mkvenv "$VP/venv-ctl" "$CTL_WHL" | tee "$OUT/ctl-version.txt"
mkvenv "$VP/venv-cand" "$CAND_WHL" greedy | tee "$OUT/cand-version.txt"
sha256sum "$CTL_WHL" "$CAND_WHL" | tee "$OUT/wheels.sha256"

exec 9>/tmp/m1-gpu.lock
flock 9
echo "lock held $(date -Is)" | tee "$OUT/lock.txt"

for mode in prune keep full; do
  if [[ $mode == prune ]]; then envs=(); else envs=(MLX_OMARCHY_GREEDY_PRUNE_TEST=$mode); fi
  env "${envs[@]}" "$VP/venv-cand/bin/python" "$VP/greedy_qual.py" "$MODEL" "$PROMPTS" \
    "$OUT/qual-$mode.json" 2>"$OUT/qual-$mode.err" | tee "$OUT/qual-$mode.txt"
done

runarm() {  # runarm <venv> <tag> <rep> [VAR=value ...]
  env "${@:4}" MLX_COMMIT_TAG="$2" "$1/bin/python" "$BENCH" --model "$MODEL" --prompts "$PROMPTS" \
    --limit 10 --warmup 2 --passes 3 --prefill-tokens 512 --label "$2-r$3" \
    --out "$OUT/contract-$2-r$3.json" 2>&1 | grep -E "ordered_records|pure_prefill" | tee -a "$OUT/contract.log"
}
for rep in 1 2 3; do
  runarm "$VP/venv-ctl" ctl "$rep"
  runarm "$VP/venv-cand" cand "$rep"
done
# Kill switch arm: candidate wheel + patch, MLX_OMARCHY_NO_GREEDY_PRUNE=1.
runarm "$VP/venv-cand" candoff 1 MLX_OMARCHY_NO_GREEDY_PRUNE=1

"$VP/venv-ctl/bin/python" "$GATE" --model "$MODEL" --prompts "$PROMPTS" --out "$OUT/logits-ctl.json" 2>"$OUT/logits-ctl.err"
"$VP/venv-cand/bin/python" "$GATE" --model "$MODEL" --prompts "$PROMPTS" --compare "$OUT/logits-ctl.json" \
  --out "$OUT/logits-cand.json" 2>"$OUT/logits-cand.err" | tee "$OUT/logits-compare.txt"
flock -u 9
echo "released $(date -Is)" | tee -a "$OUT/lock.txt"
