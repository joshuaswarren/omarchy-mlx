#!/bin/bash
# MatmulGap H6b on jw16 (inside gpu-turn): the MLX-level cells. MLX accepts a
# user ICD only when its path names Honeykrisp, so the JSONs are copied to
# *.honeykrisp_icd.json first.
set -u
B=/var/tmp/w7q-mesa-build
S=/var/tmp/w7q-venv-stage
PY=/var/tmp/w7q-venv/bin/python
MODEL=$(find "$HOME/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots" -mindepth 1 -maxdepth 1 | head -1)
export MESA_SHADER_CACHE_DISABLE=1 HF_HUB_OFFLINE=1
for t in base cand; do cp "$B/$t.icd.json" "$B/$t.honeykrisp_icd.json"; done
echo "start $(date -u +%FT%TZ) boot $(cut -c1-8 /proc/sys/kernel/random/boot_id) load $(cut -d' ' -f1 /proc/loadavg)"
for t in base cand; do
  echo "prov $t $(VK_ICD_FILENAMES=$B/$t.honeykrisp_icd.json $PY $S/mlx_provenance.py 2>&1 | tail -3 | tr '\n' ' ' | cut -c1-300)"
done
for round in 1 2 3; do
  for t in base cand; do
    echo "r$round $t lm $(VK_ICD_FILENAMES=$B/$t.honeykrisp_icd.json $PY $S/lm_cell.py "$MODEL" 2>&1 | tail -1)"
  done
done
for t in base cand; do
  VK_ICD_FILENAMES=$B/$t.honeykrisp_icd.json $PY $S/mlxcheck2.py > "$S/mlxcheck2-$t.json" 2>&1
  echo "mlxcheck2 $t $(grep -o '"tflops_4096": {[^}]*}' "$S/mlxcheck2-$t.json")"
done
$PY - "$S/mlxcheck2-base.json" "$S/mlxcheck2-cand.json" <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
diff = [k for k in a["cases"] if a["cases"][k]["sha"] != b["cases"][k]["sha"]]
print("mlxcheck2 hash diffs:", diff, "of", len(a["cases"]))
EOF
echo "end $(date -u +%FT%TZ)"
