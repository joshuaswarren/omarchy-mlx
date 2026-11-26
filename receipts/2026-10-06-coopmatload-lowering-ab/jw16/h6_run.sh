#!/bin/bash
# MatmulGap H6 on jw16 (run inside: ~/bin/gpu-turn -m 12 -- h6_run.sh).
# base vs cand ICD, alternated process by process.
set -u
B=/var/tmp/w7q-mesa-build
G=/var/tmp/w7q-gemm
S=/var/tmp/w7q-venv-stage
PY=/var/tmp/w7q-venv/bin/python
MODEL=$(ls -d $HOME/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/* | head -1)
export MESA_SHADER_CACHE_DISABLE=1 HF_HUB_OFFLINE=1
echo "start $(date -u +%FT%TZ) boot $(cut -c1-8 /proc/sys/kernel/random/boot_id) $(uname -r) load $(cut -d' ' -f1 /proc/loadavg)"
for t in base cand; do
  echo "prov $t $(VK_ICD_FILENAMES=$B/$t.icd.json $PY $S/mlx_provenance.py 2>&1 | tail -1 | cut -c1-200)"
done
D=$G/sh/matmul_coopmat_direct.comp
for round in 1 2 3; do
  for t in base cand; do
    icd=$B/$t.icd.json
    for cell in "nn4096 --mnk 4096,4096,4096 --side d=$D:64:64:0" \
                "nt4096 --mnk 4096,4096,4096 --side d=$D:64:64:1:-DB_T=1" \
                "nt512 --mnk 512,4096,4096 --side d=$D:64:64:1:-DB_T=1"; do
      set -- $cell
      name=$1
      shift
      r=$(VK_ICD_FILENAMES=$icd $G/gemm-bench --dtype f16 --reps 6 --rounds 3 "$@" 2>&1)
      echo "r$round $t gemm $name $(echo "$r" | grep -o '"out_fnv":"[0-9a-f]*"') $(echo "$r" | grep -o '"tflops":[0-9.]*')"
    done
    echo "r$round $t lm $(VK_ICD_FILENAMES=$icd $PY $S/lm_cell.py "$MODEL" 2>&1 | tail -1)"
  done
done
for t in base cand; do
  VK_ICD_FILENAMES=$B/$t.icd.json $PY $S/mlxcheck2.py > $S/mlxcheck2-$t.json 2>&1
  echo "mlxcheck2 $t $(grep -o '"tflops_4096": {[^}]*}' $S/mlxcheck2-$t.json)"
done
echo "end $(date -u +%FT%TZ)"
