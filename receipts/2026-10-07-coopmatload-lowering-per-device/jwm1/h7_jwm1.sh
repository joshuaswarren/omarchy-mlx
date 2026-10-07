#!/bin/bash
# MatmulGap H7 on jwm1 (T8103 G13G): base (01de0431ed2) vs the per-device
# gated build (6e8cde53065, gate expected ON here). Run under the GPU lock.
set -u
W=/var/tmp/w7q-icd
PY=/var/tmp/w7q-h14/venv/bin/python
export MESA_SHADER_CACHE_DISABLE=1 HF_HUB_OFFLINE=1
cd "$W" || exit 2
for t in base cand2; do
  printf '{"file_format_version":"1.0.1","ICD":{"library_path":"%s","api_version":"1.4.362","library_arch":"64"}}\n' "$W/$t.so" > "$W/$t.honeykrisp_icd.json"
done
exec 9>/tmp/m1-gpu.lock
flock -w 1800 9 || { echo LOCK-TIMEOUT; exit 2; }
echo "start $(date -u +%FT%TZ) boot $(cut -c1-8 /proc/sys/kernel/random/boot_id) $(uname -r) load $(cut -d' ' -f1 /proc/loadavg)"
echo "so base $(sha256sum base.so | cut -c1-16) cand2 $(sha256sum cand2.so | cut -c1-16)"
D=$W/matmul_coopmat_direct.comp
for t in base cand2; do
  echo "shaderdb $t $(AGX_MESA_DEBUG=shaderdb VK_ICD_FILENAMES=$W/$t.honeykrisp_icd.json ./gemm-bench --dtype f16 --mnk 128,128,128 --reps 1 --rounds 1 --side d=$D:64:64:0 2>&1 | grep 'CS shader' | cut -c1-160)"
  echo "prov $t $(VK_ICD_FILENAMES=$W/$t.honeykrisp_icd.json $PY $W/mlx_provenance.py 2>&1 | grep -E 'mx_version|version_match' | tr '\n' ' ')"
done
for round in 1 2 3; do
  for t in base cand2; do
    icd=$W/$t.honeykrisp_icd.json
    for cell in "nn4096 --mnk 4096,4096,4096 --side d=$D:64:64:0" \
                "nt4096 --mnk 4096,4096,4096 --side d=$D:64:64:1:-DB_T=1" \
                "nt512 --mnk 512,4096,4096 --side d=$D:64:64:1:-DB_T=1"; do
      set -- $cell
      name=$1
      shift
      r=$(VK_ICD_FILENAMES=$icd ./gemm-bench --dtype f16 --reps 6 --rounds 3 "$@" 2>&1)
      echo "r$round $t gemm $name $(echo "$r" | grep -o '"out_fnv":"[0-9a-f]*"') $(echo "$r" | grep -o '"tflops":[0-9.]*')"
    done
  done
done
for t in base cand2; do
  VK_ICD_FILENAMES=$W/$t.honeykrisp_icd.json $PY $W/mlxcheck2.py > "$W/mlxcheck2-$t.json" 2>&1
  echo "mlxcheck2 $t $(grep -o '"tflops_4096": {[^}]*}' "$W/mlxcheck2-$t.json")"
done
$PY - "$W/mlxcheck2-base.json" "$W/mlxcheck2-cand2.json" <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
diff = [k for k in a["cases"] if a["cases"][k]["sha"] != b["cases"][k]["sha"]]
print("mlxcheck2 hash diffs:", diff, "of", len(a["cases"]))
EOF
echo "end $(date -u +%FT%TZ)"
