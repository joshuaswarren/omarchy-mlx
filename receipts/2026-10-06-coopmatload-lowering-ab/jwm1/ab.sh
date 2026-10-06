#!/bin/bash
# MatmulGap H5 on jwm1 (G13G): base vs candidate ICD (jw16-built from
# 01de0431ed2 and agent/w7q-simdmat-addr 5ecb8b7), direct f16 GEMM.
set -u
W=/var/tmp/w7q-icd
cd $W
for t in base cand; do
  printf '{"file_format_version":"1.0.1","ICD":{"library_path":"%s","api_version":"1.4.362","library_arch":"64"}}\n' "$W/$t.so" > "$W/$t.json"
done
g++ -std=c++17 -O2 -pthread -o gemm-bench bench.cpp
exec 9>/tmp/m1-gpu.lock
flock -w 300 9 || { echo LOCK-TIMEOUT; exit 2; }
exec 3>/dev/cpu_dma_latency && printf '\x00\x00\x00\x00' >&3
export MESA_SHADER_CACHE_DISABLE=1
echo "start $(date -u +%FT%TZ) boot $(cut -c1-8 /proc/sys/kernel/random/boot_id) $(uname -r)"
for t in base cand; do
  VK_ICD_FILENAMES=$W/$t.json AGX_MESA_DEBUG=shaderdb ./gemm-bench --dtype f16 --mnk 128,128,128 --reps 1 --rounds 1 \
    --side d=$W/matmul_coopmat_direct.comp:64:64:0 2>&1 | grep "CS shader" | cut -c1-200 | sed "s/^/$t /"
done
for round in 1 2 3; do
  for t in base cand; do
    for o in 0 1; do
      r=$(VK_ICD_FILENAMES=$W/$t.json ./gemm-bench --dtype f16 --reps 6 --rounds 3 \
        --side d=$W/matmul_coopmat_direct.comp:64:64:$o:-DB_T=$o 2>&1)
      echo "round $round $t flags=$o $(echo "$r" | grep -o '"out_fnv":"[0-9a-f]*"') $(echo "$r" | grep -o '"tflops":[0-9.]*')"
    done
  done
done
echo "end $(date -u +%FT%TZ)"
