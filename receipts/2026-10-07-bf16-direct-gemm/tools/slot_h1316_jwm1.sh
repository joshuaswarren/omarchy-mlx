#!/bin/bash
# jwm1 MatmulGap H13 (b) / H14 / H15 / H16 three-arm leg, nohup with a hard end time.
# Inputs from jw16 in $H/h1316-in: the C1 and C2 wheels and omarchy_matmul_family_tests (C2).
# M = venv-h13main (already on jwm1). Driver bf16cmat01 (w7P: same release codegen as
# v3 6543eeb7df7). All GPU work under /tmp/m1-gpu.lock (H8_FLOCK=1 per round).
set -u
H=/var/tmp/w7q-h8
ICD=$HOME/.local/share/coreglass/vulkan-4b-bf16cmat01/honeykrisp_icd.aarch64.json
echo "start $(date -u +%FT%TZ) root $(df --output=pcent / | tail -1 | tr -d ' ')"
for arm in c1:fd2e384 c2:27aa673; do
  v=venv-${arm%%:*}
  whl=$(ls "$H"/h1316-in/mlx_omarchy-*"+${arm#*:}"*.whl) || exit 2
  nice -n 19 python3 -m venv "$H/$v" && nice -n 19 "$H/$v/bin/pip" install -q --force-reinstall numpy "$whl" || exit 2
  "$H/$v/bin/python" -c "import mlx.core as mx; print('$v', mx.__version__)" || exit 2
done
mkdir -p "$H/tbuild-c2/tests/omarchy" && cp "$H/h1316-in/omarchy_matmul_family_tests" "$H/tbuild-c2/tests/omarchy/" || exit 2
for r in 1 2 3 final; do
  H8_FLOCK=1 timeout -k 30 600 "$H/run_h1316.sh" "$ICD" "$r" > "$H/h1316-$r.log" 2>&1
  echo "round $r rc=$? $(date -u +%T)"
done
echo "end $(date -u +%FT%TZ)"
