#!/bin/bash
# Route traces for a gate's per-op cells (#317 or #321): the per-op run of check 1 repeated on
# profiling builds (dmain a69d9e6 and the change's dp2 91d9f8a / dp6 89b54a0;
# -DMLX_OMARCHY_GPU_PROFILING=ON) with the same environment as the gate arms, the kernels each
# build dispatched, and a check that each profiling build's outputs equal its release build's.
# Not one of the three checks; it identifies the kernels the checks ran.   usage: run_gatek.sh 317|321
set -euo pipefail
U=$HOME/gpu/upr; P=$U/probes; O=$U/gate/out; G=$U/gate
H=$HOME/.cache/huggingface/hub
mdir() { ls -d $H/models--mlx-community--$1/snapshots/*; }
case "${1:?pr}" in
317) pr=317; m=Llama-3.2-1B-Instruct-bf16; kind=linear; rel=p2; dv=dp2; env="" ;;
321) pr=321; m=granite-4.0-1b-4bit; kind=qmm; rel=p6; dv=dp6; env="HK_APPLE9_COOPMAT=0" ;;
esac
export AB_NOWAIT=1
cd $P
for v in dmain $dv; do
  [ -s $O/perop/$pr-$m-$v.kernels.txt ] && continue
  rm -f $O/perop/$pr-$m-$v.npz $O/perop/$pr-$m-$v.profile.jsonl
  AB_ENV="$env MLX_OMARCHY_GPU_PROFILE=$O/perop/$pr-$m-$v.profile.jsonl" python3 once.py $O/perop/runs.jsonl $v -- \
    perop.py run --model "$(mdir $m)" --ops $O/perop/ops-$m.npz --kind $kind --out $O/perop/$pr-$m-$v.npz
  (cd $U/wt-$v/scripts && python3 $P/kernels_used.py $O/perop/$pr-$m-$v.profile.jsonl \
    $U/wt-$v/overlay/mlx/backend/omarchy/compute.h) > $O/perop/$pr-$m-$v.kernels.txt
  gzip -f $O/perop/$pr-$m-$v.profile.jsonl
done
$U/venv-ref/bin/python $G/same_outputs.py $O/perop/$pr-$m-main.npz $O/perop/$pr-$m-dmain.npz $O/perop/$pr-$m-$rel.npz $O/perop/$pr-$m-$dv.npz > $O/perop/$pr-$m-release-vs-profiling.txt
echo "phase ${pr}k done $(date -u +%FT%TZ)"
