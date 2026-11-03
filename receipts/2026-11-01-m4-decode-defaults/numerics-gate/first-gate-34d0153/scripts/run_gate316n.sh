#!/bin/bash
# docs/numerics-gate.md for the narrowed #316 (code commit f333327, variant p1n: only
# MLX_OMARCHY_FUSED_GEMV defaults off on G16G, the trio stays on), same protocol as the first
# #316 gate (run_gate.sh arms316/perop316/analyze316): S=1, granite-4.0-1b-4bit and
# Qwen2.5-0.5B-Instruct-4bit, five fresh runs per arm interleaved by ab.py (main a69d9e6 against
# p1n f333327), the existing CPU fp32 references and captured operands (same weights, same text,
# same teacher files), per-op on release wheels plus profiling wheels dmain/dp1n for the kernel
# lists. Outputs go to new names (316n-*), so the first gate's records stay as they were.
#   run_gate316n.sh arms|perop|analyze      (each phase skips finished outputs)
set -euo pipefail
U=$HOME/gpu/upr; P=$U/probes; O=$U/gate/out; G=$U/gate
H=$HOME/.cache/huggingface/hub
mdir() { ls -d $H/models--mlx-community--$1/snapshots/*; }
GR=granite-4.0-1b-4bit; QW=Qwen2.5-0.5B-Instruct-4bit
T() { echo $O/teacher-$1.json; }
commit() { awk -v v=$1 '$1==v{print $2}' $U/commits.txt; }
arm_pairs() {  # arm_pairs <model>
  local d=$O/316n-$1 na nb need
  mkdir -p $d
  na=$(find $d -name "*-$(commit main)-*.json" | wc -l); nb=$(find $d -name "*-$(commit p1n)-*.json" | wc -l)
  need=$(( 5 - (na < nb ? na : nb) ))
  [ $need -le 0 ] && { echo "skip 316n $1: $na/$nb runs present"; return 0; }
  python3 ab.py $O/ab-316n-$1.jsonl $need main p1n -- gate_tf.py measure --model "$(mdir $1)" --teacher "$(T $1)" \
    --chunk 1 --outdir $d --label s1
}
perop_run() {  # perop_run <out.npz> <variant> <model> [AB_ENV]
  [ -s $1 ] && { echo "skip $1"; return 0; }
  AB_ENV="${4:-}" python3 once.py $O/perop/runs.jsonl $2 -- perop.py run --model "$(mdir $3)" --ops $O/perop/ops-$3.npz --kind decode --out $1
}
export AB_NOWAIT=1   # numerics runs: no idle wait (timing is not measured here)
cd $P
for m in $GR $QW; do [ -s $O/ref-$m.json ] && [ -s $O/perop/ops-$m.npz ] || { echo "missing reference or operands for $m"; exit 2; }; done
case "${1:?phase}" in
arms)
  arm_pairs $GR
  arm_pairs $QW
  ;;
perop)
  for m in $GR $QW; do
    for v in main p1n; do perop_run $O/perop/316n-$m-$v.npz $v $m; done
    for v in dmain dp1n; do
      [ -s $O/perop/316n-$m-$v.kernels.txt ] && continue
      rm -f $O/perop/316n-$m-$v.npz $O/perop/316n-$m-$v.profile.jsonl
      perop_run $O/perop/316n-$m-$v.npz $v $m "MLX_OMARCHY_GPU_PROFILE=$O/perop/316n-$m-$v.profile.jsonl"
      (cd $U/wt-$v/scripts && python3 $P/kernels_used.py $O/perop/316n-$m-$v.profile.jsonl \
        $U/wt-$v/overlay/mlx/backend/omarchy/compute.h) > $O/perop/316n-$m-$v.kernels.txt
      gzip -f $O/perop/316n-$m-$v.profile.jsonl
    done
  done
  ;;
analyze)
  R=$U/venv-ref/bin/python; D=$U/wt-main/tools/dfuse
  for m in $GR $QW; do
    $R perop.py score --model "$(mdir $m)" --ops $O/perop/ops-$m.npz --kind decode --base $O/perop/316n-$m-main.npz --cand $O/perop/316n-$m-p1n.npz --out $O/perop/score-316n-$m.json > $O/perop/score-316n-$m.txt
    $R $G/same_outputs.py $O/perop/316n-$m-main.npz $O/perop/316n-$m-dmain.npz $O/perop/316n-$m-p1n.npz $O/perop/316n-$m-dp1n.npz > $O/perop/316n-$m-release-vs-profiling.txt
    python3 $G/analyze_gate.py $O/ref-$m.json "$O/316n-$m/*-a69d9e6-*.json" "$O/316n-$m/*-f333327-*.json" $D $O/gate-316n-$m.json > $O/gate-316n-$m.txt
    b1=$(ls $O/316n-$m/*-a69d9e6-*.json | head -1); c1=$(ls $O/316n-$m/*-f333327-*.json | head -1)
    (cd $D && python3 compare_tf9.py "$b1" "$c1" > $O/compare_tf9-316n-$m.json) || true
  done
  ;;
esac
echo "phase 316n $1 done $(date -u +%FT%TZ)"
