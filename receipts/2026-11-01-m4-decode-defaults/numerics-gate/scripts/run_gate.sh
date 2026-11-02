#!/bin/bash
# docs/numerics-gate.md for #316, #317 and #321 on the M4 Mac mini, as run.
#
#   run_gate.sh facts|validate|ref [models]|arms|perop|analyze
#   run_gate.sh arms316|perop316|analyze316, arms317|perop317|analyze317, arms321|perop321|analyze321
# Every phase skips outputs that already exist, so a re-run resumes after an
# interruption: arms run only the pairs still missing (five runs per arm in all;
# the analyzer reads the first five of each arm in time order).
#
# Wheels (stamped, built on the host by build.sh from the named commits; mkvenv.sh):
#   main a69d9e6 (baseline for all three), p1 34d0153 (#316), p2 91d9f8a (#317),
#   p6 89b54a0 (#321); profiling builds dmain a69d9e6 and dp1 34d0153 for the #316
#   per-op kernel lists. Every GPU process runs through ab.py or once.py (provenance
#   gate, flock, GPU error monitor required) with AGX_APPLE9_MAX_SHARED=32768.
# Reference: stock upstream MLX 0.32.3 + mlx-cpu 0.32.3 (PyPI, sha256 checked) and stock
#   mlx-lm 0.31.3 in venv-ref; CPU stream, float32, 64-token chunks (ref_setup.sh).
# Models: the route models already on the host (not the parity-contract model):
#   granite-4.0-1b-4bit (bf16 4-bit), Qwen2.5-0.5B-Instruct-4bit, Llama-3.2-1B-Instruct-bf16.
# Text: tfref-text.txt (GPL-3, sha256 3972dc97...), ten 513-token segments per model.
# Route-sensitive pass shape per PR:
#   #316 S=1 (one token per pass: the decode GEMV and trio routes), granite and Qwen2.5-0.5B
#   #317 128-token passes (the dense bf16 linear route runs on multi-row passes), Llama bf16
#   #321 64-token passes with HK_APPLE9_COOPMAT=0 on both arms (the route runs where the
#        cooperative matrix is unavailable), granite
set -euo pipefail
U=$HOME/gpu/upr; P=$U/probes; O=$U/gate/out; G=$U/gate
H=$HOME/.cache/huggingface/hub
mdir() { ls -d $H/models--mlx-community--$1/snapshots/*; }
GR=granite-4.0-1b-4bit; QW=Qwen2.5-0.5B-Instruct-4bit; LL=Llama-3.2-1B-Instruct-bf16
T() { echo $O/teacher-$1.json; }
commit() { awk -v v=$1 '$1==v{print $2}' $U/commits.txt; }
arm_pairs() {  # arm_pairs <pr> <model> <base> <cand> <chunk> <label> [AB_ENV]
  local d=$O/$1-$2 na nb need
  mkdir -p $d
  na=$(find $d -name "*-$(commit $3)-*.json" | wc -l); nb=$(find $d -name "*-$(commit $4)-*.json" | wc -l)
  need=$(( 5 - (na < nb ? na : nb) ))
  [ $need -le 0 ] && { echo "skip $1 $2: $na/$nb runs present"; return 0; }
  AB_ENV="${7:-}" python3 ab.py $O/ab-$1-$2.jsonl $need $3 $4 -- gate_tf.py measure --model "$(mdir $2)" --teacher "$(T $2)" \
    --chunk $5 --outdir $d --label $6
}
perop_run() {  # perop_run <out.npz> <variant> <model> <kind> [AB_ENV]
  [ -s $1 ] && { echo "skip $1"; return 0; }
  AB_ENV="${5:-}" python3 once.py $O/perop/runs.jsonl $2 -- perop.py run --model "$(mdir $3)" --ops $O/perop/ops-$3.npz --kind $4 --out $1
}
export AB_NOWAIT=1   # numerics runs: no idle wait (timing is not measured here)
cd $P

case "${1:?phase}" in
facts)
  for m in $GR $QW $LL; do (cd "$(mdir $m)" && for f in $(ls | sort); do echo "$m $(sha256sum "$(readlink -f "$f")" | cut -d' ' -f1) $f"; done); done > $O/model-sha256.txt
  for v in main p1 p2 p6 dmain dp1; do echo "$v $(cd $U/dist/$v && sha256sum mlx_omarchy-*.whl)"; done > $O/wheels.txt
  cp $U/commits.txt $O/commits.txt
  bash $P/facts.sh > $O/host.txt 2>&1 || true
  journalctl -k -b --no-pager -q > /dev/null && echo "journalctl -k readable" >> $O/host.txt
  ;;
validate)
  # Harness check on the baseline wheel before any arm: S=1 PPL vs one-pass PPL.
  for m in $GR $QW $LL; do
    python3 once.py $O/validate.jsonl main -- gate_tf.py validate --model "$(mdir $m)" --teacher "$(T $m)" --out $O/validate-$m.json
  done
  AB_ENV="HK_APPLE9_COOPMAT=0" python3 once.py $O/validate.jsonl main -- gate_tf.py validate --model "$(mdir $GR)" --teacher "$(T $GR)" --out $O/validate-$GR-coopmat-hidden.json
  ;;
ref)
  for m in ${2:-$GR $QW $LL}; do
    par=5; [ $m = $LL ] && par=3   # memory: about 1.2 GB per quantized-model part, 3.5 GB per Llama fp32 part
    mkdir -p $O/ref-parts
    [ -s $O/ref-$m.json ] && { echo "skip ref $m"; continue; }
    for i in $(seq 0 9); do [ -s $O/ref-parts/$m-$i.json ] || echo $i; done | xargs -r -P $par -I{} \
      $U/venv-ref/bin/python gate_tf.py measure --cpu --model "$(mdir $m)" --teacher "$(T $m)" \
      --chunk 64 --only {} --out $O/ref-parts/$m-{}.json
    $U/venv-ref/bin/python gate_tf.py merge --model "$(mdir $m)" --parts $O/ref-parts/$m-?.json --out $O/ref-$m.json
  done
  ;;
arms)
  mkdir -p $O/316-$GR $O/316-$QW $O/317-$LL $O/321-$GR
  python3 ab.py $O/ab-316-$GR.jsonl 5 main p1 -- gate_tf.py measure --model "$(mdir $GR)" --teacher "$(T $GR)" --chunk 1 --outdir $O/316-$GR --label s1
  python3 ab.py $O/ab-316-$QW.jsonl 5 main p1 -- gate_tf.py measure --model "$(mdir $QW)" --teacher "$(T $QW)" --chunk 1 --outdir $O/316-$QW --label s1
  python3 ab.py $O/ab-317-$LL.jsonl 5 main p2 -- gate_tf.py measure --model "$(mdir $LL)" --teacher "$(T $LL)" --chunk 128 --outdir $O/317-$LL --label c128
  AB_ENV="HK_APPLE9_COOPMAT=0" python3 ab.py $O/ab-321-$GR.jsonl 5 main p6 -- gate_tf.py measure --model "$(mdir $GR)" --teacher "$(T $GR)" --chunk 64 --outdir $O/321-$GR --label c64
  ;;
arms316)
  arm_pairs 316 $GR main p1 1 s1
  arm_pairs 316 $QW main p1 1 s1
  ;;
perop316)
  mkdir -p $O/perop
  for m in $GR $QW; do
    [ -s $O/perop/ops-$m.npz ] || $U/venv-ref/bin/python perop.py capture --model "$(mdir $m)" --teacher "$(T $m)" --out $O/perop/ops-$m.npz
    for v in main p1; do perop_run $O/perop/316-$m-$v.npz $v $m decode; done
    for v in dmain dp1; do
      [ -s $O/perop/316-$m-$v.kernels.txt ] && continue
      rm -f $O/perop/316-$m-$v.npz $O/perop/316-$m-$v.profile.jsonl
      perop_run $O/perop/316-$m-$v.npz $v $m decode "MLX_OMARCHY_GPU_PROFILE=$O/perop/316-$m-$v.profile.jsonl"
      (cd $U/wt-$v/scripts && python3 $P/kernels_used.py $O/perop/316-$m-$v.profile.jsonl \
        $U/wt-$v/overlay/mlx/backend/omarchy/compute.h) > $O/perop/316-$m-$v.kernels.txt
      gzip -f $O/perop/316-$m-$v.profile.jsonl
    done
  done
  ;;
analyze316)
  R=$U/venv-ref/bin/python; D=$U/wt-main/tools/dfuse
  for m in $GR $QW; do
    $R perop.py score --model "$(mdir $m)" --ops $O/perop/ops-$m.npz --kind decode --base $O/perop/316-$m-main.npz --cand $O/perop/316-$m-p1.npz --out $O/perop/score-316-$m.json > $O/perop/score-316-$m.txt
    $R $G/same_outputs.py $O/perop/316-$m-main.npz $O/perop/316-$m-dmain.npz $O/perop/316-$m-p1.npz $O/perop/316-$m-dp1.npz > $O/perop/316-$m-release-vs-profiling.txt
    python3 $G/analyze_gate.py $O/ref-$m.json "$O/316-$m/*-a69d9e6-*.json" "$O/316-$m/*-34d0153-*.json" $D $O/gate-316-$m.json > $O/gate-316-$m.txt
    b1=$(ls $O/316-$m/*-a69d9e6-*.json | head -1); c1=$(ls $O/316-$m/*-34d0153-*.json | head -1)
    (cd $D && python3 compare_tf9.py "$b1" "$c1" > $O/compare_tf9-316-$m.json) || true
  done
  ;;
arms317)
  arm_pairs 317 $LL main p2 128 c128
  ;;
arms321)
  arm_pairs 321 $GR main p6 64 c64 "HK_APPLE9_COOPMAT=0"
  ;;
perop317)
  mkdir -p $O/perop
  [ -s $O/perop/ops-$LL.npz ] || $U/venv-ref/bin/python perop.py capture --model "$(mdir $LL)" --teacher "$(T $LL)" --out $O/perop/ops-$LL.npz
  for v in main p2; do perop_run $O/perop/317-$LL-$v.npz $v $LL linear; done
  ;;
perop321)
  mkdir -p $O/perop
  [ -s $O/perop/ops-$GR.npz ] || $U/venv-ref/bin/python perop.py capture --model "$(mdir $GR)" --teacher "$(T $GR)" --out $O/perop/ops-$GR.npz
  for v in main p6; do perop_run $O/perop/321-$GR-$v.npz $v $GR qmm "HK_APPLE9_COOPMAT=0"; done
  ;;
analyze317|analyze321)
  R=$U/venv-ref/bin/python; D=$U/wt-main/tools/dfuse
  if [ $1 = analyze317 ]; then pr=317; m=$LL; c=91d9f8a; k=linear; v=p2; else pr=321; m=$GR; c=89b54a0; k=qmm; v=p6; fi
  $R perop.py score --model "$(mdir $m)" --ops $O/perop/ops-$m.npz --kind $k --base $O/perop/$pr-$m-main.npz --cand $O/perop/$pr-$m-$v.npz --out $O/perop/score-$pr-$m.json > $O/perop/score-$pr-$m.txt
  python3 $G/analyze_gate.py $O/ref-$m.json "$O/$pr-$m/*-a69d9e6-*.json" "$O/$pr-$m/*-$c-*.json" $D $O/gate-$pr-$m.json > $O/gate-$pr-$m.txt
  b1=$(ls $O/$pr-$m/*-a69d9e6-*.json | head -1); c1=$(ls $O/$pr-$m/*-$c-*.json | head -1)
  (cd $D && python3 compare_tf9.py "$b1" "$c1" > $O/compare_tf9-$pr-$m.json) || true
  ;;
perop)
  mkdir -p $O/perop
  for m in $GR $QW $LL; do
    [ -s $O/perop/ops-$m.npz ] || $U/venv-ref/bin/python perop.py capture --model "$(mdir $m)" --teacher "$(T $m)" --out $O/perop/ops-$m.npz
  done
  for m in $GR $QW; do
    for v in main p1; do
      python3 once.py $O/perop/runs.jsonl $v -- perop.py run --model "$(mdir $m)" --ops $O/perop/ops-$m.npz --kind decode --out $O/perop/316-$m-$v.npz
    done
    for v in dmain dp1; do
      rm -f $O/perop/316-$m-$v.profile.jsonl
      AB_ENV="MLX_OMARCHY_GPU_PROFILE=$O/perop/316-$m-$v.profile.jsonl" python3 once.py $O/perop/runs.jsonl $v -- \
        perop.py run --model "$(mdir $m)" --ops $O/perop/ops-$m.npz --kind decode --out $O/perop/316-$m-$v.npz
      (cd $U/wt-$v/scripts && python3 $P/kernels_used.py $O/perop/316-$m-$v.profile.jsonl \
        $U/wt-$v/overlay/mlx/backend/omarchy/compute.h) > $O/perop/316-$m-$v.kernels.txt
      gzip -f $O/perop/316-$m-$v.profile.jsonl
    done
  done
  for v in main p2; do
    python3 once.py $O/perop/runs.jsonl $v -- perop.py run --model "$(mdir $LL)" --ops $O/perop/ops-$LL.npz --kind linear --out $O/perop/317-$LL-$v.npz
  done
  for v in main p6; do
    AB_ENV="HK_APPLE9_COOPMAT=0" python3 once.py $O/perop/runs.jsonl $v -- perop.py run --model "$(mdir $GR)" --ops $O/perop/ops-$GR.npz --kind qmm --out $O/perop/321-$GR-$v.npz
  done
  ;;
analyze)
  R=$U/venv-ref/bin/python; D=$U/wt-main/tools/dfuse
  for m in $GR $QW; do
    $R perop.py score --model "$(mdir $m)" --ops $O/perop/ops-$m.npz --kind decode --base $O/perop/316-$m-main.npz --cand $O/perop/316-$m-p1.npz --out $O/perop/score-316-$m.json > $O/perop/score-316-$m.txt
    $R $G/same_outputs.py $O/perop/316-$m-main.npz $O/perop/316-$m-dmain.npz $O/perop/316-$m-p1.npz $O/perop/316-$m-dp1.npz > $O/perop/316-$m-release-vs-profiling.txt
  done
  $R perop.py score --model "$(mdir $LL)" --ops $O/perop/ops-$LL.npz --kind linear --base $O/perop/317-$LL-main.npz --cand $O/perop/317-$LL-p2.npz --out $O/perop/score-317-$LL.json > $O/perop/score-317-$LL.txt
  $R perop.py score --model "$(mdir $GR)" --ops $O/perop/ops-$GR.npz --kind qmm --base $O/perop/321-$GR-main.npz --cand $O/perop/321-$GR-p6.npz --out $O/perop/score-321-$GR.json > $O/perop/score-321-$GR.txt
  for spec in 316:$GR:a69d9e6:34d0153 316:$QW:a69d9e6:34d0153 317:$LL:a69d9e6:91d9f8a 321:$GR:a69d9e6:89b54a0; do
    IFS=: read pr m b c <<< "$spec"
    python3 $G/analyze_gate.py $O/ref-$m.json "$O/$pr-$m/*-$b-*.json" "$O/$pr-$m/*-$c-*.json" $D $O/gate-$pr-$m.json > $O/gate-$pr-$m.txt
    b1=$(ls $O/$pr-$m/*-$b-*.json | head -1); c1=$(ls $O/$pr-$m/*-$c-*.json | head -1)
    (cd $D && python3 compare_tf9.py "$b1" "$c1" > $O/compare_tf9-$pr-$m.json) || true
  done
  ;;
esac
echo "phase $1 done $(date -u +%FT%TZ)"
