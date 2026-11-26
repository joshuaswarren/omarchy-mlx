#!/bin/bash
# usage: run_subcap.sh <label> : submission-budget sweep with the diagnostics build (profiling harness ON)
set -u
D=$HOME/mlxpr; cd $D; mkdir -p subcap
export HF_HOME=$D/hf HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
if [ ! -x venv-diag/bin/python ]; then
  python3.14 -m venv venv-diag && venv-diag/bin/pip -q install ./mlx_omarchy-*diag*.whl && venv-diag/bin/pip -q install --no-deps mlx-lm==0.31.3 && venv-diag/bin/pip -q install "transformers[sentencepiece]==5.16.1" numpy protobuf pyyaml jinja2 huggingface_hub && bash ./install.sh --patch-venv $D/venv-diag >/dev/null 2>&1
fi
P=$(ls -d venv-diag/lib/python3.14/site-packages/mlx); [ -e $P/lib/libmlx.so ] || ln -s ../lib64/libmlx.so $P/lib/libmlx.so
venv-diag/bin/python -c "import mlx.core as mx; print(mx.__version__, mx.default_device())" || exit 3
for m in SiddhJagani/Qwen3.8-2B-mlx-4Bit mlx-community/Qwen3-4B-Instruct-2507-4bit; do
  k=$(basename $m); snap=$(ls -d $HF_HOME/hub/models--${m/\//--}/snapshots/*/ | head -1)
  echo "SUBCAP $1 $k start $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg) temp $(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)"
  timeout 1200 venv-diag/bin/python subcap_calibrate.py --model $snap --tokens 64 --prefill-tokens 512 \
    --budgets 5000,8000,10000,15000,20000,40000 --profile $D/subcap/$1-$k.jsonl --venv-python $D/venv-diag/bin/python > subcap/$1-$k.out 2>&1
  echo "rc=$? end $(date -u +%T)"; rm -f $D/subcap/$1-$k.jsonl*
  tail -n 25 subcap/$1-$k.out
done
