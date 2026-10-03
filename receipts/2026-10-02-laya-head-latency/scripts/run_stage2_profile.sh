#!/bin/bash
# LayaHead2 stage 2, profile ticket (GPU): run the Laya head phases probe
# under MLX_OMARCHY_GPU_PROFILE on the freshly built diagnostics wheel in a
# fresh private venv. The profile's wall times are inflated by profiling;
# they are structure evidence, not latency numbers.
# usage: run_stage2_profile.sh OUT_DIR SERVE SUITE MODEL DIAG_WHEEL PORT
set -euo pipefail

OUT=$1; SERVE=$2; SUITE=$3; MODEL=$4; WHEEL=$5; PORT=${6:-50519}
SHARED_SP=$HOME/.local/share/mlx-omarchy/venv/lib/python3.14/site-packages
VENV=$HOME/scratch/laya2/venv-diag
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/prof"

state() {
  {
    date -u +%FT%TZ
    echo "boot $(cat /proc/sys/kernel/random/boot_id)"
    echo "uptime $(cat /proc/uptime)"
    echo "load $(cat /proc/loadavg)"
    cat /proc/pressure/cpu
  } > "$OUT/host-$1.txt"
}
state start

grep -q 'diag' <(echo "$WHEEL") || { echo "not a diagnostics wheel: $WHEEL" >&2; exit 5; }

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
VPY="$VENV/bin/python"
if ! "$VPY" -c 'import mlx' 2>/dev/null; then
  "$VPY" -m pip -q install --no-index --no-deps "$WHEEL"
fi
for pkg in numpy tokenizers; do
  if ! "$VPY" -c "import $pkg" 2>/dev/null; then
    cp -r "$SHARED_SP"/${pkg}* "$VENV/lib/python3.14/site-packages/"
  fi
done
"$VPY" -c 'import mlx.core as mx; print(mx.__version__)' > "$OUT/wheel-diag.txt" 2>&1
cat "$OUT/wheel-diag.txt"

state venv-ready

# Profiled phases run: warmups then timed calls, all inside one profile
# stream; the phases NDJSON keeps its per-call host timings for ordering.
MLX_OMARCHY_GPU_PROFILE="$OUT/prof/laya-head.jsonl" \
MLX_OMARCHY_GPU_PROFILE_LABEL=laya-head \
PYTHONPATH="$SERVE:$SCRIPTS" "$VPY" "$SCRIPTS/head_probe.py" phases \
  --serve "$SERVE" --model "$MODEL" --suite "$SUITE" --out "$OUT/phases-profiled.ndjson" \
  --n 6 --warmup 2 --arm profiled

# Same wheel, profiler off: measures the profiler's own inflation.
PYTHONPATH="$SERVE:$SCRIPTS" "$VPY" "$SCRIPTS/head_probe.py" phases \
  --serve "$SERVE" --model "$MODEL" --suite "$SUITE" --out "$OUT/phases-noprof.ndjson" \
  --n 12 --warmup 3 --arm diag-noprof

state end
ls -la "$OUT/prof/"
ps -eo pid,ppid,etime,cmd > "$OUT/ps-end.txt"
sync
echo "[stage2-profile] DONE"
