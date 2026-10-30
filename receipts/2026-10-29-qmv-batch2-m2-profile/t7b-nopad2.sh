#!/usr/bin/env bash
# Does the CURRENT mlx-lm serve patch series (ssm-maskless with the `<= 0` host test) remove the slow unpadded batch decode that the older
# packaged mlx-lm shows, and does the hoisted batch kernel lose at 2 rows under it?
#   t7b-nopad2.sh m2|hostB
# Wheels: S = venv-diag-s (a736744a, shipped kernel), H = venv-diag-h (42b30a2d, hoisted kernel); diagnostics wheels, profiler only when asked.
# mlx-lm: U = the venv's own mlx_lm (the packaged system copy on the M2); F = ~/scratch/mlxlm-series/mlx_lm (pristine 0.31.3 plus
# scripts/apply-mlx-lm-patches.sh of main at 42b30a2d). Row sets: E2 (rows 0,1: unpadded) and P2 (rows 0,2: padded).
# Cells: wheel x mlx-lm x row set = 8; three rounds, the order rotates; each cell once profiled (dispatch counts) and once unprofiled
# (64 steps: step wall time and token-stream hash). GPU, no download, cap 25 min (expect 10). Apple Honeykrisp ICD only.
set -u
HOST=${1:?host}
case "$HOST" in
  m2) MODEL=$(ls -d "$HOME"/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/* | head -1) ;;
  hostB) MODEL="$HOME/scratch/model/q38-2b" ;;
  *) echo "host?"; exit 64 ;;
esac
STAMP=$(date -u +%Y%m%dT%H%M%SZ); D="$HOME/t7/results/nopad2-$HOST-$STAMP"; mkdir -p "$D"; LOG="$D/log.txt"
echo "nopad2 $HOST $STAMP load $(cut -d' ' -f1 /proc/loadavg) kernel $(uname -r) free $(df -BG --output=avail "$HOME" | tail -1)" | tee "$LOG"
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/asahi_icd.json HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1
PP_F="$HOME/scratch/mlxlm-series"
for v in venv-diag-s venv-diag-h; do "$HOME/t7/$v/bin/python" -c 'import importlib.metadata as md; print("'"$v"'", md.version("mlx_omarchy"))' | tee -a "$LOG"; done
"$HOME/t7/venv-diag-h/bin/python" -c 'import mlx_lm; print("U mlx_lm", mlx_lm.__file__)' | tee -a "$LOG"
PYTHONPATH="$PP_F" "$HOME/t7/venv-diag-h/bin/python" -c 'import mlx_lm; print("F mlx_lm", mlx_lm.__file__)' | tee -a "$LOG"
UCACHE=$("$HOME/t7/venv-diag-h/bin/python" -c 'import mlx_lm.models.cache as c; print(c.__file__)')
echo "U ssm-maskless test: $(grep -c 'all(p <= 0' "$UCACHE") match for the <= 0 form, $(grep -c 'not any(self.left_padding_list)' "$UCACHE") for the zeros-only form" | tee -a "$LOG"
run_cell() { # wheel(S|H) lm(U|F) set(E2|P2) round
  local w=$1 l=$2 s=$3 r=$4 py b rows pp=""
  case $w in S) py="$HOME/t7/venv-diag-s/bin/python" ;; H) py="$HOME/t7/venv-diag-h/bin/python" ;; esac
  case $s in E2) b=2; rows=0,1 ;; P2) b=2; rows=0,2 ;; esac
  [ "$l" = F ] && pp="$PP_F"
  local base="$D/$w-$l-$s-r$r"
  echo "-- wheel $w mlx-lm $l set $s round $r $(date -u +%T)" | tee -a "$LOG"
  PYTHONPATH="$pp" MLX_OMARCHY_GPU_PROFILE="$base.ndjson" MLX_OMARCHY_GPU_PROFILE_LABEL="$w-$l-$s-r$r" \
    timeout 420 "$py" "$HOME/t7/profile_batch_driver2.py" --model "$MODEL" --batch "$b" --rows "$rows" --warmup-tokens 8 --tokens 32 --out "$base-windows.jsonl" 2>&1 \
    | grep -E "Error|Traceback|\[prof\]" | cut -c1-200 | tee -a "$LOG"
  python3 "$HOME/t7/b2prof_summary.py" one --profile "$base.ndjson" --windows "$base-windows.jsonl" --compute-h "$HOME/t7/compute.h" --out "$base-sum.json" 2>&1 | head -3 | tee -a "$LOG"
  PYTHONPATH="$pp" timeout 420 "$py" "$HOME/t7/profile_batch_driver2.py" --model "$MODEL" --batch "$b" --rows "$rows" --warmup-tokens 8 --tokens 64 --out "$base-u-windows.jsonl" 2>&1 \
    | grep -E "Error|Traceback" | cut -c1-200 | tee -a "$LOG"
  python3 "$HOME/t7/b2prof_summary.py" wall --windows "$base-u-windows.jsonl" --out "$base-u-wall.json" 2>&1 | tee -a "$LOG"
  python3 "$HOME/t7/b2prof_summary.py" tokhash --windows "$base-u-windows.jsonl" 2>&1 | tee -a "$LOG"
}
for r in 1 2 3; do
  case $r in
    1) O="S-U-E2 H-U-E2 S-F-E2 H-F-E2 S-U-P2 H-U-P2 S-F-P2 H-F-P2" ;;
    2) O="H-F-P2 S-F-P2 H-U-P2 S-U-P2 H-F-E2 S-F-E2 H-U-E2 S-U-E2" ;;
    3) O="S-F-E2 H-F-E2 S-U-E2 H-U-E2 H-F-P2 S-F-P2 H-U-P2 S-U-P2" ;;
  esac
  for cell in $O; do IFS=- read -r w l s <<< "$cell"; run_cell "$w" "$l" "$s" "$r"; done
done
for lm in U F; do
  for s in E2 P2; do
    echo "== mlx-lm $lm, set $s: unprofiled hoisted against shipped (speed ratio shipped/hoisted)" | tee -a "$LOG"
    python3 "$HOME/t7/b2prof_summary.py" wallcmp "$D"/H-$lm-$s-r?-u-wall.json --vs "$D"/S-$lm-$s-r?-u-wall.json 2>&1 | tee -a "$LOG"
  done
done
for s in E2 P2; do
  echo "== set $s, hoisted wheel: profiled, mlx-lm F minus U" | tee -a "$LOG"
  python3 "$HOME/t7/b2prof_summary.py" cmp "$D"/H-F-$s-r?-sum.json --vs "$D"/H-U-$s-r?-sum.json 2>&1 | head -6 | tee -a "$LOG"
  echo "== set $s, hoisted wheel: unprofiled step wall, F against U (speed ratio U/F)" | tee -a "$LOG"
  python3 "$HOME/t7/b2prof_summary.py" wallcmp "$D"/H-F-$s-r?-u-wall.json --vs "$D"/H-U-$s-r?-u-wall.json 2>&1 | tee -a "$LOG"
done
echo "done $(date -u +%T)" | tee -a "$LOG"
