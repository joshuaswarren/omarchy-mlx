#!/usr/bin/env bash
# 14-inch M2 Max: per-kernel GPU profile of batch decode, shipped (venv-diag-s, a736744a) against hoisted (venv-diag-h, 42b30a2d), for four row sets:
#   E2 = rows 0,1   (2 prompts of 12 tokens: unpadded)      P2 = rows 0,2   (12 and 13 tokens: left padded)
#   E3 = rows 0,1,4 (3 prompts of 12 tokens: unpadded)      P3 = rows 0,1,2 (12, 12, 13 tokens: left padded)
# Each arm its own process, 3 rounds, order rotates; raw profiles are kept. GPU, no download, cap 25 min (expect 12). Apple Honeykrisp ICD only.
#   t7b-b2prof2.sh m2
set -u
HOST=${1:?host}
[ "$HOST" = m2 ] || { echo "m2 only"; exit 64; }
MODEL=$(ls -d "$HOME"/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/* | head -1)
STAMP=$(date -u +%Y%m%dT%H%M%SZ); D="$HOME/t7/results/b2prof2-$STAMP"; mkdir -p "$D"; LOG="$D/log.txt"
echo "b2prof2 $HOST $STAMP load $(cut -d' ' -f1 /proc/loadavg) kernel $(uname -r) free $(df -BG --output=avail "$HOME" | tail -1)" | tee "$LOG"
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/asahi_icd.json HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1
for v in venv-diag-s venv-diag-h; do "$HOME/t7/$v/bin/python" -c 'import importlib.metadata as md; print("'"$v"'", md.version("mlx_omarchy"))' | tee -a "$LOG"; done
run_arm() { # arm(S|H) set(E2|P2|E3|P3) round
  local a=$1 s=$2 r=$3 py b rows
  case $a in S) py="$HOME/t7/venv-diag-s/bin/python" ;; H) py="$HOME/t7/venv-diag-h/bin/python" ;; esac
  case $s in E2) b=2; rows=0,1 ;; P2) b=2; rows=0,2 ;; E3) b=3; rows=0,1,4 ;; P3) b=3; rows=0,1,2 ;; esac
  local base="$D/$a-$s-r$r"
  echo "-- $a $s round $r $(date -u +%T)" | tee -a "$LOG"
  MLX_OMARCHY_GPU_PROFILE="$base.ndjson" MLX_OMARCHY_GPU_PROFILE_LABEL="$a-$s-r$r" \
    timeout 420 "$py" "$HOME/t7/profile_batch_driver2.py" --model "$MODEL" --batch "$b" --rows "$rows" --warmup-tokens 8 --tokens 32 --out "$base-windows.jsonl" 2>&1 \
    | grep -E "Error|Traceback|\[prof\]" | cut -c1-200 | tee -a "$LOG"
  python3 "$HOME/t7/b2prof_summary.py" one --profile "$base.ndjson" --windows "$base-windows.jsonl" --compute-h "$HOME/t7/compute.h" --out "$base-sum.json" 2>&1 | head -8 | tee -a "$LOG"
  timeout 420 "$py" "$HOME/t7/profile_batch_driver2.py" --model "$MODEL" --batch "$b" --rows "$rows" --warmup-tokens 8 --tokens 64 --out "$base-u-windows.jsonl" 2>&1 \
    | grep -E "Error|Traceback" | cut -c1-200 | tee -a "$LOG"
  python3 "$HOME/t7/b2prof_summary.py" wall --windows "$base-u-windows.jsonl" --out "$base-u-wall.json" 2>&1 | tee -a "$LOG"
}
for r in 1 2 3; do
  case $r in 1) O="S-E2 H-E2 S-P2 H-P2 S-E3 H-E3 S-P3 H-P3" ;; 2) O="H-P3 S-P3 H-E3 S-E3 H-P2 S-P2 H-E2 S-E2" ;; 3) O="S-P2 H-P2 S-E2 H-E2 H-P3 S-P3 H-E3 S-E3" ;; esac
  for cell in $O; do run_arm "${cell%-*}" "${cell#*-}" "$r"; done
done
for s in E2 P2 E3 P3; do
  echo "== compare $s: hoisted minus shipped" | tee -a "$LOG"
  python3 "$HOME/t7/b2prof_summary.py" cmp "$D"/H-$s-r?-sum.json --vs "$D"/S-$s-r?-sum.json 2>&1 | tee -a "$LOG"
  echo "== unprofiled $s" | tee -a "$LOG"
  python3 "$HOME/t7/b2prof_summary.py" wallcmp "$D"/H-$s-r?-u-wall.json --vs "$D"/S-$s-r?-u-wall.json 2>&1 | tee -a "$LOG"
done
echo "== compare hoisted: E2 minus P2" | tee -a "$LOG"
python3 "$HOME/t7/b2prof_summary.py" cmp "$D"/H-E2-r?-sum.json --vs "$D"/H-P2-r?-sum.json 2>&1 | tee -a "$LOG"
echo "== compare hoisted: E3 minus P3" | tee -a "$LOG"
python3 "$HOME/t7/b2prof_summary.py" cmp "$D"/H-E3-r?-sum.json --vs "$D"/H-P3-r?-sum.json 2>&1 | tee -a "$LOG"
echo "done $(date -u +%T)" | tee -a "$LOG"
