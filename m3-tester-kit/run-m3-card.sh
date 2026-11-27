#!/usr/bin/env bash
# m3-tester-kit r1 — one-command PAR-45 GPU card runner (DEFER_TOKENS=0).
#
# Builds .card-venv, installs the bundled mlx-omarchy wheel plus mlx-lm,
# applies the vendored mlx-lm patch series and the deferral check, then
# runs the PAR-45 card cell (one discarded warm-up + five timed reps,
# greedy, output-digest gated) for every model x context pair through
# wheel_card_row.py. A CPU/GPU power-state snapshot is captured before
# and after each cell into receipts/<UTC>/power/.
#
# Usage:
#   bash run-m3-card.sh                  # models 2b 4b 9b x ctx 512 2048
#   MODELS="4b" CTXS="512" bash run-m3-card.sh
#   LOCAL_ONLY=1 bash run-m3-card.sh     # use the HF cache only (offline)
#
# Nothing is uploaded. Every receipt stays under receipts/.
set -euo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEEL="$(ls "$KIT"/wheel/mlx_omarchy-*.whl 2>/dev/null | head -1 || true)"
[ -n "$WHEEL" ] || { echo "no bundled wheel under $KIT/wheel/" >&2; exit 1; }
OUT="${OUT:-$KIT/receipts/card-$(date -u +%Y%m%dT%H%M%SZ)}"
VENV="${CARD_VENV:-$KIT/.card-venv}"
MODELS="${MODELS:-2b 4b 9b}"
CTXS="${CTXS:-512 2048}"
MACHINE="${MACHINE:-$(tr '\0' '\n' < /proc/device-tree/model 2>/dev/null \
  || sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)}"
export MLX_OMARCHY_DEFER_TOKENS=0   # card rule: deferral off (PAR-45)

PY="$(command -v python3 || true)"
[ -n "$PY" ] || { echo "python3 is required" >&2; exit 1; }
mkdir -p "$OUT/power"

# power_snapshot <tag>: best-effort, read-only CPU/GPU pstate+freq dump.
power_snapshot() {
  local f="$OUT/power/$1.txt"
  {
    echo "# power snapshot $1 $(date -u +%YFT%T.%-3NZ)"
    echo "machine: $MACHINE"
    echo "uname: $(uname -srm)"
    for p in /sys/devices/system/cpu/cpufreq/policy*; do
      [ -d "$p" ] || continue
      echo "cpufreq/$(basename "$p"): gov=$(cat "$p/scaling_governor" 2>/dev/null || echo '?') cur=$(cat "$p/scaling_cur_freq" 2>/dev/null || echo '?') max=$(cat "$p/scaling_max_freq" 2>/dev/null || echo '?')"
    done
    for m in appledrm apple_gpu asahi agx; do
      [ -d "/sys/module/$m" ] && echo "module: $m loaded"
    done
    for d in /sys/class/drm/card*/device; do
      [ -d "$d" ] || continue
      echo "drm/$(basename "$(dirname "$d")"):"
      for a in "$d"/*freq* "$d"/*clock* "$d"/*pstate* "$d"/pp_* "$d"/gpu_busy*; do
        [ -f "$a" ] && echo "  $(basename "$a") = $(cat "$a" 2>/dev/null || echo unreadable)"
      done
    done
    if [ "$(uname -s)" = "Darwin" ]; then
      echo "note: one-line GPU/ANE power needs sudo — run"
      echo "      sudo powermetrics --samplers gpu_power,ane_power -n 1"
    fi
  } > "$f"
}

# [1/4] venv + wheel + mlx-lm (mirrors wheel_card_row's own recipe).
if [ ! -x "$VENV/bin/python3" ]; then
  echo "[1/4] venv $VENV"
  if command -v uv >/dev/null 2>&1; then
    uv venv "$VENV"
    uv pip install --python "$VENV/bin/python3" --quiet \
      "$WHEEL" "mlx-lm" "huggingface_hub"
  else
    "$PY" -m venv "$VENV"
    "$VENV/bin/python3" -m ensurepip --upgrade
    "$VENV/bin/python3" -m pip install --quiet \
      "$WHEEL" "mlx-lm" "huggingface_hub"
  fi
fi

# [2/4] mlx-lm patch check: vendored series, version-autodetected.
echo "[2/4] mlx-lm patch check"
if ! bash "$KIT/scripts/apply-mlx-lm-patches.sh" "$VENV"; then
  echo "mlx-lm patch series did not apply; the card cell needs the patched" >&2
  echo "generate_step. The error above says which check refused." >&2
  exit 1
fi

# [3/4] deferral guard: an unset/non-zero default while the installed
# generate_step defers would over-report decode tok/s (defer_tokens_guard).
echo "[3/4] deferral check (MLX_OMARCHY_DEFER_TOKENS=$MLX_OMARCHY_DEFER_TOKENS)"
"$VENV/bin/python3" - "$KIT" <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[1] + "/scripts")
import defer_tokens_guard  # noqa: E402
defer_tokens_guard.effective_defer_tokens(tool="m3-tester-kit")
PYEOF

# [4/4] PAR-45 cells.
for m in $MODELS; do
  for c in $CTXS; do
    tag="${m}-ctx${c}"
    echo "[4/4] PAR-45 cell $tag"
    power_snapshot "pre-$tag"
    extra=()
    [ "${LOCAL_ONLY:-0}" = "1" ] && extra+=(--local-files-only)
    "$VENV/bin/python3" "$KIT/scripts/wheel_card_row.py" --installed \
      --machine "$MACHINE" --model "$m" --ctx "$c" \
      --out "$OUT/$tag" "${extra[@]}"
    power_snapshot "post-$tag"
  done
done

echo
echo "cards + power snapshots in $OUT/"
echo "nothing was uploaded; attach $OUT when you file your report."
