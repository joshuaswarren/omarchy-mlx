#!/usr/bin/env bash
# One-command M3/M4 tester kit for omarchy-mlx.
#
# Linux (aurora 12.3, M3):
#   1. aurora's own M3 bring-up report (--m3-report: read-only,
#      sanitized by aurora before saving),
#   2. the omarchy-mlx collector archive (capability, correctness,
#      benchmark, ANE state) plus its paste-ready submission note.
# macOS (M3/M4; M4 is macOS-only today):
#   1. the same collector, native: MLX/Metal benchmark numbers, ANE
#      ioreg/DeviceTree dump, ANE firmware hashes,
#   2. optional powermetrics one-shot (printed for you to run; sudo),
#   3. optional H16G compiler proof with MKIT_HWX=1.
#
# Nothing is uploaded and nothing needs root. Review the printed
# manifest, then send with the printed --submit command.
# Why each step exists: scripts/m3m4_kit.md.
#
# Usage:
#   bash scripts/m3m4_kit.sh                # writes ./m3m4-kit-<UTC>/
#   AURORA_URL=... bash scripts/m3m4_kit.sh
#   MKIT_HWX=1 bash scripts/m3m4_kit.sh     # macOS only: H16G compile
set -euo pipefail

AURORA_URL="${AURORA_URL:-https://github.com/iconidentify/aurora-linux/releases/latest/download/install-aurora-sep.sh}"
HWX_REPO="${HWX_REPO:-https://github.com/joshuaswarren/mil-hwx-compiler}"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPTS_DIR")"
OUT="${OUT:-m3m4-kit-$(date -u +%Y%m%dT%H%M%SZ)}"
OS="$(uname -s)"

detect_chip() {
  if [ "$OS" = "Linux" ]; then
    tr '\0' '\n' < /proc/device-tree/compatible 2>/dev/null |
      grep -E '^apple,t[0-9]+$' | head -1 || true
  else
    # macOS: the marketing chip name from the brand string.
    sysctl -n machdep.cpu.brand_string 2>/dev/null || true
  fi
}

CHIP_RAW="$(detect_chip)"
CHIP=""
case "${CHIP_RAW}" in
  apple,t8122 | Apple\ M3) chip="Apple M3" ;;
  apple,t6030 | Apple\ M3\ Pro) chip="Apple M3 Pro" ;;
  apple,t6031 | apple,t6034 | Apple\ M3\ Max) chip="Apple M3 Max" ;;
  apple,t8132 | Apple\ M4) chip="Apple M4" ;;
  apple,t6040 | Apple\ M4\ Pro) chip="Apple M4 Pro" ;;
  apple,t6041 | Apple\ M4\ Max) chip="Apple M4 Max" ;;
esac
if [ -z "$chip" ]; then
  echo "This is not an M3/M4 Mac (detected: ${CHIP_RAW:-unknown})."
  echo "The M3/M4 kit is for t8122/t6030/t6031/t6034 and t8132/t6040/t6041."
  echo "For M1/M2 data, run: python3 scripts/collect_deep.py --submit"
  exit 1
fi
if [ "$OS" = "Linux" ] && [ "${chip#Apple M4}" != "$chip" ]; then
  echo "M4 on Linux: aurora does not boot M4 Macs yet; run this kit from"
  echo "macOS on the same box so we get the reference numbers."
  exit 1
fi
echo "M3/M4 kit: $chip (${CHIP_RAW}) on ${OS}; writing to $OUT/"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

collector_stage() {
  if (cd "$REPO_DIR" &&
      python3 scripts/collect_deep.py --out "$OUT/mlx-omarchy-m3m4.tar"); then
    echo "collector: OK ($OUT/mlx-omarchy-m3m4.tar)"
  else
    echo "collector did not complete (recorded above); continuing." >&2
  fi
}

if [ "$OS" = "Linux" ]; then
  # aurora's M3 bring-up report (their tool, their sanitizer).
  if (cd "$OUT" &&
      curl -fsSL "$AURORA_URL" -o install-aurora-sep.sh &&
      bash install-aurora-sep.sh --m3-report); then
    echo "aurora --m3-report: OK (tgz in $OUT/)"
  else
    echo "aurora --m3-report did not complete (recorded above); continuing." >&2
  fi
  collector_stage
  echo "(Optional, needs sudo, ~5 min, on power: aurora's power survey —"
  echo "  bash $OUT/install-aurora-sep.sh --m3-power-survey)"
else
  collector_stage
  echo "(Optional, needs sudo: one powermetrics sample for the GPU/ANE"
  echo " power channels:"
  echo "  sudo powermetrics --samplers gpu_power,ane_power -n 1 -o $OUT/powermetrics.txt)"
  if [ "${MKIT_HWX:-0}" = "1" ]; then
    echo "H16G compiler proof (mil-hwx-compiler):"
    if (git clone --depth 1 "$HWX_REPO" "$OUT/mil-hwx-compiler" &&
        make -C "$OUT/mil-hwx-compiler" mil-hwxc &&
        "$OUT/mil-hwx-compiler/build/mil-hwxc" \
          --mil "$OUT/mil-hwx-compiler/tests/fixtures/conv_relu.mil" \
          --model-root "$OUT/mil-hwx-compiler/tests/models/conv_relu" \
          --output "$OUT/hwx-conv-relu"); then
      shasum -a 256 "$OUT"/hwx-conv-relu/* 2>/dev/null |
        tee "$OUT/hwx-conv-relu/SHA256SUMS" || true
      echo "hwx: OK (program-0.hwx hashed in $OUT/hwx-conv-relu/)"
    else
      echo "hwx stage did not complete (recorded above); continuing." >&2
    fi
  else
    echo "(Optional H16G compiler proof: rerun with MKIT_HWX=1)"
  fi
fi

cat <<EOF

Next steps (nothing has been sent):
  1. Review $OUT/ contents and the printed redaction manifest.
  2. Send the collector archive:
       python3 scripts/collect_deep.py --out $OUT/mlx-omarchy-m3m4.tar --submit
     (the endpoint replies with a public receipt URL; identical content
     deduplicates by SHA-256).
  3. On Linux, also attach the aurora report tgz from $OUT/.
EOF
