#!/usr/bin/env bash
# One-command M3 tester kit for omarchy-mlx.
#
# For an M3 Mac (t8122/t6030/t6031/t6034) it gathers, into one directory:
#   1. aurora 12.3's own M3 bring-up report (--m3-report: read-only,
#      sanitized by aurora before saving),
#   2. the omarchy-mlx collector archive (capability, correctness,
#      benchmark, ANE state) plus its paste-ready submission note.
#
# Nothing is uploaded and nothing needs root. Review the printed
# manifest, then send with the printed --submit command.
#
# Usage:
#   bash scripts/m3_kit.sh            # writes ./m3-kit-<UTC>/
#   AURORA_URL=... bash scripts/m3_kit.sh
set -euo pipefail

AURORA_URL="${AURORA_URL:-https://github.com/iconidentify/aurora-linux/releases/latest/download/install-aurora-sep.sh}"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPTS_DIR")"
OUT="${OUT:-m3-kit-$(date -u +%Y%m%dT%H%M%SZ)}"

compat="$(tr '\0' '\n' < /proc/device-tree/compatible 2>/dev/null |
  grep -E '^apple,t[0-9]+$' | head -1 || true)"
case "${compat:-}" in
  apple,t8122) chip="Apple M3" ;;
  apple,t6030) chip="Apple M3 Pro" ;;
  apple,t6031 | apple,t6034) chip="Apple M3 Max" ;;
  *) chip="" ;;
esac
if [ -z "$chip" ]; then
  echo "This is not an M3 Mac (device-tree compatible: ${compat:-unreadable})."
  echo "The M3 kit is for t8122/t6030/t6031/t6034. For M1/M2 data, run:"
  echo "  python3 scripts/collect_deep.py --submit"
  exit 1
fi
echo "M3 kit: $chip ($compat); writing to $OUT/"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

# 1. aurora's M3 bring-up report (their tool, their sanitizer).
if (cd "$OUT" &&
    curl -fsSL "$AURORA_URL" -o install-aurora-sep.sh &&
    bash install-aurora-sep.sh --m3-report); then
  echo "aurora --m3-report: OK (tgz in $OUT/)"
else
  echo "aurora --m3-report did not complete (recorded above); continuing." >&2
fi

# 2. omarchy-mlx collector: preview by default, upload is explicit.
if (cd "$REPO_DIR" &&
    python3 scripts/collect_deep.py --out "$PWD/$OUT/mlx-omarchy-m3.tar"); then
  echo "collector: OK ($OUT/mlx-omarchy-m3.tar)"
else
  echo "collector did not complete (recorded above); continuing." >&2
fi

cat <<EOF

Next steps (nothing has been sent):
  1. Review $OUT/ contents and the printed redaction manifest.
  2. Send the mlx archive:
       python3 scripts/collect_deep.py --out $OUT/mlx-omarchy-m3.tar --submit
     (the endpoint replies with a public receipt URL; identical content
     deduplicates by SHA-256).
  3. Attach the aurora report tgz from $OUT/ to your message.
EOF
