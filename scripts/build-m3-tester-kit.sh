#!/usr/bin/env bash
# Build the m3-tester-kit (r1) as a standalone archive.
#
# The archive holds the kit source (m3-tester-kit/), the mlx-omarchy
# wheel, the PAR-45 card runner, the vendored mlx-lm patch series and its
# check, the deferral guard, a SHA256SUMS manifest, and LICENSE — so a
# base-M3 tester runs the card cell from the archive alone.
#
# Every staged text file is privacy-audited before sealing (same generic
# patterns as scripts/privacy_check.py: absolute home paths, RFC-1918
# addresses). A hit refuses the build.
#
# Usage:
#   build-m3-tester-kit.sh --wheel dist/mlx_omarchy-*.whl [--out DIR] [--tag T]
#   build-m3-tester-kit.sh --self-test [--out DIR]
#                          # stub wheel + full verify on a clean extract
set -euo pipefail
self="${BASH_SOURCE[0]}"
while [ -L "$self" ]; do
  link=$(readlink "$self")
  case "$link" in
    /*) self=$link ;;
    *) self=$(dirname "$self")/$link ;;
  esac
done
root="$(cd -- "$(dirname -- "$self")/.." && pwd)"
if [ ! -d "$root/m3-tester-kit" ]; then
  top=$(git -C "$(dirname -- "$self")" rev-parse --show-toplevel 2>/dev/null || true)
  [ -n "$top" ] && root="$top"
fi

wheel="" out="" tag="" self_test=0
while [ $# -gt 0 ]; do
  case "$1" in
    --wheel) wheel=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    --tag) tag=$2; shift 2 ;;
    --self-test) self_test=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
out="${out:-.}"
tag="${tag:-$(git -C "$root" describe --tags --always --dirty 2>/dev/null || echo untagged)}"

if [ "$self_test" = 1 ]; then
  # Hermetic stub wheel: a valid .whl zip with METADATA/WHEEL only.
  wheel="$(mktemp -d)/mlx_omarchy-0.0.0.dev0-py3-none-any.whl"
  python3 - "$wheel" <<'PYEOF'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1], "w") as z:
    z.writestr("mlx_omarchy-0.0.0.dev0.dist-info/METADATA",
               "Metadata-Version: 2.1\nName: mlx-omarchy\nVersion: 0.0.0.dev0\n")
    z.writestr("mlx_omarchy-0.0.0.dev0.dist-info/WHEEL",
               "Wheel-Version: 1.0\nTag: py3-none-any\n")
    z.writestr("mlx_omarchy-0.0.0.dev0.dist-info/RECORD", "")
PYEOF
  out="${out:-$(mktemp -d)}"
  [ "$out" = "." ] && out="$(mktemp -d)"
fi
[ -n "$wheel" ] || { echo "usage: $0 --wheel <whl> | --self-test" >&2; exit 2; }
[ -r "$wheel" ] || { echo "wheel not readable: $wheel" >&2; exit 1; }

# Canonicalize once: the archive step cd's into the stage dir, so a
# relative --out (the "." default) would resolve to the stage itself.
mkdir -p "$out"
out="$(cd "$out" && pwd)"

name="m3-tester-kit-r1-$tag"
stage="$out/$name"
rm -rf "$stage"
mkdir -p "$stage/scripts" "$stage/wheel" "$stage/patches"

# Kit payload: the in-tree source dir plus the in-repo pieces it names.
cp -R "$root/m3-tester-kit/." "$stage/"
cp "$wheel" "$stage/wheel/"
cp "$root/scripts/wheel_card_row.py" "$stage/scripts/"
cp "$root/scripts/defer_tokens_guard.py" "$stage/scripts/"
cp "$root/scripts/apply-mlx-lm-patches.sh" "$stage/scripts/"
for helper in "$root"/scripts/patch-mlx-lm-*.py; do
  [ -e "$helper" ] || continue
  cp "$helper" "$stage/scripts/"
done
cp -R "$root/patches/." "$stage/patches/"
cp "$root/LICENSE" "$stage/LICENSE" 2>/dev/null || true
find "$stage" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$stage" -name '*.pyc' -type f -delete 2>/dev/null || true
# Test hook: M3TK_PLANT=<file> stages an extra file so tests can prove the
# audit below refuses on planted content.
[ -n "${M3TK_PLANT:-}" ] && cp "$M3TK_PLANT" "$stage/planted.txt"

# Privacy audit: the kit ships to testers — no home paths or LAN IPs.
if grep -rIlE '/home/[a-z0-9_.-]+/|/Users/[A-Za-z0-9_.-]+/' "$stage" \
   --exclude=SHA256SUMS | grep -q . \
|| grep -rIlE '100\.(6[4-9]|[7-9][0-9]|1[0-1][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3}|10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}|192\.168\.[0-9]{1,3}\.[0-9]{1,3}' \
   "$stage" --exclude=SHA256SUMS | grep -q .; then
  echo "privacy audit REFUSED: home path or private IP in staged kit:" >&2
  grep -rIlE '/home/[a-z0-9_.-]+/|/Users/[A-Za-z0-9_.-]+/|100\.(6[4-9]|[7-9][0-9]|1[0-1][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3}|10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}|192\.168\.[0-9]{1,3}\.[0-9]{1,3}' \
    "$stage" --exclude=SHA256SUMS >&2 || true
  exit 1
fi

cd "$stage"
find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
cd "$out"
tar -cf "$name.tar" "$name"
sha256sum "$name.tar" > "$name.tar.sha256"
printf 'archive: %s\n' "$out/$name.tar"

if [ "$self_test" = 1 ]; then
  echo "-- self-test: clean-extract verification"
  verify_dir="$(mktemp -d)"
  (cd "$verify_dir" && tar -xf "$out/$name.tar")
  kitdir="$verify_dir/$name"
  (cd "$kitdir" && sha256sum -c SHA256SUMS >/dev/null) \
    && echo "  SHA256SUMS: OK ($(wc -l < "$kitdir/SHA256SUMS") files)"
  for req in run-m3-card.sh ane-preflight.sh README.md \
             scripts/wheel_card_row.py scripts/defer_tokens_guard.py \
             scripts/apply-mlx-lm-patches.sh wheel LICENSE SHA256SUMS; do
    [ -e "$kitdir/$req" ] || { echo "  MISSING: $req" >&2; exit 1; }
  done
  echo "  required files: present"
  while IFS= read -r s; do bash -n "$s"; done \
    < <(find "$kitdir" -name '*.sh' -type f)
  echo "  bash -n: clean"
  while IFS= read -r p; do python3 -m py_compile "$p"; done \
    < <(find "$kitdir" -name '*.py' -type f)
  echo "  py_compile: clean"
  grep -q 'MLX_OMARCHY_DEFER_TOKENS=0' "$kitdir/run-m3-card.sh" \
    && grep -q 'MLX_OMARCHY_DEFER_TOKENS' "$kitdir/scripts/wheel_card_row.py" \
    && echo "  DEFER_TOKENS=0 pin: present in runner + cell"
  # Tamper check: a flipped byte anywhere must break the manifest.
  (cd "$kitdir" && find . -type f ! -name SHA256SUMS | sort) > "$verify_dir/filelist"
  victim="$(sed -n '1p' "$verify_dir/filelist")"
  printf x >> "$kitdir/$victim"
  if (cd "$kitdir" && sha256sum -c SHA256SUMS >/dev/null 2>&1); then
    echo "  tamper check FAILED: manifest did not catch a flipped byte" >&2
    exit 1
  fi
  echo "  tamper check: manifest catches modification"
  echo "SELF-TEST PASS ($out/$name.tar)"
fi
