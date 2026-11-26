# On stock mlx-lm 0.31.3 with the 0.31.3 series, the qknorm patcher at BASE (default origin/main) and the one in
# this checkout must write identical trees. Usage: bash qknorm_0313_check.sh [BASE]
set -euo pipefail
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
BASE="${1:-origin/main}"
WHEEL_SHA256=758cfddf1180053b7613db76fad3d246a331a2a905808e1164a275621fc983b8
W="$(mktemp -d)"
pip download -q --no-deps --dest "$W/whl" "mlx-lm==0.31.3"
echo "$WHEEL_SHA256  $(ls "$W"/whl/mlx_lm-0.31.3-*.whl)" | sha256sum -c -
for side in base head; do
  S="$W/$side/lib/python3.14/site-packages"
  mkdir -p "$S"
  python3 -m zipfile -e "$W"/whl/mlx_lm-0.31.3-*.whl "$S"
  for p in mlx-lm-gated-delta-fast-route mlx-lm-gated-delta-fast-route-repeat mlx-lm-gated-delta-raw mlx-lm-greedy-prune \
           mlx-lm-qwen35-qk-scaled mlx-lm-qwen35-gdn-conv mlx-lm-conv-silu mlx-lm-qwen35-gated-norm \
           mlx-lm-ttft-early-submit mlx-lm-last-logits; do
    patch -s -d "$S" -p1 --forward --fuzz=0 --no-backup-if-mismatch < "$REPO/patches/$p.patch"
  done
done
git -C "$REPO" show "$BASE:scripts/patch-mlx-lm-qknorm.py" > "$W/qknorm-base.py"
python3 "$W/qknorm-base.py" "$W/base"
python3 "$REPO/scripts/patch-mlx-lm-qknorm.py" "$W/head"
diff -r -x __pycache__ "$W/base/lib" "$W/head/lib" && echo "0.31.3: $BASE and this checkout's qknorm patcher write identical trees"
