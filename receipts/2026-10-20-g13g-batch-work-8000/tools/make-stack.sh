#!/bin/bash
# make-stack.sh NAME TAG WHEEL : on the 13-inch M1, build a ledger-style release stack venv-NAME and mlx-lm-series-NAME from release TAG and its wheel.
# Mirrors ledger/build-stack.sh (STACK=release) without touching venv-release, mlx-lm-series-release or the driver. CPU only.
set -euo pipefail
NAME=${1:?name}; TAG=${2:?tag}; WHEEL=${3:?wheel}
B=$HOME/.local/share/coreglass
SITE=lib/python3.14/site-packages
BASE=${BASE:-venv}
src=$B/build/omarchy-mlx-$NAME
[ -f "$WHEEL" ] || { echo "no wheel $WHEEL"; exit 1; }
[ -d "$src/.git" ] || git clone -q --filter=blob:none https://github.com/joshuaswarren/omarchy-mlx "$src"
git -C "$src" fetch -q origin "refs/tags/$TAG:refs/tags/$TAG"
git -C "$src" reset -q --hard "$TAG"
echo "source $(git -C "$src" rev-parse --short=7 HEAD) tag $TAG"
[ -x "$B/venv-$NAME/bin/python" ] || cp -a "$B/$BASE" "$B/venv-$NAME"
for f in "$B/venv-$NAME"/bin/*; do
  [ -f "$f" ] && [ ! -L "$f" ] && head -c 2 "$f" | grep -q '#!' && sed -i "1s|^#!.*/python3\\?$|#!$B/venv-$NAME/bin/python3|" "$f" || true
done
"$B/venv-$NAME/bin/python" -m pip install -q --no-deps --force-reinstall "$WHEEL"
echo "mx $("$B/venv-$NAME/bin/python" -c 'import mlx.core as m; print(m.__version__)')"
mkdir -p "$B/mlx-lm-series-$NAME/$SITE"
rsync -a --delete "$B/$BASE/$SITE/mlx_lm/" "$B/mlx-lm-series-$NAME/$SITE/mlx_lm/"
bash "$src/scripts/apply-mlx-lm-patches.sh" "$B/mlx-lm-series-$NAME" > "$B/build/series-$NAME.log" 2>&1 || { tail -20 "$B/build/series-$NAME.log"; exit 1; }
echo "series patch lines $(grep -ciE 'applied|already' "$B/build/series-$NAME.log")"
echo "libmlx $(sha256sum "$B/venv-$NAME/$SITE/mlx/lib/libmlx.so" | cut -c1-16)"
