#!/bin/bash
# mkvenv.sh NAME WHEEL : on the 13-inch M1, venv-NAME = copy of venv-release with only the mlx-omarchy wheel replaced (series and driver stay the release's).
set -euo pipefail
NAME=${1:?}; WHEEL=${2:?}
B=$HOME/.local/share/coreglass
[ -f "$WHEEL" ] || { echo "no wheel"; exit 1; }
[ -x "$B/venv-$NAME/bin/python" ] || cp -a "$B/venv-release" "$B/venv-$NAME"
for f in "$B/venv-$NAME"/bin/*; do
  if [ -f "$f" ] && [ ! -L "$f" ] && head -c 2 "$f" | grep -q '#!'; then sed -i "1s|^#!.*/python3\\?$|#!$B/venv-$NAME/bin/python3|" "$f"; fi
done
"$B/venv-$NAME/bin/python" -m pip install -q --no-deps --force-reinstall "$WHEEL"
echo "venv-$NAME mx $("$B/venv-$NAME/bin/python" -c 'import mlx.core as m; print(m.__version__)') libmlx $(sha256sum "$B/venv-$NAME/lib/python3.14/site-packages/mlx/lib/libmlx.so" | cut -c1-16)"
