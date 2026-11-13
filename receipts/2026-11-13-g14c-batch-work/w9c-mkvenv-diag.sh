#!/usr/bin/env bash
# M2. Diagnostics venv for #422: the fnqtail2 venv's packages with mlx-omarchy swapped for the g14cdiag wheel.
set -euo pipefail
test "$(hostname)" = jwm2-linux
W=$HOME/scratch/w9c-g14cdiag-wheel
whl=$(ls "$W"/mlx_omarchy-*.whl)
V=$HOME/scratch/w9c-g14cdiag-venv
"$HOME/scratch/w9c-fnqtail2-venv/venv/bin/python" -m pip freeze | grep -v '^mlx-omarchy' > "$W/req.txt"
mkdir -p "$V"
python3.14 -m venv "$V/venv"
ionice -c3 nice -n 19 "$V/venv/bin/python" -m pip install -q -r "$W/req.txt"
ionice -c3 nice -n 19 "$V/venv/bin/python" -m pip install -q --no-deps "$whl"
echo "$("$V/venv/bin/python" -c 'import mlx_lm, mlx.core as mx; print(mx.__version__)') $(sha256sum "$whl" | cut -c1-16)"
grep -c MLX_OMARCHY_GPU_PROFILE "$V"/venv/lib/python3.14/site-packages/mlx/lib/*.so 2>/dev/null | head -n 3
echo MKVENV_DIAG_DONE
