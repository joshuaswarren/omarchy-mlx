# Is the raw decode route token-exact on mlx-lm 0.31.3 (the shipping series)? Same prompt as the 0.32 gate.
set -eu
hostname
D="$HOME/.local/share/coreglass"
SRC="$D/omarchy-mlx-src"
PY="$D/venv/bin/python"
M="$D/models/qwen3_5-4bit"
R="$D/raw0313-$(date +%s)"
mkdir -p "$R/whl"
"$D/venv/bin/pip" download -q --no-deps --dest "$R/whl" "mlx-lm==0.31.3"
for side in with-raw without-raw; do
  S="$R/$side/lib/python3.14/site-packages"
  mkdir -p "$S"
  "$PY" -m zipfile -e "$R"/whl/mlx_lm-0.31.3-*.whl "$S"
  for p in mlx-lm-gated-delta-fast-route mlx-lm-gated-delta-fast-route-repeat mlx-lm-gated-delta-raw mlx-lm-greedy-prune \
           mlx-lm-qwen35-qk-scaled mlx-lm-qwen35-gdn-conv mlx-lm-conv-silu mlx-lm-qwen35-gated-norm \
           mlx-lm-ttft-early-submit mlx-lm-last-logits; do
    [ "$side" = without-raw ] && [ "$p" = mlx-lm-gated-delta-raw ] && continue
    patch -s -d "$S" -p1 --forward --fuzz=0 < "$SRC/patches/$p.patch"
  done
  python3 "$SRC/scripts/patch-mlx-lm-rope-norm.py" "$R/$side" >/dev/null
  python3 "$SRC/scripts/patch-mlx-lm-qknorm.py" "$R/$side" >/dev/null
done
cat > /tmp/cg-hash64.py <<'PY'
import hashlib, sys
from mlx_lm import load, stream_generate
model, tok = load(sys.argv[1])
text = "Apple Silicon runs local language models on Linux through Vulkan and a reverse engineered GPU driver. "
for _ in stream_generate(model, tok, tok.encode("Warm up the GPU."), max_tokens=8):
    pass
for reps in (28, 114):
    toks = [r.token for r in stream_generate(model, tok, tok.encode(text * reps), max_tokens=64)]
    print(reps, hashlib.sha256(str(toks).encode()).hexdigest()[:12], toks[:12])
PY
for side in with-raw without-raw; do
  echo "0.31.3 $side:"; flock -w 120 "$1" env PYTHONPATH="$R/$side/lib/python3.14/site-packages" $PY /tmp/cg-hash64.py $M 2>/dev/null
done
rm -f /tmp/cg-hash64.py
