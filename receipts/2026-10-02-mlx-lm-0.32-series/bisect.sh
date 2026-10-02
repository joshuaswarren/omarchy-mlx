# Cumulative bisect of the 0.32 series on the 477-token prompt: which step first changes greedy tokens?
set -u
hostname
D="$HOME/.local/share/coreglass"
SRC="$D/omarchy-mlx-src"
PY="$D/venv/bin/python"
M="$D/models/qwen3_5-4bit"
B="$D/bisect-$(date +%s)"
STEPS="mlx-lm-gated-delta-fast-route mlx-lm-gated-delta-fast-route-repeat mlx-lm-gated-delta-raw mlx-lm-greedy-prune
mlx-lm-qwen35-qk-scaled mlx-lm-qwen35-gdn-conv mlx-lm-conv-silu mlx-lm-qwen35-gated-norm mlx-lm-ttft-early-submit
mlx-lm-last-logits rope-norm qknorm"
cat > /tmp/cg-hash64.py <<'PY'
import hashlib, sys
from mlx_lm import load, stream_generate
model, tok = load(sys.argv[1])
text = "Apple Silicon runs local language models on Linux through Vulkan and a reverse engineered GPU driver. "
for _ in stream_generate(model, tok, tok.encode("Warm up the GPU."), max_tokens=8):
    pass
toks = [r.token for r in stream_generate(model, tok, tok.encode(text * 28), max_tokens=64)]
print(hashlib.sha256(str(toks).encode()).hexdigest()[:12], toks[:12])
PY
S="$B/lib/python3.14/site-packages"
mkdir -p "$S"
cp -a "$D/venv/lib/python3.14/site-packages/mlx_lm" "$S/mlx_lm"
echo "upstream: $(flock -w 120 "$1" $PY /tmp/cg-hash64.py $M 2>/dev/null | tail -1)"
for s in $STEPS; do
  case "$s" in
    rope-norm) python3 "$SRC/scripts/patch-mlx-lm-rope-norm.py" "$B" >/dev/null ;;
    qknorm) python3 "$SRC/scripts/patch-mlx-lm-qknorm.py" "$B" >/dev/null ;;
    *) patch -s -d "$S" -p1 --forward --fuzz=0 < "$SRC/patches/mlx-lm-0.32/$s.patch" ;;
  esac
  echo "+ $s: $(flock -w 120 "$1" env PYTHONPATH=$S $PY /tmp/cg-hash64.py $M 2>/dev/null | tail -1)"
done
rm -f /tmp/cg-hash64.py
