#!/usr/bin/env bash
# Install oMLX (jundot/omlx) into a user venv on Linux on top of omarchy-mlx.
#
#   packaging/omlx-linux/install.sh --mlx-wheel /path/to/mlx_omarchy-...whl \
#       [--venv DIR] [--omlx-dir DIR] [--home DIR] [--ref <tag or full commit sha>]
#       [--with-optional-deps] [--no-mlx-lm-patches]
#
# What this does:
#   1. clones jundot/omlx at the recorded pin and applies the Linux compat
#      patch series from patches/ (never forks the source into omarchy-mlx);
#   2. runs the AST import-guard scan on the patched tree (must pass);
#   3. builds a user venv, installs the omarchy mlx wheel by path, mlx-lm at
#      the commit oMLX pins (--no-deps: its "mlx" dependency would otherwise
#      resolve to upstream Metal mlx and clobber the Vulkan install), the
#      pinned runtime dependency set, and oMLX itself (--no-deps);
#   4. applies the repo's mlx-lm serve patches (scripts/apply-mlx-lm-patches.sh
#      auto-selects patches/mlx-lm-0.32/ for the 94cdcae commit oMLX pins);
#   5. prints the server smoke command.
#
# Honest constraints, enforced here rather than hidden:
#   - oMLX declares requires-python >=3.11,<3.14. On a 3.14 interpreter the
#     installer proceeds with --ignore-requires-python and says so; the server
#     smoke is the empirical check that 3.14 works.
#   - The memory guard on Linux budgets against unified system RAM read from
#     /proc/meminfo (the GPU draws from the same RAM); there is no wired-limit
#     sysctl and mx.set_wired_limit is a no-op on the omarchy backend.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMARCHY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OMLX_REF="c0b1056b41ebde9422af316cf5038a423eb8f24c"
OMLX_PIN="c0b1056b41ebde9422af316cf5038a423eb8f24c"
MLX_LM_PIN="94cdcae13b266c337bcaca09b97b9c5a9c0e2cde"
MLX_VLM_PIN="ea79808ce1e9a19fcb915a96b0c70e37ad393a99"

VENV=""
OMLX_DIR=""
HOME_OVERRIDE=""
MLX_WHEEL=""
WITH_OPTIONAL=0
NO_PATCHES=0

while (($#)); do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    --omlx-dir) OMLX_DIR="$2"; shift 2 ;;
    --home) HOME_OVERRIDE="$2"; shift 2 ;;
    --mlx-wheel) MLX_WHEEL="$2"; shift 2 ;;
    --ref) OMLX_REF="$2"; shift 2 ;;
    --with-optional-deps) WITH_OPTIONAL=1; shift ;;
    --no-mlx-lm-patches) NO_PATCHES=1; shift ;;
    *) echo "error: unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -n $HOME_OVERRIDE ]]; then
  export HOME="$HOME_OVERRIDE"
fi
VENV="${VENV:-$HOME/.venvs/omlx}"
OMLX_DIR="${OMLX_DIR:-$HOME/.cache/omlx-linux/src/omlx}"

for tool in git python3; do
  command -v "$tool" >/dev/null 2>&1 || { echo "error: $tool not found" >&2; exit 2; }
done
# Arch ships pip only inside venvs (bootstrapped by ensurepip); the system
# interpreter has no `python3 -m pip`. Gate on ensurepip availability —
# every pip invocation below is $VENV/bin/pip from the fresh venv.
python3 -m ensurepip --version >/dev/null 2>&1 || {
  echo "error: python3 -m ensurepip unavailable; cannot bootstrap a venv pip" >&2
  exit 2
}
[[ -n $MLX_WHEEL ]] || {
  echo "error: --mlx-wheel PATH is required (the omarchy mlx wheel to install)" >&2
  exit 2
}
[[ -f $MLX_WHEEL ]] || { echo "error: no such wheel: $MLX_WHEEL" >&2; exit 2; }
[[ -f $OMARCHY_ROOT/scripts/apply-mlx-lm-patches.sh ]] || {
  echo "error: this script must run from an omarchy-mlx checkout (need scripts/ and patches/)" >&2
  exit 2
}
[[ -f $SCRIPT_DIR/apply-platform-gate.sh ]] || {
  echo "error: apply-platform-gate.sh missing in $SCRIPT_DIR" >&2; exit 2; }

PYVER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
IGNORE_PY=()
case "$PYVER" in
  3.11|3.12|3.13) : ;;
  *)
    IGNORE_PY=(--ignore-requires-python)
    echo "note: python $PYVER is outside oMLX's declared >=3.11,<3.14 range;" \
         "installing with --ignore-requires-python (see README)" >&2
    ;;
esac

# OMLX_REF is a release tag, or a full commit sha when upstream has not tagged the commit we pin (GitHub serves any
# reachable sha). The pin check below is what makes either form safe.
fetch_ref() {
  if [[ $OMLX_REF =~ ^[0-9a-f]{40}$ ]]; then
    git -C "$OMLX_DIR" fetch --quiet origin "$OMLX_REF"
  else
    git -C "$OMLX_DIR" fetch --quiet origin "refs/tags/$OMLX_REF:refs/tags/$OMLX_REF"
  fi
}
echo "==> cloning oMLX at $OMLX_REF (pin $OMLX_PIN) into $OMLX_DIR"
if [[ -d $OMLX_DIR/.git ]]; then
  fetch_ref || true
else
  git clone --quiet https://github.com/jundot/omlx "$OMLX_DIR"
  fetch_ref
fi
git -C "$OMLX_DIR" checkout --quiet "$OMLX_REF"
HEAD_SHA="$(git -C "$OMLX_DIR" rev-parse HEAD)"
if [[ $HEAD_SHA != "$OMLX_PIN" ]]; then
  echo "error: $OMLX_REF resolves to $HEAD_SHA, expected pin $OMLX_PIN" >&2
  exit 3
fi
if [[ -n $(git -C "$OMLX_DIR" status --porcelain) ]]; then
  echo "error: oMLX work tree is dirty; refusing to patch" >&2
  exit 3
fi

echo "==> applying platform-gate series (16 gate sites across mx.metal.is_available() calls)"
bash "$SCRIPT_DIR/apply-platform-gate.sh" "$OMLX_DIR"

echo "==> applying Linux compat patches (3 macOS-only helpers)"
for p in "$SCRIPT_DIR"/patches/*.patch; do
  if git -C "$OMLX_DIR" apply --check "$p" 2>/dev/null; then
    git -C "$OMLX_DIR" apply "$p"
    echo "  applied $(basename "$p")"
  elif git -C "$OMLX_DIR" apply --check --reverse "$p" >/dev/null 2>&1; then
    echo "  already applied $(basename "$p")"
  else
    echo "error: patch does not apply: $p" >&2
    exit 4
  fi
done

echo "==> AST import-guard scan"
python3 "$SCRIPT_DIR/ast_import_scan.py" "$OMLX_DIR/omlx" --allow patches \
  || { echo "error: unguarded imports in patched tree" >&2; exit 5; }

echo "==> creating venv $VENV (python $PYVER)"
if [[ ! -x $VENV/bin/python ]]; then
  PYTHONNOUSERSITE=1 python3 -m venv "$VENV"
fi
PIP=("$VENV/bin/pip")

echo "==> installing omarchy mlx wheel"
"${PIP[@]}" install --quiet "$MLX_WHEEL"

# Constraints: keep the resolver from quietly upgrading anything that would
# break the omarchy pairing. There is deliberately no "mlx" pin: the omarchy
# wheel's distribution name is mlx-omarchy and it must never be shadowed by
# upstream Metal mlx from PyPI.
MLX_WHEEL_VERSION="$("${PIP[@]}" show mlx-omarchy | sed -n 's/^Version: //p')"
CONSTRAINTS="$VENV/omlx-linux-constraints.txt"
cat > "$CONSTRAINTS" <<EOF
# Generated by packaging/omlx-linux/install.sh — do not edit.
mlx-omarchy==$MLX_WHEEL_VERSION
transformers>=5.14.0,<5.18
huggingface-hub>=1.19.0
numpy>=1.24.0,<2.4
markitdown==0.1.7
EOF

echo "==> installing mlx-lm at the commit oMLX pins (--no-deps; its mlx dep would pull upstream Metal mlx)"
"${PIP[@]}" install --quiet --no-deps "${IGNORE_PY[@]}" \
  "mlx-lm @ git+https://github.com/ml-explore/mlx-lm@$MLX_LM_PIN"

echo "==> installing oMLX runtime dependencies"
"${PIP[@]}" install --quiet -c "$CONSTRAINTS" "${IGNORE_PY[@]}" \
  "transformers[sentencepiece]>=5.14.0,<5.18" \
  "mistral-common>=1.10" tokenizers "huggingface-hub>=1.19.0" \
  "numpy>=1.24.0,<2.4" tqdm pyyaml itsdangerous jinja2 "rich>=13.0.0" \
  sentencepiece tiktoken protobuf "requests>=2.28.0" "httpx>=0.27.0,<1" \
  "socksio>=1.0.0" "ddgs==9.16.0" tabulate "psutil>=5.9.0" \
  "setproctitle>=1.3.3" "fastapi>=0.108.0" "uvicorn>=0.23.0" \
  "python-multipart>=0.0.5" "jsonschema>=4.0.0" openai-harmony \
  "cohere_melody>=0.9.0" "Pillow>=9.0.0" regex \
  "mcp>=2.0.0,<3" \
  "markitdown[pdf,docx,pptx]==0.1.7"

echo "==> installing mlx-vlm at oMLX's pin (--no-deps: its resolver would pull mlx/mlx-lm)"
"${PIP[@]}" install --quiet --no-deps "${IGNORE_PY[@]}" \
  "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm@$MLX_VLM_PIN"

if (( WITH_OPTIONAL )); then
  echo "==> optional guarded deps (embeddings/dflash/audio; absent = feature honestly disabled)"
  "${PIP[@]}" install --quiet --no-deps -c "$CONSTRAINTS" "${IGNORE_PY[@]}" \
    "mlx-embeddings @ git+https://github.com/Blaizzy/mlx-embeddings@32981fa4e8064ed664b52071789dd18271fe4206" \
    "dflash-mlx @ git+https://github.com/jundot/dflash-mlx@71f7c2cae42a968ddc981972a8ad35368dc587a9" \
    "mlx-audio @ git+https://github.com/Blaizzy/mlx-audio@49596ac8b69b9ed377db311a73df838795f38a3d"
else
  echo "    skipped: mlx-embeddings/dflash-mlx/mlx-audio (all import-guarded; use --with-optional-deps)"
fi

echo "==> installing oMLX (--no-deps; deps were installed explicitly above)"
"${PIP[@]}" install --quiet --no-deps "${IGNORE_PY[@]}" "$OMLX_DIR"

if (( ! NO_PATCHES )); then
  echo "==> applying mlx-lm serve patches (0.32 series auto-selected for the pinned commit)"
  MLX_OMARCHY_CONV_RING="${MLX_OMARCHY_CONV_RING:-0}" \
    bash "$OMARCHY_ROOT/scripts/apply-mlx-lm-patches.sh" "$VENV"
fi

echo
echo "==> installed. oMLX commit: $HEAD_SHA  mlx-omarchy: $MLX_WHEEL_VERSION"
"${VENV}/bin/python" -m pip show omlx mlx-lm mlx-omarchy 2>/dev/null | grep -E '^(Name|Version):' || true
echo
echo "Smoke (GPU hosts: run inside a gpu-turn ticket):"
echo "  $VENV/bin/omlx serve --host 127.0.0.1 --port 8900 --model <model-id>"
