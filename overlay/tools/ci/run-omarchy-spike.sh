#!/usr/bin/env bash
# U2 spike gate: build and run the real Vulkan matmul and attention kernels
# against the pinned llama.cpp comparator, then enforce the 80 percent
# go/no-go. This is tools/ci/run-omarchy-spike.sh from the plan.
#
# Two comparator inputs are REQUIRED and both fail closed:
#   1. benchmarks/omarchy/llama_cpp_reference.json - the pinned llama.cpp
#      identity (commit, build, model hashes) and model-level pp512/tg128
#      throughput as context only. Fill it from llama-bench output.
#   2. benchmarks/omarchy/ggml_op_comparator.json - matched per-op kernel
#      timings (mul_mat F16, mul_mat Q4_K, flash_attn_ext) generated
#      automatically by building benchmarks/omarchy/ggml_op_probe.cpp
#      against the pinned llama.cpp tree (LLAMA_CPP_DIR, LLAMA_CPP_BUILD).
#      The spike's 0.80 gates compare every row against these timings.
#
# Usage:
#   tools/ci/run-omarchy-spike.sh [--comparator FILE] [extra benchmark args]
#
# Environment:
#   MLX_OMARCHY_BUILD_DIR   build directory (default build-omarchy)
#   MLX_OMARCHY_ALLOW_NON_APPLE=1  development devices only; receipts then
#                                  record a dev device, not Honeykrisp.
#   MLX_OMARCHY_SOURCE_COMMIT  source commit; required for snapshot trees
#   MLX_OMARCHY_SOURCE_DIRTY   0 or 1; required with a snapshot source commit
#
# Exit codes: 0 GO, 3 NO-GO, 2 comparator/config error, 1 runtime error.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${MLX_OMARCHY_BUILD_DIR:-$ROOT/build-omarchy}"
SPIKE_BUILD="$BUILD_DIR/spike"
COMPARATOR="$ROOT/benchmarks/omarchy/llama_cpp_reference.json"

extra_args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --comparator)
      COMPARATOR="$2"
      extra_args+=(--comparator "$2")
      shift 2
      ;;
    --comparator=*)
      COMPARATOR="${1#*=}"
      extra_args+=(--comparator "$COMPARATOR")
      shift
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

echo "== comparator pre-check =="
if [ ! -f "$COMPARATOR" ]; then
  cat >&2 <<EOF
ERROR: pinned llama.cpp comparator file not found: $COMPARATOR

The U2 spike reports go/no-go only against real same-machine llama.cpp
Vulkan numbers; it never invents them. Produce the comparator first:

  1. Build the pinned llama.cpp release v0.3.0 (commit
     c1d0e7a004015f23bc0233470b747b596f29b264; the llama-vulkan-build unit
     on m1-test-host-linux produces it) with Vulkan.
  2. On the same M1 machine, run
       llama-bench -m ~/ane-models/Qwen3.8-2B-Q4_K_M.gguf -p 512 -n 128 -ngl 99 -r 5
  3. Copy benchmarks/omarchy/llama_cpp_reference.example.json to
     benchmarks/omarchy/llama_cpp_reference.json and fill in every field
     from the llama-bench output.
  4. Re-run tools/ci/run-omarchy-spike.sh
EOF
  exit 2
fi
if command -v python3 >/dev/null 2>&1; then
  if ! python3 - "$COMPARATOR" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
missing = []
try:
    if d["prefill"]["tps"] in (None, 0):
        missing.append("prefill.tps (pp512 t/s from llama-bench)")
except (KeyError, TypeError):
    missing.append("prefill.tps (pp512 t/s from llama-bench)")
try:
    if d["decode"]["tps"] in (None, 0):
        missing.append("decode.tps (tg128 t/s from llama-bench)")
except (KeyError, TypeError):
    missing.append("decode.tps (tg128 t/s from llama-bench)")
if missing:
    print("comparator fields not filled in: " + "; ".join(missing), file=sys.stderr)
    sys.exit(1)
PY
  then
    echo "ERROR: fill the comparator fields listed above, then re-run." >&2
    exit 2
  fi
fi

echo "== build mlx runtime (release-equivalent, CPU backend for explicit CPU streams) =="
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
  cmake -S "$ROOT" -B "$BUILD_DIR" \
    -DMLX_BUILD_OMARCHY=ON \
    -DMLX_BUILD_CPU=ON \
    -DMLX_BUILD_METAL=OFF \
    -DMLX_BUILD_CUDA=OFF \
    -DMLX_BUILD_TESTS=ON \
    -DMLX_BUILD_EXAMPLES=OFF \
    -DMLX_BUILD_BENCHMARKS=OFF \
    -DMLX_BUILD_PYTHON_BINDINGS=OFF
fi
cmake --build "$BUILD_DIR" --target mlx -j

echo "== build spike =="
cmake -S "$ROOT/benchmarks/omarchy" -B "$SPIKE_BUILD" \
  -DMLX_ROOT="$ROOT" -DMLX_BUILD_DIR="$BUILD_DIR"
cmake --build "$SPIKE_BUILD" -j

echo "== pinned ggml per-op comparator (matched kernels) =="
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/src/llama.cpp}"
LLAMA_CPP_BUILD="${LLAMA_CPP_BUILD:-$LLAMA_CPP_DIR/build-vulkan}"
GGML_COMPARATOR="$ROOT/benchmarks/omarchy/ggml_op_comparator.json"

if [ ! -f "$LLAMA_CPP_DIR/ggml/include/ggml.h" ]; then
  cat >&2 <<EOF
ERROR: pinned llama.cpp tree not found at $LLAMA_CPP_DIR

The 80 percent gate compares matched kernels: the spike's GEMM/GEMV/SDPA
rows against GGML_OP_FLASH_ATTN_EXT and ggml_mul_mat timings from the pinned
llama.cpp Vulkan backend at the same shapes. Point the runner at the pinned
checkout with LLAMA_CPP_DIR=/path/to/llama.cpp (built with GGML_VULKAN=ON,
build dir LLAMA_CPP_BUILD, default <tree>/build-vulkan), then re-run.
EOF
  exit 2
fi
if [ ! -f "$LLAMA_CPP_BUILD/bin/libggml.so" ] \
   && [ ! -f "$LLAMA_CPP_BUILD/bin/libggml-vulkan.so" ] \
   && [ ! -f "$LLAMA_CPP_BUILD/src/libggml.so" ]; then
  cat >&2 <<EOF
ERROR: built ggml libraries not found under $LLAMA_CPP_BUILD

Build the pinned llama.cpp with Vulkan (cmake -B build-vulkan
-DGGML_VULKAN=ON && cmake --build build-vulkan) and re-run, or point
LLAMA_CPP_BUILD at the existing build directory.
EOF
  exit 2
fi

pin_commit=""
if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git not found; the pinned llama.cpp commit cannot be verified." >&2
  exit 2
fi
pin_commit="$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD 2>/dev/null || true)"
if [ -z "$pin_commit" ]; then
  echo "ERROR: cannot determine the commit of $LLAMA_CPP_DIR (not a git checkout?). The comparator pin must be verifiable." >&2
  exit 2
fi
source_commit="${MLX_OMARCHY_SOURCE_COMMIT:-}"
source_dirty="${MLX_OMARCHY_SOURCE_DIRTY:-}"
if [ -z "$source_commit" ]; then
  source_commit="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"
fi
if [ -z "$source_commit" ]; then
  echo "ERROR: set MLX_OMARCHY_SOURCE_COMMIT when the source tree has no Git metadata." >&2
  exit 2
fi
source_args=(--source-commit "$source_commit")
if [ "$source_dirty" = "1" ]; then
  source_args+=(--source-dirty)
elif [ -z "$source_dirty" ] && { ! git -C "$ROOT" diff --quiet --ignore-submodules -- || ! git -C "$ROOT" diff --cached --quiet --ignore-submodules --; }; then
  source_args+=(--source-dirty)
elif [ -n "$source_dirty" ] && [ "$source_dirty" != "0" ]; then
  echo "ERROR: MLX_OMARCHY_SOURCE_DIRTY must be 0 or 1." >&2
  exit 2
fi
# Stale-build check: any ggml source newer than the built libraries means the
# libs do not match the checkout.
newest_lib=""
for lib in "$LLAMA_CPP_BUILD/bin/libggml-base.so" "$LLAMA_CPP_BUILD/src/libggml-base.so" "$LLAMA_CPP_BUILD/bin/libggml.so" "$LLAMA_CPP_BUILD/src/libggml.so"; do
  if [ -f "$lib" ]; then
    newest_lib="$lib"
    break
  fi
done
if [ -n "$newest_lib" ] && [ -n "$(find "$LLAMA_CPP_DIR/ggml/src" "$LLAMA_CPP_DIR/ggml/include" -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.cu' \) -newer "$newest_lib" -print -quit 2>/dev/null)" ]; then
  echo "ERROR: ggml sources in $LLAMA_CPP_DIR are newer than $newest_lib; the build does not match the checkout. Rebuild and re-run." >&2
  exit 2
fi
expected_commit=""
if command -v python3 >/dev/null 2>&1; then
  expected_commit="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("pin",{}).get("commit",""))' "$COMPARATOR" 2>/dev/null || true)"
fi
if [ -n "$expected_commit" ] && [ -n "$pin_commit" ] && [ "$pin_commit" != "$expected_commit" ]; then
  cat >&2 <<EOF
ERROR: pinned-commit mismatch

llama_cpp_reference.json pins llama.cpp commit $expected_commit but
$LLAMA_CPP_DIR is at $pin_commit. Check out the pinned commit, rebuild,
and re-run.
EOF
  exit 2
fi

LIB_DIRS="$LLAMA_CPP_BUILD/bin $LLAMA_CPP_BUILD/src $LLAMA_CPP_BUILD"
g++ -O2 -std=c++17 \
  -I"$LLAMA_CPP_DIR/ggml/include" \
  "$ROOT/benchmarks/omarchy/ggml_op_probe.cpp" \
  $(for d in $LIB_DIRS; do echo -n "-L$d "; done) \
  $(for d in $LIB_DIRS; do echo -n "-Wl,-rpath,$d "; done) \
  -lggml -lggml-base -lggml-vulkan \
  -o "$SPIKE_BUILD/ggml_op_probe"

probe_warmup=5
probe_reps=30
probe_rounds=10
prev=""
for a in ${extra_args+"${extra_args[@]}"}; do
  case "$prev" in
    --warmup) probe_warmup="$a" ;;
    --reps) probe_reps="$a" ;;
    --rounds) probe_rounds="$a" ;;
  esac
  prev="$a"
done
"$SPIKE_BUILD/ggml_op_probe" \
  --pin-commit "$pin_commit" \
  --output "$GGML_COMPARATOR" \
  --warmup "$probe_warmup" --reps "$probe_reps" --rounds "$probe_rounds"
echo "== run spike =="
RECEIPT_DIR="$ROOT/benchmarks/omarchy/receipts"
mkdir -p "$RECEIPT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RECEIPT="$RECEIPT_DIR/spike-$STAMP.json"

set +e
"$SPIKE_BUILD/omarchy_matmul_attention" \
  --comparator "$COMPARATOR" \
  --ggml-comparator "$GGML_COMPARATOR" \
  --output "$RECEIPT" \
  "${source_args[@]}" \
  ${extra_args+"${extra_args[@]}"}
code=$?
set -e

echo "== receipt =="
echo "$RECEIPT"
if [ -s "$RECEIPT" ]; then
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$RECEIPT" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
g = d.get("go_no_go", {})
print("verdict: %s  prefill_ratio=%.3f  decode_ratio=%.3f  attention_ratio=%.3f  threshold=%.2f"
      % (g.get("verdict"), g.get("prefill_ratio", 0), g.get("decode_ratio", 0),
         g.get("attention_ratio", 0), g.get("threshold", 0.8)))
PY
  fi
fi

exit $code
