#!/usr/bin/env bash
# prefill_census.sh MODEL_DIR PY [OUT_DIR] — traced-prefill dispatch census.
#
# Runs one model forward pass (prefill only, max_tokens=1) at two prompt
# lengths, SMALL=512 and BIG=4096, under MLX_OMARCHY_TRACE_DISPATCH=1, then
# writes ONE tarball: $OUT/prefill-census-<ts>.tar.gz. Read it with
# prefill_gap_reader.py — it names whether the coopmat-off tile fallback or
# GEMM coverage owns the prefill gap on this machine.
#
#   bash prefill_census.sh /path/to/model ~/.venvs/omarchy/bin/python
#
# MODEL_DIR: a local mlx_lm model directory (weights + config + tokenizer).
# PY: a python with the omarchy mlx wheel AND mlx_lm installed.
# Sizes overridable: SMALL=256 BIG=2048 bash prefill_census.sh ...
set -euo pipefail
MODEL=${1:?usage: prefill_census.sh MODEL_DIR PY [OUT_DIR]}
PY=${2:?usage: prefill_census.sh MODEL_DIR PY [OUT_DIR]}
OUT=${3:-$PWD/prefill-census}
SMALL=${SMALL:-512}
BIG=${BIG:-4096}
TS=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
TARBALL="$OUT/prefill-census-$TS.tar.gz"

hist() { # stderr file -> histogram file (same key as SC-20 census_ab.sh)
	grep -a '\[rtmod\] DISPATCH' "$1" \
	  | sed -n 's/.*DISPATCH \(kernel=[0-9]* count=[0-9]* gx=[0-9]* gy=[0-9]* gz=[0-9]*\).*/\1/p' \
	  | sort | uniq -c | sort -k2,2 > "$2"
}

run() { # label tokens
	local label=$1 n=$2
	MLX_OMARCHY_TRACE_DISPATCH=1 HF_HUB_OFFLINE=1 timeout 600 "$PY" "$HERE/prefill_probe.py" \
	  "$MODEL" "$n" > "$OUT/$label.out" 2> "$OUT/$label.err"
	hist "$OUT/$label.err" "$OUT/hist-$label.txt"
	echo "census $label tokens=$n rc=0 dispatches=$(awk '{s+=$1} END {print s+0}' "$OUT/hist-$label.txt")"
}

run "p$SMALL" "$SMALL"
run "p$BIG" "$BIG"

# provenance block: what ran, on which wheel, on which device
{
	echo "host=$(uname -n) kernel=$(uname -r) boot_id=$(cat /proc/sys/kernel/random/boot_id)"
	echo "utc=$TS small=$SMALL big=$BIG"
	"$PY" -c 'import mlx.core as mx; print("mlx_version=" + getattr(mx, "__version__", "?"))' 2>/dev/null || true
	command -v vulkaninfo >/dev/null && vulkaninfo --summary 2>/dev/null | grep -m1 deviceName || true
} > "$OUT/meta.txt"

tar -C "$OUT" -czf "$TARBALL" \
	"meta.txt" "hist-p$SMALL.txt" "hist-p$BIG.txt" \
	"p$SMALL.out" "p$SMALL.err" "p$BIG.out" "p$BIG.err"
echo "TARBALL $TARBALL"
