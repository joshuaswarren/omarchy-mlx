#!/bin/bash
# census_ab.sh KEY MODEL_DIR: dispatch census per decoded token, bench_decode path (A) against the mlx_lm.server generator in-process (G) and
# against G with the all-True mask removed (GM). UNTIMED: MLX_OMARCHY_TRACE_DISPATCH=1 prints one line per dispatch, so no rate is read here.
# A: bench_decode with --tokens 6 and 10 (its 4-token warm-up is in both, so the difference is 4 decoded tokens).
# G, GM: servegen_inproc.py with PAIRS=1 and TOKENS 24 and 40 (requests of 8 + 6 + 24 and 8 + 10 + 40 tokens: the difference is 20 decoded tokens;
# every request pays its own prefill, three requests in both runs, so prefill cancels). Per-token histogram = (big - small) / delta tokens.
# Histogram key = "kernel=<ComputeKernel id> count=<n> gx gy gz" of the "[rtmod] DISPATCH" line (overlay/mlx/backend/omarchy/encoder.cpp:575).
# Nothing is written outside $WORK/servepath/out-census-*. No governor write, no root, no kill. Same wheel/tree/prompt as flip_ab.sh.
set -u
KEY=${1:?KEY}
MODEL=${2:?MODEL_DIR}
S=${GOV_S:-$WORK/score}
CPUF=${GOV_CPUF:-/sys/devices/system/cpu/cpufreq}
PYB=${PYB:-$S/venv/bin/python}
TREE=${TREE:-main81}
SERVEPKG=${SERVEPKG:-$WORK/serve-pkg/serve}
PROMPT="Write a short story about a lighthouse keeper who finds a message in a bottle."
HERE=$(cd "$(dirname "$0")" && pwd)
T=census-$KEY-$(date -u +%Y%m%dT%H%M%SZ)
OUT=$HERE/out-$T
mkdir -p "$OUT"
L=$OUT/ticket.log
say() { echo "CENSUS $T $(date -u +%T) $*" | tee -a "$L"; }
die() { say "ABORT: $*"; exit 1; }
for p in "$CPUF"/policy*; do [ "$(cat "$p/scaling_governor")" = performance ] || die "governor of $(basename "$p") is $(cat "$p/scaling_governor"), not performance"; done
nane=$(pgrep -fc '[a]ne-run|[i]dle-guard-ane-job' || true)
[ "${nane:-0}" -eq 0 ] || die "$nane ANE process(es) running; the filler must be off"
for f in bench_decode.py run_clamped.py mlx_provenance.py caps_sim_guard.py servegen_inproc.py thread_table.py census_analyze.py; do
	[ -f "$HERE/$f" ] || die "missing $f next to the script"
done
[ -d "$MODEL" ] || die "model directory $MODEL does not exist"
[ -d "$S/$TREE/mlx" ] || die "no tree $S/$TREE"
export PYTHONPATH="$S/$TREE:$S/mlxlm-latest-deps:$SERVEPKG" HF_HUB_OFFLINE=1 MESA_SHADER_CACHE_DISABLE=1 MLX_OMARCHY_ALLOW_UNPINNED_MLXLM=1 PYTHONDONTWRITEBYTECODE=1
unset VK_ICD_FILENAMES VK_DRIVER_FILES MLX_OMARCHY_UCLAMP_MIN
say "start key $KEY model $MODEL boot $(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel $(uname -r) load1 $(cut -d' ' -f1 /proc/loadavg)"
say "tree $S/$TREE libmlx $(sha256sum "$S/$TREE/mlx/lib/libmlx.so" 2> /dev/null | cut -c1-16)"
say "sha256 prefixes: script $(sha256sum "$0" | cut -c1-16)$(for f in bench_decode.py run_clamped.py servegen_inproc.py census_analyze.py; do printf ' %s %s' "$f" "$(sha256sum "$HERE/$f" | cut -c1-16)"; done)"
hist() { # stderr file -> histogram file
	grep -a '\[rtmod\] DISPATCH' "$1" | sed -n 's/.*DISPATCH \(kernel=[0-9]* count=[0-9]* gx=[0-9]* gy=[0-9]* gz=[0-9]*\).*/\1/p' | sort | uniq -c | sort -k2,2 > "$2"
}
run_a() { # tokens
	sleep 10
	MLX_OMARCHY_TRACE_DISPATCH=1 timeout 900 "$PYB" "$HERE/run_clamped.py" thread "$HERE/bench_decode.py" --model "$MODEL" --prompt "$PROMPT" --tokens "$1" --temp 0 --seed 0 > "$OUT/A-$1.out" 2> "$OUT/A-$1.err"
	rc=$?
	hist "$OUT/A-$1.err" "$OUT/hist-A-$1.txt"
	say "A tokens=$1 rc=$rc dispatches=$(awk '{s+=$1} END {print s+0}' "$OUT/hist-A-$1.txt") keys=$(wc -l < "$OUT/hist-A-$1.txt") $(grep -h 'generated_ids' "$OUT/A-$1.out" | head -1 | cut -c1-90)"
	[ "$rc" -eq 0 ] || die "A tokens=$1 exited $rc"
}
run_g() { # arm tokens
	sleep 10
	MLX_OMARCHY_TRACE_DISPATCH=1 MLX_OMARCHY_SERVE_CONTEXT_LIMIT=4096 timeout 900 "$PYB" "$HERE/servegen_inproc.py" "$1" "$MODEL" 1 "$2" "$PROMPT" > "$OUT/$1-$2.out" 2> "$OUT/$1-$2.err"
	rc=$?
	hist "$OUT/$1-$2.err" "$OUT/hist-$1-$2.txt"
	say "$1 tokens=$2 rc=$rc dispatches=$(awk '{s+=$1} END {print s+0}' "$OUT/hist-$1-$2.txt") keys=$(wc -l < "$OUT/hist-$1-$2.txt") $(grep -h '"pair"' "$OUT/$1-$2.out" | head -1 | grep -o '"ids_sha16": "[0-9a-f]*"') $(grep -h '^FLIP' "$OUT/$1-$2.out")"
	[ "$rc" -eq 0 ] || die "$1 tokens=$2 exited $rc"
}
run_a 6
run_a 10
run_g G 24
run_g G 40
run_g GM 24
run_g GM 40
"$PYB" "$HERE/census_analyze.py" "$OUT" 2>&1 | tee -a "$L"
say "DONE"
