#!/bin/bash
# flip_ab.sh KEY MODEL_DIR: do the two measured costs add up, and where does the rest of the generator-loop gap go? (13-inch, E4B)
#   A     bench_decode through run_clamped.py thread mode (anchor, as in servepath_ab.sh)
#   G     the mlx_lm.server generator in-process, no HTTP (servegen_inproc.py G): the loop under test
#   GM    G without the mask array of the sliding-window batch cache when it would be all True        (cache.py:1447-1470 vs 557-585)
#   GL    G without the per-token logprob read in the server loop                                       (server.py:867)
#   GML   G with BOTH removals (additivity test)
#   GMLT  GML with phase timers (BatchGenerator.next, mx.eval, mx.async_eval): where the remaining time per token goes; its rate is not judged
#   (GS, a new thread-local stream, was measured in the previous ticket: NOTHING; arm names in ARMS are free: ARMS="A G GM GL GS" reruns it)
# Same prompt, 128 tokens, greedy, one wheel (tree main81), governor performance READ not written, ANE filler off. ROUNDS rounds; the order
# of each round is a rotation of the arm list. G-type arms run REQS request pairs per process. REGISTERED RULE: flip_analyze.py header.
# No governor write, no root, no kill. env: ARMS, PYB, TREE, ROUNDS(4), REQS(2), TOKENS(128), GOV_SETTLE(15), GOV_MAXLOAD(0.5), GOV_LOADWAIT(300), GOV_SAMP, GOV_CSV
set -u
KEY=${1:?KEY (a short label such as e4b)}
MODEL=${2:?MODEL_DIR (a local model snapshot directory)}
S=${GOV_S:-$WORK/score}
CPUF=${GOV_CPUF:-/sys/devices/system/cpu/cpufreq}
PYB=${PYB:-$S/venv/bin/python}
TREE=${TREE:-main81}
SERVEPKG=${SERVEPKG:-$WORK/serve-pkg/serve}
ROUNDS=${ROUNDS:-4}
REQS=${REQS:-2}
TOKENS=${TOKENS:-128}
SETTLE=${GOV_SETTLE:-15}
PROMPT="Write a short story about a lighthouse keeper who finds a message in a bottle."
HERE=$(cd "$(dirname "$0")" && pwd)
T=flip-$KEY-$(date -u +%Y%m%dT%H%M%SZ)
OUT=$HERE/out-$T
mkdir -p "$OUT"
L=$OUT/ticket.log
SAMP=${GOV_SAMP:-/var/tmp/sampler/row-sampler.sh}
CSV=${GOV_CSV:-/var/tmp/sampler/rows-$T.csv}
say() { echo "FLIP $T $(date -u +%T) $*" | tee -a "$L"; }
die() { say "ABORT: $*"; exit 1; }
cleanup() { [ -f "$SAMP" ] && bash "$SAMP" stop "$CSV" > /dev/null 2>&1; }
trap cleanup EXIT
trap 'exit 130' INT TERM

for p in "$CPUF"/policy*; do [ "$(cat "$p/scaling_governor")" = performance ] || die "governor of $(basename "$p") is $(cat "$p/scaling_governor"), not performance"; done
nane=$(pgrep -fc '[a]ne-run|[i]dle-guard-ane-job' || true)
[ "${nane:-0}" -eq 0 ] || die "$nane ANE process(es) running; the filler must be off"
LOADWAIT=${GOV_LOADWAIT:-300}
waited=0
while :; do
	load=$(cut -d' ' -f1 /proc/loadavg)
	awk -v l="$load" -v m="${GOV_MAXLOAD:-0.5}" 'BEGIN{exit !(l < m)}' && break
	[ "$waited" -lt "$LOADWAIT" ] || die "load1 $load is not under ${GOV_MAXLOAD:-0.5} after waiting ${waited}s"
	sleep 10
	waited=$((waited + 10))
done
[ "$waited" -eq 0 ] || say "waited ${waited}s for load1 to settle (now $load)"
for f in bench_decode.py run_clamped.py mlx_provenance.py caps_sim_guard.py servegen_inproc.py thread_table.py flip_analyze.py; do
	[ -f "$HERE/$f" ] || die "missing $f next to the script"
done
[ -d "$MODEL" ] || die "model directory $MODEL does not exist"
[ -d "$S/$TREE/mlx" ] || die "no tree $S/$TREE"
export PYTHONPATH="$S/$TREE:$S/mlxlm-latest-deps:$SERVEPKG" HF_HUB_OFFLINE=1 MESA_SHADER_CACHE_DISABLE=1 MLX_OMARCHY_ALLOW_UNPINNED_MLXLM=1 PYTHONDONTWRITEBYTECODE=1
unset VK_ICD_FILENAMES VK_DRIVER_FILES MLX_OMARCHY_UCLAMP_MIN
say "start key $KEY model $MODEL boot $(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel $(uname -r) load1 $load tokens $TOKENS reqs $REQS rounds $ROUNDS"
say "governors: $(cat "$CPUF"/policy*/scaling_governor | sort | uniq -c | tr '\n' ' ') (read only)"
say "tree $S/$TREE libmlx $(sha256sum "$S/$TREE/mlx/lib/libmlx.so" 2> /dev/null | cut -c1-16)"
say "sha256 prefixes: script $(sha256sum "$0" | cut -c1-16)$(for f in bench_decode.py run_clamped.py servegen_inproc.py thread_table.py flip_analyze.py; do printf ' %s %s' "$f" "$(sha256sum "$HERE/$f" | cut -c1-16)"; done)"
[ -f "$SAMP" ] && { bash "$SAMP" start "$CSV" >> "$L" 2>&1; say "sampler csv $CSV"; }

read -ra AR <<< "${ARMS:-A G GM GL GML GMLT}"
rot() { local i n=${#AR[@]}; for ((i = 0; i < n; i++)); do printf '%s ' "${AR[$(((i + $1) % n))]}"; done; }
r=0
while [ "$r" -lt "$ROUNDS" ]; do
	for a in $(rot "$r"); do
		sleep "$SETTLE"
		f=$OUT/run-$a$((r + 1))
		t0=$(date +%s)
		say "round $((r + 1)) arm $a load1=$(cut -d' ' -f1 /proc/loadavg) start_epoch=$t0"
		if [ "$a" = A ]; then
			timeout 900 "$PYB" "$HERE/run_clamped.py" thread "$HERE/bench_decode.py" --model "$MODEL" --prompt "$PROMPT" --tokens "$TOKENS" --temp 0 --seed 0 > "$f.out" 2> "$f.err"
			rc=$?
			say "round $((r + 1)) arm A rc=$rc $(grep -h 'decode .* tok/s over' "$f.out" | head -1) $(grep -h 'generated_ids' "$f.out" | head -1 | cut -c1-90)"
			[ "$rc" -eq 0 ] || die "round $((r + 1)) arm A exited $rc: $(grep -v '^CLAMP' "$f.err" | tail -2 | tr '\n' ' ' | cut -c1-300)"
		else
			env MLX_OMARCHY_SERVE_CONTEXT_LIMIT=4096 timeout 900 "$PYB" "$HERE/servegen_inproc.py" "$a" "$MODEL" "$REQS" "$TOKENS" "$PROMPT" > "$f.out" 2> "$f.err"
			rc=$?
			grep -h -E '^(provenance|ARM)' "$f.out" | cut -c1-300 >> "$L"
			say "round $((r + 1)) arm $a rc=$rc pairs $(grep -c '"pair"' "$f.out")"
			[ "$rc" -eq 0 ] || die "round $((r + 1)) arm $a exited $rc: $(grep -v 'INFO -' "$f.err" | tail -2 | tr '\n' ' ' | cut -c1-300)"
		fi
		say "round $((r + 1)) arm $a end_epoch=$(date +%s)"
	done
	r=$((r + 1))
done
[ -f "$SAMP" ] && bash "$SAMP" stop "$CSV" >> "$L" 2>&1
"$PYB" "$HERE/flip_analyze.py" "$OUT" "$ROUNDS" "$REQS" "$CSV" "$L" 2>&1 | tee -a "$L"
say "DONE"
