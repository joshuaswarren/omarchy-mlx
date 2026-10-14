#!/bin/bash
# runwhl.sh <tsv> <python> <key> <rounds> <old_tree> <new_tree>: PR 124 wheel A/B. Same-wheel arms on the installed package
# driver (SB_SYSTEM=1) so the only change between the two is the gather_qmm route:
#   V = new tree + MLX_OMARCHY_GATHER_QMM_DIRECT=0  -> OLD staged gather_qmm (the PR 124 control, same wheel)
#   P = new tree default (env unset)                -> NEW direct gather_qmm (the PR 124 change)
#   O = old tree default (env unset)                -> previously published wheel (extra reference; run with BLK_ARMS="V P O")
# The env var is passed per-call (inline prefix), so it is set for arm V only and never reaches arm P or O.
# Interleaved V P [O] ... for <rounds> rounds. Same RESULT/ROWTIME/NOISY output as runblk.sh, with the literal
# "ratio P/V" label so the scoreboard delta parser still matches; ratio P/V is the gain of the new direct route
# over the same-wheel staged control. NOISY is computed on V (control spread > 5%).
set -uo pipefail
S=$HOME/scratch/w71/score
TSV=$1 PYB=$2 KEY=$3 ROUNDS=${4:-3} OLD=$5 NEW=${6:-main124}
H=$HOME/.cache/huggingface/hub
cd "$S" || exit 1
repo=$(awk -F'\t' -v k="$KEY" '$1==k {print $3}' "$S/$TSV")
d=$H/models--${repo//\//--}
pre=0; [ -d "$d" ] && pre=1
T=whl-$KEY-$(date -u +%H%M)
L=$S/sb-$T.log
hw() { for z in /sys/class/hwmon/hwmon*; do [ "$(cat "$z/name" 2> /dev/null)" = macsmc_hwmon ] && echo "power_W=$(awk '{printf "%.1f", $1/1000000}' "$z/power1_input" 2> /dev/null) fan1=$(cat "$z/fan1_input" 2> /dev/null)"; done; }
echo "BLK $T start boot $(cut -c1-8 /proc/sys/kernel/random/boot_id) words: $(grep -o 'asahi\.[a-z_0-9]*=[^ ]*' /proc/cmdline | tr '\n' ' ') pkg-lib $(sha256sum /usr/lib/libvulkan_asahi.so | cut -c1-12) old_tree=$OLD new_tree=$NEW arms=${BLK_ARMS:-V P} direct_V=0 direct_P=<unset> cached_before=$pre $(hw)" >> "$L"
if [ "$pre" = 0 ]; then
  export SCORE_TSV=$TSV
  PYTHONPATH=$S/mlxlm-latest-deps "$PYB" "$S/dl.py" "$KEY-blk" mlx "$repo" >> "$L" 2>&1
fi
grep -q "^DL FAIL" "$L" && { echo "BLK $T ABORT: download failed (key or tsv wrong); no rows run" >> "$L"; exit 1; }
SAMP=/var/tmp/w73-q/row-sampler.sh; CSV=/var/tmp/w73-q/rows-$T.csv
[ -f "$SAMP" ] && { trap 'bash "$SAMP" stop "$CSV" > /dev/null 2>&1' EXIT; ( for f in $(seq 3 30); do eval "exec $f>&-" 2> /dev/null; done; exec bash "$SAMP" start "$CSV" ) >> "$L" 2>&1; echo "BLK $T sampler csv $CSV" >> "$L"; }
for r in $(seq 1 "$ROUNDS"); do
  for a in ${BLK_ARMS:-V P}; do
    case "$a" in
      V) tree=$NEW;  direct=0 ;;
      P) tree=$NEW;  unset direct ;;
      O) tree=$OLD;  unset direct ;;
      *) echo "BLK $T unknown arm $a; skipping" >> "$L"; continue ;;
    esac
    echo "BLK $T round $r $a tree=$tree direct=${direct:-<default>} $(date -u +%T) $(hw) ane-run=$(pgrep -fc '[a]ne-run|[i]dle-guard-ane-job')" >> "$L"
    t0=$(date +%s)
    if [ "${direct:-}" = 0 ]; then
      MLX_OMARCHY_GATHER_QMM_DIRECT=0 SB_SYSTEM=1 bash "$S/runsb.sh" "$TSV" "$T-$a$r" "$PYB" "$tree" "$KEY"
    else
      SB_SYSTEM=1 bash "$S/runsb.sh" "$TSV" "$T-$a$r" "$PYB" "$tree" "$KEY"
    fi
    echo "BLK $T ROWTIME $a$r start=$t0 end=$(date +%s) $(grep -ho 'DECODE [0-9.]* tok/s' "$S/sb-$T-$a$r.log" 2> /dev/null | head -1)" >> "$L"
  done
done
[ -f "$SAMP" ] && bash "$SAMP" stop "$CSV" >> "$L" 2>&1
python3 - "$S" "$T" "$ROUNDS" "$KEY" >> "$L" << 'PY'
import re, statistics as st, sys
S, T, R, K = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
def row(a, r):
    try:
        t = open(f"{S}/sb-{T}-{a}{r}.log").read()
    except OSError:
        return None
    m = re.search(r"DECODE ([\d.]+) tok/s\s+PREFILL ([\d.]+) tok/s", t)
    return (float(m.group(1)), float(m.group(2))) if m else None
arms = [a for a in ("V", "P", "O") if row(a, 1) is not None]
buckets = {a: [x for x in (row(a, r) for r in range(1, R + 1)) if x] for a in arms}
if not buckets.get("V") or not buckets.get("P"):
    print(f"BLK {T} RESULT incomplete V={buckets.get('V')} P={buckets.get('P')} O={buckets.get('O')}")
else:
    for i, n in enumerate(("decode", "prefill")):
        v = [x[i] for x in buckets["V"]]
        p = [x[i] for x in buckets["P"]]
        spread = (max(v) - min(v)) / st.median(v) * 100
        flag = " NOISY(control spread > 5%: re-run, no gain/loss claimed)" if spread > 5 else ""
        line = f"BLK {T} RESULT {K} {n}: V median {st.median(v):.2f} {['%.2f' % x for x in v]} P median {st.median(p):.2f} {['%.2f' % x for x in p]} ratio P/V (new direct / old staged) {st.median(p) / st.median(v):.3f} control spread {spread:.1f}%{flag}"
        if buckets.get("O"):
            o = [x[i] for x in buckets["O"]]
            line += f" O median {st.median(o):.2f} ratio P/O {st.median(p) / st.median(o):.3f}"
        print(line)
PY
echo "BLK $T DONE $(date -u +%T)" >> "$L"
if [ "$pre" = 0 ]; then
  case "$d" in */hub/models--*--*) [ -d "$d" ] && rm -r "$d" && echo "BLK $T purged $d" >> "$L";; esac
  PYTHONPATH=$S/mlxlm-latest-deps "$PYB" "$S/orphans.py" "$H" --go --owner "models--${repo//\//--}" 2> /dev/null | tail -1 >> "$L"
fi
