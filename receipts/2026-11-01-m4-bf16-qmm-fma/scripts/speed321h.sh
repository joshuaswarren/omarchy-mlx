#!/bin/bash
# #321 current-head speed A/B: main a69d9e6 against the head's code commit 89b54a0 (wheel p6),
# the receipt's protocol (two interleaved pairs, load below 0.5 and 20 s idle before each
# process). The receipt's cells with the cooperative matrix hidden (HK_APPLE9_COOPMAT=0: the
# route), plus no-regression cells with the driver's default (cooperative matrix exposed: the
# route is not taken). Waits for the #316n queue so nothing else uses the host. Resumable per cell.
set -uo pipefail
U=$HOME/gpu/upr; O=$U/gate/out/speed321h; mkdir -p $O; cd $U/probes
while pgrep -f "^bash ~/gpu/upr/gate/queue316n.sh" > /dev/null; do sleep 30; done
cell() {  # cell <name> <env or ""> <probe args...>
  local n=$1 e=$2; shift 2
  [ -s $O/done-$n ] && { echo "skip $n"; return 0; }
  AB_ENV="$e" python3 ab.py $O/perf-p6h.jsonl 2 main p6 -- "$@" || { echo "AB_FAILED $n"; exit 3; }
  date -u +%FT%TZ > $O/done-$n
}
cell hidden-granite-128 "HK_APPLE9_COOPMAT=0" prefill_bench.py mlx-community/granite-4.0-1b-4bit 128 4
cell hidden-granite-512 "HK_APPLE9_COOPMAT=0" prefill_bench.py mlx-community/granite-4.0-1b-4bit 512 4
cell hidden-qwen35-512 "HK_APPLE9_COOPMAT=0" prefill_bench.py mlx-community/Qwen3.5-2B-MLX-4bit 512 4
cell hidden-granite-decode "HK_APPLE9_COOPMAT=0" decode_bench.py mlx-community/granite-4.0-1b-4bit 64 4
cell default-granite-512 "" prefill_bench.py mlx-community/granite-4.0-1b-4bit 512 4
cell default-granite-decode "" decode_bench.py mlx-community/granite-4.0-1b-4bit 64 4
echo "SPEED321H_DONE $(date -u +%FT%TZ)"
