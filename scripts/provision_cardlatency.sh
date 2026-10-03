#!/usr/bin/env bash
# Re-provision /tmp/CardLatency on jw14m2 from the dev-box worktree.
# Usage: provision_cardlatency.sh   (run from the dev box)
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
H=${CARDLATENCY_HOST:-jw14m2-linux}
D=${CARDLATENCY_DIR:-/tmp/CardLatency}
ssh $H "mkdir -p $D/base/serve $D/base/scripts $D/after/serve $D/after/scripts"
# base/serve must be pristine origin/main -- never the lever worktree.
git -C $W archive origin/main serve | ssh $H "tar -x -C $D/base"
for build in base after; do
  [ "$build" = after ] && rsync -a --delete --exclude __pycache__ $W/serve/ $H:$D/$build/serve/
  rsync -a $W/scripts/pair_memory_v2.py $W/scripts/card_timing_boot.py \
        $W/scripts/card_decompose.py \
        $W/scripts/mlx_provenance.py $W/scripts/dev_subset.py \
        $H:$D/$build/scripts/
  rsync -a $W/tests/fixtures/cards_dev.json $W/tests/fixtures/cards_held_out_v4.json \
        $H:$D/$build/scripts/
done
rsync -a $W/scripts/lane_wait.sh $W/scripts/gap_lane.sh $W/scripts/ticket_card_decompose.sh $W/scripts/ticket_card_final.sh \
      $W/scripts/ticket_card_after.sh $W/scripts/ticket_dev_suite.sh \
      $W/scripts/ticket_v4_suite.sh $H:$D/
ssh $H "mkdir -p $D/receipt"
rsync -a $W/receipts/2026-10-02-card-latency/run_card_latency_suite.py $H:$D/receipt/
ssh $H "chmod +x $D/*.sh && python3 $D/after/scripts/dev_subset.py $D/after/scripts/cards_dev.json $D/dev_subset.json"
ssh $H "diff <(grep -c '' $D/base/serve/mlx_omarchy_assistant/coordinator.py) <(grep -c '' $D/after/serve/mlx_omarchy_assistant/coordinator.py) >/dev/null && echo 'BUILDS IDENTICAL — after sync is stale' || echo 'builds differ as expected'"
echo provisioned
