set -eu
N=$HOME/scratch/w7q-d27/nll
T=$HOME/scratch/w7q-ratio3-wt/receipts/2026-10-19-gdn-bf16a-untiled/tools
O=$HOME/scratch/w7q-r3-stage/nll-cmp.log
{
  echo "nll_cmp $(sha256sum "$T/nll_cmp.py" | cut -c1-16)"
  python3 "$T/nll_cmp.py" "$N/U0.npz,$N/U1.npz" "$N/D0.npz,$N/D1.npz" --label "U minus D"
  python3 "$T/nll_cmp.py" "$N/D2.npz" "$N/D0.npz" --label "D2 minus D (prefixes 0-31)"
} > "$O" 2>&1
cut -c1-200 "$O"
