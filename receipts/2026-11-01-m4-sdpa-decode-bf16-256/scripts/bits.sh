#!/bin/bash
# usage: bits.sh <arm> dump|compare <ref-arm>
# Reopen check, then the four bit-exactness probes; outputs under bits/.
U=$HOME/gpu/upr; A=$1; MODE=$2; REF=${3:-$1}
cd $U/probes; mkdir -p $U/bits; L=$U/logs/bits-$A-$MODE.jsonl
export AB_NOWAIT=1
python3 once.py $L $A -- reopen.py || exit 9
for p in qmv rope sdpa; do
  python3 once.py $L $A -- ${p}_bits.py $MODE $U/bits/$p-$REF.npz || exit 9
done
AB_ENV="HK_APPLE9_COOPMAT=0" python3 once.py $L $A -- qmm_fma_bits.py $MODE $U/bits/qmmfma-$REF.npz || exit 9
python3 once.py $L $A -- reopen.py || exit 9
echo BITS_DONE
