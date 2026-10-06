#!/bin/bash
# f16 linear-layer m/n sweep on the main-tip wheel: direct route vs
# MLX_OMARCHY_NO_COOPMAT=1 (register-blocked tile), both orientations.
cd /var/tmp/w7q-h12 || exit 1
P=/var/tmp/w7q-h11/venv/bin/python
for o in nt nn; do
  $P ms.py float16 $o > out/ms16-$o-direct.json 2>&1
  MLX_OMARCHY_NO_COOPMAT=1 $P ms.py float16 $o > out/ms16-$o-rb.json 2>&1
done
