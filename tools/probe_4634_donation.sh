#!/bin/bash
# lane UpstreamMlx - 1008b (a) step 1: foreign-buffer donation probe (mlx#4634)
set -uo pipefail
mkdir -p "$HOME/u1008b-probe"
export VK_DRIVER_FILES=/usr/lib/omarchy-mlx/vulkan/honeykrisp_icd.aarch64.json
PY=/usr/lib/omarchy-mlx/venv/bin/python
"$PY" - > "$HOME/u1008b-probe/4634-donation.log" 2>&1 << 'PROBE'
import os, tempfile
import numpy as np
import mlx.core as mx

print("probe: mlx#4634 foreign-buffer donation on the omarchy Vulkan wheel")
print("mlx module:", mx.__file__)

a = np.arange(4096, dtype=np.float32)
before = a.copy()

# CASE 1: explicit zero-copy import
try:
    x = mx.asarray(a, copy=False)
    y = mx.exp(x)
    mx.eval(y)
    print("CASE1 asarray(copy=False): imported ok; numpy_overwritten=%s"
          % (not np.array_equal(a, before)))
except Exception as exc:
    print("CASE1 asarray(copy=False): RAISED %s: %s"
          % (type(exc).__name__, str(exc)[:140]))

# CASE 2: default import (copy=None adopt-or-copy)
x2 = mx.array(a)
y2 = mx.exp(x2)
mx.eval(y2)
print("CASE2 mx.array(default): numpy_overwritten=%s"
      % (not np.array_equal(a, before)))

# CASE 3: read-only memmap
path = os.path.join(tempfile.gettempdir(), "u1008b-mm.bin")
np.arange(4096, dtype=np.float32).tofile(path)
mm = np.memmap(path, dtype=np.float32, mode="r", shape=(4096,))
snapshot = np.array(mm)
try:
    xm = mx.asarray(mm, copy=False)
    ym = mx.exp(xm)
    mx.eval(ym)
    print("CASE3 memmap(mode=r) asarray(copy=False): memmap_intact=%s"
          % np.array_equal(np.array(mm), snapshot))
except Exception as exc:
    print("CASE3 memmap(mode=r): RAISED %s: %s"
          % (type(exc).__name__, str(exc)[:140]))

# CASE 4: default import of the memmap
xm2 = mx.array(mm)
ym2 = mx.exp(xm2)
mx.eval(ym2)
print("CASE4 memmap(default): memmap_intact=%s"
      % np.array_equal(np.array(mm), snapshot))
print("probe done")
PROBE
echo "exit=$?" >> "$HOME/u1008b-probe/4634-donation.log"
cat "$HOME/u1008b-probe/4634-donation.log"
