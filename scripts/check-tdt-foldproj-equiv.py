# Bit-exact check of the TDT chain fold_proj kernel: run against two coreml package roots (e.g. installed vs new) and compare the printed digests.
# usage: python check-tdt-foldproj-equiv.py <dir containing coreml/> [trials]
import hashlib
import sys

sys.path.insert(0, sys.argv[1])
import mlx.core as mx
import numpy as np
from coreml import vulkan_tdt_chain as tc

print("MODULE", tc.__file__, flush=True)
rng = np.random.default_rng(2024)
kern = tc._fold_proj_kernel()
h = hashlib.sha256()
trials = int(sys.argv[2])
for t in range(trials):
    scale = [0.05, 0.5, 2.0, 8.0][t % 4]
    bsum = (rng.standard_normal(25600) * scale).astype(np.float16)
    biases = (rng.standard_normal(5120) * scale).astype(np.float16)
    luts = np.concatenate([1 / (1 + np.exp(-np.linspace(-12, 12, 65536))), np.tanh(np.linspace(-6, 6, 65536))]).astype(np.float16)
    if t % 3 == 0:
        luts = (rng.standard_normal(131072) * 0.5).astype(np.float16)
    c_state = (rng.standard_normal(1280) * scale).astype(np.float32)
    h0 = (rng.standard_normal(640) * scale).astype(np.float32).astype(np.float16).astype(np.float32)
    c0 = (rng.standard_normal(640) * scale).astype(np.float32)
    h_state = (rng.standard_normal(1280) * scale).astype(np.float32)
    pj16_prev = (rng.standard_normal(640) * scale).astype(np.float16)
    projector = (rng.standard_normal(640 * 640 + 640) * (0.05 if t % 2 else 1.0)).astype(np.float16)
    ctl = np.array([0, 0, 0, 0, 0, 0, 0, 1, -1], dtype=np.int32)  # run=1, done=0, skip=0
    outs = kern(
        inputs=[mx.array(bsum), mx.array(biases), mx.array(luts), mx.array(c_state), mx.array(h0), mx.array(c0),
                mx.array(h_state), mx.array(c_state), mx.array(pj16_prev), mx.array(projector), mx.array(ctl)],
        output_shapes=[(1280,), (1280,), (640,), (640,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float16],
        grid=(640, 1, 1), threadgroup=(640, 1, 1), stream=mx.gpu)
    mx.eval(*outs)
    for o in outs:
        h.update(np.array(o).tobytes())
print("FOLDPROJ", trials, h.hexdigest()[:32])
