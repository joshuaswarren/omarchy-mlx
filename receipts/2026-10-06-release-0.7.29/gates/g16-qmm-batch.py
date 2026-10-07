#!/usr/bin/env python3
"""g16-qmm-batch — installed-wheel probe for the batched bf16 coopmat QMM
row-mixing fix (cf41e9b6f, the X_F32 batch-stride defect that shipped in
v0.7.29). Runs on the INSTALLED wheel and asserts BATCH INDEPENDENCE:
quantized_matmul over [B, T, K] must be BIT-IDENTICAL per batch row to the
same row run alone (B in {2, 4}; the w7B failing shape N=6144 plus a
non-tile-multiple T=17). The fp32/fp64 absolute reference lives in
g16b-qmm-route-probe.py (per-row error vs fp64 dequantize-on-CPU, fail at
>3x the same shape's flat max). This probe prints the wheel version, the
mlx libmlx.so sha256, the ICD json + libvulkan sha256 + driverInfo, and
uname -r so the receipt names the exact artifact and driver."""
import hashlib
import os
import json
import platform
import sys

import mlx.core as mx

K, N = 512, 6144
G64, BITS = 64, 4
mx.random.seed(0x0730)

import importlib.metadata as md

WHEEL_VERSION = md.version("mlx_omarchy")
import mlx
import pathlib

mlx_pkg_dir = pathlib.Path(list(mlx.__path__)[0])
libmlx = mlx_pkg_dir / "lib" / "libmlx.so"
LIBMLX_SHA = hashlib.sha256(libmlx.read_bytes()).hexdigest() if libmlx.exists() else "absent"

ICD_JSON = os.environ.get(
    "VK_DRIVER_FILES", "/usr/lib/omarchy-mlx/vulkan/honeykrisp_icd.aarch64.json")
LIBV_SHA = "unavailable"
try:
    libv = json.load(open(ICD_JSON))["ICD"]["library_path"]
    libv = libv.replace("$DEST", "")
    if libv.startswith("~"):
        libv = os.path.expanduser(libv)
    LIBV_SHA = hashlib.sha256(open(libv, "rb").read()).hexdigest()
except Exception as exc:
    LIBV_SHA = f"error: {exc}"

print(json.dumps({
    "wheel_version": WHEEL_VERSION,
    "libmlx_sha256": LIBMLX_SHA,
    "icd_json": ICD_JSON,
    "libvulkan_sha256": LIBV_SHA,
    "uname": platform.uname().release,
}, indent=2))

w = mx.random.normal((N, K)).astype(mx.bfloat16)
wq, scales, biases = mx.quantize(w, group_size=G64, bits=BITS)

failures = []
digests = []
cases = 0
for B in (2, 4):
    for T in (16, 17):
        x = mx.random.normal((B, T, K)).astype(mx.bfloat16)
        batched = mx.quantized_matmul(x, wq, scales, biases, transpose=True,
                                      group_size=G64, bits=BITS)
        for i in range(B):
            alone = mx.quantized_matmul(x[i:i + 1], wq, scales, biases,
                                        transpose=True, group_size=G64, bits=BITS)
            same = mx.array_equal(batched[i], alone[0])
            import numpy as np

            d = hashlib.sha256(
                np.asarray(batched[i].astype(mx.float32),
                           dtype=np.float32).tobytes()).hexdigest()[:16]
            digests.append({"case": f"B{B}T{T}", "row": i,
                            "bit_identical_to_alone": bool(same), "digest": d})
            if not same:
                failures.append(f"B{B}T{T} row {i}: batched != alone")
        cases += 1

ok = not failures and bool(digests)
print(json.dumps({"gate": "g16-qmm-batch", "cases": cases,
                  "rows": len(digests), "failures": failures,
                  "digests": digests, "pass": bool(ok)}, indent=2))
sys.exit(0 if ok else 1)
