"""Captured-operand doctest for the MLX_OMARCHY_QMV_GEOM qmv geometry twins.

The geometry variants (ROWS_PER_SLOT 4x2 and 8x1, see compute.h) must keep
every output row's accumulation chain identical to the shipped 2x4 kernels:
same words per lane, same quad order, same subgroupAdd pairing. This test
freezes one operand set per Qwen3.8-2B decode shape (plus the non-512-K and
N%8 tail shapes), computes a sha256 digest of the outputs in one subprocess
per arm (the env is read once per process), and requires every geometry arm
to agree bit for bit with the shipped kernels.

Note: the scalar nibble path (MLX_OMARCHY_QMM_VEC_Q4_WORD=0) is NOT an
equality arm - its per-lane k grouping differs from the packed word path,
and the two converge only under honeykrisp's 32-lane reduction (measured,
receipts/2026-09-09-q4-gemv-order); on other drivers the digests legitimately
differ (observed on llvmpipe, which this test's dev-box run uses).
"""

import hashlib
import json
import os
import subprocess
import sys
import textwrap
import unittest

# Qwen3.8-2B decode GEMV shapes (name, N, K) plus order-edge shapes:
# K=2112 (K%64==0 but not the 512 fast tile) and N=2044 (N%8!=0, the
# clamped-row tail both geometries must share).
SHAPES = [
    ("gdn_qkv", 6144, 2048),
    ("gdn_z", 2048, 2048),
    ("attn_q", 4096, 2048),
    ("attn_k", 512, 2048),
    ("attn_o", 2048, 2048),
    ("mlp_down", 2048, 6144),
    ("lm_head", 248320, 2048),
    ("tail_k", 2048, 2112),
    ("tail_n", 2044, 2048),
]

ARMS = [
    ("shipped", {}),
    ("geom42", {"MLX_OMARCHY_QMV_GEOM": "42"}),
    ("geom81", {"MLX_OMARCHY_QMV_GEOM": "81"}),
]

GROUP_SIZE = 64

_PROBE = textwrap.dedent(
    """
    import hashlib, json, os
    import mlx.core as mx
    import numpy as np

    mx.set_default_device(mx.gpu)

    group_size = int(os.environ["QMV_TEST_GS"])
    shapes = json.loads(os.environ["QMV_TEST_SHAPES"])
    dt = mx.bfloat16 if os.environ["QMV_TEST_DTYPE"] == "bf16" else mx.float16
    out = {}
    for name, n, k in shapes:
        mx.random.seed(0)
        w = mx.random.randint(0, 2**31 - 1, (n, k // 8)).astype(mx.uint32)
        sc = (mx.random.uniform(shape=(n, k // group_size)) * 0.02 + 0.001).astype(dt)
        bi = (mx.random.uniform(shape=(n, k // group_size)) * 0.01).astype(dt)
        x = mx.random.normal((1, k)).astype(dt)
        y = mx.quantized_matmul(x, w, sc, bi, transpose=True,
                                group_size=group_size, bits=4)
        mx.eval(y)
        out[name] = hashlib.sha256(
            np.asarray(y.view(mx.uint16)).tobytes()).hexdigest()
    print(json.dumps(out))
    """
)


class QmvGeomDigest(unittest.TestCase):
    def _digests(self, dtype_env, extra_env):
        env = dict(os.environ)
        env.update(
            QMV_TEST_GS=str(GROUP_SIZE),
            QMV_TEST_SHAPES=json.dumps(SHAPES),
            QMV_TEST_DTYPE=dtype_env,
        )
        for key, value in extra_env.items():
            env[key] = value
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if proc.returncode != 0:
            self.fail(f"arm failed ({extra_env}): {proc.stderr[-2000:]}")
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def _check_dtype(self, dtype_env):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy unavailable for byte-exact digesting")
        per_arm = {}
        for name, extra in ARMS:
            arm_env = dict(extra)
            per_arm[name] = self._digests(dtype_env, arm_env)
        reference = per_arm["shipped"]
        for name, digests in per_arm.items():
            for shape, digest in digests.items():
                self.assertEqual(
                    digest,
                    reference[shape],
                    f"{name} digest differs from shipped on {shape} "
                    f"({dtype_env})",
                )

    def test_bf16_digest_identity(self):
        self._check_dtype("bf16")

    def test_f16_digest_identity(self):
        self._check_dtype("f16")


if __name__ == "__main__":
    unittest.main()
