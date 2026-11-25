"""Run under the hardware guard on a device advertising a bf16 subgroup
cooperative-matrix shape (ANV/Xe2: 8x16x16 bf16 x bf16 -> f32).

MLX_OMARCHY_QMM_INTEL=1 routes bf16 q4_g64 transposed prefill QMMs to
QmmPrefillIntelBF16 instead of QmmPrefillFmaBF16. The route flag is read
once per process, so each arm runs in a subprocess. The intel kernel
rounds its weight operands to bf16 while the FMA kernel keeps them f32,
so the outputs are not bit-identical; the check is that the intel
route's error against the f32 dequantize-matmul reference is no worse
than the shipped route's.

DEC-19 adds QmmPrefillCoopmat16XeBF16 (fp16 operands on the same ANV
8x16x16 family, staged x so ragged matrix_m works): arms here pin
MLX_OMARCHY_NO_QMM_XE16=1 to keep the Intel-vs-FMA contract, and
test_xe16_covers_ragged_m covers the ragged-m shapes it newly takes.
"""

import os
import subprocess
import sys
import tempfile
import unittest

import mlx.core as mx
import numpy as np

ARM = """
import sys
import mlx.core as mx
import numpy as np

m_rows = int(sys.argv[2])
mx.random.seed(0)
w = mx.random.normal((512, 512)).astype(mx.bfloat16)
wq, scales, biases = mx.quantize(w, group_size=64, bits=4)
x = mx.random.normal((m_rows, 512)).astype(mx.bfloat16)
y = mx.quantized_matmul(
    x, wq, scales, biases, transpose=True, group_size=64, bits=4)
wd = mx.dequantize(wq, scales, biases, group_size=64, bits=4)
ref = x.astype(mx.float32) @ wd.astype(mx.float32).T
mx.eval(y, ref)
np.savez(sys.argv[1], y=np.array(y.astype(mx.float32)),
         ref=np.array(ref))
"""


class QmmIntelParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not mx.device_info().get("cooperative_matrix_bf16"):
            raise unittest.SkipTest(
                "device advertises no bf16 subgroup coopmat shape")

    def run_arm(self, intel, m_rows=64, no_xe16=True):
        env = dict(
            os.environ,
            MLX_OMARCHY_QMM_INTEL="1" if intel else "0",
            MLX_OMARCHY_NO_QMM_XE16="1" if no_xe16 else "0")
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "y.npz")
            subprocess.run(
                [sys.executable, "-c", ARM, out, str(m_rows)], env=env,
                check=True, capture_output=True, timeout=120)
            d = np.load(out)
            return d["y"], d["ref"]

    def test_intel_matches_current_kernel(self):
        ctl_y, ref = self.run_arm(False)
        intel_y, _ = self.run_arm(True)
        # Intel error vs the f32 dequant reference must not exceed the
        # shipped kernel's by more than the bf16 rounding slack.
        ctl_err = np.abs(ctl_y - ref).max()
        intel_err = np.abs(intel_y - ref).max()
        # bf16 operands carry ~1.7x the f32 route's max error on this
        # shape (measured 0.446 vs 0.262); 2x leaves margin for driver
        # rounding differences while still failing a broken kernel.
        self.assertLessEqual(intel_err, ctl_err * 2.0 + 1e-3)
        np.testing.assert_allclose(intel_y, ctl_y, rtol=5e-2, atol=0.6)
        flips = (intel_y.argmax(axis=-1) != ctl_y.argmax(axis=-1)).sum()
        self.assertLessEqual(flips, 4)

    def test_non_16_m_falls_back(self):
        # matrix_m is not a multiple of the kernel's 16-row tile: the
        # intel route must decline the call (its global A loads read
        # whole 16-row blocks) and the FMA path below must answer it.
        # MLX_OMARCHY_NO_QMM_XE16=1 (the run_arm default) keeps the
        # staged-x Xe16 variant out of this comparison.
        ctl_y, ref = self.run_arm(False, m_rows=20)
        intel_y, _ = self.run_arm(True, m_rows=20)
        np.testing.assert_allclose(intel_y, ctl_y, rtol=1e-3, atol=1e-3)

    def test_xe16_covers_ragged_m(self):
        # QmmPrefillCoopmat16XeBF16 (fp16 operands on the ANV 8x16x16
        # shape) stages x through shared, so a ragged matrix_m tile
        # zero-fills: the m=20 call the intel route declines lands on
        # it. fp16 operand rounding puts it in the same error class as
        # the intel route, not the FMA route's bit-level agreement.
        ctl_y, ref = self.run_arm(False, m_rows=20)
        xe_y, _ = self.run_arm(False, m_rows=20, no_xe16=False)
        ctl_err = np.abs(ctl_y - ref).max()
        xe_err = np.abs(xe_y - ref).max()
        self.assertLessEqual(xe_err, ctl_err * 2.0 + 1e-3)
        np.testing.assert_allclose(xe_y, ctl_y, rtol=5e-2, atol=0.6)
        flips = (xe_y.argmax(axis=-1) != ctl_y.argmax(axis=-1)).sum()
        self.assertLessEqual(flips, 4)


if __name__ == "__main__":
    unittest.main()
