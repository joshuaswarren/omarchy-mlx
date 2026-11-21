"""Run under the hardware guard with the wheel under test installed."""

import math
import unittest

import mlx.core as mx


class ElementwiseGpuParity(unittest.TestCase):
    def check_values(self, cpu, gpu):
        for i, (want, got) in enumerate(zip(cpu, gpu, strict=True)):
            with self.subTest(index=i, cpu=want, gpu=got):
                if math.isnan(want):
                    self.assertTrue(math.isnan(got))
                elif want == 0.0:
                    self.assertEqual(got, want)
                    self.assertEqual(math.copysign(1.0, got), math.copysign(1.0, want))
                else:
                    self.assertTrue(math.isclose(got, want, rel_tol=1e-6))

    def test_power(self):
        # Exactness argument (Aurora, 2026-11-21): promote f16 to f32, round
        # at 32 bits, demote back is exact for every f16 input, including
        # ties, both signed zeros and infinities; NaN may quiet, which
        # Vulkan allows. Integers above 1024 are already exact in f16, so
        # no double rounding. The f16 rows below confirm it on hardware.
        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            with self.subTest(dtype=dtype):
                a = mx.array([4, 9, -2, -2, -2, 0.0, -0.0, -0.0, 0, -2], dtype=dtype)
                b = mx.array([0.5, 0.5, 3, 4, -3, 0, 3, -3, 2, 0.5], dtype=dtype)
                cpu, gpu = [mx.power(a, b, stream=d).tolist() for d in (mx.cpu, mx.gpu)]
                self.assertEqual(cpu[:7], [2, 3, -8, 16, -0.125, 1, -0.0])
                self.check_values(cpu, gpu)

    def test_power_large(self):
        # 65504 is the largest finite f16: its square overflows to infinity
        # on every stream, and the negative base keeps the sign through the
        # odd exponent.
        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            with self.subTest(dtype=dtype):
                a = mx.array([65504, 2, -65504], dtype=dtype)
                b = mx.array([2, 10, 3], dtype=dtype)
                cpu, gpu = [mx.power(a, b, stream=d).tolist() for d in (mx.cpu, mx.gpu)]
                self.check_values(cpu, gpu)

    def test_power_broadcast(self):
        a = mx.array([[-2.0], [4.0]])
        b = mx.array([[0.0, 1.0, 2.0, 3.0]])
        cpu, gpu = [mx.power(a, b, stream=d).tolist() for d in (mx.cpu, mx.gpu)]
        for want, got in zip(cpu, gpu, strict=True):
            self.check_values(want, got)

    def test_ceil(self):
        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            with self.subTest(dtype=dtype):
                a = mx.array([-2.5, -1, -0.5, -0.0, 0.0, 0.5, 2.5,
                              1024, 65504, -65504,
                              -math.inf, math.inf, math.nan], dtype=dtype)
                cpu, gpu = [mx.ceil(a, stream=d).tolist() for d in (mx.cpu, mx.gpu)]
                self.check_values(cpu, gpu)


if __name__ == '__main__':
    unittest.main(verbosity=2)
