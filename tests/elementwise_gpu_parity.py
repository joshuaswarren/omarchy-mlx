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

    def test_int64_divmod_compiles(self):
        # G18G regression (G626 battery 2026-11-23): the int64 div-mod
        # lowering emits an AGX extr bitfield extract that the AGX3 binary
        # packer cannot encode (agx3_pack_binary.c:19234, "no grounded AGX3
        # encoding for extr"), so pipeline creation failed. The word-pair
        # div-mod in the shader keeps every operation on the uint32 halves.
        an = [x for x in range(-97, 98)]
        bn = [y for y in range(1, 97)]
        a = mx.array([an[i % len(an)] for i in range(256)], dtype=mx.int64)
        b = mx.array([bn[i % len(bn)] for i in range(256)], dtype=mx.int64)
        gpu_q = mx.floor_divide(a, b, stream=mx.gpu)
        gpu_r = mx.remainder(a, b, stream=mx.gpu)
        cpu_q = mx.floor_divide(a, b, stream=mx.cpu)
        cpu_r = mx.remainder(a, b, stream=mx.cpu)
        self.assertEqual(gpu_q.tolist(), cpu_q.tolist())
        self.assertEqual(gpu_r.tolist(), cpu_r.tolist())
        gpu_dq, gpu_dr = mx.divmod(a, b, stream=mx.gpu)
        cpu_dq, cpu_dr = mx.divmod(a, b, stream=mx.cpu)
        self.assertEqual(gpu_dq.tolist(), cpu_dq.tolist())
        self.assertEqual(gpu_dr.tolist(), cpu_dr.tolist())
        identity = [(q * bi + r) for q, bi, r in
                    zip(gpu_q.tolist(), b.tolist(), gpu_r.tolist(), strict=True)]
        self.assertEqual(identity, a.tolist())

    def test_uint64_divmod_magnitude_path(self):
        # uint64 lane (review, wAW): operands at or above 2^63 exercise the
        # unsigned magnitude path, which the first submission mis-negated
        # (1 // (2^64-1) returned 1, must be 0).
        au = [2 ** 64 - 1, 2 ** 64 - 2, 2 ** 63, 2 ** 63 - 1, 7, 3, 1, 0]
        bu = [2 ** 63, 5, 1, 9, 3]
        a_u = mx.array([au[i % len(au)] for i in range(64)], dtype=mx.uint64)
        b_u = mx.array([bu[i % len(bu)] for i in range(64)], dtype=mx.uint64)
        gpu_qu = mx.floor_divide(a_u, b_u, stream=mx.gpu)
        gpu_ru = mx.remainder(a_u, b_u, stream=mx.gpu)
        cpu_qu = mx.floor_divide(a_u, b_u, stream=mx.cpu)
        cpu_ru = mx.remainder(a_u, b_u, stream=mx.cpu)
        self.assertEqual(gpu_qu.tolist(), cpu_qu.tolist())
        self.assertEqual(gpu_ru.tolist(), cpu_ru.tolist())
        gpu_dqu, gpu_dru = mx.divmod(a_u, b_u, stream=mx.gpu)
        self.assertEqual(gpu_dqu.tolist(), cpu_qu.tolist())
        self.assertEqual(gpu_dru.tolist(), cpu_ru.tolist())
        # 1 // (2^64-1) must be 0 (the review's exact case).
        self.assertEqual(
            mx.floor_divide(mx.array([1], dtype=mx.uint64),
                            mx.array([2 ** 64 - 1], dtype=mx.uint64),
                            stream=mx.gpu).tolist(), [0])

    def test_sign(self):
        # G18G regression (issue 544, G626 battery 2026-11-23): with the sign
        # case inside the elementwise switch, the AGX3 packer aliased it with
        # the remainder case and every nonzero lane read 0.0 (253 of 256
        # lanes wrong on the M6). Upstream Sign compares against zero, so NaN
        # maps to 0 on both streams; everything else must match the CPU.
        samples = [-6.0, -1.0, -0.25, -0.0, 0.0, 0.25, 1.0, 6.0,
                   1024.0, -1024.0, math.inf, -math.inf, math.nan]
        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            with self.subTest(dtype=dtype):
                gpu = mx.sign(mx.array(samples, dtype=dtype), stream=mx.gpu)
                cpu = mx.sign(mx.array(samples, dtype=dtype), stream=mx.cpu)
                for want, got in zip(cpu.tolist(), gpu.tolist(), strict=True):
                    if math.isnan(want):
                        continue  # NaN may quiet to 0 on the GPU path
                    with self.subTest(want=want, got=got):
                        self.assertEqual(got, want)
                        if want != 0.0:
                            self.assertEqual(math.copysign(1.0, got),
                                             math.copysign(1.0, want))
        # Wide lane sweep: every nonzero lane must read ±1 on the GPU.
        rng_samples = [x / 7.0 for x in range(-253, 254) if x != 0]
        gpu = mx.sign(mx.array(rng_samples, dtype=mx.float32), stream=mx.gpu)
        self.assertEqual(sorted(set(gpu.tolist())), [-1.0, 1.0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
