import unittest

from bf16_ulp import bf16_ulp, within_one_bf16_ulp


class Bf16UlpTests(unittest.TestCase):
    def test_normal_binades(self):
        self.assertEqual(bf16_ulp(24), 0.125)
        self.assertEqual(bf16_ulp(12), 0.0625)
        self.assertEqual(bf16_ulp(32), 0.25)

    def test_subnormal_spacing(self):
        self.assertEqual(bf16_ulp(0), 2.0 ** -133)
        self.assertEqual(bf16_ulp(2.0 ** -127), 2.0 ** -133)

    def test_inclusive_one_ulp_boundary(self):
        self.assertTrue(within_one_bf16_ulp(0.125, 24))
        self.assertFalse(within_one_bf16_ulp(0.1250001, 24))

    def test_nonfinite_magnitude_rejected(self):
        with self.assertRaises(ValueError):
            bf16_ulp(float("inf"))


if __name__ == "__main__":
    unittest.main()
