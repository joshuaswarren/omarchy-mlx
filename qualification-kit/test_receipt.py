import unittest
from qualify import counts, ane_result, benchmark_result


class ReceiptTest(unittest.TestCase):
    def test_counts_never_passing_empty_or_inconsistent(self):
        self.assertIsNone(counts('<testsuite tests="0"/>'))
        self.assertIsNone(counts('<testsuite tests="2" failures="2" skipped="1"/>'))
        self.assertEqual(counts('<testsuite tests="5" failures="1" errors="1" skipped="1"/>'),
                         dict(passed=2, failed=2, skipped=1))

    def test_ane_unsupported_is_not_pass(self):
        self.assertEqual(ane_result(2, {}, False)['status'], 'unavailable')
        self.assertEqual(ane_result(0, {'available': True, 'errors': 0, 'sha256': []}, False)['status'], 'fail')
        self.assertEqual(ane_result(0, {'available': True, 'errors': 0, 'sha256': ['a'] * 20, 'golden_sha256': 'a'}, False)['status'], 'pass')
        self.assertEqual(ane_result(0, {'available': True}, True)['status'], 'coreml_pass_placement_unverified')

    def test_decode_needs_full_run_and_finite_rates(self):
        r = dict(generated=64, prompt_tokens=12, decode_tps=12, prefill_tps=50,
                 prefill_s=0.24, ids_sha256_16='0123456789abcdef', device='Apple M1')
        self.assertIsNotNone(benchmark_result(r))
        self.assertIsNone(benchmark_result(dict(r, generated=63)))
        self.assertIsNone(benchmark_result(dict(r, decode_tps=float('nan'))))
        self.assertIsNone(benchmark_result(dict(r, device='llvmpipe')))
