import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import qualify
from qualify import ane_result, benchmark_result, counts


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


class QualificationRunTest(unittest.TestCase):
    def run_qualification(self, gradient, probe_exit=0):
        expected = [[-0.25, 0.0], [0.125, 0.0]]
        ops = [{'op': name, 'pass': True, 'expected': 1, 'actual': None} for name in
               ('add', 'matmul', 'softmax', 'argmax', 'cumsum')]
        ops.append({'op': 'value_and_grad', 'pass': gradient == expected,
                    'expected': expected, 'actual': gradient})
        golden = {'available': True, 'errors': 0, 'sha256': ['a'] * 20, 'golden_sha256': 'a'}
        decode = {'generated': 64, 'prompt_tokens': 12, 'decode_tps': 12, 'prefill_tps': 50,
                  'prefill_s': 0.24, 'ids_sha256_16': '0123456789abcdef', 'device': 'Apple M1'}
        suite = {'files': {'test_array.py': {'ids_complete': True, 'exit': 0}},
                 'expected_files': 1, 'passed': 1, 'failed': 0, 'failures': {}, 'skips': []}
        args = SimpleNamespace(icd=None, model_dir=Path('pinned-model'))
        receipt = {'versions': {}}
        with (patch.object(qualify, 'setup', return_value=('python', None)),
              patch.object(qualify, 'verify', return_value={}),
              patch.object(qualify, 'command', side_effect=[
                  (probe_exit, json.dumps({'available': True, 'ops': ops})),
                  (0, json.dumps(golden)), (0, json.dumps(decode))]),
              patch.object(qualify, 'load_model', return_value=args.model_dir),
              patch.object(qualify, 'suite', return_value=suite)):
            status = qualify.run(args, Path('.'), False, {}, receipt)
        return status, receipt

    def test_wrong_gpu_gradient_is_reported_without_losing_decode_or_suite(self):
        zeros = [[0.0, 0.0], [0.0, 0.0]]
        status, receipt = self.run_qualification(zeros)
        self.assertEqual(status, 'gpu_correctness_failed')
        self.assertEqual(receipt['mlx_checks']['failures'], {
            'value_and_grad': {'expected': [[-0.25, 0.0], [0.125, 0.0]], 'actual': zeros}})
        self.assertEqual(receipt['benchmark']['decode_tps'], 12)
        self.assertEqual(receipt['suite']['passed'], 1)

    def test_clean_gpu_is_still_qualified(self):
        status, receipt = self.run_qualification([[-0.25, 0.0], [0.125, 0.0]])
        self.assertEqual(status, 'qualified')
        self.assertEqual(receipt['mlx_checks']['passed'], 6)

    def test_interrupted_probe_still_aborts(self):
        with self.assertRaisesRegex(RuntimeError, 'gpu_correctness_failed'):
            self.run_qualification([[0.0, 0.0], [0.0, 0.0]], probe_exit=124)
