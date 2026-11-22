import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from qualification_reference import comparison


class ReferenceTest(unittest.TestCase):
    def test_assertion_change_is_new_and_known_skip_is_not_a_pass(self):
        old = {'failures': {'sample': 'vulkan_refusal'}, 'skips': ['not-run']}
        result = comparison(old, {'sample': 'assertion', 'fresh': 'assertion'}, ['not-run', 'fresh-skip'])
        self.assertEqual(result['new_failures'], {'sample': 'assertion', 'fresh': 'assertion'})
        self.assertEqual(result['new_skips'], ['fresh-skip'])
        self.assertEqual(result['matched_skips'], ['not-run'])
        known = comparison(old, {'sample': 'vulkan_refusal'}, [])
        self.assertFalse(known['new_failures'])
        self.assertEqual(known['matched_failures'], {'sample': 'vulkan_refusal'})

    def test_real_pytest_captures_independent_subtests_and_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'test_smoke.py').write_text('''import pytest

def test_twice(subtests):
    with subtests.test(arm=1):
        raise RuntimeError('[omarchy] Sort dtype is not implemented')
    with subtests.test(arm=2):
        assert False

@pytest.mark.skip(reason='fixture skip')
def test_skip():
    assert False
''')
            dest = path / 'events.json'
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parent),
                       PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', QUALIFY_SUITE_EVENTS=str(dest))
            out = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'suite_events',
                                  'test_smoke.py'], cwd=path, env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
            found = json.loads(dest.read_text())
            self.assertEqual(len(found['failures']), 3)
            self.assertEqual(set(found['failures'].values()),
                             {'vulkan_refusal', 'assertion', 'upstream_subtest_summary'})
            self.assertTrue(all(k.startswith('test_smoke.py::test_twice') for k in found['failures']))
            self.assertEqual(found['skips'], ['test_smoke.py::test_skip'])


class ReferenceFromReceiptTest(unittest.TestCase):
    def receipt(self):
        return {
            'schema': 'omarchy-mlx-qualification-v4', 'utc': '2026-11-22T18:00:00+00:00',
            'suite_source': 'a' * 40, 'icd_manifest_sha256': 'c' * 64,
            'versions': {'mlx': '0.32.4', 'os': '7.1'},
            'wheel': {'name': 'w.whl', 'sha256': 'b' * 64, 'vendor_sha256': 'd' * 64},
            'benchmark': {'device': 'Apple M1 Max (G13C C0)'},
            'suite': {'files': {'test_a.py': {'seconds': 1.5, 'ids_complete': True},
                                'test_b.py': {'seconds': 0.25, 'ids_complete': True}},
                      'expected_files': 2,
                      'failures': {'test_a.py::x': 'assertion', 'test_b.py::y': 'vulkan_refusal',
                                   'test_b.py::z': 'vulkan_refusal'},
                      'skips': ['test_a.py::s'], 'passed': 10, 'failed': 3, 'skipped': 1}}

    def test_reference_carries_the_receipt_lists_counts_and_identity(self):
        from qualification_reference import reference_from_receipt
        ref = reference_from_receipt(self.receipt(), 'r7', 'b' * 64, 'e' * 64)
        self.assertEqual(ref['schema'], 'omarchy-mlx-m1-max-reference-v1')
        self.assertEqual(ref['release'], 'r7')
        self.assertEqual(ref['capture_sha256'], 'e' * 64)
        self.assertEqual(ref['wheel_sha256'], 'b' * 64)
        self.assertEqual(ref['suite_source'], 'a' * 40)
        self.assertEqual(ref['device'], 'Apple M1 Max (G13C C0)')
        self.assertEqual(ref['counts'], {'failed': 3, 'passed': 10, 'skipped': 1})
        self.assertEqual(ref['failure_classes'], {'assertion': 1, 'vulkan_refusal': 2})
        self.assertEqual(ref['seconds_per_file'], {'test_a.py': 1.5, 'test_b.py': 0.25})
        self.assertEqual(ref['skips'], ['test_a.py::s'])
        self.assertEqual(len(ref['failures']), 3)

    def test_incomplete_suite_or_other_wheel_is_refused(self):
        from qualification_reference import reference_from_receipt
        r = self.receipt()
        r['suite']['files']['test_b.py']['ids_complete'] = False
        with self.assertRaises(ValueError):
            reference_from_receipt(r, 'r7', 'b' * 64, 'e' * 64)
        with self.assertRaises(ValueError):
            reference_from_receipt(self.receipt(), 'r7', 'f' * 64, 'e' * 64)
        r = self.receipt()
        r['suite']['expected_files'] = 3
        with self.assertRaises(ValueError):
            reference_from_receipt(r, 'r7', 'b' * 64, 'e' * 64)
