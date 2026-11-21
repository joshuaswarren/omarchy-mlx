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
