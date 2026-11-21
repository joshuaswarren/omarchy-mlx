"""Issue 537 defect 1: the qualification suite runner must survive a slow or dying file.

The r3 runner used a fixed 35 s per-file limit and broke out of its loop on
any exit outside (0, 1), so one slow file (test_array.py needs 44 to 47 s on
an M4 Pro) or one SIGABRT file ended the suite with nothing counted.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parents[1] / 'qualification-kit'

CLEAN_EVENTS = {'invalid': 0, 'failures': {}, 'skips': []}
CLEAN_XML = '<testsuite tests="1" failures="0" errors="0" skipped="0"/>'


def load_kit(monkeypatch):
    monkeypatch.syspath_prepend(str(KIT))
    spec = importlib.util.spec_from_file_location('kit_qualify', KIT / 'qualify.py')
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_kit_import_survives_another_reference_module(monkeypatch):
    path = KIT.parent / 'overlay/tools/coreml/reference.py'
    spec = importlib.util.spec_from_file_location('reference', path)
    assert spec is not None and spec.loader is not None
    overlay = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, 'reference', overlay)
    spec.loader.exec_module(overlay)
    mod = load_kit(monkeypatch)
    assert mod.comparison({'failures': {}, 'skips': []}, {'test_x': 'assertion'}, [])[
        'new_failures'
    ] == {'test_x': 'assertion'}
    assert sys.modules['reference'] is overlay


@pytest.fixture()
def kit(tmp_path, monkeypatch):
    monkeypatch.delenv('QUALIFY_PER_FILE_TIMEOUT', raising=False)
    mod = load_kit(monkeypatch)
    suite_dir = tmp_path / 'suite'
    suite_dir.mkdir()
    for name in ('test_a.py', 'test_b.py'):
        (suite_dir / name).write_text(f'def test_{name[5]}():\n    assert True\n')
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    calls = []
    scripts = {}

    def fake_command(argv, env, seconds, cwd=None):
        name = Path(argv[-1]).name
        calls.append({'file': name, 'seconds': seconds})
        script = scripts[name]
        if script.get('events'):
            (tmp_path / 'suite-events.json').write_text(json.dumps(script['events']))
            (tmp_path / 'suite.xml').write_text(script['xml'])
        return script['rc'], ''

    monkeypatch.setattr(mod, 'command', fake_command)
    return mod, calls, scripts


def run_suite(kit, tmp_path):
    mod, calls, _ = kit
    result = mod.suite('python3', tmp_path, {})
    return result, calls


def test_measured_default_and_recorded_seconds(kit, tmp_path):
    _, _, scripts = kit
    for name in ('test_a.py', 'test_b.py'):
        scripts[name] = {'rc': 0}
    result, calls = run_suite(kit, tmp_path)
    assert [call['seconds'] for call in calls] == [300, 300]
    assert all(row['seconds'] >= 0 for row in result['files'].values())


def test_timeout_recorded_and_suite_continues(kit, tmp_path):
    mod, calls, scripts = kit
    scripts['test_a.py'] = {'rc': 124}
    scripts['test_b.py'] = {'rc': 0}
    result, calls = run_suite(kit, tmp_path)
    assert [c['file'] for c in calls] == ['test_a.py', 'test_b.py'], \
        'a timed-out file must not end the suite'
    assert result['files']['test_a.py']['timed_out'] is True
    assert result['files']['test_b.py']['exit'] == 0


def test_abort_recorded_and_suite_continues(kit, tmp_path):
    mod, calls, scripts = kit
    scripts['test_a.py'] = {'rc': -6}
    scripts['test_b.py'] = {'rc': 0}
    result, calls = run_suite(kit, tmp_path)
    assert [c['file'] for c in calls] == ['test_a.py', 'test_b.py'], \
        'an aborting file must not end the suite'
    assert result['files']['test_a.py']['aborted'] is True
    assert result['files']['test_b.py']['exit'] == 0


def test_complete_files_accumulate_diff_inputs(kit, tmp_path):
    mod, calls, scripts = kit
    scripts['test_a.py'] = {'rc': 0, 'events': {
        'invalid': 0,
        'failures': {'test_a.py::test_x': 'boom'},
        'skips': ['test_a.py::test_s']},
        'xml': '<testsuite tests="2" failures="1" errors="0" skipped="1"/>'}
    scripts['test_b.py'] = {'rc': 0, 'events': CLEAN_EVENTS, 'xml': CLEAN_XML}
    result, _ = run_suite(kit, tmp_path)
    assert result['failures'] == {'test_a.py::test_x': 'boom'}
    assert result['skips'] == ['test_a.py::test_s']
    assert result['files']['test_a.py']['ids_complete'] is True
    assert result['files']['test_a.py']['counts'] == {
        'passed': 0, 'failed': 1, 'skipped': 1}
