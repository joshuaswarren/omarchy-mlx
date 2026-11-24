import contextlib
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import qualify
from qualify import ane_result, benchmark_result, counts

from scripts import bench_decode, mlx_provenance

DEEP_JSON = '[' * 200_000 + ']' * 200_000


def on_pinned_stack(fn):
    """Run fn on a thread with a 4 MiB stack, so nesting limits do not follow
    the runner's ulimit -s (Python 3.14 bounds recursion by the C stack)."""
    outcome = {}

    def target():
        try:
            outcome['value'] = fn()
        except BaseException as exc:  # re-raised in the caller
            outcome['error'] = exc

    previous = threading.stack_size(4 * 1024 * 1024)
    try:
        thread = threading.Thread(target=target)
        thread.start()
    finally:
        threading.stack_size(previous)
    thread.join()
    if 'error' in outcome:
        raise outcome['error']
    return outcome['value']


def run_main(*argv):
    """Run the CLI on the pinned stack with a strict UTF-8 stdout, like a real pipe."""
    out = io.TextIOWrapper(io.BytesIO(), encoding='utf-8', write_through=True)
    with contextlib.redirect_stdout(out), patch.object(sys, 'argv', ['qualify.py', *map(str, argv)]):
        rc = on_pinned_stack(qualify.main)
    return rc, out.buffer.getvalue().decode()


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



class CardRowTest(unittest.TestCase):
    LINUX = {
        'arm': 'linux-release-wheel', 'status': 'new_suite_failures', 'release': 'qualification-r6',
        'machine': {'model': 'apple,j616s', 'chip': 'apple,t6040'},
        'versions': {'os': '7.1.12-aurora', 'mlx': '0.32.4', 'mlx_lm': '0.28.4', 'python': '3.14.0'},
        'mlx_checks': {'passed': 6, 'total': 6, 'failed_ops': []},
        'suite': {'expected_files': 34, 'files': {f'test_{i}.py': {} for i in range(34)},
                  'passed': 13211, 'failed': 81, 'skipped': 81,
                  'new_failures': ['test_export.py::TestExport::test_export_unique'], 'new_skips': []},
        'benchmark': {'decode_tps': 75.712, 'prefill_tps': 160.04, 'generated': 64,
                      'prompt_tokens': 25, 'ids_sha256_16': '0123456789abcdef'},
        'ane': {'status': 'unavailable', 'exit': 127},
    }

    def test_full_linux_receipt_prints_every_field(self):
        self.assertEqual(
            qualify.card_row(self.LINUX),
            'apple,t6040 (apple,j616s), Linux 7.1.12-aurora, mlx-omarchy 0.32.4, kit qualification-r6: '
            'new_suite_failures. MLX checks 6/6. Tests 13211 passed, 81 failed, 81 skipped, 34/34 files; '
            'new against the M1 Max reference: 1 failure (test_export.py::TestExport::test_export_unique), '
            '0 skips. Decode 75.71 tok/s, prefill 160.04 tok/s, 64 tokens, ids 0123456789abcdef. '
            'Neural Engine: unavailable.')

    def test_incomplete_receipt_names_what_did_not_run(self):
        r = {'arm': 'linux-release-wheel', 'status': 'incomplete', 'release': 'qualification-r6',
             'machine': {'model': 'apple,j615', 'chip': 'apple,t8122'}, 'versions': {'os': '7.2.11'},
             'mlx_checks': None, 'suite': None, 'benchmark': None, 'ane': None,
             'error': 'gpu_or_wheel_verification_exit_1'}
        self.assertEqual(
            qualify.card_row(r),
            'apple,t8122 (apple,j615), Linux 7.2.11, kit qualification-r6: incomplete '
            '(gpu_or_wheel_verification_exit_1). MLX checks not run. Tests not run. Decode not run. '
            'Neural Engine: not run.')

    def test_row_flag_prints_the_row_of_a_saved_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'linux.json'
            path.write_text(json.dumps(self.LINUX))
            out = subprocess.run([sys.executable, str(Path(qualify.__file__)), '--row', str(path)],
                                 capture_output=True, text=True, check=True).stdout
        self.assertEqual(out.strip(), qualify.card_row(self.LINUX))

    def test_row_refuses_a_receipt_it_cannot_read_or_render(self):
        deep_field = '{"a": ' * 30_000 + '1' + '}' * 30_000
        with self.assertRaises(RecursionError):
            on_pinned_stack(lambda: json.loads(DEEP_JSON))
        with self.assertRaises(RecursionError):  # 3.14 decodes it and overflows in str(); 3.11 at decode
            on_pinned_stack(lambda: str(json.loads(deep_field)))
        self.addCleanup(sys.set_int_max_str_digits, sys.get_int_max_str_digits())
        sys.set_int_max_str_digits(4300)
        bench = {'decode_tps': 1, 'prefill_tps': 1, 'generated': 64, 'ids_sha256_16': '0'}
        receipts = {
            'missing': (None, 'FileNotFoundError'),
            'not_an_object': (b'[1]', 'ValueError: invalid_receipt'),
            'invalid_utf8': (b'{"status": "ok\xff"}', 'UnicodeDecodeError'),
            'invalid_json': (b'{not json', 'JSONDecodeError'),
            'huge_int': (b'{"status": ' + b'1' * 5000 + b'}', 'ValueError'),
            'deep': (DEEP_JSON.encode(), 'RecursionError'),
            'deep_field': (b'{"machine": {"chip": ' + deep_field.encode() + b'}}', 'RecursionError'),
            'surrogate': (b'{"status": "\\ud800"}', 'UnicodeEncodeError'),
            'machine_not_an_object': (b'{"machine": [1]}', 'AttributeError'),
            'checks_not_an_object': (b'{"mlx_checks": [1]}', 'TypeError'),
            'checks_missing_total': (b'{"mlx_checks": {"passed": 1}}', 'KeyError'),
            'rate_not_a_number': (json.dumps({'benchmark': dict(bench, decode_tps='x')}).encode(), 'ValueError'),
            'rate_overflows_float': (json.dumps({'benchmark': dict(bench, decode_tps=10 ** 400)}).encode(),
                                     'OverflowError'),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, (data, error) in receipts.items():
                with self.subTest(name=name):
                    path = Path(tmp) / f'{name}.json'
                    if data is not None:
                        path.write_bytes(data)
                    rc, out = run_main('--row', path)
                    self.assertEqual(rc, 1)
                    self.assertTrue(out.startswith(f'row refused: {error}'), out[:200])
            path = Path(tmp) / 'list.json'
            path.write_text('[1]')
            result = subprocess.run([sys.executable, str(Path(qualify.__file__)), '--row', str(path)],
                                    capture_output=True, text=True, check=False)
        self.assertEqual((result.returncode, result.stdout, result.stderr),
                         (1, 'row refused: ValueError: invalid_receipt\n', ''))


class PatchStateTest(unittest.TestCase):
    """The kit must know whether the mlx-lm it benchmarks carries the series."""

    def test_unpatched_tree_is_stock(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            (site / 'mlx_lm').mkdir()
            (site / 'mlx_lm' / 'generate.py').write_text('x = 1\n')
            state = qualify.patch_state(site)
            self.assertEqual(state['state'], 'stock')
            self.assertFalse(state['series_present'])
            self.assertIsNone(state['record_sha256'])
            self.assertEqual(state['eager_swiglu_markers'], 0)

    def test_patched_tree_reports_record_and_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            (site / 'mlx_lm' / 'models').mkdir(parents=True)
            (site / 'mlx_lm' / 'models' / 'qwen3.py').write_text(
                'flag = "_EAGER_SWIGLU"\nother = "_EAGER_SWIGLU"\n')
            record = site / 'mlx_lm' / qualify.PATCH_RECORD
            record.write_text('series patches\n')
            state = qualify.patch_state(site)
            self.assertEqual(state['state'], 'patched')
            self.assertTrue(state['series_present'])
            self.assertEqual(state['record_sha256'], qualify.sha(record))
            self.assertEqual(state['eager_swiglu_markers'], 2)

    def test_missing_site_is_unreadable(self):
        state = qualify.patch_state(None)
        self.assertEqual(state['state'], 'unreadable')
        self.assertEqual(state['eager_swiglu_markers'], 0)

    def test_record_without_markers_is_unreadable(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            (site / 'mlx_lm').mkdir()
            (site / 'mlx_lm' / qualify.PATCH_RECORD).write_text('series patches\n')
            state = qualify.patch_state(site)
            self.assertEqual(state['state'], 'unreadable')
            self.assertTrue(state['series_present'])
            self.assertEqual(state['eager_swiglu_markers'], 0)

    def test_stock_arm_refuses_a_patched_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            (site / 'mlx_lm' / 'models').mkdir(parents=True)
            (site / 'mlx_lm' / 'models' / 'qwen3.py').write_text('flag = "_EAGER_SWIGLU"\n')
            (site / 'mlx_lm' / qualify.PATCH_RECORD).write_text('series patches\n')
            with self.assertRaises(RuntimeError) as caught:
                qualify.stock_state(site)
            self.assertIn('patch_series', str(caught.exception))

    def test_stock_arm_refuses_an_unreadable_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as caught:
                qualify.stock_state(None)
            self.assertIn('unreadable', str(caught.exception))

    def test_stock_arm_reports_a_clean_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            (site / 'mlx_lm').mkdir()
            (site / 'mlx_lm' / 'generate.py').write_text('x = 1\n')
            state = qualify.stock_state(site)
            self.assertEqual(state['state'], 'stock')


class ArchiveTest(unittest.TestCase):
    """The sealed kit archive has to carry the patch series."""

    def test_sealed_archive_holds_the_series_and_the_kit_finds_it(self):
        root = Path(__file__).resolve().parent.parent
        seal = root / 'scripts' / 'seal-qualification-kit.sh'
        self.assertTrue(seal.is_file())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = subprocess.run(['bash', str(seal), str(out), 'test-tag'],
                                    capture_output=True, text=True, check=True)
            self.assertIn('test-tag', result.stdout)
            archive = out / 'omarchy-mlx-qualification-kit-test-tag.tar'
            self.assertTrue(archive.is_file())
            names = subprocess.run(['tar', '-tf', str(archive)], capture_output=True,
                                   text=True, check=True).stdout.splitlines()
            self.assertIn('omarchy-mlx-qualification-kit-test-tag/scripts/apply-mlx-lm-patches.sh', names)
            self.assertIn('omarchy-mlx-qualification-kit-test-tag/qualification-kit/qualify.py', names)
            self.assertTrue(any(name.endswith('patches/mlx-lm-last-logits.patch') for name in names))
            self.assertFalse(any('__pycache__' in name for name in names))
            extracted = out / 'extracted'
            extracted.mkdir()
            subprocess.run(['tar', '-xf', str(archive), '-C', str(extracted)], check=True)
            kit_root = extracted / 'omarchy-mlx-qualification-kit-test-tag'
            script = kit_root / 'scripts' / 'apply-mlx-lm-patches.sh'
            self.assertTrue(script.is_file())
            self.assertTrue((kit_root / 'patches').is_dir())
            found = qualify.patch_series_script(kit_root / 'qualification-kit')
            self.assertEqual(found, script)
            self.assertTrue((found.parent.parent / 'patches').is_dir())

    def test_archive_holds_every_helper_the_apply_script_names(self):
        root = Path(__file__).resolve().parent.parent
        apply_script = (root / 'scripts' / 'apply-mlx-lm-patches.sh').read_text()
        named = set(re.findall(r'patch-mlx-lm-[a-z0-9-]+\.py', apply_script))
        self.assertTrue(named, 'the apply script names no helper')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            subprocess.run(['bash', str(root / 'scripts' / 'seal-qualification-kit.sh'),
                            str(out), 'helper-tag'], capture_output=True, text=True, check=True)
            archive = out / 'omarchy-mlx-qualification-kit-helper-tag.tar'
            names = subprocess.run(['tar', '-tf', str(archive)], capture_output=True,
                                   text=True, check=True).stdout.splitlines()
            for helper in named:
                self.assertIn(f'omarchy-mlx-qualification-kit-helper-tag/scripts/{helper}', names)

    def test_sealed_archive_manifest_verifies(self):
        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            subprocess.run(['bash', str(root / 'scripts' / 'seal-qualification-kit.sh'),
                            str(out), 'manifest-tag'], capture_output=True, text=True, check=True)
            archive = out / 'omarchy-mlx-qualification-kit-manifest-tag.tar'
            extracted = out / 'extracted'
            extracted.mkdir()
            subprocess.run(['tar', '-xf', str(archive), '-C', str(extracted)], check=True)
            kit_root = extracted / 'omarchy-mlx-qualification-kit-manifest-tag'
            result = subprocess.run(['sha256sum', '-c', '--quiet', 'SHA256SUMS'],
                                    cwd=kit_root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn('SHA256SUMS', result.stdout)

    def test_archive_without_the_series_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            kit_root = Path(tmp) / 'kit'
            (kit_root / 'qualification-kit').mkdir(parents=True)
            os.environ.pop('MLX_OMARCHY_PATCH_SERIES', None)
            found = qualify.patch_series_script(kit_root / 'qualification-kit')
            self.assertFalse(found.is_file())


class PairTest(unittest.TestCase):
    def test_pair_requires_matching_generated_token_digests(self):
        base = {'schema': qualify.SCHEMA, 'machine_id': '0123456789abcdef',
                'workload': qualify.WORKLOAD}
        ids = list(range(64))
        benchmark = json.loads(bench_decode.result_line(12, ids, 460_000_000, 23, 'Apple M5 Max'))
        digest = benchmark.pop('ids_sha256_16')
        linux = dict(base, arm='linux-release-wheel',
                     benchmark=dict(benchmark, ids_sha256_16=digest))
        with tempfile.TemporaryDirectory() as tmp:
            linux_path = Path(tmp) / 'linux.json'
            mac_path = Path(tmp) / 'macos.json'
            linux_path.write_text(json.dumps(linux))
            for name, fields, expected in (
                    ('matched', {'ids_sha256_16': digest}, 'matched'),
                    ('missing', {}, 'invalid_benchmark'),
                    ('wrong_device', {'ids_sha256_16': digest, 'device': 'llvmpipe'}, 'invalid_benchmark'),
                    ('wrong_ids', {'ids_sha256_16': digest, 'generated_ids': ids[::-1]},
                     'invalid_benchmark'),
                    ('different', {'ids_sha256_16': bench_decode.ids_digest(ids[::-1]),
                                   'generated_ids': ids[::-1]}, 'generated_ids_differ'),
                    ('wrong_prompt', {'ids_sha256_16': digest, 'prompt_tokens': 15}, 'workloads_differ')):
                with self.subTest(name=name):
                    mac = dict(base, arm='macos-baseline', benchmark=dict(benchmark, **fields))
                    mac_path.write_text(json.dumps(mac))
                    result = subprocess.run(
                        [sys.executable, qualify.__file__, '--compare', linux_path, mac_path],
                        capture_output=True, text=True, check=False)
                    self.assertIn(expected, result.stdout)
                    self.assertEqual(result.returncode, 0 if name == 'matched' else 1)

    def test_pair_refuses_a_receipt_nested_too_deeply_to_decode(self):
        with self.assertRaises(RecursionError):
            on_pinned_stack(lambda: json.loads(DEEP_JSON))
        benchmark = json.loads(bench_decode.result_line(12, list(range(64)), 460_000_000, 23, 'Apple M5 Max'))
        linux = {'schema': qualify.SCHEMA, 'machine_id': '0123456789abcdef', 'workload': qualify.WORKLOAD,
                 'arm': 'linux-release-wheel', 'benchmark': benchmark}
        with tempfile.TemporaryDirectory() as tmp:
            deep = Path(tmp) / 'deep.json'
            deep.write_text(DEEP_JSON)
            other = Path(tmp) / 'linux.json'
            other.write_text(json.dumps(linux))
            for paths in ((deep, other), (other, deep)):
                with self.subTest(deep=paths.index(deep)):
                    rc, out = run_main('--compare', *paths)
                    self.assertEqual(rc, 1)
                    self.assertRegex(out, r'^pair refused: .* while decoding a JSON array')


class QualificationRunTest(unittest.TestCase):
    def run_qualification(self, gradient, probe_exit=0):
        expected = [[-0.25, 0.0], [0.125, 0.0]]
        ops = [{'op': name, 'pass': True, 'expected': 1, 'actual': None} for name in
               ('add', 'matmul', 'softmax', 'argmax', 'cumsum')]
        ops.append({'op': 'value_and_grad', 'pass': gradient == expected,
                    'expected': expected, 'actual': gradient})
        golden = {'available': True, 'errors': 0, 'sha256': ['a'] * 20, 'golden_sha256': 'a'}
        decode = bench_decode.result_line(12, list(range(64)), 240_000_000, 12, 'Apple M1')
        suite = {'files': {'test_array.py': {'ids_complete': True, 'exit': 0}},
                 'expected_files': 1, 'passed': 1, 'failed': 0, 'failures': {}, 'skips': []}
        args = SimpleNamespace(icd=None, model_dir=Path('pinned-model'), benchmark_only=False)
        receipt = {'versions': {}}
        with (patch.object(qualify, 'setup', return_value=('python', None)),
              patch.object(qualify, 'verify', return_value={}),
              patch.object(qualify, 'command', side_effect=[
                  (probe_exit, json.dumps({'available': True, 'ops': ops})),
                  (0, json.dumps(golden)), (0, decode)]),
              patch.object(qualify, 'load_model', return_value=args.model_dir),
              patch.object(qualify, 'suite', return_value=suite)):
            status = qualify.run(args, Path('.'), False, {}, receipt)
        return status, receipt

    def test_benchmark_only_uses_the_same_runner_on_both_os(self):
        args = SimpleNamespace(icd=None, model_dir=Path('pinned-model'), benchmark_only=True)
        ids = list(range(64))
        for mac, wheel in ((True, None), (False, Path('pinned.whl'))):
            with (
                self.subTest(mac=mac),
                patch.object(qualify, 'setup', return_value=('python', wheel)),
                patch.object(qualify, 'verify', return_value={}),
                patch.object(qualify, 'load_model', return_value=args.model_dir),
                patch.object(qualify, 'command', return_value=(
                    0, bench_decode.result_line(12, ids, 240_000_000, 12, 'Apple M1'))) as command,
            ):
                receipt = {'versions': {}}
                self.assertEqual(qualify.run(args, Path('.'), mac, {}, receipt),
                                 'benchmark_complete')
                self.assertEqual(receipt['benchmark']['generated_ids'], ids)
                self.assertEqual(receipt['benchmark']['ids_sha256_16'], bench_decode.ids_digest(ids))
                self.assertEqual(receipt['benchmark']['prefill_s'], 0.24)
                command.assert_called_once_with(
                    ['python', qualify.BENCH, '--model', args.model_dir, '--prompt', qualify.PROMPT,
                     '--tokens', '64', '--temp', '0', '--seed', '0', '--warmup-tokens', '4',
                     *(['--wheel', wheel] if wheel else [])], {}, 180, Path('.'))

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


class ProvenanceTest(unittest.TestCase):
    def test_unhashed_binary_does_not_clear_an_earlier_record_mismatch(self):
        core_rel = 'mlx/core/core.cpython-311-x86_64-linux-gnu.so'
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp) / 'site'
            (site / 'mlx' / 'lib').mkdir(parents=True)
            (site / 'mlx' / 'core').mkdir()
            (site / core_rel).write_bytes(b'core')
            (site / 'mlx' / 'lib' / 'libmlx.so').write_bytes(b'lib')
            wheel = Path(tmp) / 'claimed.whl'
            with zipfile.ZipFile(wheel, 'w') as zf:
                zf.writestr(core_rel, b'core')
                zf.writestr('mlx/lib/libmlx.so', b'lib')
            # The extension's RECORD hash is stale; libmlx.so has no entry.
            records = {core_rel: hashlib.sha256(b'stale-core').hexdigest()}
            mlx = SimpleNamespace(core=SimpleNamespace(__version__='1.0'))
            dist = SimpleNamespace(version='1.0', locate_file=lambda rel: site / rel)
            for argv in ([], ['--expect-wheel', str(wheel)]):
                out = io.StringIO()
                with (self.subTest(argv=argv),
                      patch.dict(sys.modules, {'mlx': mlx, 'mlx.core': mlx.core}),
                      patch.object(mlx_provenance.importlib.metadata, 'distribution',
                                   return_value=dist),
                      patch.object(mlx_provenance, '_record_hashes', return_value=records),
                      patch.object(sys, 'argv', ['mlx_provenance.py', *argv]),
                      contextlib.redirect_stdout(out)):
                    with self.assertRaises(SystemExit) as exit_:
                        mlx_provenance.main()
                    self.assertEqual(exit_.exception.code, 3)
                    report = json.loads(out.getvalue())
                    self.assertEqual([f['match'] for f in report['files']], [False, None])
                    self.assertEqual(report['verified'], 'mismatch')
                    self.assertIn('RECORD says', report['mismatch'])


class PackagedIcdTest(unittest.TestCase):
    """Default Linux --icd discovery prefers Borealis and falls back to Honeykrisp."""

    def _run_main(self, argv, *, system, vulkan_dir, dt_dir=None, run_status='qualified'):
        real_path = Path
        packaged_root = real_path('/usr/lib/omarchy-mlx/vulkan')
        dt_root = real_path('/proc/device-tree')
        captured = {}

        def fake_path(value, *parts):
            path = real_path(value, *parts)
            if path.is_relative_to(packaged_root):
                return vulkan_dir / path.relative_to(packaged_root)
            if dt_dir is not None and path == dt_root:
                return dt_dir
            return path

        def fake_run(args, work, mac, env, receipt):
            captured.update(icd=args.icd, mac=mac)
            return run_status

        stdout, stderr = io.StringIO(), io.StringIO()
        with (patch.object(sys, 'argv', ['qualify.py', *argv]),
              patch.object(qualify.platform, 'system', return_value=system),
              patch.object(qualify, 'Path', fake_path),
              patch.object(qualify, 'text', return_value='Apple M1'),
              patch.object(qualify, 'run', side_effect=fake_run) as run_mock,
              contextlib.redirect_stdout(stdout),
              contextlib.redirect_stderr(stderr)):
            try:
                rc = qualify.main()
            except SystemExit as exc:
                return exc.code, stderr.getvalue(), captured, run_mock
        return rc, stderr.getvalue(), captured, run_mock

    def _make_vulkan(self, root, entries):
        vulkan_dir = root / 'vulkan'
        if entries is not None:
            vulkan_dir.mkdir()
            for name, kind in entries.items():
                entry = vulkan_dir / name
                if kind == 'dir':
                    entry.mkdir()
                else:
                    entry.write_text('{}\n')
        dt_dir = root / 'device-tree'
        dt_dir.mkdir()
        (dt_dir / 'compatible').write_bytes(b'apple,j293\0apple,t8103\0')
        return vulkan_dir, dt_dir

    def test_linux_default_icd_discovery(self):
        borealis, honeykrisp = 'borealis_icd.aarch64.json', 'honeykrisp_icd.aarch64.json'
        cases = (
            ('borealis_only', {borealis: 'file'}, borealis),
            ('honeykrisp_only', {honeykrisp: 'file'}, honeykrisp),
            ('both_present', {borealis: 'file', honeykrisp: 'file'}, borealis),
            ('borealis_dir_honeykrisp_file', {borealis: 'dir', honeykrisp: 'file'}, honeykrisp),
            ('borealis_dir_only', {borealis: 'dir'}, None),
            ('empty_dir', {}, None),
            ('missing_dir', None, None),
        )
        for case, entries, expected in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                vulkan_dir, dt_dir = self._make_vulkan(root, entries)
                out = root / 'receipt.json'
                rc, stderr, captured, run_mock = self._run_main(
                    ['--machine-id', '0123456789abcdef', '--out', str(out)],
                    system='Linux', vulkan_dir=vulkan_dir, dt_dir=dt_dir)
                if expected is None:
                    self.assertEqual(rc, 2)
                    self.assertIn('set --icd to the installed Apple Vulkan JSON manifest', stderr)
                    self.assertFalse(out.exists())
                    run_mock.assert_not_called()
                else:
                    self.assertEqual(rc, 0, stderr)
                    run_mock.assert_called_once()
                    self.assertEqual(captured['icd'], (vulkan_dir / expected).resolve())
                    self.assertIs(captured['mac'], False)

    def test_linux_explicit_icd_overrides_packaged_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vulkan_dir, dt_dir = self._make_vulkan(
                root, {'borealis_icd.aarch64.json': 'file', 'honeykrisp_icd.aarch64.json': 'file'})
            custom = root / 'custom_icd.json'
            custom.write_text('{}\n')
            rc, stderr, captured, run_mock = self._run_main(
                ['--machine-id', '0123456789abcdef', '--out', str(root / 'explicit.json'),
                 '--icd', str(custom)],
                system='Linux', vulkan_dir=vulkan_dir, dt_dir=dt_dir)
            self.assertEqual(rc, 0, stderr)
            run_mock.assert_called_once()
            self.assertEqual(captured['icd'], custom.resolve())
            self.assertIs(captured['mac'], False)

    def test_macos_ignores_packaged_manifests_and_rejects_icd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vulkan_dir, _ = self._make_vulkan(
                root, {'borealis_icd.aarch64.json': 'file', 'honeykrisp_icd.aarch64.json': 'file'})
            rc, stderr, captured, run_mock = self._run_main(
                ['--machine-id', '0123456789abcdef', '--out', str(root / 'macos.json')],
                system='Darwin', vulkan_dir=vulkan_dir, run_status='macos_baseline')
            self.assertEqual(rc, 0, stderr)
            run_mock.assert_called_once()
            self.assertIsNone(captured['icd'])
            self.assertIs(captured['mac'], True)

            out_rejected = root / 'macos-icd.json'
            rc, stderr, _, run_mock = self._run_main(
                ['--machine-id', '0123456789abcdef', '--out', str(out_rejected),
                 '--icd', str(vulkan_dir / 'borealis_icd.aarch64.json')],
                system='Darwin', vulkan_dir=vulkan_dir, run_status='macos_baseline')
            self.assertEqual(rc, 2)
            self.assertIn('--icd is Linux only', stderr)
            self.assertFalse(out_rejected.exists())
            run_mock.assert_not_called()
