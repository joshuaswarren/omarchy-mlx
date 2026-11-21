#!/usr/bin/env python3
"""Run the release qualification on one OS; use the same random ID on both."""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from qualification_reference import comparison

ROOT = Path(__file__).resolve().parent
PER_FILE_TIMEOUT = int(os.environ.get('QUALIFY_PER_FILE_TIMEOUT', '900'))
TAG = 'v0.7.47'
SOURCE = '83cf322bb3c238ed9781434573b3f133f84273e6'
WHEEL = 'mlx_omarchy-0.32.4.dev202611181701+83cf322-cp314-cp314-linux_aarch64.whl'
VENDOR = 'omarchy-mlx-vendor-wheels-v0.7.47-cp314-aarch64.tar'
ASSETS = {WHEEL: 'b7cb0244e300e518450af635995629aa9e52c7cde7e7613b53ee9a8e819752e1',
          VENDOR: '31a9ed2075865d41e84729b850c1bdf9f66530f9b12df7bfb50cd80c472555f4'}
MODEL = 'mlx-community/Qwen3-4B-Instruct-2507-4bit'
REV = '50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b'
PROMPT = 'Explain how a transistor can function as a switch in a simple circuit.'
TOKENS = 64
BENCH = ROOT / 'scripts/bench_decode.py'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def command(argv, env, seconds, cwd=None):
    try:
        p = subprocess.run(argv, env=env, cwd=cwd, text=True, errors='replace',
                           capture_output=True, timeout=seconds, check=False)
        if p.returncode and (log := env.get('QUALIFY_PRIVATE_LOG')):
            with open(log, 'a', encoding='utf-8') as f:
                f.write(f'\n{Path(argv[0]).name} exit {p.returncode}\n{p.stdout}\n{p.stderr}')
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired:
        return 124, ''
    except OSError:
        return 127, ''


def text(argv):
    rc, out = command(argv, os.environ, 10)
    return out.strip() if rc == 0 else None


def read_json(out):
    try:
        result = json.loads(out.strip().splitlines()[-1])
        return result if isinstance(result, dict) else {}
    except (IndexError, ValueError):
        return {}


def counts(xml):
    try:
        root = ET.fromstring(xml)
        suites = [root] if root.tag == 'testsuite' else root.findall('testsuite')
        n, fail, err, skip = (sum(int(s.get(k, '0')) for s in suites)
                              for k in ('tests', 'failures', 'errors', 'skipped'))
        if min(n, fail, err, skip) < 0 or n == 0 or fail + err + skip > n:
            return None
        return {'passed': n - fail - err - skip, 'failed': fail + err, 'skipped': skip}
    except (ValueError, ET.ParseError):
        return None


def ane_result(rc, result, mac):
    summary = {'status': 'unavailable' if rc in (2, 127) else 'fail', 'exit': rc}
    if rc == 0 and result.get('available') is True:
        if mac:
            summary.update(status='coreml_pass_placement_unverified', passed=20, failed=0)
        else:
            hashes = result.get('sha256')
            golden = result.get('golden_sha256')
            if (result.get('errors') == 0 and isinstance(hashes, list) and len(hashes) == 20
                    and golden and all(h == golden for h in hashes)):
                summary.update(status='pass', passed=20, failed=0, golden_sha256=golden)
    for key in ('min_ms', 'median_ms'):
        value = result.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
            summary[key] = value
    return summary


def benchmark_result(result):
    device = result.get('device', '')
    digest = result.get('ids_sha256_16', '')
    if (result.get('generated') != TOKENS or not isinstance(device, str)
            or not re.match(r'^(Apple M[1-9]|Honey[Kk]risp)', device, re.I)
            or not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{16}', digest)
            or not isinstance(result.get('prompt_tokens'), int) or result['prompt_tokens'] <= 0):
        return None
    metrics = ('decode_tps', 'prefill_tps', 'prefill_s')
    if any(not isinstance(result.get(k), (int, float)) or not math.isfinite(result[k])
           or result[k] <= 0 for k in metrics):
        return None
    return {k: result[k] for k in (*metrics, 'generated', 'prompt_tokens', 'ids_sha256_16', 'device')}


def asset(work, cached, name):
    path = cached / name if cached else work / name
    if not cached:
        url = f'https://github.com/joshuaswarren/omarchy-mlx/releases/download/{TAG}/{name.replace("+", "%2B")}'
        subprocess.run(['curl', '-fLsS', '--max-time', '300', '-o', path, url],
                       check=True, timeout=310)
    if sha(path) != ASSETS[name]:
        raise RuntimeError('release_asset_hash_mismatch')
    return path


def setup(work, args, mac, env, receipt):
    needed = (3, 12) if mac else (3, 14)
    arch = 'arm64' if mac else 'aarch64'
    if platform.machine() != arch or sys.version_info[:2] != needed:
        raise RuntimeError(f'need_{arch}_python_{needed[0]}_{needed[1]}')
    if args.reuse_runtime:
        code = ('import importlib.metadata as m; '
                'assert all(m.version(k) == v for k,v in '
                '{"mlx":"0.32.3","mlx-lm":"0.31.3","pytest":"9.0.1","coremltools":"9.0"}.items())')
        rc, _ = command([sys.executable, '-c', code], env, 15)
        if rc:
            raise RuntimeError('existing_macos_runtime_versions_do_not_match')
        return Path(sys.executable), None
    print('Setting up isolated Python environment.', flush=True)
    rc, _ = command([sys.executable, '-m', 'venv', work / 'venv'], env, 45)
    if rc:
        raise RuntimeError(f'venv_setup_exit_{rc}')
    py = work / 'venv/bin/python'
    pip = [py, '-m', 'pip', 'install', '--disable-pip-version-check',
           '--no-input', '--only-binary=:all:']
    if mac:
        rc, _ = command([*pip, 'mlx==0.32.3', 'mlx-lm==0.31.3', 'pytest==9.0.1',
                         'coremltools==9.0'], env, 240)
        wheel = None
    else:
        wheel = asset(work, args.asset_dir, WHEEL)
        archive = asset(work, args.asset_dir, VENDOR)
        with tarfile.open(archive) as tar:
            tar.extractall(work, filter='data')
        opts = ['--no-index', '--find-links', work / 'vendor-0747']
        rc, _ = command([*pip, *opts, wheel, 'mlx-lm==0.31.3', 'numpy'], env, 240)
        if rc == 0:
            rc, _ = command([*pip, 'pytest==9.0.1'], env, 120)
        receipt['wheel'] = {'name': WHEEL, 'sha256': ASSETS[WHEEL], 'vendor_sha256': ASSETS[VENDOR]}
    if rc:
        raise RuntimeError(f'pip_install_exit_{rc}')
    return py, wheel


def verify(py, wheel, env):
    code = '''import importlib.metadata as m, json, sys
import mlx.core as mx
from mlx_provenance import installed_provenance, native_provenance
from pathlib import Path
mx.set_default_device(mx.gpu)
mx.eval(mx.array([1]) + 2)
native = len(sys.argv) == 1
p = native_provenance() if native else installed_provenance(expect_wheel=Path(sys.argv[1]))
print(json.dumps({'verified': p['verified'], 'mlx': m.version('mlx' if native else 'mlx-omarchy'),
                  'mlx_lm': m.version('mlx-lm'), 'python': sys.version.split()[0]}))
'''
    rc, out = command([py, '-c', code, *([wheel] if wheel else [])], env, 80, BENCH.parent)
    p = read_json(out)
    if rc or not p.get('mlx') or (wheel and p.get('verified') != 'match'):
        raise RuntimeError(f'gpu_or_wheel_verification_exit_{rc}')
    return p


def suite(py, work, env):
    rows, failures, skips = {}, {}, []
    paths = sorted((ROOT / 'suite').glob('test_*.py'))
    xml, events = work / 'suite.xml', work / 'suite-events.json'
    suite_env = dict(env, PYTHONPATH=str(ROOT), QUALIFY_SUITE_EVENTS=str(events))
    for path in paths:
        print(f'Suite: {path.name}', flush=True)
        xml.unlink(missing_ok=True)
        events.unlink(missing_ok=True)
        started = time.monotonic()
        rc, _ = command([py, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                         '-p', 'suite_events', '--junitxml', xml, path],
                        suite_env, PER_FILE_TIMEOUT, work)
        elapsed = time.monotonic() - started
        num = counts(xml.read_text()) if xml.exists() else None
        data = json.loads(events.read_text()) if events.exists() else None
        complete = bool(num and data and data['invalid'] == 0
                        and data is not None
                        and num['failed'] == len(data['failures'])
                        and num['skipped'] == len(set(data['skips'])))
        row = {'exit': rc, 'counts': num, 'ids_complete': complete,
               'seconds': round(elapsed, 1)}
        if rc == 124:
            row['timed_out'] = True
        elif rc not in (0, 1):
            row['aborted'] = True
        rows[path.name] = row
        if complete and data is not None:
            failures.update(data['failures'])
            skips.extend(data['skips'])
    return {'files': rows, 'expected_files': len(paths), 'failures': failures,
            'skips': sorted(skips), **{
                k: sum(r['counts'][k] for r in rows.values() if r['counts'])
                for k in ('passed', 'failed', 'skipped')}}


def load_model(py, work, env, path):
    if path is None:
        code = '''from huggingface_hub import snapshot_download
import sys
print(snapshot_download(sys.argv[1], revision=sys.argv[2],
    allow_patterns=['*.safetensors', '*.json', '*.txt', '*.model', '*.jinja']))'''
        rc, out = command([py, '-c', code, MODEL, REV], env, 600, work)
        if rc or not out.strip():
            raise RuntimeError(f'model_download_exit_{rc}')
        path = Path(out.strip().splitlines()[-1])
    if path.name != REV or not (path / 'config.json').is_file() or not list(path.glob('*.safetensors')):
        raise RuntimeError('model_snapshot_invalid')
    return path


def run(args, work, mac, env, r):
    if args.icd:
        env.update(VK_DRIVER_FILES=str(args.icd), VK_ICD_FILENAMES=str(args.icd))
        r['icd_manifest_sha256'] = sha(args.icd)
        owner = text(['pacman', '-Qoq', args.icd])
        if owner and re.fullmatch(r'[a-z0-9_.+-]+', owner):
            version = text(['pacman', '-Q', owner])
            if version and re.fullmatch(r'[a-z0-9_.+-]+ [A-Za-z0-9_.+:-]+', version):
                r['icd_package'] = version
    py, wheel = setup(work, args, mac, env, r)
    r['versions'].update(verify(py, wheel, env))
    code = 'import mlx.core as mx; mx.set_default_device(mx.gpu); exec(open("correctness.py").read())'
    rc, out = command([py, '-c', code], env, 100, BENCH.parent)
    ops = read_json(out).get('ops', [])
    r['mlx_checks'] = {'exit': rc, 'passed': sum(o.get('pass') is True for o in ops),
                       'total': len(ops), 'failed_ops': [o['op'] for o in ops if not o.get('pass')]}
    if rc or len(ops) != 6 or r['mlx_checks']['passed'] != 6:
        raise RuntimeError('gpu_correctness_failed')
    ane = [py, BENCH.parent / 'coreml_smoke.py'] if mac else ['omarchy-ane-smoke']
    rc, out = command(ane, env, 105, work)
    r['ane'] = ane_result(rc, read_json(out), mac)
    if rc not in (0, 1, 2, 127):
        raise RuntimeError('ane_timeout_or_signal')
    model = load_model(py, work, env, args.model_dir)
    argv = [py, BENCH, '--model', model, '--prompt', PROMPT, '--tokens', str(TOKENS),
            '--temp', '0', '--seed', '0', '--warmup-tokens', '4']
    rc, out = command([*argv, *(['--wheel', wheel] if wheel else [])], env, 180, work)
    r['benchmark'] = benchmark_result(read_json(out)) if not rc else None
    if not r['benchmark']:
        raise RuntimeError(f'decode_benchmark_exit_{rc}')
    return suite_verdict(r, suite(py, work, env), mac)


def suite_verdict(r, results, mac):
    r['suite'] = results
    rows = results['files']
    if (len(rows) != results['expected_files'] or not results['passed']
            or any(not row['ids_complete'] or row['exit'] not in (0, 1)
                   for row in rows.values())):
        return 'suite_incomplete'
    expected_ane = 'coreml_pass_placement_unverified' if mac else 'pass'
    if mac:
        return ('macos_baseline' if r['ane']['status'] == expected_ane
                and not results['failed'] else 'macos_baseline_with_gaps')
    ref_path = ROOT / 'm1-max-reference.json'
    if not ref_path.is_file():
        return 'reference_missing'
    baseline = json.loads(ref_path.read_text())
    if (baseline.get('schema') != 'omarchy-mlx-m1-max-reference-v1'
            or baseline.get('release') != TAG or baseline.get('suite_source') != SOURCE
            or baseline.get('wheel_sha256') != ASSETS[WHEEL]
            or len(baseline.get('failures', {})) != 80
            or len(set(baseline.get('skips', []))) != 81):
        return 'reference_invalid'
    results['reference'] = {'schema': baseline['schema'], 'device': baseline['device'],
                            'sha256': sha(ref_path)}
    diff = comparison(baseline, results['failures'], results['skips'])
    results.update(diff)
    if diff['new_failures']:
        return 'new_suite_failures'
    if diff['new_skips']:
        return 'new_suite_skips'
    return 'qualified' if r['ane']['status'] == expected_ane else 'ane_not_qualified'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--machine-id', required=True, help='same random 16–32 hex characters on both OSes')
    parser.add_argument('--out', type=Path, required=True, help='new JSON receipt path')
    parser.add_argument('--asset-dir', type=Path, help='optional cache of the two exact GitHub release assets')
    parser.add_argument('--icd', type=Path, help='installed Apple Vulkan ICD manifest (Linux only)')
    parser.add_argument('--model-dir', type=Path, help='existing directory for the exact pinned model snapshot')
    parser.add_argument('--reuse-runtime', action='store_true',
                        help='macOS only: use the calling interpreter with exact pinned packages, no install')
    parser.add_argument('--diagnostics', type=Path, help='optional private failure log; do not share')
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{16,32}', args.machine_id):
        parser.error('machine-id must be 16–32 lowercase hex characters; use a random ID, never a serial')
    if args.out.exists():
        parser.error('receipt already exists; choose a fresh path')
    kind = platform.system()
    if kind not in ('Linux', 'Darwin'):
        parser.error('Linux or macOS required')
    mac = kind == 'Darwin'
    if not mac and not args.icd:
        packaged = Path('/usr/lib/omarchy-mlx/vulkan/honeykrisp_icd.aarch64.json')
        if packaged.is_file():
            args.icd = packaged
        else:
            parser.error('set --icd to the installed Apple Vulkan JSON manifest')
    if mac and args.icd:
        parser.error('--icd is Linux only')
    if args.reuse_runtime and (not mac or not args.model_dir):
        parser.error('--reuse-runtime requires macOS and an existing --model-dir')
    for option in ('asset_dir', 'icd', 'model_dir'):
        path = getattr(args, option)
        if path:
            setattr(args, option, path.resolve())
    env = {k: v for k, v in os.environ.items() if not k.startswith(
        ('MLX_', 'VK_', 'MESA_', 'AGX_', 'GALLIUM_', 'PYTEST_', 'PIP_', 'DYLD_', 'LD_'))
        and k not in ('PYTHONPATH', 'PYTHONHOME', 'DEVICE', 'VIRTUAL_ENV', 'QUALIFY_PRIVATE_LOG')}
    env.update(DEVICE='gpu', MLX_ENABLE_TF32='0', PYTHONNOUSERSITE='1',
               PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PIP_CONFIG_FILE=os.devnull,
               PIP_NO_CACHE_DIR='1', PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1')
    if args.model_dir:
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if args.diagnostics:
        fd = os.open(args.diagnostics, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        env['QUALIFY_PRIVATE_LOG'] = str(args.diagnostics.resolve())
    if mac:
        machine = {'model': text(['sysctl', '-n', 'hw.model']),
                   'chip': text(['sysctl', '-n', 'machdep.cpu.brand_string'])}
    else:
        dtroot = Path('/proc/device-tree')
        compat = (dtroot / 'compatible').read_bytes().decode().split('\0') if dtroot.exists() else []
        machine = {'model': next((x for x in compat if re.fullmatch(r'apple,j\w+', x)), None),
                   'chip': next((x for x in compat if re.fullmatch(r'apple,t\d+', x)), None)}
    r = {'schema': 'omarchy-mlx-qualification-v3', 'machine_id': args.machine_id,
         'machine': machine, 'arm': 'macos-baseline' if mac else 'linux-release-wheel',
         'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
         'release': TAG, 'suite_source': SOURCE,
         'versions': {'os': platform.mac_ver()[0] if mac else platform.release()},
         'workload': {'model': MODEL, 'revision': REV, 'prompt': PROMPT,
                      'tokens': TOKENS, 'warmup': 4, 'temp': 0, 'seed': 0},
         'ane': None, 'mlx_checks': None, 'suite': None, 'benchmark': None}
    with tempfile.TemporaryDirectory(prefix='mlx-qualification-', dir=args.out.parent) as tmp:
        try:
            if not machine['chip']:
                raise RuntimeError('apple_chip_unidentified')
            r['status'] = run(args, Path(tmp), mac, env, r)
        except (OSError, ValueError, RuntimeError, tarfile.TarError, subprocess.SubprocessError) as exc:
            r['status'] = 'incomplete'
            r['error'] = str(exc) if type(exc) is RuntimeError else type(exc).__name__
    with args.out.open('x') as f:
        json.dump(r, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    print(f"{r['arm']}: {r['status']} — {args.out}")
    return 0 if r['status'] in ('qualified', 'macos_baseline') else 1


if __name__ == '__main__':
    raise SystemExit(main())
