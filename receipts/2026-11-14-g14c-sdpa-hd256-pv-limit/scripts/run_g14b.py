"""G14C width-256 SDPA value-phase pass 1 (MLX_OMARCHY_SDPA_SPLIT_PV 0 vs 1), op level, by workgroup count (kv heads x 128-key blocks). One process per config."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('<HOME>/scratch/w9c-m2-profile-run2-20261114')
IN = ROOT / 'inputs-g14b'
PY = ROOT / 'venv-m6bf/bin/python'
ICD = os.environ['VK_ICD_FILENAMES']
ENV = {'HOME': '<HOME>', 'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'PYTHONNOUSERSITE': '1', 'VK_DRIVER_FILES': ICD,
       'MLX_OMARCHY_BATCH_WORK': '100000'}
CONFIGS = [(4, 1, 128 * w) for w in (68, 69, 70, 71, 72, 73, 74, 75)]
for qh, kvh, n in CONFIGS:
    env = dict(ENV)
    env.update({'SDPA_QH': str(qh), 'SDPA_KVH': str(kvh), 'SDPA_CTX': str(n), 'SDPA_NTOK': '128'})
    r = subprocess.run([str(PY), '-I', '-B', str(IN / 'sdpa_probe_g14.py')], env=env, capture_output=True, text=True, timeout=300)
    line = [x for x in r.stdout.splitlines() if x.startswith('{')]
    print('ROW', qh, kvh, n, 'rc', r.returncode, line[-1] if line else r.stderr[-300:], flush=True)
    if r.returncode != 0:
        sys.exit(3)
print('SUMMARY done rows', len(CONFIGS), flush=True)
os._exit(0)
