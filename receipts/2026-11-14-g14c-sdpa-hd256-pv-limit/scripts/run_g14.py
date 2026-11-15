"""G14C width-256 SDPA value-phase pass 1 (MLX_OMARCHY_SDPA_SPLIT_PV 0 vs 1), op level, by workgroup count (kv heads x 128-key blocks). One process per config."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('<HOME>/scratch/w9c-m2-profile-run2-20261114')
IN = ROOT / 'inputs-g14'
PY = ROOT / 'venv-m6bf/bin/python'
ICD = '<HOME>/scratch/w9c-mesa-main/asahi_icd.main.json'
ENV = {'HOME': '<HOME>', 'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'PYTHONNOUSERSITE': '1', 'VK_DRIVER_FILES': ICD,
       'MLX_OMARCHY_BATCH_WORK': '100000'}
CONFIGS = [(4, 1, n) for n in (4096, 7680, 7808, 9728, 9856, 12288)] + \
          [(8, 2, n) for n in (1024, 2048, 3840, 3968, 4096, 4864, 4992, 6144, 7680, 7808)] + \
          [(16, 4, n) for n in (1024, 1920, 2048, 2176, 2432, 2560, 3072)]
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
