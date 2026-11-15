"""Qwen3-4B-4bit cast to f16, decode ABBA: MLX_OMARCHY_FUSED_TRIO default (on) vs 0. One process per arm."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('<HOME>/scratch/w9c-m2-profile-run2-20261114')
IN = ROOT / 'inputs-wd3b'
OUT = ROOT / 'output-wd3b'
PYD = ROOT / 'venv-m6bf/bin/python'
PYN = ROOT / 'venv-m6bf/bin/python'
MODELS = (('9b', '<HOME>/.cache/huggingface/hub/models--nvythong--Qwen3.8-9B-Distill-mlx-4Bit/snapshots/e827c31fbd588828f43180a87ab34415a6d8a4bf'), ('2b', '<HOME>/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e7c97b86b34b381'))
ICD = '<HOME>/scratch/w9c-mesa-main/asahi_icd.main.json'
ENV = {'HOME': '<HOME>', 'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'PYTHONNOUSERSITE': '1',
       'HF_HOME': '<HOME>/.cache/huggingface', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
       'TOKENIZERS_PARALLELISM': 'false', 'VK_DRIVER_FILES': ICD, 'MLX_OMARCHY_BATCH_WORK': '100000'}
ORDER = ('D1', 'N1', 'N2', 'D2')
CTXS = ((4096, 'prompt-ids-4096.json'),)

assert not OUT.exists()
OUT.mkdir()
for mname, MODEL in MODELS:
    summ = {}
    for ctx, ids in CTXS:
        rows = {}
        for name in ORDER:
            PY = PYD
            env = dict(ENV)
            if name[0] == 'N':
                env['MLX_OMARCHY_SDPA_SPLIT_PV'] = '1'
            arm = f'{name}-{mname}-c{ctx}'
            cmd = [str(PY), '-I', '-B', str(IN / 'driver_bf16d.py'), '--model', MODEL, '--prompt-ids', str(IN / ids),
                   '--ctx', str(ctx), '--output', str(OUT / arm), '--tokens', '128']
            rc = subprocess.run(cmd, env=env, timeout=600).returncode
            res = json.loads((OUT / arm / 'result.json').read_text()) if rc == 0 else {}
            rows[name] = (rc, res.get('decode_tps'), res.get('ids_sha'))
            print('ARM', arm, 'py', PY.parent.parent.name, 'rc', rc, 'tps', res.get('decode_tps'), 'ids', res.get('ids_sha'), flush=True)
            if rc != 0:
                sys.exit(3)
        d = (rows['D1'][1] + rows['D2'][1]) / 2
        n = (rows['N1'][1] + rows['N2'][1]) / 2
        summ[ctx] = 1000 / d - 1000 / n
        print('SUMMARY', mname, 'ctx', ctx, 'default_tps', round(d, 2), 'pv_tps', round(n, 2), 'pv_saves_ms', round(summ[ctx], 3), 'pv_gain_pct', round(100 * (n / d - 1), 2),
              'ids_equal', len({v[2] for v in rows.values()}) == 1, flush=True)
os._exit(0)
