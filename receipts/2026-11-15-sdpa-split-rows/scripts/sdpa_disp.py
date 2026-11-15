# Dispatches per verify call with MLX_OMARCHY_SDPA_SPLIT_ROWS=1 and unset, counted from the backend's own
# MLX_OMARCHY_TRACE_DISPATCH lines (one "[rtmod] DISPATCH kernel=<id> count=<n>" line per dispatch) between two markers.
# Expected with the flag set: 2 * rows dispatches, two kernel ids, each rows times (the one-token split-K pair per row).
import collections
import os
import re
import subprocess
import sys

CASES = ((32, 8, 128, 500, 2), (32, 8, 128, 2048, 4), (32, 8, 128, 3000, 16), (16, 4, 256, 700, 3), (16, 16, 128, 130, 8))
CHILD = '''
import os
import sys
import mlx.core as mx
h, kv, d, n, t, flag = (int(a) for a in sys.argv[1:7])
os.environ['MLX_OMARCHY_SDPA_DECODE_SPLIT'] = '1'
mx.random.seed(3)
q = (mx.random.normal((1, h, t, d)) * 0.5).astype(mx.bfloat16)
k = (mx.random.normal((1, kv, n, d)) * 0.5).astype(mx.bfloat16)
v = (mx.random.normal((1, kv, n, d)) * 0.5).astype(mx.bfloat16)
mx.eval(q, k, v)
if flag:
    os.environ['MLX_OMARCHY_SDPA_SPLIT_ROWS'] = '1'
mx.eval(mx.fast.scaled_dot_product_attention(q, k, v, scale=d ** -0.5, mask='causal'))
print('MARK_A', file=sys.stderr, flush=True)
mx.eval(mx.fast.scaled_dot_product_attention(q, k, v, scale=d ** -0.5, mask='causal'))
print('MARK_B', file=sys.stderr, flush=True)
'''
PAT = re.compile(r'\[rtmod\] DISPATCH kernel=(\d+) count=\d+')
bad = 0
for h, kv, d, n, t in CASES:
    for flag in (1, 0):
        env = dict(os.environ, MLX_OMARCHY_TRACE_DISPATCH='1')
        p = subprocess.run([sys.executable, '-I', '-B', '-c', CHILD, *map(str, (h, kv, d, n, t, flag))], env=env, capture_output=True, text=True, timeout=300)
        text = p.stderr
        window = text.split('MARK_A', 1)[-1].split('MARK_B', 1)[0] if 'MARK_B' in text else ''
        ids = collections.Counter(PAT.findall(window))
        total = sum(ids.values())
        ok = (total == 2 * t and len(ids) == 2 and set(ids.values()) == {t}) if flag else None
        bad += ok is False or p.returncode != 0
        print('DISP', 'heads', h, 'kv', kv, 'width', d, 'keys', n, 'rows', t, 'flag', flag, 'rc', p.returncode, 'dispatches', total,
              'kernel_ids', dict(ids), 'expected_when_set', 2 * t, 'ok', ok, flush=True)
print('DISP_DONE', 'bad', bad, flush=True)
sys.exit(1 if bad else 0)
