# M2 end-to-end: patched mlx_lm stream_generate, greedy, plain vs prompt lookup K=3 with MLX_OMARCHY_SDPA_SPLIT_ROWS unset (the patch sets it per verify pass) vs set to 0.
# usage: e2e_m2d.py <target repo> <max tokens> <draft repo>; prints E2E lines (tok/s per arm, ABBA mean, ids identical to plain greedy)
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'pb'))
import mlx_lm  # noqa: E402
from mlx_lm import load, stream_generate  # noqa: E402

assert mlx_lm.__file__.startswith(sys.path[0]), mlx_lm.__file__
exec(open(os.path.join(HERE, 'prompts.py')).read())
repo = sys.argv[1]
maxt = int(sys.argv[2])
model, tok = load(repo)
draft = load(sys.argv[3])[0]
ARMS = {'base': (False, None), 'spec': (True, None), 'spec0': (True, '1')}  # True = draft model k=3
ORDER = ('base', 'spec', 'spec0', 'spec0', 'spec', 'base')


def run(p, arm):
    lookup, st = ARMS[arm]
    os.environ.pop('MLX_OMARCHY_PROMPT_LOOKUP', None)
    kw = {'draft_model': draft, 'num_draft_tokens': 3} if lookup else {}
    if st is None:
        os.environ.pop('MLX_OMARCHY_SDPA_SPLIT_ROWS', None)
    else:
        os.environ['MLX_OMARCHY_SDPA_SPLIT_ROWS'] = st
    out = []
    last = None
    for r in stream_generate(model, tok, p, max_tokens=maxt, **kw):
        out.append(r.token)
        last = r
    return out, last.generation_tps


for name, msg in PROMPTS.items():
    p = enc(tok, msg)
    run(p, 'base')
    run(p, 'spec')
    res = {}
    ids = {}
    for arm in ORDER:
        o, t = run(p, arm)
        res.setdefault(arm, []).append(t)
        ids.setdefault(arm, []).append(o)
    mean = {a: sum(v) / len(v) for a, v in res.items()}
    ref = ids['base'][0]
    print('E2E', json.dumps({
        'model': repo.rsplit('/', 3)[-3][:40], 'prompt': name, 'base_tps': round(mean['base'], 1), 'spec_tps': round(mean['spec'], 1),
        'spec0_tps': round(mean['spec0'], 1), 'spec_speedup': round(mean['spec'] / mean['base'], 3), 'spec0_speedup': round(mean['spec0'] / mean['base'], 3),
        'base_pair_spread_pct': round(100 * abs(res['base'][0] - res['base'][1]) / mean['base'], 2),
        'spec_pair_spread_pct': round(100 * abs(res['spec'][0] - res['spec'][1]) / mean['spec'], 2),
        'base_deterministic': ids['base'][0] == ids['base'][1], 'spec_identical': all(o == ref for o in ids['spec']),
        'spec0_identical': all(o == ref for o in ids['spec0']), 'gen': len(ref),
        'sha': hashlib.sha256(json.dumps(ref).encode()).hexdigest()[:12]}), flush=True)
os._exit(0)
