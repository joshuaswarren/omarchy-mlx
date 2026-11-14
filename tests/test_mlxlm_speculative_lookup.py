"""Speculative verify and prompt lookup (mlx-lm 0.31.3 series).

The last-logits patches trim a cached forward to its final position, which left
the upstream draft-model path with one row of logits for a multi-token verify
pass. speculative_generate_step now runs every verify pass with full logits and
the short-pass GEMV route, rolls recurrent (non-trimmable) layers back by
replaying the accepted tokens, and drafts from earlier text when no draft model
is given (MLX_OMARCHY_PROMPT_LOOKUP=K through stream_generate). These tests pin
the helpers and, where mlx and mlx-lm import, check that greedy output equals a
plain greedy loop on a toy model with a trimmable and with a recurrent cache."""
import ast
import os
import subprocess
import sys

import pytest

from tests.test_apply_mlxlm_series import SCRIPT, _pristine_venv


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    root = _pristine_venv(tmp_path_factory.mktemp("spec-lookup"), "0.31.3")
    r = subprocess.run(["bash", str(SCRIPT), str(root)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-400:] + r.stderr[-400:]
    return root / "lib" / "python3.11" / "site-packages" / "mlx_lm"


def helpers(site, *names):
    tree = ast.parse((site / "generate.py").read_text())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in keep} == set(names)
    for n in keep:
        n.decorator_list = []
    module = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"os": os}
    exec(compile(module, "generate.py", "exec"), ns)
    return [ns[n] for n in names]


def test_lookup_takes_the_longest_latest_match(site):
    (draft,) = helpers(site, "_omarchy_lookup_draft")
    h = [1, 2, 3, 9, 9, 1, 2, 3, 7, 8, 5, 2, 3]
    assert draft(h, 3, 3, 2) == [7, 8, 5]  # suffix (2, 3): latest earlier match at 6
    assert draft(h + [7], 2, 3, 2) == [8, 5]  # suffix (2, 3, 7)
    assert draft([4, 5, 6], 3, 3, 2) == []
    assert draft([4, 5, 4], 3, 3, 1) == [5, 4]
    assert draft([4, 5, 4], 3, 3, 2) == []


def test_verify_pass_sets_and_restores_the_routes(site, monkeypatch):
    (verify,) = helpers(site, "_omarchy_verify_pass")
    import contextlib

    verify = contextlib.contextmanager(verify)
    monkeypatch.delenv("MLX_OMARCHY_FULL_LOGITS", raising=False)
    monkeypatch.delenv("MLX_OMARCHY_QMV_SMALL_T", raising=False)
    with verify(4):
        assert os.environ["MLX_OMARCHY_FULL_LOGITS"] == "1"
        assert os.environ["MLX_OMARCHY_QMV_SMALL_T"] == "4"
    assert "MLX_OMARCHY_FULL_LOGITS" not in os.environ
    assert "MLX_OMARCHY_QMV_SMALL_T" not in os.environ
    with verify(12):
        assert os.environ["MLX_OMARCHY_QMV_SMALL_T"] == "8"
    with verify(1):
        assert "MLX_OMARCHY_QMV_SMALL_T" not in os.environ
    monkeypatch.setenv("MLX_OMARCHY_QMV_SMALL_T", "0")
    monkeypatch.setenv("MLX_OMARCHY_FULL_LOGITS", "0")
    with verify(4):
        assert os.environ["MLX_OMARCHY_QMV_SMALL_T"] == "0"
    assert os.environ["MLX_OMARCHY_FULL_LOGITS"] == "0"


TOY = r"""
import os, sys
import mlx.core as mx
from mlx_lm.generate import speculative_generate_step
from mlx_lm.models.cache import KVCache, ArraysCache

V = 11
NEXT = [3, 5, 7, 2, 9, 4, 1, 8, 6, 10, 0]

class Toy:
    # next token = NEXT[(tok + s) % V]; only the recurrent variant keeps a
    # running state: c tokens seen, s = (c // 8) % 2
    def __init__(self, recurrent):
        self.recurrent = recurrent
        self.layers = [0, 0]
    def make_cache(self):
        return [KVCache(), ArraysCache(size=1) if self.recurrent else KVCache()]
    def __call__(self, x, cache=None):
        toks = x[0].tolist()
        kv, st = cache
        kv.update_and_fetch(mx.zeros((1, 1, len(toks), 1)), mx.zeros((1, 1, len(toks), 1)))
        if self.recurrent:
            c = st[0].item() if st[0] is not None else 0
        else:
            st.update_and_fetch(mx.zeros((1, 1, len(toks), 1)), mx.zeros((1, 1, len(toks), 1)))
            s = 0
        rows = []
        for t in toks:
            if self.recurrent:
                c += 1
                s = (c // 8) % 2
            rows.append(NEXT[(t + s) % V])
        if self.recurrent:
            st[0] = mx.array([c])
        logits = mx.full((1, len(toks), V), -1e4)
        for i, r in enumerate(rows):
            logits[0, i, r] = 0.0
        return logits

def plain(model, prompt, n):
    c = len(prompt)
    t, out = prompt[-1], []
    for _ in range(n):
        s = (c // 8) % 2 if model.recurrent else 0
        y = NEXT[(t + s) % V]
        out.append(y); t = y; c += 1
    return out

for recurrent, rate in ((False, 0.0), (True, 0.0), (True, 0.3), (True, 0.99)):
    m = Toy(recurrent)
    prompt = [3, 2, 7, 1, 3, 2, 7, 1, 3, 2]
    want = plain(m, prompt, 40)
    got, drafted = [], 0
    for tok, _, from_draft in speculative_generate_step(mx.array(prompt), m, None, num_draft_tokens=3,
                                                        max_tokens=40, lookup_ngram_min=1, lookup_min_hit_rate=rate):
        got.append(int(tok)); drafted += bool(from_draft)
    assert got == want, (recurrent, rate, got, want)
    if rate < 0.5:
        assert drafted > 0, (recurrent, rate)
    assert "MLX_OMARCHY_FULL_LOGITS" not in os.environ
print("ok")
"""


def test_greedy_output_matches_plain_decode(site, tmp_path):
    pytest.importorskip("mlx.core")
    pytest.importorskip("transformers")
    env = dict(os.environ, PYTHONPATH=str(site.parent) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    env.pop("MLX_OMARCHY_FULL_LOGITS", None)
    r = subprocess.run([sys.executable, "-c", TOY], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout[-600:] + r.stderr[-1200:]
