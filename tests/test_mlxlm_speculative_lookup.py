"""Speculative verify and prompt lookup (mlx-lm 0.31.3 and 0.32 series).

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


@pytest.fixture(scope="module", params=["0.31.3", "0.32.0"])
def site(request, tmp_path_factory):
    root = _pristine_venv(tmp_path_factory.mktemp("spec-lookup"), request.param)
    r = subprocess.run(["bash", str(SCRIPT), str(root)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-400:] + r.stderr[-400:]
    return root / "lib" / "python3.11" / "site-packages" / "mlx_lm"


def helpers(site, *names, extra_ns=None):
    tree = ast.parse((site / "generate.py").read_text())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in keep} == set(names)
    for n in keep:
        n.decorator_list = []
    module = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"os": os, **(extra_ns or {})}
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
    # MLX_OMARCHY_SDPA_SPLIT_ROWS is never written by the patch: unset stays unset (plain
    # path) in and after every pass, and an explicit value stays fixed for the whole process.
    (verify,) = helpers(site, "_omarchy_verify_pass")
    import contextlib

    verify = contextlib.contextmanager(verify)
    monkeypatch.delenv("MLX_OMARCHY_FULL_LOGITS", raising=False)
    monkeypatch.delenv("MLX_OMARCHY_QMV_SMALL_T", raising=False)
    monkeypatch.delenv("MLX_OMARCHY_SDPA_SPLIT_ROWS", raising=False)
    with verify(4):
        assert os.environ["MLX_OMARCHY_FULL_LOGITS"] == "1"
        assert os.environ["MLX_OMARCHY_QMV_SMALL_T"] == "4"
        assert "MLX_OMARCHY_SDPA_SPLIT_ROWS" not in os.environ
    assert "MLX_OMARCHY_FULL_LOGITS" not in os.environ
    assert "MLX_OMARCHY_QMV_SMALL_T" not in os.environ
    assert "MLX_OMARCHY_SDPA_SPLIT_ROWS" not in os.environ
    with verify(12):
        assert os.environ["MLX_OMARCHY_QMV_SMALL_T"] == "8"
    with verify(1):
        assert "MLX_OMARCHY_QMV_SMALL_T" not in os.environ
        assert "MLX_OMARCHY_SDPA_SPLIT_ROWS" not in os.environ
    monkeypatch.setenv("MLX_OMARCHY_QMV_SMALL_T", "0")
    monkeypatch.setenv("MLX_OMARCHY_FULL_LOGITS", "0")
    assert os.environ["MLX_OMARCHY_FULL_LOGITS"] == "0"
    for fixed in ("0", "1"):
        monkeypatch.setenv("MLX_OMARCHY_SDPA_SPLIT_ROWS", fixed)
        for rows in (1, 4, 12):
            with verify(rows):
                assert os.environ["MLX_OMARCHY_SDPA_SPLIT_ROWS"] == fixed
            assert os.environ["MLX_OMARCHY_SDPA_SPLIT_ROWS"] == fixed
    monkeypatch.setenv("MLX_OMARCHY_QMV_SMALL_T", "0")
    with verify(4):
        assert os.environ["MLX_OMARCHY_QMV_SMALL_T"] == "0"


def test_verify_pass_sets_and_restores_the_gdn_sequential_route(site, monkeypatch):
    # Qwen3.5 GDN layers in a verify pass take the S == 1 route one token at a time
    # (the S > 1 route changed a hybrid model's greedy ids). Same contract as the
    # short-pass GEMV variable: set only for rows > 1, capped at 8, restored after,
    # and a value already in the environment is left as it is.
    (verify,) = helpers(site, "_omarchy_verify_pass")
    import contextlib

    verify = contextlib.contextmanager(verify)
    monkeypatch.delenv("MLX_OMARCHY_GDN_SEQ_ROWS", raising=False)
    with verify(4):
        assert os.environ["MLX_OMARCHY_GDN_SEQ_ROWS"] == "4"
    assert "MLX_OMARCHY_GDN_SEQ_ROWS" not in os.environ
    with verify(12):
        assert os.environ["MLX_OMARCHY_GDN_SEQ_ROWS"] == "8"
    with verify(1):
        assert "MLX_OMARCHY_GDN_SEQ_ROWS" not in os.environ
    monkeypatch.setenv("MLX_OMARCHY_GDN_SEQ_ROWS", "0")
    with verify(4):
        assert os.environ["MLX_OMARCHY_GDN_SEQ_ROWS"] == "0"


def test_speculation_engages_split_rows_once_per_process(site):
    # Verify-pass attention rows match single-token decode only on the split-K rows
    # route; speculative_generate_step sets MLX_OMARCHY_SDPA_SPLIT_ROWS=1 when unset,
    # before its loop, and never per verify pass (the verify-pass test above pins that).
    src = (site / "generate.py").read_text()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "speculative_generate_step")
    body = ast.get_source_segment(src, fn)
    guard = 'if os.environ.get("MLX_OMARCHY_SDPA_SPLIT_ROWS") is None:'
    assert guard in body
    assert 'os.environ["MLX_OMARCHY_SDPA_SPLIT_ROWS"] = "1"' in body
    assert body.index(guard) < body.index("while True:")


GDN_ROWS_PROBE = r"""
import os, sys
sys.path.insert(0, sys.argv[1])
import mlx.core as mx
from mlx_lm.models import qwen3_5

class Layer:
    sharding_group = None
    def __init__(self):
        self.seen = []
    def __call__(self, inputs, mask=None, cache=None):
        if inputs.shape[1] > 1:
            return qwen3_5.GatedDeltaNet.__call__(self, inputs, mask=mask, cache=cache)
        self.seen.append(inputs.shape[1])
        return inputs * 2

layer = Layer()
x = mx.arange(12, dtype=mx.float32).reshape(1, 3, 4)
os.environ["MLX_OMARCHY_GDN_SEQ_ROWS"] = "4"
out = layer(x, mask=None, cache=object())
assert layer.seen == [1, 1, 1], layer.seen
assert mx.array_equal(out, x * 2).item()
print("GDN-SEQ-OK")
"""


def test_gdn_layer_feeds_verify_rows_one_at_a_time(site):
    # Inside a verify pass (MLX_OMARCHY_GDN_SEQ_ROWS set) a 3-token GDN call is three
    # S == 1 calls, concatenated in order. Without the patch the layer takes the S > 1
    # route (here: an AttributeError on the stub, which has no projections).
    pytest.importorskip("mlx.core")
    r = subprocess.run([sys.executable, "-c", GDN_ROWS_PROBE, str(site.parent)],
                       capture_output=True, text=True, timeout=120)
    assert "GDN-SEQ-OK" in r.stdout, r.stdout[-400:] + r.stderr[-800:]


def test_snapshot_detaches_tuple_state_from_the_live_cache(site):
    # 0.32 ArraysCache.state is a tuple holding the live cache list, and
    # CacheList.state nests sub-cache states; a verify pass writes into those
    # containers, so the snapshot must clone every list/tuple level or a
    # rejected draft restores corrupted state.
    trimmed = []

    def trim(caches, n):
        trimmed.append(n)

    cache_mod = type("cache", (), {"trim_prompt_cache": staticmethod(trim)})
    snap, restore = helpers(
        site,
        "_omarchy_snapshot",
        "_omarchy_restore",
        extra_ns={"trim_prompt_cache": trim, "cache": cache_mod},
    )

    class Arrays:
        def __init__(self):
            self.cache = ["a"]

        def is_trimmable(self):
            return False

        @property
        def state(self):
            return (self.cache, None, None)

        @state.setter
        def state(self, v):
            self.cache, _, _ = v

    class Wrap:  # CacheList shape: [(state, name), ...]
        def __init__(self, inner):
            self.inner = inner

        def is_trimmable(self):
            return False

        @property
        def state(self):
            return [(self.inner.state, "ArraysCache")]

        @state.setter
        def state(self, v):
            ((st, _),) = v
            self.inner.state = st

    for cache in (Arrays(), Wrap(Arrays())):
        snapshot = snap([cache])
        inner = getattr(cache, "inner", cache)
        inner.cache.append("verify-write")
        restore([cache], snapshot, 1)
        assert inner.cache == ["a"], inner.cache


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


ROUTE = r"""
import os
import mlx.core as mx
import importlib
g = importlib.import_module("mlx_lm.generate")  # the name mlx_lm.generate is the function
from mlx_lm.tokenizer_utils import TokenizerWrapper

calls = []


def fake(name):
    def step(prompt, model, *a, **kw):
        calls.append((name, dict(kw)))
        for t in range(3):
            yield (mx.array(t + 1), mx.zeros(4), False) if name == "spec" else (mx.array(t + 1), mx.zeros(4))
    return step


class Det:
    last_segment = ""
    def add_token(self, t): pass
    def finalize(self): pass


class Tok(TokenizerWrapper):
    def __init__(self): pass
    eos_token_ids = set()
    detokenizer = Det()


g.speculative_generate_step = fake("spec")
g.generate_step = fake("plain")
os.environ["MLX_OMARCHY_PROMPT_LOOKUP"] = "3"


def route(**kw):
    calls.clear()
    list(g.stream_generate(None, Tok(), [1, 2, 3], max_tokens=3, **kw))
    return calls[0]


name, kw = route()
assert name == "spec" and kw["num_draft_tokens"] == 3, (name, kw)
name, kw = route(max_kv_size=64)
assert name == "plain" and kw["max_kv_size"] == 64 and "num_draft_tokens" not in kw, (name, kw)
print("ok")
"""


def test_prompt_lookup_keeps_the_max_kv_size_bound(site):
    # speculative_generate_step builds an unbounded KVCache, so a set max_kv_size must stay on generate_step.
    pytest.importorskip("mlx.core")
    pytest.importorskip("transformers")
    env = dict(os.environ, PYTHONPATH=str(site.parent) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, "-c", ROUTE], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout[-600:] + r.stderr[-1200:]
