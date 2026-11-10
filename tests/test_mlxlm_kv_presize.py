"""KV reserve: generate_step tells each KVCache how many tokens the call needs, so prefill allocates the cache once.

Upstream KVCache grows by one 256-token block per prefill chunk and builds the larger buffer while the old one is alive. A
16384-token Qwen3-4B 4-bit prompt failed twice at a growth step on a 13-inch M1 and completed with the cache allocated once.
The patch gives KVCache a `reserve` (total tokens the next growth must cover; 0 keeps the upstream rule) and has generate_step
set it to the prompt plus up to 1024 decode tokens. The prefill chunk is chosen before the cache exists, so generate_step chooses
it again once the first chunk has made the reserved cache resident, from the heap headroom that is left."""
import ast
import os
import shutil
import subprocess
import sys

import pytest

from tests.test_apply_mlxlm_series import REPO, SCRIPT, VERSIONS, _pristine_venv

sys.path.insert(0, str(REPO / "serve"))
from mlx_omarchy_serve import budget  # noqa: E402

GIB = 1024 ** 3


@pytest.fixture(scope="module", params=VERSIONS)
def site(request, tmp_path_factory):
    root = _pristine_venv(tmp_path_factory.mktemp("kv-presize"), request.param)
    r = subprocess.run(["bash", str(SCRIPT), str(root)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-400:] + r.stderr[-400:]
    return root / "lib" / "python3.11" / "site-packages" / "mlx_lm"


def test_kvcache_allocates_once_for_a_reserve(site, tmp_path):
    """The real patched cache.py: how many tokens the buffer holds after each of four 512-token prefill chunks."""
    pytest.importorskip("mlx.core")
    pkg = tmp_path / "kv_pkg"
    (pkg / "models").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "models" / "__init__.py").write_text("")
    for name in ("base.py", "cache.py"):
        shutil.copy(site / "models" / name, pkg / "models" / name)
    code = """
import mlx.core as mx
from kv_pkg.models.cache import KVCache


def capacities(reserve):
    c = KVCache()
    c.reserve = reserve
    caps = []
    for i in range(4):
        chunk = mx.full((1, 2, 512, 8), float(i))
        k, v = c.update_and_fetch(chunk, chunk)
        caps.append(c.keys.shape[2])
    assert k.shape[2] == v.shape[2] == 2048
    for i in range(4):
        assert mx.all(k[:, :, i * 512:(i + 1) * 512] == float(i)).item()
    return caps


assert capacities(0) == [512, 1024, 1536, 2048]
assert capacities(1300) == [1536, 1536, 1536, 2048]
assert capacities(2048) == [2048] * 4
"""
    env = dict(os.environ, PYTHONPATH=str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-600:]


class KVCache:
    reserve = 0

    def __init__(self, offset=0):
        self.offset = offset


class RotatingKVCache:
    def __init__(self):
        self.offset = 0


def reserve_fn(site):
    """The patch's helper on its own, against stand-in cache classes."""
    tree = ast.parse((site / "generate.py").read_text())
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name == "_omarchy_reserve_kv")
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "_OMARCHY_KV_DECODE_RESERVE" for t in n.targets))]
    module = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"KVCache": KVCache, "os": os}
    exec(compile(module, "generate.py", "exec"), ns)
    return ns["_omarchy_reserve_kv"]


def test_reserve_is_the_prompt_plus_the_first_decode_tokens(site, monkeypatch):
    monkeypatch.delenv("MLX_OMARCHY_KV_PRESIZE", raising=False)
    fresh, reused, rotating = KVCache(), KVCache(offset=100), RotatingKVCache()
    reserve_fn(site)([fresh, reused, rotating], 5000, 128, None)
    assert fresh.reserve == 5128
    assert reused.reserve == 5228
    assert not hasattr(rotating, "reserve")


def test_decode_part_is_bounded(site, monkeypatch):
    monkeypatch.delenv("MLX_OMARCHY_KV_PRESIZE", raising=False)
    for max_tokens, decode in ((10**6, 1024), (-1, 1024), (1024, 1024), (200, 200), (0, 0)):
        cache = KVCache()
        reserve_fn(site)([cache], 5000, max_tokens, None)
        assert cache.reserve == 5000 + decode, max_tokens


def test_quantized_kv_and_the_kill_switch_keep_upstream_growth(site, monkeypatch):
    monkeypatch.delenv("MLX_OMARCHY_KV_PRESIZE", raising=False)
    quantized = KVCache()
    reserve_fn(site)([quantized], 5000, 128, 4)
    assert quantized.reserve == 0
    monkeypatch.setenv("MLX_OMARCHY_KV_PRESIZE", "0")
    off = KVCache()
    reserve_fn(site)([off], 5000, 128, None)
    assert off.reserve == 0
    monkeypatch.setenv("MLX_OMARCHY_KV_PRESIZE", "1")
    on = KVCache()
    reserve_fn(site)([on], 5000, 128, None)
    assert on.reserve == 5128


def test_reserve_reports_whether_the_cache_will_be_resident(site, monkeypatch):
    monkeypatch.delenv("MLX_OMARCHY_KV_PRESIZE", raising=False)
    reserve = reserve_fn(site)
    assert reserve([KVCache(), RotatingKVCache()], 5000, 128, None) is True
    assert reserve([RotatingKVCache()], 5000, 128, None) is False
    assert reserve([KVCache()], 5000, 128, 4) is False
    monkeypatch.setenv("MLX_OMARCHY_KV_PRESIZE", "0")
    assert reserve([KVCache()], 5000, 128, None) is False


def rechoose_fn(site, headroom):
    """The patch's chooser and re-chooser on their own, with the heap reader answering `headroom`."""
    tree = ast.parse((site / "generate.py").read_text())
    names = {"_omarchy_prefill_step", "_omarchy_rechoose_step"}
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in names)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "").startswith("_OMARCHY_PREFILL") for t in n.targets))]
    module = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"_omarchy_prefill_headroom_bytes": lambda: headroom}
    exec(compile(module, "generate.py", "exec"), ns)
    return ns["_omarchy_rechoose_step"], ns["_OMARCHY_PREFILL_WORKSPACE_BYTES"]


def test_workspace_is_the_serve_budget_floor(site):
    assert rechoose_fn(site, None)[1] == budget.WORKSPACE_MIN_BYTES


def test_rechoose_leaves_the_workspace_free_for_the_chunk(site):
    context, workspace = 16384, budget.WORKSPACE_MIN_BYTES
    caches = [KVCache(offset=100), KVCache(offset=1024)]
    fits_256 = workspace + budget.prefill_scratch_bytes(256, context)
    assert rechoose_fn(site, fits_256 + 1)[0](2048, caches, context - 1024) == 256
    assert rechoose_fn(site, fits_256 - 1)[0](2048, caches, context - 1024) == 128


def test_rechoose_never_grows_the_chunk(site):
    caches = [KVCache(offset=1024)]
    assert rechoose_fn(site, 100 * GIB)[0](2048, caches, 15360) == 2048
    assert rechoose_fn(site, 100 * GIB)[0](128, caches, 15360) == 128


def test_rechoose_keeps_the_chunk_without_a_heap_report(site):
    assert rechoose_fn(site, None)[0](512, [KVCache(offset=1024)], 15360) == 512


LOOP = """
import importlib, os, sys
import mlx.core as mx
try:
    g = importlib.import_module("mlx_lm.generate")
    from mlx_lm.models import qwen3
except ImportError as error:
    print(error)
    sys.exit(77)
mx.set_default_device(mx.cpu)
args = qwen3.ModelArgs(
    model_type="qwen3", hidden_size=64, num_hidden_layers=2, intermediate_size=128, num_attention_heads=4,
    num_key_value_heads=2, rms_norm_eps=1e-6, vocab_size=100, head_dim=16, max_position_embeddings=8192,
    rope_theta=10000.0, tie_word_embeddings=True,
)
model = qwen3.Model(args)
mx.eval(model.parameters())
seen = []
call = qwen3.Model.__call__


def spy(self, inputs, *a, **k):
    seen.append(int(inputs.shape[1]))
    return call(self, inputs, *a, **k)


qwen3.Model.__call__ = spy
prompt = mx.array([(i * 7) % 90 + 1 for i in range(6000)])
BIG = 10**12


def chunks(headrooms, **kwargs):
    asked = []

    def headroom():
        asked.append(1)
        return headrooms[min(len(asked), len(headrooms)) - 1]

    g._omarchy_prefill_headroom_bytes = headroom
    del seen[:]
    next(g.generate_step(prompt, model, max_tokens=1, **kwargs))
    return [n for n in seen if n > 1]


assert chunks([BIG, SMALL]) == [2048] + [512] * 7 + [367], chunks([BIG, SMALL])
assert chunks([BIG, BIG]) == [2048, 2048, 1903]
assert chunks([None]) == [2048, 2048, 1903]
assert chunks([BIG, SMALL], prefill_step_size=1024) == [1024] * 5 + [879]
os.environ["MLX_OMARCHY_KV_PRESIZE"] = "0"
assert chunks([BIG, SMALL]) == [2048, 2048, 1903]
"""


def test_first_chunk_decides_the_rest_of_the_prefill(site):
    """The real patched generate_step on a tiny CPU model: the chunk lengths the model sees."""
    pytest.importorskip("mlx.core")
    workspace = rechoose_fn(site, None)[1]
    small = workspace + budget.prefill_scratch_bytes(512, 6000) + 1
    env = dict(os.environ, PYTHONPATH=str(site.parent) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, "-c", f"SMALL = {small}\n" + LOOP], env=env, capture_output=True, text=True,
                       timeout=300)
    if r.returncode == 77:
        pytest.skip("mlx_lm is not importable here: " + r.stdout.strip()[-160:])
    assert r.returncode == 0, r.stderr[-800:]
