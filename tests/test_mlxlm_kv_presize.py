"""KV reserve: generate_step tells each KVCache how many tokens the call needs, so prefill allocates the cache once.

Upstream KVCache grows by one 256-token block per prefill chunk and builds the larger buffer while the old one is alive. A
16384-token Qwen3-4B 4-bit prompt failed twice at a growth step on a 13-inch M1 and completed with the cache allocated once.
The patch gives KVCache a `reserve` (total tokens the next growth must cover; 0 keeps the upstream rule) and has generate_step
set it to the prompt plus up to 1024 decode tokens."""
import ast
import os
import shutil
import subprocess
import sys

import pytest

from tests.test_apply_mlxlm_series import SCRIPT, VERSIONS, _pristine_venv


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
