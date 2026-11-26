# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Behavior regression for scripts/patch-mlx-lm-rotkv-host-offset.py and patch-mlx-lm-gemma4-rope-offset.py.

BatchRotatingKVCache keeps a host mirror of its offset so the RoPE trig gate reads a host-built array instead of joining the
stream (the sliding-window half of patch-mlx-lm-kv-host-offset.py). The mirror must never change a result, so these tests
run the real patcher chain on a copy of the installed stock cache.py and compare two module objects loaded from the same
patched file, one with MLX_OMARCHY_KV_HOST_OFFSET=0 (device ops) and one with 1 (mirror), step by step through decode,
window rotation and a filter() between prefill and decode (what PromptProcessingBatch.split does on every request):

  1. keys, values, offset, left_padding are bit-identical at every step, and the mask route agrees,
  2. the mirror equals the arrays after every decode step, including after filter() cleared it,
  3. the patchers are idempotent, refuse to run before their prerequisites, and are wired into apply-mlx-lm-patches.sh;
     the gemma-4 patcher hands an array offset to rope unchanged and keeps the int path.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
CHAIN = [SCRIPTS / "patch-mlx-lm-ssm-maskless.py",
         SCRIPTS / "patch-mlx-lm-kv-maskless.py",
         SCRIPTS / "patch-mlx-lm-kv-host-offset.py",
         SCRIPTS / "patch-mlx-lm-rotkv-maskless.py"]
ROTKV = SCRIPTS / "patch-mlx-lm-rotkv-host-offset.py"
GEMMA4 = SCRIPTS / "patch-mlx-lm-gemma4-rope-offset.py"


def run(script, root, check=True):
    return subprocess.run([sys.executable, str(script), str(root)], check=check, capture_output=True, text=True)


def make_venv(base, files):
    site = base / "lib" / "python3" / "site-packages"
    models = site / "mlx_lm_copy" / "models"
    models.mkdir(parents=True)
    (site / "mlx_lm_copy" / "__init__.py").write_text("")
    (models / "__init__.py").write_text("")
    src = Path(importlib.util.find_spec("mlx_lm.models.cache").origin).parent
    for name in files:
        if (src / name).exists():
            shutil.copy(src / name, models / name)
    (site / "mlx_lm").symlink_to(site / "mlx_lm_copy")
    return site, models


@pytest.fixture(scope="module")
def patched(tmp_path_factory):
    src = Path(importlib.util.find_spec("mlx_lm.models.cache").origin)
    if "BatchRotatingKVCache host offset mirror" in src.read_text():
        pytest.skip(f"installed mlx_lm cache.py is already rotkv-host-offset patched: {src}")
    root = tmp_path_factory.mktemp("rotkvho")
    _, models = make_venv(root, ("cache.py", "base.py", "gemma4_text.py"))
    for patcher in CHAIN:
        run(patcher, root)
    before = (models / "cache.py").read_text()
    run(ROTKV, root)
    return root, models, before


def load(models, flag, tag):
    """One module object per flag: the methods read the module global at call time, so a reload would flip both caches."""
    os.environ["MLX_OMARCHY_KV_HOST_OFFSET"] = flag
    name = f"mlx_lm_copy.models.cache_{tag}_{flag}"
    spec = importlib.util.spec_from_file_location(name, models / "cache.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    assert module._KV_HOST_OFFSET == (flag != "0")
    return module


def same(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert bool(mx.array_equal(a, b).item())


@pytest.fixture
def site_path(patched):
    root = patched[0]
    site = str(root / "lib" / "python3" / "site-packages")
    sys.path.insert(0, site)
    yield
    sys.path.remove(site)
    os.environ.pop("MLX_OMARCHY_KV_HOST_OFFSET", None)


def drive(patched, padding, filtered, tag):
    _, models, _ = patched
    dev, host = load(models, "0", tag), load(models, "1", tag)
    max_size, prefill, steps = 8, 5, 30
    cache_dev = dev.BatchRotatingKVCache(max_size, list(padding))
    cache_host = host.BatchRotatingKVCache(max_size, list(padding))
    key = mx.random.key(7)
    batch = len(padding)
    for step, length in enumerate([prefill] + [1] * steps):
        key, sub = mx.random.split(key)
        keys = mx.random.normal((batch, 2, length, 4), key=sub)
        values = keys + 1.0
        if filtered and step == 1:
            cache_dev.filter(list(range(batch)))
            cache_host.filter(list(range(batch)))
        out_dev = cache_dev.update_and_fetch(keys, values)
        out_host = cache_host.update_and_fetch(keys, values)
        mx.eval(out_dev, out_host)
        same(out_dev[0], out_host[0])
        same(out_dev[1], out_host[1])
        same(cache_dev.offset, cache_host.offset)
        same(cache_dev.left_padding, cache_host.left_padding)
        if length == 1:
            assert cache_host.offset_list == cache_host.offset.tolist()
            assert cache_host.left_padding_list == cache_host.left_padding.tolist()
        mask_dev, mask_host = cache_dev.make_mask(1), cache_host.make_mask(1)
        assert (mask_dev is None) == (mask_host is None)
        if mask_dev is not None:
            same(mask_dev, mask_host)
    assert cache_host.rotated
    return cache_host


@pytest.mark.parametrize("padding", [[0], [0, 0], [2, 0], [3, 1, 0]])
@pytest.mark.parametrize("filtered", [False, True])
def test_mirror_matches_device_path_through_rotation(patched, site_path, padding, filtered):
    cache = drive(patched, padding, filtered, f"{len(padding)}{int(filtered)}")
    assert cache.offset_list is not None


def test_outside_assignment_clears_mirror_and_decode_rebuilds_it(patched, site_path):
    _, models, _ = patched
    host = load(models, "1", "outside")
    cache = host.BatchRotatingKVCache(8, [0])
    keys = mx.random.normal((1, 2, 5, 4))
    cache.update_and_fetch(keys, keys)
    assert cache.offset_list == [5]
    cache.offset = mx.array([5])
    assert cache.offset_list is None
    step = mx.random.normal((1, 2, 1, 4))
    cache.update_and_fetch(step, step)
    assert cache.offset_list == [6] and cache.offset.tolist() == [6]


def test_kill_switch_keeps_the_device_offset(patched, site_path):
    _, models, _ = patched
    dev = load(models, "0", "kill")
    cache = dev.BatchRotatingKVCache(8, [0])
    keys = mx.random.normal((1, 2, 3, 4))
    cache.update_and_fetch(keys, keys)
    step = mx.random.normal((1, 2, 1, 4))
    cache.update_and_fetch(step, step)
    assert cache.offset.tolist() == [4]


def test_rotkv_patcher_is_idempotent_and_needs_its_prerequisites(patched, tmp_path):
    root, models, before = patched
    assert run(ROTKV, root).stdout.startswith("already patched")
    assert (models / "cache.py").read_text() != before
    bare, _ = make_venv(tmp_path / "bare", ("cache.py", "base.py"))
    refusal = run(ROTKV, tmp_path / "bare", check=False)
    assert refusal.returncode != 0 and "kv-host-offset" in (refusal.stderr + refusal.stdout)


def test_gemma4_patcher_passes_an_array_offset_through(patched):
    root, models, _ = patched
    path = models / "gemma4_text.py"
    if not path.exists():
        pytest.skip("installed mlx_lm has no gemma4_text.py")
    stock = path.read_text()
    if "isinstance(cache.offset, mx.array)" in stock:
        pytest.skip("gemma4_text.py already patched")
    assert "offset = mx.array(cache.offset) if cache is not None else 0" in stock
    run(GEMMA4, root)
    text = path.read_text()
    assert "offset = cache.offset if isinstance(cache.offset, mx.array) else mx.array(cache.offset)" in text
    assert text.count("mx.array(cache.offset)") == 1
    assert run(GEMMA4, root).stdout.startswith("already patched")


def test_patchers_are_wired_after_the_prerequisites():
    apply = (SCRIPTS / "apply-mlx-lm-patches.sh").read_text()
    order = [apply.index(name) for name in ("patch-mlx-lm-kv-host-offset.py", "patch-mlx-lm-rotkv-maskless.py",
                                           "patch-mlx-lm-rotkv-host-offset.py", "patch-mlx-lm-gemma4-rope-offset.py")]
    assert order == sorted(order) and len(set(order)) == 4
