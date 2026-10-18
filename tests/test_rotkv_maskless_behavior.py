# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Behavior regression for scripts/patch-mlx-lm-rotkv-maskless.py.

The patch lets BatchRotatingKVCache.make_mask(1) return None while the
sliding window is not binding and no row is padded, so batched decode keeps
the native SDPA decode kernel on every sliding-window layer. Dropping a mask
that is NOT all-True would let attention read padding keys or keys outside
the window, so these tests run the real patchers (ssm-maskless, kv-maskless,
kv-host-offset, rotkv-maskless, the apply-script order) on a copy of the
installed stock mlx_lm cache.py and require:

  1. a dropped mask is all-True and SDPA output is bit-equal with and without
     it, across the window boundary; padded (left, right after finalize),
     rotated, prefill, window-bound and return_array states keep the mask,
  2. the host mirror equals left_padding after every in-class mutation
     (init, update, prepare, finalize, filter, extend, merge), and any
     assignment from outside the class (or a state restore) disables the
     guard,
  3. MLX_OMARCHY_ROTKV_MASKLESS=0 keeps the mask, and the patcher stays
     wired into apply-mlx-lm-patches.sh after kv-host-offset.
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
PATCHERS = [REPO / "scripts" / "patch-mlx-lm-ssm-maskless.py",
            REPO / "scripts" / "patch-mlx-lm-kv-maskless.py",
            REPO / "scripts" / "patch-mlx-lm-kv-host-offset.py",
            REPO / "scripts" / "patch-mlx-lm-rotkv-maskless.py"]
H, D = 2, 8
W = 16  # cache max_size == sliding window: the boundary sits at _offset == W


@pytest.fixture(scope="module")
def pkg(tmp_path_factory):
    """A patched copy of the stock cache.py (rot_mlx_lm.models.cache) and a
    pristine stock copy (rot_stock_mlx_lm.models.cache) for reference masks."""
    spec = importlib.util.find_spec("mlx_lm.models.cache")
    src = Path(spec.origin).parent
    if "MLX_OMARCHY_ROTKV_MASKLESS" in (src / "cache.py").read_text():
        pytest.skip(f"installed mlx_lm cache.py is already rotkv-maskless patched: {src}")
    root = tmp_path_factory.mktemp("rotkv")
    site = root / "lib" / "python3" / "site-packages"
    for name in ("rot_mlx_lm", "rot_stock_mlx_lm"):
        models = site / name / "models"
        models.mkdir(parents=True)
        (site / name / "__init__.py").write_text("")
        (models / "__init__.py").write_text("")
        for f in ("cache.py", "base.py"):
            shutil.copy(src / f, models / f)
    # The patchers look for mlx_lm/models under the venv; point them at the copy.
    (site / "mlx_lm").symlink_to(site / "rot_mlx_lm")
    for patcher in PATCHERS:
        subprocess.run([sys.executable, str(patcher), str(root)], check=True)
    sys.path.insert(0, str(site))
    yield site, importlib.import_module("rot_stock_mlx_lm.models.cache"), \
        importlib.import_module("rot_mlx_lm.models.cache")
    sys.path.remove(str(site))


def fill(c, n, seed):
    mx.random.seed(seed)
    B = c.left_padding.shape[0]
    c.update_and_fetch(mx.random.normal((B, H, n, D)), mx.random.normal((B, H, n, D)))


def step(c, seed):
    fill(c, 1, seed)


def maskless(c, stock):
    """Decode order: build the mask, append the step's key/value, attend. Checks the mirror and, when the mask is
    dropped, that it was all-True and that attention is unchanged."""
    assert c.left_padding_list is None or c.left_padding_list == c.left_padding.tolist()
    m = c.make_mask(1)
    real = stock.BatchRotatingKVCache.make_mask(c, 1)
    if m is None:
        assert bool(mx.all(real).item()), "dropped a mask that is not all-True"
        step(c, 99)
        k, v = c.keys_and_values()
        q = mx.random.normal((k.shape[0], H, 1, D))
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.3, mask=real)
        b = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.3, mask=None)
        assert bool(mx.array_equal(a, b).item())
        c.trim(1)
    return m is None


def test_unpadded_decode_drops_only_the_decode_mask(pkg):
    _, stock, cache = pkg
    c = cache.BatchRotatingKVCache(W, [0, 0])
    assert maskless(c, stock)
    fill(c, 5, 1)
    assert maskless(c, stock)
    assert c.make_mask(2) is not None
    assert c.make_mask(1, return_array=True) is not None
    assert c.make_mask(1, window_size=3) is not None  # the window binds: 5 keys >= 3
    real8 = stock.BatchRotatingKVCache.make_mask(c, 1, 8)
    assert c.make_mask(1, window_size=8) is None and bool(mx.all(real8).item())
    while c._offset < W - 1:
        step(c, 20 + c._offset)
    assert maskless(c, stock)  # _offset == W - 1: still every key inside the window
    step(c, 40)
    assert c._offset == W and c.make_mask(1) is not None  # the window binds now
    step(c, 41)
    assert c.rotated and c.make_mask(1) is not None
    assert c.left_padding_list == c.left_padding.tolist()


def test_padded_states_keep_the_mask(pkg):
    _, stock, cache = pkg
    c = cache.BatchRotatingKVCache(W, [1, 0, 2])
    fill(c, 5, 2)
    assert not maskless(c, stock)

    c = cache.BatchRotatingKVCache(W, [0, 0, 0])
    c.prepare(left_padding=[0, 0, 0], lengths=[6, 4, 6], right_padding=[0, 2, 0])
    fill(c, 6, 3)
    c.finalize()
    assert not maskless(c, stock)

    c = cache.BatchRotatingKVCache(W, [0, 0])
    c.prepare(left_padding=[0, 0], lengths=[6, 6], right_padding=[0, 0])
    fill(c, 6, 4)
    c.finalize()
    assert maskless(c, stock)


def test_filter_extend_merge_track_the_mirror(pkg):
    _, stock, cache = pkg
    c = cache.BatchRotatingKVCache(W, [2, 0, 1])
    fill(c, 6, 5)
    c.filter([0, 2])
    assert c.left_padding_list == [2, 1] and not maskless(c, stock)
    c.filter([1])
    assert c.left_padding_list == [1] and c.make_mask(1) is not None

    c = cache.BatchRotatingKVCache(W, [0, 0])
    fill(c, 6, 5)
    c.filter([0])
    assert maskless(c, stock)
    c.filter(mx.array([0]))
    assert c.left_padding_list is None and c.make_mask(1) is not None

    a, b = cache.BatchRotatingKVCache(W, [0, 0]), cache.BatchRotatingKVCache(W, [0])
    fill(a, 4, 6)
    fill(b, 7, 7)
    a.extend(b)
    assert a.left_padding_list == [3, 3, 0] and not maskless(a, stock)

    a, b = cache.BatchRotatingKVCache(W, [0]), cache.BatchRotatingKVCache(W, [0])
    a.extend(b)
    fill(a, 3, 8)
    assert a.left_padding_list == [0, 0] and maskless(a, stock)

    m = cache.BatchRotatingKVCache.merge(
        [cache.BatchRotatingKVCache(W, [0]), cache.BatchRotatingKVCache(W, [0])]
    )
    assert m.left_padding_list == [0, 0] and maskless(m, stock)


def test_outside_writes_disable_the_guard(pkg):
    _, stock, cache = pkg
    c = cache.BatchRotatingKVCache(W, [0, 0])
    fill(c, 4, 11)
    c.left_padding = c.left_padding + 1
    assert c.left_padding_list is None and not maskless(c, stock)

    c = cache.BatchRotatingKVCache(W, [0, 0])
    fill(c, 4, 12)
    c.state = c.state
    assert c.left_padding_list is None and c.make_mask(1) is not None


def test_kill_switch_and_wiring(pkg):
    site, _, _ = pkg
    code = ("from rot_mlx_lm.models.cache import BatchRotatingKVCache; import mlx.core as mx; "
            "c = BatchRotatingKVCache(16, [0]); "
            "c.update_and_fetch(mx.zeros((1, 2, 3, 8)), mx.zeros((1, 2, 3, 8))); "
            "assert c.make_mask(1) is not None")
    env = dict(os.environ, MLX_OMARCHY_ROTKV_MASKLESS="0", PYTHONPATH=str(site))
    assert subprocess.run([sys.executable, "-c", code], env=env).returncode == 0
    apply = (REPO / "scripts" / "apply-mlx-lm-patches.sh").read_text()
    assert (apply.index("patch-mlx-lm-ssm-maskless.py") < apply.index("patch-mlx-lm-kv-maskless.py")
            < apply.index("patch-mlx-lm-kv-host-offset.py")
            < apply.index("patch-mlx-lm-rotkv-maskless.py"))
    assert "rotkv-maskless: 0.32 series only" in apply
