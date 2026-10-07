# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Behavior regression for scripts/patch-mlx-lm-kv-maskless.py.

The patch lets BatchKVCache.make_mask(1) return None when no row is padded,
so batched decode reaches the native SDPA decode kernel. Dropping a mask that
is NOT all-True would let attention read padding keys, so these tests run the
real patchers (ssm-maskless, then kv-maskless) on a copy of the installed
stock mlx_lm cache.py and require:

  1. a dropped mask is all-True and SDPA output is bit-equal with and without
     it; every padded state (left, right after finalize, filter, extend,
     merge) keeps the mask, and so do prefill, windowed and return_array calls,
  2. the host mirror equals left_padding after every in-class mutation, and
     any assignment from outside the class (or a state restore) disables the
     guard,
  3. MLX_OMARCHY_KV_MASKLESS=0 keeps the mask, and the patcher stays wired
     into apply-mlx-lm-patches.sh.
"""

import importlib
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
            REPO / "scripts" / "patch-mlx-lm-kv-maskless.py"]
H, D = 2, 8


@pytest.fixture(scope="module")
def pkg(tmp_path_factory):
    """A patched copy of the stock cache.py, importable as kvm_mlx_lm.models.cache."""
    spec = importlib.util.find_spec("mlx_lm.models.cache")
    src = Path(spec.origin).parent
    if "MLX_OMARCHY_KV_MASKLESS" in (src / "cache.py").read_text():
        pytest.skip(f"installed mlx_lm cache.py is already kv-maskless patched: {src}")
    root = tmp_path_factory.mktemp("kvm")
    models = root / "lib" / "python3" / "site-packages" / "kvm_mlx_lm" / "models"
    models.mkdir(parents=True)
    (models.parent / "__init__.py").write_text("")
    (models / "__init__.py").write_text("")
    for name in ("cache.py", "base.py"):
        shutil.copy(src / name, models / name)
    # The patchers look for mlx_lm/models under the venv; point them at the copy.
    (models.parent.parent / "mlx_lm").symlink_to(models.parent)
    for patcher in PATCHERS:
        subprocess.run([sys.executable, str(patcher), str(root)], check=True)
    site = str(models.parent.parent)
    sys.path.insert(0, site)
    yield site, importlib.import_module("kvm_mlx_lm.models.cache"), \
        importlib.import_module("kvm_mlx_lm.models.base")
    sys.path.remove(site)


def fill(c, n, seed):
    mx.random.seed(seed)
    B = c.left_padding.shape[0]
    c.update_and_fetch(mx.random.normal((B, H, n, D)), mx.random.normal((B, H, n, D)))


def maskless(c, base):
    """Decode order: build the mask, append the step's key/value, attend. Checks the mirror and, when the mask is
    dropped, that it was all-True and that attention is unchanged."""
    assert c.left_padding_list is None or c.left_padding_list == c.left_padding.tolist()
    m = c.make_mask(1)
    real = base.create_causal_mask(1, offset=c._idx, left_padding=c.left_padding)
    if m is None:
        assert bool(mx.all(real).item()), "dropped a mask that is not all-True"
        fill(c, 1, 99)
        k, v = c.keys_and_values()
        q = mx.random.normal((k.shape[0], H, 1, D))
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.3, mask=real)
        b = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.3, mask=None)
        assert bool(mx.array_equal(a, b).item())
        c.trim(1)
    return m is None


def test_unpadded_decode_drops_only_the_decode_mask(pkg):
    _, cache, base = pkg
    c = cache.BatchKVCache([0, 0, 0])
    fill(c, 5, 1)
    assert maskless(c, base)
    assert c.make_mask(2) is not None
    assert c.make_mask(1, return_array=True) is not None
    assert c.make_mask(1, window_size=4) is not None


def test_padded_states_keep_the_mask(pkg):
    _, cache, base = pkg
    c = cache.BatchKVCache([1, 0, 2])
    fill(c, 5, 2)
    assert not maskless(c, base)

    c = cache.BatchKVCache([0, 0, 0])
    c.prepare(left_padding=[0, 0, 0], right_padding=[0, 2, 0])
    fill(c, 6, 3)
    c.finalize()
    assert not maskless(c, base)

    c = cache.BatchKVCache([0, 0, 0])
    c.prepare(left_padding=[0, 0, 0], right_padding=[0, 0, 0])
    fill(c, 6, 4)
    c.finalize()
    assert maskless(c, base)


def test_filter_extend_merge_track_the_mirror(pkg):
    _, cache, base = pkg
    c = cache.BatchKVCache([2, 0, 1])
    fill(c, 6, 5)
    c.filter([0, 2])
    assert c.left_padding_list == [1, 0] and not maskless(c, base)
    c.filter([1])
    assert maskless(c, base)

    a, b = cache.BatchKVCache([0, 0]), cache.BatchKVCache([0])
    fill(a, 4, 6)
    fill(b, 7, 7)
    a.extend(b)
    assert a.left_padding_list == [3, 3, 0] and not maskless(a, base)

    a, b = cache.BatchKVCache([0]), cache.BatchKVCache([0])
    a.extend(b)
    fill(a, 3, 8)
    assert a.left_padding_list == [0, 0] and maskless(a, base)

    rows = []
    for n, seed in ((3, 9), (5, 10)):
        k = cache.KVCache()
        mx.random.seed(seed)
        k.update_and_fetch(mx.random.normal((1, H, n, D)), mx.random.normal((1, H, n, D)))
        rows.append(k)
    m = cache.BatchKVCache.merge(rows)
    assert m.left_padding_list == [2, 0] and not maskless(m, base)


def test_outside_writes_disable_the_guard(pkg):
    _, cache, base = pkg
    c = cache.BatchKVCache([0, 0])
    fill(c, 4, 11)
    c.left_padding = c.left_padding + 1
    assert c.left_padding_list is None and not maskless(c, base)

    c = cache.BatchKVCache([0, 0])
    fill(c, 4, 12)
    c.state = c.state
    assert c.left_padding_list is None and c.make_mask(1) is not None


def test_kill_switch_and_wiring(pkg):
    site, _, _ = pkg
    code = ("from kvm_mlx_lm.models.cache import BatchKVCache; import mlx.core as mx; c = BatchKVCache([0]); "
            "c.update_and_fetch(mx.zeros((1, 2, 3, 8)), mx.zeros((1, 2, 3, 8))); "
            "assert c.make_mask(1) is not None")
    env = dict(os.environ, MLX_OMARCHY_KV_MASKLESS="0", PYTHONPATH=site)
    assert subprocess.run([sys.executable, "-c", code], env=env).returncode == 0
    apply = (REPO / "scripts" / "apply-mlx-lm-patches.sh").read_text()
    assert apply.index("patch-mlx-lm-ssm-maskless.py") < apply.index("patch-mlx-lm-kv-maskless.py")
