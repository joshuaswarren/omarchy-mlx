# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Behavior regression for scripts/patch-mlx-lm-ssm-maskless.py.

ArraysCache.make_mask may return None only when the mask it replaces is
all-True. The mask is pos >= left_padding (and pos < lengths when set), and
advance(N) lowers left_padding by N every step, so the guard must hold for any
left_padding <= 0, not only zeros: a zeros-only guard kept a real mask on every
decode step of an unpadded BatchGenerator batch and sent the fused GDN decode
kernel to its fallback (2026-10-07). These tests run the real patcher on a copy
of the installed stock mlx_lm cache.py and require:

  1. whenever make_mask returns None, the unpatched formula gives all-True,
     over a sweep of paddings, steps and query lengths,
  2. unpadded batches drop the mask at decode after advance, padded rows keep
     it until their padding is consumed, and right padding (lengths) keeps it,
  3. MLX_OMARCHY_SSM_MASKLESS=0 keeps the mask.
"""

import importlib
import importlib.util
import itertools
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

REPO = Path(__file__).resolve().parent.parent
PATCHER = REPO / "scripts" / "patch-mlx-lm-ssm-maskless.py"


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    """A patched copy of the stock cache.py, importable as ssm_mlx_lm.models.cache."""
    spec = importlib.util.find_spec("mlx_lm.models.cache")
    src = Path(spec.origin).parent
    if "MLX_OMARCHY_SSM_MASKLESS" in (src / "cache.py").read_text():
        pytest.skip(f"installed mlx_lm cache.py is already patched: {src}")
    root = tmp_path_factory.mktemp("ssm")
    models = root / "lib" / "python3" / "site-packages" / "ssm_mlx_lm" / "models"
    models.mkdir(parents=True)
    (models.parent / "__init__.py").write_text("")
    (models / "__init__.py").write_text("")
    for name in ("cache.py", "base.py"):
        shutil.copy(src / name, models / name)
    (models.parent.parent / "mlx_lm").symlink_to(models.parent)
    subprocess.run([sys.executable, str(PATCHER), str(root)], check=True)
    path = str(models.parent.parent)
    sys.path.insert(0, path)
    yield path, importlib.import_module("ssm_mlx_lm.models.cache")
    sys.path.remove(path)


def reference(c, n):
    """The unpatched make_mask formula on the cache's own arrays."""
    pos = mx.arange(n)
    mask = pos >= c.left_padding[:, None]
    if c.lengths is not None:
        mask = mask & (pos < c.lengths[:, None])
    return mask


def test_dropped_masks_are_all_true(site):
    _, cache = site
    for padding, steps, n in itertools.product(([0, 0, 0], [2, 0, 1], [3, 3, 0], [1]), range(5), (1, 3)):
        c = cache.ArraysCache(size=2, left_padding=padding)
        c.advance(steps)
        assert c.left_padding_list == c.left_padding.tolist()
        if c.make_mask(n) is None:
            assert bool(mx.all(reference(c, n)).item()), (padding, steps, n)
        else:
            assert bool(mx.array_equal(c.make_mask(n), reference(c, n)).item())


def test_decode_drops_the_mask_once_padding_is_consumed(site):
    _, cache = site
    merged = cache.ArraysCache.merge([cache.ArraysCache(size=2) for _ in range(4)])
    merged.advance(442)
    assert merged.make_mask(1) is None

    padded = cache.ArraysCache(size=2, left_padding=[2, 0, 1])
    padded.advance(1)
    assert padded.make_mask(1) is not None
    padded.advance(1)
    assert padded.make_mask(1) is None

    right = cache.ArraysCache(size=2, left_padding=[0, 0])
    right.prepare(lengths=[3, 5])
    right.advance(1)
    assert right.make_mask(1) is not None


def test_kill_switch(site):
    path, _ = site
    code = ("from ssm_mlx_lm.models.cache import ArraysCache; c = ArraysCache(size=2, left_padding=[0, 0]); "
            "c.advance(4); assert c.make_mask(1) is not None")
    env = dict(os.environ, MLX_OMARCHY_SSM_MASKLESS="0", PYTHONPATH=path)
    assert subprocess.run([sys.executable, "-c", code], env=env).returncode == 0


def test_rerun_upgrades_a_zeros_only_guard(site, tmp_path):
    path, _ = site
    root = tmp_path / "old"
    models = root / "lib" / "python3" / "site-packages" / "mlx_lm" / "models"
    models.mkdir(parents=True)
    patched = (Path(path) / "ssm_mlx_lm" / "models" / "cache.py").read_text()
    new = "and max(self.left_padding_list) <= 0"
    assert patched.count(new) == 1
    (models / "cache.py").write_text(patched.replace(new, "and not any(self.left_padding_list)"))
    out = subprocess.run([sys.executable, str(PATCHER), str(root)], check=True, capture_output=True, text=True)
    assert "upgraded guard" in out.stdout
    assert (models / "cache.py").read_text() == patched
