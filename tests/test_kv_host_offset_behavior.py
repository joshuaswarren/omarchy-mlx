# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Behavior regression for scripts/patch-mlx-lm-kv-host-offset.py.

With MLX_OMARCHY_KV_HOST_OFFSET=1 the patched BatchKVCache stores offset as a
host-built array so the omarchy RoPE gate needs no host join. A wrong mirror
would rotate keys at the wrong position, so these tests run the real patchers
(ssm-maskless, kv-maskless, kv-host-offset) on a copy of the installed stock
mlx_lm cache.py, drive the same mutation script with the switch off and on in
child processes, and require:

  1. every offset value, dtype and shape is identical with the switch off and
     on, after init, update, prepare/finalize, trim, filter, extend and merge,
  2. with the switch on, offset_list equals offset after each of those steps,
  3. an assignment from outside the class (or a state restore) clears the
     mirror and keeps the assigned array, and
  4. the patcher refuses a cache.py without kv-maskless and stays wired after
     it in apply-mlx-lm-patches.sh.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

REPO = Path(__file__).resolve().parent.parent
PATCHERS = [REPO / "scripts" / name for name in (
    "patch-mlx-lm-ssm-maskless.py", "patch-mlx-lm-kv-maskless.py", "patch-mlx-lm-kv-host-offset.py")]

SCRIPT = r"""
import json
import mlx.core as mx
from kvo_mlx_lm.models.cache import BatchKVCache, KVCache

H, D = 2, 8
log = []


def snap(tag, c):
    o = c.offset
    log.append({"tag": tag, "values": o.tolist(), "dtype": str(o.dtype), "shape": list(o.shape),
                "mirror": c.offset_list})


def fill(c, n):
    B = c.offset.shape[0]
    c.update_and_fetch(mx.zeros((B, H, n, D)), mx.zeros((B, H, n, D)))


c = BatchKVCache([2, 0, 1])
snap("init", c)
fill(c, 5)
snap("prefill", c)
fill(c, 1)
snap("decode", c)
c.trim(2)
snap("trim", c)
c.filter([0, 2])
snap("filter", c)

c = BatchKVCache([0, 0, 0])
c.prepare(left_padding=[1, 0, 2], right_padding=[0, 3, 0])
snap("prepare", c)
fill(c, 6)
c.finalize()
snap("finalize", c)

a, b = BatchKVCache([0, 0]), BatchKVCache([1])
a.extend(b)
snap("extend-empty", a)
a, b = BatchKVCache([0, 0]), BatchKVCache([0])
fill(a, 4)
fill(b, 7)
a.extend(b)
snap("extend", a)
fill(a, 1)
snap("extend-decode", a)

rows = []
for n in (3, 5):
    k = KVCache()
    k.update_and_fetch(mx.zeros((1, H, n, D)), mx.zeros((1, H, n, D)))
    rows.append(k)
m = BatchKVCache.merge(rows)
snap("merge", m)
fill(m, 1)
snap("merge-decode", m)

c = BatchKVCache([0])
fill(c, 4)
outside = mx.array([9])
c.offset = outside
log.append({"tag": "outside", "kept": c.offset is outside, "mirror": c.offset_list})
fill(c, 1)
snap("outside-decode", c)
c = BatchKVCache([0])
fill(c, 4)
c.state = c.state
snap("state", c)
print(json.dumps(log))
"""


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    """A patched copy of the stock cache.py, importable as kvo_mlx_lm.models.cache."""
    spec = importlib.util.find_spec("mlx_lm.models.cache")
    src = Path(spec.origin).parent
    if "MLX_OMARCHY_KV_MASKLESS" in (src / "cache.py").read_text():
        pytest.skip(f"installed mlx_lm cache.py is already patched: {src}")
    root = tmp_path_factory.mktemp("kvo")
    models = root / "lib" / "python3" / "site-packages" / "kvo_mlx_lm" / "models"
    models.mkdir(parents=True)
    (models.parent / "__init__.py").write_text("")
    (models / "__init__.py").write_text("")
    for name in ("cache.py", "base.py"):
        shutil.copy(src / name, models / name)
    (models.parent.parent / "mlx_lm").symlink_to(models.parent)
    for patcher in PATCHERS:
        subprocess.run([sys.executable, str(patcher), str(root)], check=True)
    return root, str(models.parent.parent)


def run(site, switch):
    env = dict(os.environ, MLX_OMARCHY_KV_HOST_OFFSET=switch, PYTHONPATH=site)
    out = subprocess.run([sys.executable, "-c", SCRIPT], env=env, check=True, capture_output=True, text=True)
    return {row["tag"]: row for row in json.loads(out.stdout)}


def test_offsets_identical_and_mirrored(site):
    off, on = run(site[1], "0"), run(site[1], "1")
    assert off.keys() == on.keys()
    for tag, row in on.items():
        if tag == "outside":
            continue
        assert {k: row[k] for k in ("values", "dtype", "shape")} == \
            {k: off[tag][k] for k in ("values", "dtype", "shape")}, tag
    for tag in ("init", "prefill", "decode", "trim", "filter", "prepare", "finalize", "extend-empty", "extend",
                "extend-decode", "merge", "merge-decode"):
        assert on[tag]["mirror"] == on[tag]["values"], tag
    assert on["decode"]["values"] == [4, 6, 5]
    assert on["finalize"]["values"] == [5, 3, 4]


def test_outside_writes_clear_the_mirror(site):
    on = run(site[1], "1")
    assert on["outside"] == {"tag": "outside", "kept": True, "mirror": None}
    assert on["outside-decode"]["mirror"] is None and on["outside-decode"]["values"] == [10]
    assert on["state"]["mirror"] is None and on["state"]["values"] == [4]


def test_refuses_without_kv_maskless_and_stays_wired(site, tmp_path):
    root = tmp_path / "plain"
    models = root / "lib" / "python3" / "site-packages" / "mlx_lm" / "models"
    models.mkdir(parents=True)
    spec = importlib.util.find_spec("mlx_lm.models.cache")
    shutil.copy(spec.origin, models / "cache.py")
    result = subprocess.run([sys.executable, str(PATCHERS[2]), str(root)], capture_output=True, text=True)
    assert result.returncode != 0 and "kv-maskless" in result.stderr
    apply = (REPO / "scripts" / "apply-mlx-lm-patches.sh").read_text()
    assert apply.index("patch-mlx-lm-kv-maskless.py") < apply.index("patch-mlx-lm-kv-host-offset.py")
