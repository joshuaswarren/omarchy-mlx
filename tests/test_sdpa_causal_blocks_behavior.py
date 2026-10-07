# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Behavior regression for scripts/patch-mlx-lm-sdpa-causal-blocks.py.

The opt-in lever splits a causal prompt into row blocks that each attend only
to their key prefix. A wrong key bound or offset changes which keys a row
sees, so these tests run the real patcher on a copy of the installed stock
mlx_lm base.py (imported as scb_mlx_lm) and require, on CPU MLX:

  1. blocked output equals the single full-square causal call (fresh prompts
     and chunked-prefill continuations with keys longer than queries, GQA,
     block sizes that do and do not divide the prompt),
  2. the blocked path runs one fast SDPA call per block, and only for causal
     string masks without sinks and longer than the block,
  3. unset/0 keeps the upstream call, and the patcher stays wired.
"""

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

REPO = Path(__file__).resolve().parent.parent
PATCHER = REPO / "scripts" / "patch-mlx-lm-sdpa-causal-blocks.py"
H, KV, D = 8, 2, 32


@pytest.fixture(scope="module")
def pkg(tmp_path_factory):
    spec = importlib.util.find_spec("mlx_lm.models.base")
    src = Path(spec.origin).parent
    if "MLX_OMARCHY_SDPA_CAUSAL_BLOCK" in (src / "base.py").read_text():
        pytest.skip(f"installed mlx_lm base.py is already patched: {src}")
    root = tmp_path_factory.mktemp("scb")
    models = root / "lib" / "python3" / "site-packages" / "scb_mlx_lm" / "models"
    models.mkdir(parents=True)
    (models.parent / "__init__.py").write_text("")
    (models / "__init__.py").write_text("")
    shutil.copy(src / "base.py", models / "base.py")
    (models.parent.parent / "mlx_lm").symlink_to(models.parent)
    subprocess.run([sys.executable, str(PATCHER), str(root)], check=True)
    site = str(models.parent.parent)
    sys.path.insert(0, site)
    yield site, importlib.import_module("scb_mlx_lm.models.base")
    sys.path.remove(site)


def qkv(L, S, seed):
    mx.random.seed(seed)
    q = mx.random.normal((1, H, L, D))
    k = mx.random.normal((1, KV, S, D))
    v = mx.random.normal((1, KV, S, D))
    return q, k, v


def full(q, k, v):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5, mask="causal")


@pytest.mark.parametrize("L,S,block", [(130, 130, 64), (256, 256, 64), (200, 237, 64), (300, 300, 97)])
def test_blocks_equal_the_full_causal_call(pkg, monkeypatch, L, S, block):
    _, base = pkg
    monkeypatch.setattr(base, "_SDPA_CAUSAL_BLOCK", block)
    q, k, v = qkv(L, S, L + S + block)
    got = base.scaled_dot_product_attention(q, k, v, None, scale=D**-0.5, mask="causal")
    assert got.shape == (1, H, L, D)
    assert bool(mx.allclose(got, full(q, k, v), atol=1e-5, rtol=1e-5).item())


def test_only_long_causal_calls_take_the_blocks(pkg, monkeypatch):
    _, base = pkg
    monkeypatch.setattr(base, "_SDPA_CAUSAL_BLOCK", 64)
    calls = []
    real = mx.fast.scaled_dot_product_attention

    def count(*args, **kwargs):
        calls.append(kwargs.get("mask"))
        return real(*args, **kwargs)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", count)
    q, k, v = qkv(200, 200, 1)
    base.scaled_dot_product_attention(q, k, v, None, scale=D**-0.5, mask="causal")
    assert calls == ["causal"] * 4
    for L, mask, sinks in ((64, "causal", None), (200, None, None),
                           (200, mx.ones((200, 200), dtype=mx.bool_), None),
                           (200, "causal", mx.zeros((H,)))):
        calls.clear()
        q, k, v = qkv(L, L, 2)
        base.scaled_dot_product_attention(q, k, v, None, scale=D**-0.5, mask=mask, sinks=sinks)
        assert len(calls) == 1


def test_kill_switch_and_wiring(pkg):
    site, _ = pkg
    code = ("import importlib; b = importlib.import_module('scb_mlx_lm.models.base'); "
            "assert b._SDPA_CAUSAL_BLOCK == 0")
    env = {k: v for k, v in os.environ.items() if k != "MLX_OMARCHY_SDPA_CAUSAL_BLOCK"}
    assert subprocess.run([sys.executable, "-c", code], env=dict(env, PYTHONPATH=site)).returncode == 0
    apply = (REPO / "scripts" / "apply-mlx-lm-patches.sh").read_text()
    assert "patch-mlx-lm-sdpa-causal-blocks.py" in apply
