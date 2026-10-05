# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Behavior regression for the qwen3 dense rope-norm fold with batched caches.

The v0.7.25..27 fold crashed every B>1 leg with
'[broadcast_shapes] (B,8,L,64) and (B,1,L,128)' (the C++ fallback composed
D-wide trig from the norm weight), and the interim v0.7.27 patcher fence
dodged it by refusing every non-int offset. These tests run the real
patcher against a real mlx_lm qwen3.py and require, for B in {1, 2, 4}
with per-request array offsets (BatchKVCache left padding):

  1. patched logits are bit-equal to the unpatched eager chain,
  2. the fold dispatches by default (kill switches default OFF the fold,
     never ON), and the patcher stays wired into apply-mlx-lm-patches.sh.

Needs the omarchy wheel that ships mx.fast.rope_rms_norm; on hardware
(Apple GPU) and on llvmpipe dev boxes alike, via the usual
MLX_OMARCHY_ALLOW_NON_APPLE=1.
"""

import importlib.util
import os
import subprocess
import types
from pathlib import Path

import mlx.core as mx
import pytest

REPO = Path(__file__).resolve().parent.parent
PATCHER = REPO / "scripts" / "patch-mlx-lm-qwen3-rope-norm.py"
APPLY_SCRIPT = REPO / "scripts" / "apply-mlx-lm-patches.sh"

CONFIG = {
    "model_type": "qwen3",
    "vocab_size": 64,
    "hidden_size": 128,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "intermediate_size": 256,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "max_position_embeddings": 128,
    "tie_word_embeddings": True,
}


def _qwen3_source() -> str:
    spec = importlib.util.find_spec("mlx_lm.models.qwen3")
    assert spec is not None and spec.origin is not None, "mlx_lm qwen3 not installed"
    return Path(spec.origin).read_text()


def _exec_qwen3(source: str, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__name__ = name
    module.__package__ = "mlx_lm.models"
    spec = importlib.util.find_spec("mlx_lm.models.qwen3")
    module.__file__ = spec.origin
    exec(compile(source, name, "exec"), module.__dict__)  # noqa: S102
    return module


def _patched_source(tmp_path: Path) -> str:
    """Run the repo patcher against a copy of the installed qwen3.py."""
    fake_venv = tmp_path / "venv"
    models = fake_venv / "lib" / "python3.11" / "site-packages" / "mlx_lm" / "models"
    models.mkdir(parents=True)
    (models / "qwen3.py").write_text(_qwen3_source())
    subprocess.run(
        ["python3", str(PATCHER), str(fake_venv)],
        check=True,
        capture_output=True,
        text=True,
    )
    out = (models / "qwen3.py").read_text()
    assert "MLX_OMARCHY_ROPE_NORM_FUSE" in out, "patcher did not patch qwen3.py"
    return out


@pytest.fixture(scope="module")
def modules(tmp_path_factory):
    if not hasattr(mx.fast, "rope_rms_norm"):
        pytest.skip("wheel has no mx.fast.rope_rms_norm")
    tmp = tmp_path_factory.mktemp("rope_norm_batch")
    return {
        "unpatched": _exec_qwen3(_qwen3_source(), "mlx_lm.models._q3_unpatched"),
        "patched": _exec_qwen3(_patched_source(tmp), "mlx_lm.models._q3_patched"),
    }


def _logits(mod, B: int, left_padding, T: int, seed: int):
    from mlx_lm.models.cache import BatchKVCache

    mx.random.seed(seed)
    model = mod.Model(mod.ModelArgs.from_dict(CONFIG))
    model.set_dtype(mx.bfloat16)
    mx.random.seed(seed + 17)
    ids = mx.random.randint(0, CONFIG["vocab_size"], (B, T))
    cache = [BatchKVCache(list(left_padding))]
    out = model(ids, cache=cache)
    mx.eval(out)
    return out


@pytest.mark.parametrize("B,left_padding", [(1, [0]), (2, [0, 2]), (4, [0, 1, 3, 5])])
def test_patched_logits_bit_equal_unpatched_with_array_offsets(
    modules, B, left_padding
):
    """Per-request array offsets (BatchKVCache left padding) must give the
    same bits through the fold as through the eager chain, B>1 included."""
    ref = _logits(modules["unpatched"], B, left_padding, 5, seed=B)
    got = _logits(modules["patched"], B, left_padding, 5, seed=B)
    ref32 = mx.astype(ref, mx.float32)
    got32 = mx.astype(got, mx.float32)
    mx.eval(ref32, got32)
    assert ref32.shape == got32.shape
    assert mx.array_equal(ref32, got32), (
        f"B={B}: patched rope-norm logits diverge from the eager chain"
    )


def test_fold_dispatches_by_default_and_respects_kill_switches(modules):
    """Default env: the fold runs (rope_rms_norm is called). Kill switch:
    the fold is skipped and the eager chain serves the same bits."""
    import mlx_lm.models.cache as cache_mod

    calls = {"n": 0}
    real = mx.fast.rope_rms_norm

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    mx.fast.rope_rms_norm = counting
    try:
        os.environ.pop("MLX_OMARCHY_QWEN3_ROPE_NORM_FUSE", None)
        os.environ.pop("MLX_OMARCHY_ROPE_NORM_FUSE", None)
        _logits(modules["patched"], 2, [0, 2], 1, seed=42)
        assert calls["n"] > 0, "fold must dispatch by default (kill switch =0)"

        calls["n"] = 0
        os.environ["MLX_OMARCHY_QWEN3_ROPE_NORM_FUSE"] = "0"
        ref = _logits(modules["unpatched"], 2, [0, 2], 1, seed=42)
        got = _logits(modules["patched"], 2, [0, 2], 1, seed=42)
        assert calls["n"] == 0, "kill switch must disable the fold"
        assert mx.array_equal(
            mx.astype(ref, mx.float32), mx.astype(got, mx.float32)
        ), "kill-switch route must equal the eager chain"
    finally:
        os.environ.pop("MLX_OMARCHY_QWEN3_ROPE_NORM_FUSE", None)
        mx.fast.rope_rms_norm = real
        del cache_mod


def test_patcher_stays_wired_and_default_on(tmp_path):
    text = APPLY_SCRIPT.read_text()
    assert "patch-mlx-lm-qwen3-rope-norm.py" in text, (
        "qwen3 rope-norm patcher must stay invoked by apply-mlx-lm-patches.sh"
    )
    src = _patched_source(tmp_path)
    assert 'os.environ.get("MLX_OMARCHY_QWEN3_ROPE_NORM_FUSE"' in src, (
        "patcher must keep the per-patcher kill env"
    )
    assert (
        'os.environ.get("MLX_OMARCHY_QWEN3_ROPE_NORM_FUSE",\n'
        '                                os.environ.get("MLX_OMARCHY_ROPE_NORM_FUSE", "1"))'
        in src
    ), "kill switches must default the fold ON (fallback default \"1\")"
