# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Behavior regression for the RMSNorm-into-GEMV prologue fold.

With MLX_OMARCHY_NORM_FOLD=1 the decode RMSNorm that feeds the qkv group
and the gate/up group is recomputed per GEMV workgroup instead of being
dispatched. The fold must change no bit: every decode logit of a small
random 4-bit qwen3 (hidden 2048 and 2560, the two widths the fold serves)
equals the unfused run, byte for byte.

Each arm runs in its own process, because the fold flag is read when the
decode tape is planned. Set MLX_OMARCHY_TEST_EXPECT_NORM_FOLD=1 on a part
that serves the fold (G13C, G14C, G13G at 2048) to also require that the
fold engaged (the [norm-fold] trace line appears); without it the test
only proves the equality, which a never-engaging fold also satisfies.

Needs the omarchy wheel and mlx_lm; on hardware or an llvmpipe dev box via
MLX_OMARCHY_ALLOW_NON_APPLE=1.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

RUNNER = textwrap.dedent(
    """
    import hashlib
    import sys

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import qwen3
    from mlx_lm.models.cache import make_prompt_cache

    hidden = int(sys.argv[1])
    args = qwen3.ModelArgs.from_dict({
        "model_type": "qwen3",
        "vocab_size": 512,
        "hidden_size": hidden,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 3072,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 256,
        "tie_word_embeddings": False,
    })
    mx.random.seed(11)
    model = qwen3.Model(args)
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())
    cache = make_prompt_cache(model)
    mx.random.seed(23)
    prompt = mx.random.randint(0, 512, (1, 8))
    logits = model(prompt, cache=cache)
    mx.eval(logits)
    digest = hashlib.sha256()
    token = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
    for _ in range(6):
        logits = model(token, cache=cache)
        mx.eval(logits)
        digest.update(bytes(memoryview(mx.astype(logits, mx.float32))))
        token = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
    print("DIGEST", digest.hexdigest())
    """
)


def _run(tmp_path: Path, hidden: int, fold: str) -> tuple[str, str]:
    script = tmp_path / "norm_fold_runner.py"
    script.write_text(RUNNER)
    env = dict(os.environ)
    env["MLX_OMARCHY_NORM_FOLD"] = fold
    env["MLX_OMARCHY_NORM_FOLD_TRACE"] = "1"
    done = subprocess.run(
        [sys.executable, str(script), str(hidden)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    lines = [ln for ln in done.stdout.splitlines() if ln.startswith("DIGEST ")]
    assert len(lines) == 1, done.stdout + done.stderr
    return lines[0].split()[1], done.stderr


@pytest.mark.parametrize("hidden", [2048, 2560])
def test_fold_changes_no_decode_logit_bit(tmp_path, hidden):
    ref, ref_err = _run(tmp_path, hidden, "0")
    got, got_err = _run(tmp_path, hidden, "1")
    assert "[norm-fold]" not in ref_err, "fold must stay off with the flag at 0"
    assert got == ref, f"hidden {hidden}: folded decode logits differ from unfused"
    if os.environ.get("MLX_OMARCHY_TEST_EXPECT_NORM_FOLD") == "1":
        assert "[norm-fold] adopted" in got_err, (
            f"hidden {hidden}: the fold never engaged on this part"
        )


def test_fold_default_is_off(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "MLX_OMARCHY_NORM_FOLD"}
    env["MLX_OMARCHY_NORM_FOLD_TRACE"] = "1"
    script = tmp_path / "norm_fold_runner.py"
    script.write_text(RUNNER)
    done = subprocess.run(
        [sys.executable, str(script), "2048"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert "[norm-fold]" not in done.stderr
