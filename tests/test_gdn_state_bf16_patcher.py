# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""scripts/patch-mlx-lm-gdn-state-bf16.py: both raw sites, idempotence, refusal.

The backend's bf16 recurrent state default (#377) only engages when the
caller passes no initial state; the mlx-lm wrapper always hands it an
fp32 zeros state or the fp32 prefill carry, so without this patcher the
state stays fp32 end to end. The test pins the patcher contract on both
vendored raw sites (0.31.3 and 0.32), the os import guard, idempotence,
and refusal on a tree without the raw route.
"""
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

PATCHER = Path(__file__).resolve().parents[1] / "scripts" / "patch-mlx-lm-gdn-state-bf16.py"

# The exact raw site the 0.31.3 series writes (patches/mlx-lm-gated-delta-raw.patch
# plus the conv/norm patches around it; the routing tail is the anchor's home).
RAW_0313 = '''import glob
import os

import mlx.core as mx


def gated_delta_update(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=True):
    if not use_kernel or mx.default_device() != mx.gpu:
        return gated_delta_ops(q, k, v, g, beta, state, mask)
    if hasattr(mx.fast, "gated_delta_update"):
        if q.shape[1] == 1 and hasattr(mx.fast, "gated_delta_update_raw"):
            return mx.fast.gated_delta_update_raw(
                q, k, v, a, b, A_log, dt_bias, state, mask
            )
        out, st = mx.fast.gated_delta_update(q, k, v, g, beta, state, mask)
        return out, st
    return gated_delta_kernel(q, k, v, g, beta, state, mask)
'''

# The 0.32 raw site: same return shape, extra guard conditions in the branch above it.
RAW_032 = RAW_0313.replace(
    '        if q.shape[1] == 1 and hasattr(mx.fast, "gated_delta_update_raw"):',
    '        if (\n'
    '            q.shape[1] == 1\n'
    '            and lower_bound is None\n'
    '            and not allow_neg_eigval\n'
    '            and hasattr(mx.fast, "gated_delta_update_raw")\n'
    '        ):',
)

# A stock tree without the raw route must be refused, not silently patched.
NO_RAW = RAW_0313.replace(
    '''        if q.shape[1] == 1 and hasattr(mx.fast, "gated_delta_update_raw"):
            return mx.fast.gated_delta_update_raw(
                q, k, v, a, b, A_log, dt_bias, state, mask
            )
''',
    "",
)

CAST_MARK = 'os.environ.get("MLX_OMARCHY_GDN_STATE_F32", "0") != "1"'
CAST_BLOCK = "state = state.astype(mx.bfloat16)"


def make_venv(tmp_path, text):
    models = tmp_path / "lib" / "python3.14" / "site-packages" / "mlx_lm" / "models"
    models.mkdir(parents=True)
    target = models / "gated_delta.py"
    target.write_text(text)
    return tmp_path, target


def run(venv):
    return subprocess.run([sys.executable, str(PATCHER), str(venv)],
                          capture_output=True, text=True, check=False)


@pytest.mark.parametrize("raw", [RAW_0313, RAW_032], ids=["0313", "032"])
def test_fresh_patch_adds_the_cast_with_an_opt_out(tmp_path, raw):
    venv, target = make_venv(tmp_path, raw)
    first = run(venv)
    assert first.returncode == 0 and "patched:" in first.stdout, first.stderr
    text = target.read_text()
    assert CAST_MARK in text and CAST_BLOCK in text
    # The cast sits inside the raw branch only: the composed and kernel tails are untouched.
    assert text.count(CAST_BLOCK) == 1
    assert "state.astype" not in raw
    py_compile.compile(str(target), doraise=True)
    second = run(venv)
    assert second.returncode == 0 and "already patched" in second.stdout
    assert target.read_text() == text


def test_os_import_is_added_only_when_missing(tmp_path):
    # A tree without any os import still gets patched: the patcher adds it.
    raw = RAW_032.replace("import glob\nimport os\n", "import glob\n")
    assert "import os" not in raw
    venv, target = make_venv(tmp_path, raw)
    assert run(venv).returncode == 0
    assert target.read_text().count("import os") == 1


def test_refuses_a_tree_without_the_raw_route(tmp_path):
    venv, target = make_venv(tmp_path, NO_RAW)
    result = run(venv)
    assert result.returncode != 0
    assert "raw site" in result.stderr
    assert target.read_text() == NO_RAW


def test_already_patched_tree_is_left_alone(tmp_path):
    venv, target = make_venv(tmp_path, RAW_0313)
    assert run(venv).returncode == 0
    text = target.read_text()
    result = run(venv)
    assert result.returncode == 0 and "already patched" in result.stdout
    assert target.read_text() == text
