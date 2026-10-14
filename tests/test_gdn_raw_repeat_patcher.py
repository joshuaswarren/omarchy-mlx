"""scripts/patch-mlx-lm-gdn-raw-repeat.py: fresh patch, migration of the older default-ON block, idempotence and refusal."""
import py_compile
import subprocess
import sys
from pathlib import Path

PATCHER = Path(__file__).resolve().parents[1] / "scripts" / "patch-mlx-lm-gdn-raw-repeat.py"

FRESH = """from functools import partial

import mlx.core as mx


def gated_delta_update(q, k, v, g, beta, state, mask, use_kernel):
    if not use_kernel or mx.default_device() != mx.gpu:
        return gated_delta_ops(q, k, v, g, beta, state, mask)
    return None
"""

# The block the previous version of the patcher wrote: the Python q/k expansion ON by default.
OLD_BLOCK = """    # mlx-omarchy decode fast-route (GQA repeat): the fused raw decode
    # kernel requires Hk == Hv (omarchy GatedDeltaUpdate::use_fallback);
    # expand q/k to the value-head count so Hk<Hv models take the fused
    # route instead of the composed per-token fallback. Gate:
    # MLX_OMARCHY_GDN_RAW_REPEAT (default ON; set to 0 to opt out).
    if (
        os.environ.get("MLX_OMARCHY_GDN_RAW_REPEAT", "1") != "0"
        and use_kernel
        and q.shape[1] == 1
    ):
        q = mx.repeat(q, v.shape[-2] // q.shape[-2], -2)
"""

OLD_GATE = 'os.environ.get("MLX_OMARCHY_GDN_RAW_REPEAT", "1") != "0"'
NEW_GATE = 'os.environ.get("MLX_OMARCHY_GDN_RAW_REPEAT", "0") == "1"'


def make_venv(tmp_path, text):
    models = tmp_path / "lib" / "python3.14" / "site-packages" / "mlx_lm" / "models"
    models.mkdir(parents=True)
    target = models / "gated_delta.py"
    target.write_text(text)
    return tmp_path, target


def run(venv):
    return subprocess.run([sys.executable, str(PATCHER), str(venv)], capture_output=True, text=True, check=False)


def test_fresh_patch_writes_the_opt_in_default(tmp_path):
    venv, target = make_venv(tmp_path, FRESH)
    first = run(venv)
    assert first.returncode == 0 and "patched:" in first.stdout
    text = target.read_text()
    assert NEW_GATE in text and OLD_GATE not in text
    assert "import os" in text
    py_compile.compile(str(target), doraise=True)
    second = run(venv)
    assert second.returncode == 0 and "already patched" in second.stdout
    assert target.read_text() == text


def test_old_default_on_block_is_migrated(tmp_path):
    old = FRESH.replace("from functools import partial\n", "from functools import partial\nimport os\n").replace(
        "    if not use_kernel or mx.default_device() != mx.gpu:\n", OLD_BLOCK + "    if not use_kernel or mx.default_device() != mx.gpu:\n"
    )
    assert OLD_GATE in old
    venv, target = make_venv(tmp_path, old)
    first = run(venv)
    assert first.returncode == 0 and "migrated" in first.stdout
    text = target.read_text()
    assert NEW_GATE in text and OLD_GATE not in text
    assert "(default OFF; set to 1 to expand here)" in text
    py_compile.compile(str(target), doraise=True)
    second = run(venv)
    assert second.returncode == 0 and "already patched" in second.stdout
    assert target.read_text() == text


def test_refuses_unrecognized_content(tmp_path):
    venv, target = make_venv(tmp_path, "def gated_delta_update():\n    pass\n")
    before = target.read_text()
    result = run(venv)
    assert result.returncode != 0 and "refusing" in (result.stdout + result.stderr)
    assert target.read_text() == before
