"""scripts/patch-mlx-lm-gdn-raw-repeat.py: fresh patch, migration of the older default-ON block, idempotence and refusal."""
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

PATCHER = Path(__file__).resolve().parents[1] / "scripts" / "patch-mlx-lm-gdn-raw-repeat.py"

FRESH = """from functools import partial

import mlx.core as mx


def gated_delta_update(q, k, v, g, beta, state, mask, use_kernel):
    if not use_kernel or mx.default_device() != mx.gpu:
        return gated_delta_ops(q, k, v, g, beta, state, mask)
    return None
"""

# The three default-ON blocks earlier versions of the patcher wrote (history: ad2809a35; cd4a8c6f9 and 111b61da2).
def old_block(gate, comment):
    return (
        "    # mlx-omarchy decode fast-route (GQA repeat): the fused raw decode\n"
        "    # kernel requires Hk == Hv (omarchy GatedDeltaUpdate::use_fallback);\n"
        "    # expand q/k to the value-head count so Hk<Hv models take the fused\n"
        "    # route instead of the composed per-token fallback. Gate:\n"
        "    # MLX_OMARCHY_GDN_RAW_REPEAT " + comment + "\n"
        "    if (\n"
        "        " + gate + "\n"
        "        and use_kernel\n"
        "        and q.shape[1] == 1\n"
        "    ):\n"
        "        q = mx.repeat(q, v.shape[-2] // q.shape[-2], -2)\n"
    )


OLD_FORMS = {
    "default-on-not-equal": old_block('os.environ.get("MLX_OMARCHY_GDN_RAW_REPEAT", "1") != "0"', "(default ON; set to 0 to opt out)."),
    "default-on-equal-one": old_block('os.environ.get("MLX_OMARCHY_GDN_RAW_REPEAT", "1") == "1"', "(default 0 = exact existing dispatch)."),
}
OLD_GATES = ('os.environ.get("MLX_OMARCHY_GDN_RAW_REPEAT", "1") != "0"', 'os.environ.get("MLX_OMARCHY_GDN_RAW_REPEAT", "1") == "1"')
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
    assert NEW_GATE in text and not any(g in text for g in OLD_GATES)
    assert "import os" in text
    py_compile.compile(str(target), doraise=True)
    second = run(venv)
    assert second.returncode == 0 and "already patched" in second.stdout
    assert target.read_text() == text


@pytest.mark.parametrize("form", sorted(OLD_FORMS))
def test_old_default_on_block_is_migrated(tmp_path, form):
    block = OLD_FORMS[form]
    old = FRESH.replace("from functools import partial\n", "from functools import partial\nimport os\n").replace(
        "    if not use_kernel or mx.default_device() != mx.gpu:\n", block + "    if not use_kernel or mx.default_device() != mx.gpu:\n"
    )
    assert any(g in old for g in OLD_GATES)
    venv, target = make_venv(tmp_path, old)
    first = run(venv)
    assert first.returncode == 0 and "migrated" in first.stdout
    text = target.read_text()
    assert NEW_GATE in text and not any(g in text for g in OLD_GATES)
    assert "(default OFF; set to 1 to expand here)" in text
    assert "default ON" not in text and "exact existing dispatch" not in text
    py_compile.compile(str(target), doraise=True)
    second = run(venv)
    assert second.returncode == 0 and "already patched" in second.stdout
    assert target.read_text() == text


def test_opt_in_blocks_are_left_alone(tmp_path):
    # 159f93f35 and bc51a7bd4 already wrote the opt-in gate.
    for comment in ("(default OFF; set to 1 to opt in).", "(default 0 = exact existing dispatch)."):
        block = old_block(NEW_GATE, comment)
        text = FRESH.replace("from functools import partial\n", "from functools import partial\nimport os\n").replace(
            "    if not use_kernel or mx.default_device() != mx.gpu:\n", block + "    if not use_kernel or mx.default_device() != mx.gpu:\n"
        )
        sub = tmp_path / comment[:10].strip("(").replace(" ", "_")
        sub.mkdir()
        venv, target = make_venv(sub, text)
        result = run(venv)
        assert result.returncode == 0 and "already patched" in result.stdout
        assert target.read_text() == text


def test_refuses_unrecognized_content(tmp_path):
    venv, target = make_venv(tmp_path, "def gated_delta_update():\n    pass\n")
    before = target.read_text()
    result = run(venv)
    assert result.returncode != 0 and "refusing" in (result.stdout + result.stderr)
    assert target.read_text() == before
