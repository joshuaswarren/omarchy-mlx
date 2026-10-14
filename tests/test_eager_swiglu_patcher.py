"""scripts/patch-mlx-lm-qwen3-next-eager-swiglu.py: anchor, idempotence, refusal and the decode-only guard."""
import py_compile
import subprocess
import sys
import textwrap
from pathlib import Path

PATCHER = Path(__file__).resolve().parents[1] / "scripts" / "patch-mlx-lm-qwen3-next-eager-swiglu.py"

STUB = textwrap.dedent(
    '''\
    from __future__ import annotations
    import os

    import mlx.core as mx
    import mlx.nn as nn

    from .activations import swiglu


    class Qwen3NextMLP(nn.Module):
        def __init__(self, dim, hidden_dim):
            super().__init__()
            self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
            self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
            self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

        def __call__(self, x) -> mx.array:
            return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))
    '''
)


def make_venv(tmp_path, text):
    models = tmp_path / "lib" / "python3.14" / "site-packages" / "mlx_lm" / "models"
    models.mkdir(parents=True)
    target = models / "qwen3_next.py"
    target.write_text(text)
    return tmp_path, target


def run(venv):
    return subprocess.run([sys.executable, str(PATCHER), str(venv)], capture_output=True, text=True, check=False)


def test_patches_once_and_keeps_prefill_compiled(tmp_path):
    venv, target = make_venv(tmp_path, STUB)
    first = run(venv)
    assert first.returncode == 0 and "patched:" in first.stdout
    text = target.read_text()
    assert '_EAGER_SWIGLU = os.environ.get("MLX_OMARCHY_EAGER_SWIGLU", "1") != "0"' in text
    # Decode rows only: the eager product is behind the single-row guard.
    assert "if _EAGER_SWIGLU and x.shape[-2] == 1:" in text
    assert "gate * mx.sigmoid(gate) * self.up_proj(x)" in text
    # Prefill keeps the compiled swiglu, unchanged.
    assert text.count("return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))") == 1
    py_compile.compile(str(target), doraise=True)
    before = text
    second = run(venv)
    assert second.returncode == 0 and "already patched" in second.stdout
    assert target.read_text() == before


def test_refuses_unrecognized_content(tmp_path):
    venv, target = make_venv(tmp_path, STUB.replace("swiglu(self.gate_proj(x), self.up_proj(x))", "other(self.gate_proj(x))"))
    before = target.read_text()
    result = run(venv)
    assert result.returncode != 0 and "refusing" in (result.stdout + result.stderr)
    assert target.read_text() == before


def test_refuses_without_os_import(tmp_path):
    venv, target = make_venv(tmp_path, STUB.replace("import os\n", ""))
    before = target.read_text()
    result = run(venv)
    assert result.returncode != 0 and "refusing" in (result.stdout + result.stderr)
    assert target.read_text() == before
