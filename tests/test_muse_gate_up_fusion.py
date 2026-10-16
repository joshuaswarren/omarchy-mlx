"""Muse adapter: fuse_gate_up gives the same routed-expert output as separate gate and up calls (CPU, tiny shapes)."""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "serve") not in sys.path:
    sys.path.insert(0, str(REPO / "serve"))

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_lm")

import mlx_omarchy_muse
from mlx_lm.models.switch_layers import SwitchGLU


class _Layer(nn.Module):
    def __init__(self, glu):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.switch_mlp = glu


class _Stack(nn.Module):
    def __init__(self, glu):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = [_Layer(glu)]


def _glu(seed=0):
    mx.random.seed(seed)
    glu = SwitchGLU(128, 128, 8)
    mx.eval(glu.parameters())
    return glu


@pytest.mark.parametrize("bits,group", [(2, 128), (4, 64)])
@pytest.mark.parametrize("tokens", [1, 24])
def test_fused_output_is_identical_to_separate_calls(bits, group, tokens):
    """One token (8 routes) takes the unsorted path; 24 tokens (192 routes, at least 64) take the sorted path."""
    glu = _glu()
    nn.quantize(glu, group_size=group, bits=bits)
    mx.eval(glu.parameters())
    mx.random.seed(3)
    x = mx.random.normal((1, tokens, 128))
    idx = mx.random.randint(0, 8, (1, tokens, 8))
    before = glu(x, idx)
    mx.eval(before)
    assert mlx_omarchy_muse.fuse_gate_up(_Stack(glu)) == 1
    assert "gate_up_proj" in glu and "gate_proj" not in glu and "up_proj" not in glu
    assert glu.gate_up_proj.weight.shape[1] == 256  # 2 x the intermediate size 128
    after = glu(x, idx)
    mx.eval(after)
    assert mx.array_equal(before, after).item()


def test_unquantized_layer_is_left_alone():
    glu = _glu()
    assert mlx_omarchy_muse.fuse_gate_up(_Stack(glu)) == 0
    assert "gate_proj" in glu and "up_proj" in glu


def test_layer_with_different_bits_is_left_alone():
    glu = _glu()
    glu.gate_proj = glu.gate_proj.to_quantized(group_size=64, bits=4)
    glu.up_proj = glu.up_proj.to_quantized(group_size=64, bits=8)
    mx.eval(glu.parameters())
    assert mlx_omarchy_muse.fuse_gate_up(_Stack(glu)) == 0
    assert "gate_proj" in glu and "up_proj" in glu
