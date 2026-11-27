"""Tests for scripts/patch-mlx-lm-adapter-switch.py and in-place LoRA switching.

Covers both the patcher contract (0.31.3 & 0.32.x server.py anchors,
idempotence, refusal on unrecognized content, apply-mlx-lm-patches.sh wiring)
and the runtime invariants of _OmarchyAdapterRegistry + ModelProvider:
- base weights load exactly once across base <-> A <-> B <-> base;
- resident adapters wrap shared base layer objects (zero weight copies);
- switching back to no adapter restores exact unwrapped base objects and
  bit-for-bit identical base output;
- A, base, A, base yields two distinct, stable outputs (with dropout eval'd);
- partial layer subsets (num_layers) restore non-overlapping slots to base;
- invalid / mismatched / base-overwriting adapters fail atomically without
  mutating the active model;
- LRU eviction bounds resident adapter count without reloading base;
- full fine-tune and MLX_OMARCHY_INPLACE_ADAPTER_SWITCH=0 fall back cleanly.
"""

from __future__ import annotations

import argparse
import json
import py_compile
import re
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "patch-mlx-lm-adapter-switch.py"
APPLY_SCRIPT = REPO / "scripts" / "apply-mlx-lm-patches.sh"

SERVER_STUB_031 = textwrap.dedent(
    '''\
    import argparse
    import json
    import logging
    import time
    from pathlib import Path
    from typing import Any, Dict, List, Optional, Tuple

    import mlx.core as mx

    from .models.cache import make_prompt_cache
    from .utils import load, sharded_load


    class ModelProvider:
        def __init__(self, cli_args: argparse.Namespace):
            """Load models on demand and persist them across the whole process."""
            self.cli_args = cli_args
            self.model_key = None
            self.model = None
            self.tokenizer = None
            self.draft_model = None
            self.is_batchable = False

            group = mx.distributed.init()
            self.pipeline_group = group if group.size() > 1 and cli_args.pipeline else None
            self.tensor_group = (
                group if group.size() > 1 and not cli_args.pipeline else None
            )
            self.is_distributed = group.size() > 1

            self._model_map = {}
            self._adapter_map = {}
            self._draft_model_map = {}
            self._model_map["default_model"] = self.cli_args.model
            self._adapter_map["default_model"] = self.cli_args.adapter_path
            self._draft_model_map["default_model"] = self.cli_args.draft_model

            self._tokenizer_config = {
                "trust_remote_code": True if cli_args.trust_remote_code else None
            }
            if cli_args.chat_template:
                self._tokenizer_config["chat_template"] = cli_args.chat_template

        def _load(self, model_path, adapter_path=None, draft_model_path=None):
            if self.is_distributed and (
                adapter_path is not None or draft_model_path is not None
            ):
                raise ValueError(
                    "Loading with adapters or draft models not supported in distributed mode"
                )

            # Remove the old model if it exists.
            self.model_key = None
            self.model = None
            self.tokenizer = None
            self.draft_model = None

            if self.is_distributed:
                model, tokenizer = sharded_load(
                    model_path,
                    pipeline_group=self.pipeline_group,
                    tensor_group=self.tensor_group,
                    tokenizer_config=self._tokenizer_config,
                )
            else:
                model, tokenizer = load(
                    model_path,
                    adapter_path=adapter_path,
                    tokenizer_config=self._tokenizer_config,
                )

            draft_model = None
            if draft_model_path is not None:
                draft_model, _ = load(draft_model_path)

            is_batchable = draft_model is None
            is_batchable = is_batchable and all(
                hasattr(c, "merge") for c in make_prompt_cache(model)
            )

            self.model_key = (model_path, adapter_path, draft_model_path)
            self.model = model
            self.tokenizer = tokenizer
            self.draft_model = draft_model
            self.is_batchable = is_batchable

        def load_default(self):
            if self._model_map["default_model"] is not None:
                self.load("default_model", None, "default_model")

        def load(self, model_path, adapter_path=None, draft_model_path=None):
            model_path = self._model_map.get(model_path, model_path)
            adapter_path = self._adapter_map.get(model_path, adapter_path)
            draft_model_path = self._draft_model_map.get(draft_model_path, draft_model_path)

            model_key = (model_path, adapter_path, draft_model_path)
            if self.model_key != model_key:
                self._load(*model_key)

            return self.model, self.tokenizer
    '''
)

SERVER_STUB_032 = textwrap.dedent(
    '''\
    import argparse
    import gc
    import json
    import logging
    import time
    from pathlib import Path
    from typing import Any, Dict, List, Optional, Tuple

    import mlx.core as mx

    from .models.cache import make_prompt_cache
    from .utils import load, sharded_load


    class ModelProvider:
        def __init__(self, cli_args: argparse.Namespace):
            """Load models on demand and persist them across the whole process."""
            self.cli_args = cli_args
            self.model_key = None
            self.model = None
            self.tokenizer = None
            self.draft_model = None
            self.is_batchable = False

            group = mx.distributed.init()
            self.pipeline_group = group if group.size() > 1 and cli_args.pipeline else None
            self.tensor_group = (
                group if group.size() > 1 and not cli_args.pipeline else None
            )
            self.is_distributed = group.size() > 1

            self._model_map = {}
            self._adapter_map = {}
            self._draft_model_map = {}
            self._model_map["default_model"] = self.cli_args.model
            self._adapter_map["default_model"] = self.cli_args.adapter_path
            self._draft_model_map["default_model"] = self.cli_args.draft_model

            self._tokenizer_config = {"trust_remote_code": cli_args.trust_remote_code}
            if cli_args.chat_template:
                self._tokenizer_config["chat_template"] = cli_args.chat_template

        def reset(self) -> None:
            self.model_key = None
            self.model = None
            self.tokenizer = None
            self.draft_model = None
            self.is_batchable = False

        def _load(self, model_path, adapter_path=None, draft_model_path=None):
            if self.is_distributed and (
                adapter_path is not None or draft_model_path is not None
            ):
                raise ValueError(
                    "Loading with adapters or draft models not supported in distributed mode"
                )

            self.reset()
            gc.collect()
            mx.clear_cache()

            if self.is_distributed:
                model, tokenizer = sharded_load(
                    model_path,
                    pipeline_group=self.pipeline_group,
                    tensor_group=self.tensor_group,
                    tokenizer_config=self._tokenizer_config,
                    trust_remote_code=self.cli_args.trust_remote_code,
                )
            else:
                model, tokenizer = load(
                    model_path,
                    adapter_path=adapter_path,
                    tokenizer_config=self._tokenizer_config,
                    trust_remote_code=self.cli_args.trust_remote_code,
                )

            draft_model = None
            if draft_model_path is not None:
                draft_model, _ = load(draft_model_path)

            is_batchable = draft_model is None
            is_batchable = is_batchable and all(
                hasattr(c, "merge") for c in make_prompt_cache(model)
            )

            self.model_key = (model_path, adapter_path, draft_model_path)
            self.model = model
            self.tokenizer = tokenizer
            self.draft_model = draft_model
            self.is_batchable = is_batchable

        def load_default(self):
            if self._model_map["default_model"] is not None:
                self.load("default_model", None, "default_model")

        def load(self, model_path, adapter_path=None, draft_model_path=None):
            adapter_path = self._adapter_map.get(model_path, adapter_path)
            model_path = self._model_map.get(model_path, model_path)
            draft_model_path = self._draft_model_map.get(draft_model_path, draft_model_path)

            model_key = (model_path, adapter_path, draft_model_path)
            if self.model_key != model_key:
                self._load(*model_key)

            return self.model, self.tokenizer
    '''
)


def _make_venv(tmp_path: Path, text: str) -> tuple[Path, Path]:
    pkg = tmp_path / "lib" / "python3.14" / "site-packages" / "mlx_lm"
    pkg.mkdir(parents=True)
    target = pkg / "server.py"
    target.write_text(text, encoding="utf-8")
    return tmp_path, target


def _run_patcher(venv: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PATCHER), str(venv)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "series,stub",
    [("0.31.3", SERVER_STUB_031), ("0.32.0", SERVER_STUB_032)],
    ids=["0.31.3", "0.32.0"],
)
def test_patcher_applies_once_and_is_idempotent(tmp_path, series, stub):
    venv, target = _make_venv(tmp_path, stub)
    first = _run_patcher(venv)
    assert first.returncode == 0, first.stdout + first.stderr
    assert "patched:" in first.stdout
    py_compile.compile(str(target), doraise=True)
    after_first = target.read_text(encoding="utf-8")
    assert "class _OmarchyAdapterRegistry:" in after_first
    assert "_INPLACE_ADAPTER_SWITCH" in after_first
    if series == "0.32.0":
        assert "def reset(self) -> None:\n        self.model_key = None" in after_first

    second = _run_patcher(venv)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "already patched:" in second.stdout
    assert target.read_text(encoding="utf-8") == after_first


def test_patcher_refuses_unrecognized_content(tmp_path):
    bad = SERVER_STUB_031.replace("self._load(*model_key)", "self._reload(*model_key)")
    venv, target = _make_venv(tmp_path, bad)
    res = _run_patcher(venv)
    assert res.returncode != 0
    assert "refusing" in (res.stdout + res.stderr)
    assert target.read_text(encoding="utf-8") == bad


def test_wired_into_apply_script_and_install_fetch_pattern():
    apply_text = APPLY_SCRIPT.read_text(encoding="utf-8")
    assert 'scripts/patch-mlx-lm-adapter-switch.py' in apply_text
    # install.sh and test_documented_patch_paths.py discover helpers with this exact regex.
    helpers = set(re.findall(r"scripts/([a-z0-9_-]+\.py)", apply_text))
    assert "patch-mlx-lm-adapter-switch.py" in helpers


# ---------------------------------------------------------------------------
# Behavioral tests for the patched ModelProvider + _OmarchyAdapterRegistry
# (runs everywhere without requiring MLX hardware/wheels).


class _FakeTensor:
    """Minimal 2D float matrix supporting @, +, *, shape, and exact value equality."""

    def __init__(self, rows: list[list[float]]):
        self.rows = [[float(v) for v in r] for r in rows]
        self.shape = (len(self.rows), len(self.rows[0]) if self.rows else 0)

    def __matmul__(self, other: "_FakeTensor") -> "_FakeTensor":
        assert self.shape[1] == other.shape[0], (self.shape, other.shape)
        out = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                s = 0.0
                for k in range(self.shape[1]):
                    s += self.rows[i][k] * other.rows[k][j]
                row.append(s)
            out.append(row)
        return _FakeTensor(out)

    def __add__(self, other: "_FakeTensor") -> "_FakeTensor":
        assert self.shape == other.shape
        return _FakeTensor(
            [
                [self.rows[i][j] + other.rows[i][j] for j in range(self.shape[1])]
                for i in range(self.shape[0])
            ]
        )

    def __rmul__(self, scalar: float) -> "_FakeTensor":
        return _FakeTensor([[scalar * v for v in r] for r in self.rows])

    @property
    def T(self) -> "_FakeTensor":
        return _FakeTensor(
            [[self.rows[i][j] for i in range(self.shape[0])] for j in range(self.shape[1])]
        )

    def tolist(self) -> list[list[float]]:
        return [list(r) for r in self.rows]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeTensor) and self.rows == other.rows


class _FakeModule:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False
        for _, child in self.named_modules():
            if child is not self and isinstance(child, _FakeModule):
                child.training = False
        return self

    def named_modules(self, prefix: str = ""):
        out = [(prefix, self)]
        for k, v in self.__dict__.items():
            if k.startswith("_") or k == "training":
                continue
            sub = f"{prefix}.{k}" if prefix else k
            if isinstance(v, _FakeModule):
                out.extend(v.named_modules(sub))
            elif isinstance(v, list):
                for idx, elem in enumerate(v):
                    if isinstance(elem, _FakeModule):
                        out.extend(elem.named_modules(f"{sub}.{idx}"))
        return out

    def apply_to_modules(self, fn):
        for p, m in self.named_modules():
            if p:
                fn(p, m)

    def load_weights(self, weights, strict=True):
        for k, v in weights:
            setattr(self, k, v)


class _FakeLinear(_FakeModule):
    def __init__(self, weight: _FakeTensor):
        super().__init__()
        self.weight = weight

    def __call__(self, x: _FakeTensor) -> _FakeTensor:
        return x @ self.weight.T


class _FakeLoRALinear(_FakeModule):
    @staticmethod
    def from_base(linear: _FakeLinear, r: int = 2, dropout: float = 0.0, scale: float = 1.0):
        out_dim, in_dim = linear.weight.shape
        mod = _FakeLoRALinear(in_dim, out_dim, r=r, dropout=dropout, scale=scale)
        mod.linear = linear
        return mod

    def __init__(self, in_dim: int, out_dim: int, r: int = 2, dropout: float = 0.0, scale: float = 1.0):
        super().__init__()
        self.linear = None
        self.dropout_p = dropout
        self.scale = scale
        self.lora_a = _FakeTensor([[0.0] * r for _ in range(in_dim)])
        self.lora_b = _FakeTensor([[0.0] * out_dim for _ in range(r)])

    def __call__(self, x: _FakeTensor) -> _FakeTensor:
        if self.training and self.dropout_p > 0.0:
            raise AssertionError("LoRA module executed in training mode with dropout > 0")
        y = self.linear(x)
        z = (x @ self.lora_a) @ self.lora_b
        return y + (self.scale * z)


class _FakeSwitchLinear(_FakeModule):
    pass


class _FakeEmbedding(_FakeModule):
    pass


class _FakeBlock(_FakeModule):
    def __init__(self, idx: int):
        super().__init__()
        # Deterministic distinct 2x2 weights per layer
        self.self_attn = _FakeModule()
        self.self_attn.q_proj = _FakeLinear(_FakeTensor([[1.0 + 0.1 * idx, 0.2], [0.0, 1.0]]))
        self.self_attn.v_proj = _FakeLinear(_FakeTensor([[1.0, 0.0], [0.3, 1.0 + 0.1 * idx]]))

    def __call__(self, x: _FakeTensor) -> _FakeTensor:
        return self.self_attn.v_proj(self.self_attn.q_proj(x))


class _FakeTransformer(_FakeModule):
    def __init__(self, n_layers: int = 4, tag: str = "base"):
        super().__init__()
        self.tag = tag
        self.model = _FakeModule()
        self.model.layers = [_FakeBlock(i) for i in range(n_layers)]
        self.eval()

    @property
    def layers(self):
        return self.model.layers

    def __call__(self, x: _FakeTensor) -> _FakeTensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h


def _write_adapter(
    adapter_dir: Path,
    *,
    weights: dict[str, list[list[float]]],
    num_layers: int = -1,
    rank: int = 2,
    scale: float = 2.0,
    dropout: float = 0.0,
    keys: list[str] | None = None,
    fine_tune_type: str = "lora",
) -> Path:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    lora_params: dict[str, object] = {"rank": rank, "scale": scale, "dropout": dropout}
    if keys is not None:
        lora_params["keys"] = keys
    cfg = {
        "fine_tune_type": fine_tune_type,
        "num_layers": num_layers,
        "lora_parameters": lora_params,
    }
    (adapter_dir / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (adapter_dir / "adapters.safetensors").write_text(json.dumps(weights), encoding="utf-8")
    return adapter_dir


def _load_patched_server_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    stub: str = SERVER_STUB_031,
    env_overrides: dict[str, str] | None = None,
):
    """Patch a server.py stub and execute it against lightweight fake mlx/mlx_lm modules."""
    if env_overrides:
        for k, v in env_overrides.items():
            monkeypatch.setenv(k, v)

    venv, server_file = _make_venv(tmp_path / "venv", stub)
    res = _run_patcher(venv)
    assert res.returncode == 0, res.stdout + res.stderr

    eval_calls: list[int] = []
    load_calls: list[tuple[str, str | None]] = []

    fake_mx = types.ModuleType("mlx")
    fake_mx_core = types.ModuleType("mlx.core")

    class _FakeGroup:
        def size(self):
            return 1

    fake_mx_core.distributed = types.SimpleNamespace(init=lambda: _FakeGroup())
    fake_mx_core.clear_cache = lambda: None

    def _fake_mx_load(path: str):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return {k: _FakeTensor(v) for k, v in raw.items()}

    def _fake_mx_eval(*arrays):
        eval_calls.append(len(arrays))

    fake_mx_core.load = _fake_mx_load
    fake_mx_core.eval = _fake_mx_eval

    fake_mx_nn = types.ModuleType("mlx.nn")
    fake_mx_nn.Module = _FakeModule
    fake_mx_nn.Linear = _FakeLinear
    fake_mx_nn.QuantizedLinear = _FakeLinear
    fake_mx_nn.Embedding = _FakeEmbedding
    fake_mx_nn.QuantizedEmbedding = _FakeEmbedding

    fake_pkg = types.ModuleType("mlx_lm_testpkg")
    fake_pkg.__path__ = []
    fake_models = types.ModuleType("mlx_lm_testpkg.models")
    fake_models.__path__ = []
    fake_cache = types.ModuleType("mlx_lm_testpkg.models.cache")
    fake_cache.make_prompt_cache = lambda model: [types.SimpleNamespace(merge=lambda *a: None)]
    fake_switch = types.ModuleType("mlx_lm_testpkg.models.switch_layers")
    fake_switch.SwitchLinear = _FakeSwitchLinear
    fake_switch.QuantizedSwitchLinear = _FakeSwitchLinear
    fake_tuner = types.ModuleType("mlx_lm_testpkg.tuner")
    fake_tuner.__path__ = []
    fake_lora = types.ModuleType("mlx_lm_testpkg.tuner.lora")
    fake_lora.LoRALinear = _FakeLoRALinear
    fake_lora.LoRASwitchLinear = _FakeLoRALinear
    fake_lora.LoRAEmbedding = _FakeLoRALinear
    fake_dora = types.ModuleType("mlx_lm_testpkg.tuner.dora")
    fake_dora.DoRALinear = _FakeLoRALinear
    fake_dora.DoRAEmbedding = _FakeLoRALinear

    fake_utils = types.ModuleType("mlx_lm_testpkg.utils")

    def _fake_load(model_path, adapter_path=None, **_kwargs):
        load_calls.append((model_path, adapter_path))
        return _FakeTransformer(n_layers=4, tag=f"{model_path}:{adapter_path}"), object()

    fake_utils.load = _fake_load
    fake_utils.sharded_load = _fake_load

    for name, mod in {
        "mlx": fake_mx,
        "mlx.core": fake_mx_core,
        "mlx.nn": fake_mx_nn,
        "mlx_lm_testpkg": fake_pkg,
        "mlx_lm_testpkg.models": fake_models,
        "mlx_lm_testpkg.models.cache": fake_cache,
        "mlx_lm_testpkg.models.switch_layers": fake_switch,
        "mlx_lm_testpkg.tuner": fake_tuner,
        "mlx_lm_testpkg.tuner.lora": fake_lora,
        "mlx_lm_testpkg.tuner.dora": fake_dora,
        "mlx_lm_testpkg.utils": fake_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    server_mod = types.ModuleType("mlx_lm_testpkg.server")
    server_mod.__package__ = "mlx_lm_testpkg"
    exec(compile(server_file.read_text(encoding="utf-8"), str(server_file), "exec"), server_mod.__dict__)
    return server_mod, load_calls, eval_calls


def _cli_args(model="base-model", adapter_path=None, draft_model=None):
    return argparse.Namespace(
        model=model,
        adapter_path=adapter_path,
        draft_model=draft_model,
        pipeline=False,
        trust_remote_code=False,
        chat_template=None,
        use_default_chat_template=False,
    )


def _adapter_a_weights(scale_val: float = 0.25) -> dict[str, list[list[float]]]:
    w = {}
    for i in range(4):
        w[f"model.layers.{i}.self_attn.q_proj.lora_a"] = [[scale_val, 0.0], [0.0, scale_val]]
        w[f"model.layers.{i}.self_attn.q_proj.lora_b"] = [[1.0, 0.5], [-0.25, 1.0]]
    return w


def _adapter_b_weights(scale_val: float = -0.4) -> dict[str, list[list[float]]]:
    # Only last 2 layers (layers 2 and 3)
    w = {}
    for i in (2, 3):
        w[f"model.layers.{i}.self_attn.v_proj.lora_a"] = [[scale_val, 0.1], [0.0, scale_val]]
        w[f"model.layers.{i}.self_attn.v_proj.lora_b"] = [[0.5, -1.0], [0.75, 0.25]]
    return w


@pytest.mark.parametrize("stub", [SERVER_STUB_031, SERVER_STUB_032], ids=["0.31.3", "0.32.0"])
def test_inplace_switching_single_base_load_bitexact_base_and_stable_alternation(
    tmp_path, monkeypatch, stub
):
    server_mod, load_calls, eval_calls = _load_patched_server_module(tmp_path, monkeypatch, stub=stub)
    adapter_a = _write_adapter(
        tmp_path / "adapter_a",
        weights=_adapter_a_weights(0.25),
        num_layers=-1,
        keys=["self_attn.q_proj"],
        dropout=0.2,  # must be eval'd so inference never triggers training dropout
    )
    adapter_b = _write_adapter(
        tmp_path / "adapter_b",
        weights=_adapter_b_weights(-0.4),
        num_layers=2,
        keys=["self_attn.v_proj"],
    )

    provider = server_mod.ModelProvider(_cli_args())
    x = _FakeTensor([[1.0, -2.0]])

    # 1. Initial unadapted base
    model, _ = provider.load("default_model", None, "default_model")
    assert load_calls == [("base-model", None)]
    assert provider._base_load_count == 1
    orig_q = [layer.self_attn.q_proj for layer in model.layers]
    orig_v = [layer.self_attn.v_proj for layer in model.layers]
    out_base_0 = model(x)
    key_base = provider.model_key

    # 2. A, base, A, base sequence + B
    model_a1, _ = provider.load("default_model", str(adapter_a), "default_model")
    assert model_a1 is model
    assert len(load_calls) == 1  # NO base reload
    assert provider._base_load_count == 1
    assert len(eval_calls) == 1  # adapter A tensors evaluated once on cold load
    # Shared base layers (zero copy)
    for i, layer in enumerate(model.layers):
        assert isinstance(layer.self_attn.q_proj, _FakeLoRALinear)
        assert layer.self_attn.q_proj.linear is orig_q[i]
        assert layer.self_attn.v_proj is orig_v[i]
    out_a1 = model(x)
    key_a = provider.model_key
    assert key_a != key_base
    assert out_a1 != out_base_0

    # Switch back to base
    model_b1, _ = provider.load("default_model", None, "default_model")
    assert model_b1 is model
    assert len(load_calls) == 1
    assert provider.model_key == key_base
    for i, layer in enumerate(model.layers):
        assert layer.self_attn.q_proj is orig_q[i]
        assert layer.self_attn.v_proj is orig_v[i]
    out_base_1 = model(x)
    assert out_base_1 == out_base_0  # bit-for-bit identical

    # Switch back to A (warm — trailing slash must hit the same canonical resident key!)
    model_a2, _ = provider.load("default_model", str(adapter_a) + "/", "default_model")
    assert model_a2 is model
    assert len(load_calls) == 1
    assert len(eval_calls) == 1  # warm hit, no re-load / re-eval
    assert provider.model_key == key_a
    out_a2 = model(x)
    assert out_a2 == out_a1

    # Switch directly A -> B (different layer subset: B only adapts layers 2..3 v_proj)
    provider.load("default_model", str(adapter_b), "default_model")
    assert len(load_calls) == 1
    assert len(eval_calls) == 2
    for i, layer in enumerate(model.layers):
        # All q_proj slots from A must be restored to original base!
        assert layer.self_attn.q_proj is orig_q[i]
        if i in (2, 3):
            assert isinstance(layer.self_attn.v_proj, _FakeLoRALinear)
            assert layer.self_attn.v_proj.linear is orig_v[i]
        else:
            assert layer.self_attn.v_proj is orig_v[i]
    out_b = model(x)
    assert out_b != out_base_0
    assert out_b != out_a1

    # Final switch B -> base (via explicit empty string "")
    provider.load("default_model", "", "default_model")
    assert len(load_calls) == 1
    assert provider.model_key == key_base
    for i, layer in enumerate(model.layers):
        assert layer.self_attn.q_proj is orig_q[i]
        assert layer.self_attn.v_proj is orig_v[i]
    out_base_2 = model(x)
    assert out_base_2 == out_base_0


def test_atomic_rollback_on_invalid_or_mismatched_adapter(tmp_path, monkeypatch):
    server_mod, load_calls, _ = _load_patched_server_module(tmp_path, monkeypatch)
    adapter_a = _write_adapter(
        tmp_path / "adapter_a",
        weights=_adapter_a_weights(0.3),
        keys=["self_attn.q_proj"],
    )
    bad_shape = _write_adapter(
        tmp_path / "bad_shape",
        weights={"model.layers.0.self_attn.q_proj.lora_a": [[1.0, 2.0, 3.0]]},  # wrong shape
        keys=["self_attn.q_proj"],
    )
    bad_base_overwrite = _write_adapter(
        tmp_path / "bad_overwrite",
        weights={"model.layers.0.self_attn.q_proj.linear.weight": [[9.0, 9.0], [9.0, 9.0]]},
        keys=["self_attn.q_proj"],
    )
    bad_unmatched = _write_adapter(
        tmp_path / "bad_unmatched",
        weights={"nonexistent.layer.lora_a": [[1.0, 0.0], [0.0, 1.0]]},
        keys=["self_attn.q_proj"],
    )

    provider = server_mod.ModelProvider(_cli_args())
    x = _FakeTensor([[1.0, 1.0]])
    model, _ = provider.load("default_model", str(adapter_a), "default_model")
    out_a = model(x)
    key_a = provider.model_key

    # 1. Missing directory
    with pytest.raises(FileNotFoundError):
        provider.load("default_model", str(tmp_path / "does_not_exist"), "default_model")
    assert provider.model_key == key_a
    assert model(x) == out_a

    # 2. Wrong tensor shape
    with pytest.raises(ValueError, match="shape"):
        provider.load("default_model", str(bad_shape), "default_model")
    assert provider.model_key == key_a
    assert model(x) == out_a

    # 3. Attempting to overwrite base linear.weight
    with pytest.raises(ValueError, match="base layer weights"):
        provider.load("default_model", str(bad_base_overwrite), "default_model")
    assert provider.model_key == key_a
    assert model(x) == out_a

    # 4. Zero matched weights
    with pytest.raises(ValueError, match="matched no target layers"):
        provider.load("default_model", str(bad_unmatched), "default_model")
    assert provider.model_key == key_a
    assert model(x) == out_a
    assert len(load_calls) == 1


def test_default_cli_adapter_path_and_explicit_base_override(tmp_path, monkeypatch):
    server_mod, load_calls, _ = _load_patched_server_module(tmp_path, monkeypatch)
    adapter_a = _write_adapter(
        tmp_path / "adapter_a",
        weights=_adapter_a_weights(0.5),
        keys=["self_attn.q_proj"],
    )
    provider = server_mod.ModelProvider(_cli_args(adapter_path=str(adapter_a)))
    x = _FakeTensor([[2.0, -1.0]])

    # load_default() loads clean base once, then activates default adapter A
    provider.load_default()
    assert load_calls == [("base-model", None)]
    out_default_a = provider.model(x)
    assert provider.model_key == ("base-model", str(adapter_a.resolve()), None)

    # Explicit "" requests plain base without reloading base weights
    provider.load("default_model", "", "default_model")
    assert len(load_calls) == 1
    out_base = provider.model(x)
    assert out_base != out_default_a
    assert provider.model_key == ("base-model", None, None)

    # Omitting adapter (None) returns to CLI default A without reloading
    provider.load("default_model", None, "default_model")
    assert len(load_calls) == 1
    assert provider.model(x) == out_default_a


def test_lru_eviction_and_reload_without_base_reload(tmp_path, monkeypatch):
    server_mod, load_calls, eval_calls = _load_patched_server_module(
        tmp_path,
        monkeypatch,
        env_overrides={"MLX_OMARCHY_MAX_RESIDENT_ADAPTERS": "2"},
    )
    a = _write_adapter(tmp_path / "a", weights=_adapter_a_weights(0.1), keys=["self_attn.q_proj"])
    b = _write_adapter(tmp_path / "b", weights=_adapter_a_weights(0.2), keys=["self_attn.q_proj"])
    c = _write_adapter(tmp_path / "c", weights=_adapter_a_weights(0.3), keys=["self_attn.q_proj"])

    provider = server_mod.ModelProvider(_cli_args())
    provider.load("default_model", str(a), "default_model")
    provider.load("default_model", str(b), "default_model")
    assert len(provider._adapter_registry.resident_adapters) == 2

    # Loading C evicts LRU (a) while keeping base loaded once
    provider.load("default_model", str(c), "default_model")
    assert len(load_calls) == 1
    assert len(provider._adapter_registry.resident_adapters) == 2
    assert str(a.resolve()) not in provider._adapter_registry.resident_adapters
    assert str(b.resolve()) in provider._adapter_registry.resident_adapters
    assert str(c.resolve()) in provider._adapter_registry.resident_adapters

    # Re-activating a reloads only a's tensors, still 1 base load
    provider.load("default_model", str(a), "default_model")
    assert len(load_calls) == 1
    assert len(eval_calls) == 4


def test_kill_switch_and_full_finetune_fallback(tmp_path, monkeypatch):
    # 1. Kill switch MLX_OMARCHY_INPLACE_ADAPTER_SWITCH=0 -> upstream reload on every change
    server_off, load_calls_off, _ = _load_patched_server_module(
        tmp_path / "off",
        monkeypatch,
        env_overrides={"MLX_OMARCHY_INPLACE_ADAPTER_SWITCH": "0"},
    )
    a = _write_adapter(tmp_path / "a", weights=_adapter_a_weights(0.2), keys=["self_attn.q_proj"])
    prov_off = server_off.ModelProvider(_cli_args())
    prov_off.load("default_model", None, "default_model")
    prov_off.load("default_model", str(a), "default_model")
    prov_off.load("default_model", None, "default_model")
    assert len(load_calls_off) == 3

    # 2. fine_tune_type="full" falls back to _load, then base request reloads clean base
    server_on, load_calls_on, _ = _load_patched_server_module(
        tmp_path / "on",
        monkeypatch,
        env_overrides={"MLX_OMARCHY_INPLACE_ADAPTER_SWITCH": "1"},
    )
    full_ft = _write_adapter(
        tmp_path / "full_ft",
        weights=_adapter_a_weights(0.2),
        fine_tune_type="full",
    )
    prov_on = server_on.ModelProvider(_cli_args())
    prov_on.load("default_model", None, "default_model")
    prov_on.load("default_model", str(full_ft), "default_model")
    prov_on.load("default_model", None, "default_model")
    assert load_calls_on == [
        ("base-model", None),
        ("base-model", str(full_ft.resolve())),
        ("base-model", None),
    ]
