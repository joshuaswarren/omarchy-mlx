#!/usr/bin/env python3
"""In-place per-request LoRA/DoRA adapter switching for mlx_lm.server.

Upstream mlx_lm.server's ModelProvider.load() keys on
(model_path, adapter_path, draft_model_path) and calls _load() whenever
adapter_path changes, dropping self.model and reloading every base weight from
disk on every base <-> adapter alternation.

This patcher injects _OmarchyAdapterRegistry into mlx_lm/server.py and updates
ModelProvider so:
- Base weights load once and remain resident for the lifetime of (model, draft).
- Each requested LoRA/DoRA adapter loads only its low-rank tensors once and
  wraps the shared base layers (zero base weight copies), kept resident in an
  LRU registry (MLX_OMARCHY_MAX_RESIDENT_ADAPTERS, default 16).
- Switching between base (None / "") and resident adapters rebinds module slots
  in place in O(#adapted_layers) Python attribute writes (~0.1 ms).
- A request with no adapter restores the exact original unwrapped base module
  objects, giving bit-for-bit identical output to unadapted base and keeping
  every quantized fast path intact.
- Full fine-tune adapters (fine_tune_type == "full") fall back to _load().
- Invalid/missing adapters fail atomically without mutating the active model.
- Kill switch: MLX_OMARCHY_INPLACE_ADAPTER_SWITCH=0 restores upstream reload.

Supports both the mlx-lm 0.31.3 and 0.32.x server.py shapes.
Idempotent; refuses unrecognized content. Usage: patch <venv>
"""

import glob
import sys

MARKER = "class _OmarchyAdapterRegistry:"

HELPER_BLOCK = '''import os as _os_adapter_switch
from collections import OrderedDict as _AdapterOrderedDict

_INPLACE_ADAPTER_SWITCH = (
    _os_adapter_switch.environ.get("MLX_OMARCHY_INPLACE_ADAPTER_SWITCH", "1") != "0"
)
try:
    _MAX_RESIDENT_ADAPTERS = max(
        1, int(_os_adapter_switch.environ.get("MLX_OMARCHY_MAX_RESIDENT_ADAPTERS", "16"))
    )
except ValueError:
    _MAX_RESIDENT_ADAPTERS = 16


def _normalize_adapter_key(adapter_path: Optional[str]) -> Optional[str]:
    if adapter_path is None:
        return None
    text = str(adapter_path).strip()
    if not text:
        return None
    return _os_adapter_switch.path.realpath(_os_adapter_switch.path.expanduser(text))


def _peek_adapter_fine_tune_type(adapter_key: str) -> str:
    adapter_dir = Path(adapter_key)
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"The adapter path does not exist: {adapter_dir}")
    config_file = adapter_dir / "adapter_config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"Adapter config not found: {config_file}")
    with open(config_file, "r", encoding="utf-8") as fid:
        config = json.load(fid)
    if not isinstance(config, dict):
        raise ValueError(f"Adapter config must be a JSON object: {config_file}")
    return str(config.get("fine_tune_type", "lora"))


def _resolve_module_slot(root, dotted_path: str):
    parts = dotted_path.split(".")
    cur = root
    for part in parts[:-1]:
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            cur = getattr(cur, part)
    last = parts[-1]
    if isinstance(cur, list):
        return cur, int(last), True
    return cur, last, False


def _set_module_slot(parent, key, is_list: bool, module) -> None:
    if is_list:
        parent[key] = module
    elif isinstance(parent, dict) and not hasattr(parent, "named_modules"):
        parent[key] = module
    else:
        setattr(parent, key, module)


def _iter_lora_target_types():
    import mlx.nn as nn
    from .models.switch_layers import QuantizedSwitchLinear, SwitchLinear

    return (
        nn.Linear,
        nn.QuantizedLinear,
        SwitchLinear,
        QuantizedSwitchLinear,
        nn.Embedding,
        nn.QuantizedEmbedding,
    )


def _wrap_base_module_for_lora(layer, config: Dict[str, Any], use_dora: bool = False):
    import mlx.nn as nn
    from .models.switch_layers import QuantizedSwitchLinear, SwitchLinear
    from .tuner.dora import DoRAEmbedding, DoRALinear
    from .tuner.lora import LoRAEmbedding, LoRALinear, LoRASwitchLinear

    rank = config["rank"]
    scale = config["scale"]
    dropout = config.get("dropout", 0.0)
    if not use_dora and hasattr(layer, "to_lora"):
        return layer.to_lora(r=rank, scale=scale, dropout=dropout)

    if isinstance(layer, (nn.Linear, nn.QuantizedLinear)):
        lora_cls = DoRALinear if use_dora else LoRALinear
    elif isinstance(layer, (SwitchLinear, QuantizedSwitchLinear)):
        if use_dora:
            raise ValueError(f"{type(layer).__name__} doesn't support DoRA yet.")
        lora_cls = LoRASwitchLinear
    elif isinstance(layer, (nn.Embedding, nn.QuantizedEmbedding)):
        lora_cls = DoRAEmbedding if use_dora else LoRAEmbedding
    else:
        raise ValueError(
            f"Can't convert layer of type {type(layer).__name__} to LoRA"
        )
    return lora_cls.from_base(layer, r=rank, scale=scale, dropout=dropout)


class _OmarchyAdapterRegistry:
    """Keep base weights loaded and switch resident LoRA/DoRA layers in place."""

    def __init__(self, model, max_resident: int = _MAX_RESIDENT_ADAPTERS):
        self._model = model
        self._max_resident = max(1, int(max_resident))
        self._base_slots: Dict[str, Tuple[Any, Any, bool, Any]] = {}
        self._resident: "_AdapterOrderedDict[str, Dict[str, Any]]" = (
            _AdapterOrderedDict()
        )
        self._active_key: Optional[str] = None
        self._active_paths: set = set()

    @property
    def active_adapter(self) -> Optional[str]:
        return self._active_key

    @property
    def resident_adapters(self) -> Tuple[str, ...]:
        return tuple(self._resident.keys())

    def _bind(self, adapter_key: Optional[str]) -> None:
        target = {} if adapter_key is None else self._resident[adapter_key]
        for path in self._active_paths:
            if path not in target:
                parent, key, is_list, base_mod = self._base_slots[path]
                _set_module_slot(parent, key, is_list, base_mod)
        for path, lora_mod in target.items():
            parent, key, is_list, _ = self._base_slots[path]
            _set_module_slot(parent, key, is_list, lora_mod)
        self._active_key = adapter_key
        self._active_paths = set(target.keys())
        if adapter_key is not None:
            self._resident.move_to_end(adapter_key)

    def _select_targets(
        self, num_layers: int, lora_params: Dict[str, Any]
    ) -> List[Tuple[str, Any]]:
        id_to_path = {
            id(mod): path for path, mod in self._model.named_modules() if path
        }
        layers = list(getattr(self._model, "layers", ()))
        keys = lora_params.get("keys", None)
        if keys is None:
            keys_set = set()
            target_types = _iter_lora_target_types()

            def _collect(rel_path, mod):
                if hasattr(mod, "to_lora") or isinstance(mod, target_types):
                    keys_set.add(rel_path)

            for layer in layers:
                layer.apply_to_modules(_collect)
        else:
            keys_set = set(keys)

        targets: List[Tuple[str, Any]] = []
        seen = set()
        if layers:
            layer_slice = layers[-max(int(num_layers), 0) :]
            for layer in layer_slice:
                layer_prefix = id_to_path.get(id(layer), "")
                for rel_k, mod in layer.named_modules():
                    if rel_k in keys_set:
                        full_path = f"{layer_prefix}.{rel_k}" if layer_prefix else rel_k
                        if full_path not in seen:
                            seen.add(full_path)
                            targets.append((full_path, mod))

        for full_k, mod in self._model.named_modules():
            if full_k in keys_set and full_k not in seen:
                seen.add(full_k)
                targets.append((full_k, mod))

        return targets

    def _load_adapter(self, adapter_key: str) -> Dict[str, Any]:
        adapter_dir = Path(adapter_key)
        if not adapter_dir.is_dir():
            raise FileNotFoundError(f"The adapter path does not exist: {adapter_dir}")
        config_file = adapter_dir / "adapter_config.json"
        if not config_file.is_file():
            raise FileNotFoundError(f"Adapter config not found: {config_file}")
        with open(config_file, "r", encoding="utf-8") as fid:
            config = json.load(fid)
        if not isinstance(config, dict):
            raise ValueError(f"Adapter config must be a JSON object: {config_file}")
        if not isinstance(config.get("lora_parameters"), dict):
            raise ValueError(f"Adapter config missing lora_parameters object: {config_file}")

        fine_tune_type = str(config.get("fine_tune_type", "lora"))
        if fine_tune_type not in ("lora", "dora"):
            raise ValueError(
                f"In-place adapter switching supports lora/dora, got {fine_tune_type!r}"
            )
        weights_file = adapter_dir / "adapters.safetensors"
        if not weights_file.is_file():
            raise FileNotFoundError(f"Adapter weights not found: {weights_file}")

        weights = mx.load(str(weights_file))
        if not isinstance(weights, dict):
            raise ValueError(f"Adapter weights must be a mapping: {weights_file}")

        prev_key = self._active_key
        if prev_key is not None:
            self._bind(None)

        try:
            num_layers = config.get("num_layers", -1)
            lora_params = dict(config["lora_parameters"])
            use_dora = fine_tune_type == "dora"
            targets = self._select_targets(num_layers, lora_params)
            if not targets:
                raise ValueError(
                    f"Adapter at {adapter_dir} selected no target layers on the loaded model"
                )

            target_paths = {p for p, _ in targets}
            weights_by_target: Dict[str, List[Tuple[str, Any]]] = {}
            for w_key, w_val in weights.items():
                parent_path, sep, rel_key = w_key.rpartition(".")
                if sep and parent_path in target_paths:
                    if rel_key in ("linear", "embedding"):
                        raise ValueError(
                            f"Adapter weight {w_key!r} attempts to overwrite base layer weights"
                        )
                    weights_by_target.setdefault(parent_path, []).append(
                        (rel_key, w_val)
                    )
                    continue
                for full_path, _ in targets:
                    pref = full_path + "."
                    if w_key.startswith(pref):
                        rel_sub = w_key[len(pref) :]
                        if rel_sub.startswith(("linear.", "embedding.")) or rel_sub in (
                            "linear",
                            "embedding",
                        ):
                            raise ValueError(
                                f"Adapter weight {w_key!r} attempts to overwrite base layer weights"
                            )
                        weights_by_target.setdefault(full_path, []).append(
                            (rel_sub, w_val)
                        )
                        break

            new_slots: Dict[str, Tuple[Any, Any, bool, Any]] = {}
            adapter_modules: Dict[str, Any] = {}
            eval_arrays = []
            matched_keys = 0

            for full_path, base_mod in targets:
                if full_path not in self._base_slots:
                    parent, key, is_list = _resolve_module_slot(self._model, full_path)
                    new_slots[full_path] = (parent, key, is_list, base_mod)

                lora_mod = _wrap_base_module_for_lora(
                    base_mod, lora_params, use_dora=use_dora
                )
                if hasattr(lora_mod, "eval"):
                    lora_mod.eval()

                sub_weights = []
                for rel_w_key, w_val in weights_by_target.get(full_path, ()):
                    existing = getattr(lora_mod, rel_w_key, None)
                    if (
                        existing is not None
                        and hasattr(existing, "shape")
                        and hasattr(w_val, "shape")
                        and tuple(existing.shape) != tuple(w_val.shape)
                    ):
                        raise ValueError(
                            f"Adapter weight {full_path + '.' + rel_w_key!r} shape {tuple(w_val.shape)} "
                            f"does not match expected {tuple(existing.shape)}"
                        )
                    sub_weights.append((rel_w_key, w_val))
                    eval_arrays.append(w_val)

                if sub_weights:
                    lora_mod.load_weights(sub_weights, strict=False)
                    matched_keys += len(sub_weights)

                adapter_modules[full_path] = lora_mod

            if weights and matched_keys == 0:
                raise ValueError(
                    f"Adapter weights at {weights_file} matched no target layers on the loaded model"
                )

            if eval_arrays:
                mx.eval(*eval_arrays)

            self._base_slots.update(new_slots)
            self._resident[adapter_key] = adapter_modules
            self._resident.move_to_end(adapter_key)
            while len(self._resident) > self._max_resident:
                evict_key = next(
                    (k for k in self._resident if k != adapter_key and k != prev_key),
                    None,
                )
                if evict_key is None:
                    break
                del self._resident[evict_key]
            return adapter_modules
        except Exception:
            if prev_key is not None:
                self._bind(prev_key)
            raise

    def activate(self, adapter_key: Optional[str]) -> None:
        if adapter_key is not None and adapter_key not in self._resident:
            self._load_adapter(adapter_key)
        self._bind(adapter_key)


class ModelProvider:
'''

CLASS_ANCHOR = "class ModelProvider:\n"

INIT_OLD = '''        if cli_args.chat_template:
            self._tokenizer_config["chat_template"] = cli_args.chat_template
'''

INIT_NEW = '''        if cli_args.chat_template:
            self._tokenizer_config["chat_template"] = cli_args.chat_template

        # mlx-omarchy: resident base + in-place per-request adapter switching state
        self._base_key = None
        self._adapter_registry = None
        self._last_adapter_switch_ms = 0.0
        self._base_load_count = 0
'''

RESET_032_OLD = '''    def reset(self) -> None:
        self.model_key = None
        self.model = None
        self.tokenizer = None
        self.draft_model = None
        self.is_batchable = False
'''

RESET_032_NEW = '''    def reset(self) -> None:
        self.model_key = None
        self.model = None
        self.tokenizer = None
        self.draft_model = None
        self.is_batchable = False
        self._base_key = None
        self._adapter_registry = None
'''

RESET_031_OLD = '''        # Remove the old model if it exists.
        self.model_key = None
        self.model = None
        self.tokenizer = None
        self.draft_model = None
'''

RESET_031_NEW = '''        # Remove the old model if it exists.
        self.model_key = None
        self.model = None
        self.tokenizer = None
        self.draft_model = None
        self._base_key = None
        self._adapter_registry = None
'''

LOAD_031_OLD = '''    def load(self, model_path, adapter_path=None, draft_model_path=None):
        model_path = self._model_map.get(model_path, model_path)
        adapter_path = self._adapter_map.get(model_path, adapter_path)
        draft_model_path = self._draft_model_map.get(draft_model_path, draft_model_path)

        model_key = (model_path, adapter_path, draft_model_path)
        if self.model_key != model_key:
            self._load(*model_key)

        return self.model, self.tokenizer
'''

LOAD_032_OLD = '''    def load(self, model_path, adapter_path=None, draft_model_path=None):
        adapter_path = self._adapter_map.get(model_path, adapter_path)
        model_path = self._model_map.get(model_path, model_path)
        draft_model_path = self._draft_model_map.get(draft_model_path, draft_model_path)

        model_key = (model_path, adapter_path, draft_model_path)
        if self.model_key != model_key:
            self._load(*model_key)

        return self.model, self.tokenizer
'''

LOAD_NEW = '''    def load(self, model_path, adapter_path=None, draft_model_path=None):
        raw_model_path = model_path
        model_path = self._model_map.get(model_path, model_path)
        if adapter_path is None:
            adapter_path = self._adapter_map.get(
                raw_model_path, self._adapter_map.get(model_path, None)
            )
        draft_model_path = self._draft_model_map.get(draft_model_path, draft_model_path)

        if not _INPLACE_ADAPTER_SWITCH:
            adapter_path = adapter_path or None
            model_key = (model_path, adapter_path, draft_model_path)
            if self.model_key != model_key:
                self._load(*model_key)
                self._base_load_count += 1
            return self.model, self.tokenizer

        adapter_key = _normalize_adapter_key(adapter_path)
        model_key = (model_path, adapter_key, draft_model_path)
        if self.model_key == model_key:
            return self.model, self.tokenizer

        if self.is_distributed and (
            adapter_key is not None or draft_model_path is not None
        ):
            raise ValueError(
                "Loading with adapters or draft models not supported in distributed mode"
            )

        if adapter_key is not None and (
            self._adapter_registry is None
            or adapter_key not in self._adapter_registry.resident_adapters
        ):
            if _peek_adapter_fine_tune_type(adapter_key) == "full":
                self._load(model_path, adapter_key, draft_model_path)
                self._base_load_count += 1
                return self.model, self.tokenizer

        base_key = (model_path, draft_model_path)
        if (
            self.model is None
            or self._base_key != base_key
            or self._adapter_registry is None
        ):
            self._load(model_path, None, draft_model_path)
            self._base_key = base_key
            self._adapter_registry = _OmarchyAdapterRegistry(self.model)
            self._base_load_count += 1

        t0 = time.perf_counter()
        self._adapter_registry.activate(adapter_key)
        self._last_adapter_switch_ms = (time.perf_counter() - t0) * 1000.0
        self.model_key = model_key
        logging.info(
            "In-place adapter switch -> %s (%.3f ms)",
            adapter_key or "base",
            self._last_adapter_switch_ms,
        )
        return self.model, self.tokenizer
'''


def patch_text(text: str, path: str = "<memory>") -> tuple[str, str]:
    """Patch mlx_lm/server.py source text. Returns (new_text, status)."""
    if MARKER in text:
        if LOAD_NEW in text and INIT_NEW in text and HELPER_BLOCK in text:
            return text, "already patched"
        raise RuntimeError(
            f"patched content not recognized in {path}; refusing to patch"
        )

    if text.count(CLASS_ANCHOR) != 1:
        raise RuntimeError(
            f"ModelProvider anchor not unique in {path}; refusing to patch"
        )
    if text.count(INIT_OLD) != 1:
        raise RuntimeError(
            f"ModelProvider.__init__ anchor not unique in {path}; refusing to patch"
        )

    if text.count(RESET_032_OLD) == 1:
        reset_old, reset_new = RESET_032_OLD, RESET_032_NEW
    elif text.count(RESET_031_OLD) == 1:
        reset_old, reset_new = RESET_031_OLD, RESET_031_NEW
    else:
        raise RuntimeError(
            f"ModelProvider reset anchor not unique in {path}; refusing to patch"
        )

    if text.count(LOAD_031_OLD) == 1:
        load_old = LOAD_031_OLD
    elif text.count(LOAD_032_OLD) == 1:
        load_old = LOAD_032_OLD
    else:
        raise RuntimeError(
            f"ModelProvider.load anchor not unique in {path}; refusing to patch"
        )

    patched = (
        text.replace(CLASS_ANCHOR, HELPER_BLOCK, 1)
        .replace(INIT_OLD, INIT_NEW, 1)
        .replace(reset_old, reset_new, 1)
        .replace(load_old, LOAD_NEW, 1)
    )
    return patched, "patched"


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    venv = args[0] if args else "."
    sites = glob.glob(venv.rstrip("/") + "/lib/python3*/site-packages/mlx_lm")
    if not sites:
        sys.exit("mlx_lm not found under " + venv)
    path = sites[0] + "/server.py"
    with open(path, "r", encoding="utf-8") as fid:
        text = fid.read()
    try:
        new_text, status = patch_text(text, path)
    except RuntimeError as exc:
        sys.exit(str(exc))
    if status == "already patched":
        print("already patched:", path)
        return 0
    with open(path, "w", encoding="utf-8") as fid:
        fid.write(new_text)
    print("patched:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
