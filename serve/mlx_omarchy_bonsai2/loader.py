"""Strict loader for prism_hadamard_qwen35 text packs.

Loads a Bonsai-2 style checkpoint (single model.safetensors + config.json,
schema 2) into an mlx_lm qwen3_5 TextModel with `Packed` Hadamard modules,
without importing or executing any code shipped inside the pack.

Contract enforced here (everything else is refused, never approximated):

- config.json: model_type == prism_hadamard_qwen35, schema_version == 2,
  base_model_type == qwen3_5, quantization == affine 2-bit group-128.
- tensor namespace "language_model." (mlx-vlm-qwen3_5); keys outside it
  (vision tower, MTP drafts) are excluded from residency and reported,
  never loaded.
- every packed module record is validated against the stock module it
  replaces (kind, shapes, storage dtypes, sign vectors) before any
  substitution; consumed keys must cover the text model's parameter tree
  exactly (`strict=True` load on top of an explicit accounting pass).

No float dequantization happens at load or serve time: packed tensors
stay U32/F16 and run through mx.quantized_matmul / row-gathered
mx.dequantize inside `Packed`.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx import nn

from .packed import ALLOWED_BLOCKS, BITS, GROUP_SIZE, Packed, PackedError

PACK_MODEL_TYPE = "prism_hadamard_qwen35"
BASE_MODEL_TYPE = "qwen3_5"
SCHEMA_VERSION = 2
LM_PREFIX = "language_model."
RECORD_FIELDS = ("path", "block", "embedding", "dtype")
PACKED_SUFFIXES = ("weight", "scales", "biases", "signs")


class PackError(ValueError):
    """The checkpoint does not satisfy the pack contract."""


def _fail(msg: str) -> None:
    raise PackError(msg)


def config_sha256(pack_dir: Path) -> str:
    return hashlib.sha256((pack_dir / "config.json").read_bytes()).hexdigest()


def _safetensors_header(path: Path) -> dict:
    """Parse the safetensors header: name -> (dtype, shape, nbytes).

    Reads only the header bytes, never the tensor data; gives exact
    per-tensor sizes for residency accounting without touching weights.
    """
    with path.open("rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    out = {}
    for name, info in header.items():
        begin, end = info["data_offsets"]
        out[name] = (info["dtype"], tuple(info["shape"]), end - begin)
    return out


def _check_config(config: dict) -> None:
    if config.get("model_type") != PACK_MODEL_TYPE:
        _fail(f"model_type must be {PACK_MODEL_TYPE!r}, got {config.get('model_type')!r}")
    if config.get("schema_version") != SCHEMA_VERSION:
        _fail(f"schema_version must be {SCHEMA_VERSION}, got {config.get('schema_version')!r}")
    if config.get("base_model_type") != BASE_MODEL_TYPE:
        _fail(f"base_model_type must be {BASE_MODEL_TYPE!r}")
    quant = config.get("quantization")
    if quant != {"bits": BITS, "group_size": GROUP_SIZE, "mode": "affine"}:
        _fail(f"unsupported quantization {quant!r}; this loader implements affine 2-bit group-128 only")
    if not isinstance(config.get("text_config"), dict):
        _fail("config is missing text_config")
    records = config.get("modules")
    if not isinstance(records, list) or not records:
        _fail("config is missing the packed-module list")
    namespace = config.get("tensor_namespace")
    if namespace is not None and namespace != "mlx-vlm-qwen3_5":
        _fail(f"unsupported tensor_namespace {namespace!r}")


def _resolve_parent(model, parts, where: str):
    parent = model
    for part in parts:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent


def _kv_bytes_per_token(text_config):
    """f16 KV-cache bytes per token: 2 (K+V) per full-attention layer.

    GDN layers carry constant state, not per-token KV, and are covered by
    the budget's workspace margin, not this number.
    """
    layers = text_config.get("num_hidden_layers")
    interval = text_config.get("full_attention_interval")
    kv_heads = text_config.get("num_key_value_heads")
    head_dim = text_config.get("head_dim")
    if not all(isinstance(v, int) and v > 0 for v in (layers, interval, kv_heads, head_dim)):
        return None
    return (layers // interval) * 2 * kv_heads * head_dim * 2


def pack_footprint(pack_dir):
    """Pre-load facts from config.json + the safetensors header only.

    Reads zero tensor bytes; gives the exact text-only weights size and
    KV rate so memory admission can happen BEFORE the model is loaded.
    """
    pack_dir = Path(pack_dir)
    config_path = pack_dir / "config.json"
    if not config_path.is_file():
        _fail(f"{pack_dir}: no config.json")
    config = json.loads(config_path.read_text())
    _check_config(config)
    safetensors = pack_dir / "model.safetensors"
    if not safetensors.is_file():
        _fail(f"{pack_dir}: no model.safetensors")
    header = _safetensors_header(safetensors)
    lm_bytes = sum(n for k, (_, _, n) in header.items() if k.startswith(LM_PREFIX))
    total_header_bytes = sum(n for _, (_, _, n) in header.items())
    if not lm_bytes:
        _fail("no language_model.* tensors in checkpoint")
    return {
        # Conservative load-phase bound: mx.load is lazy/mmap-backed
        # (verified empirically 2026-09-20: RSS and MemAvailable stay flat
        # across mx.load of the 8.6 GB pack), but the conservative
        # admission bound assumes the whole file could materialize.
        "total_header_bytes": total_header_bytes,
        "live_weights_bytes": lm_bytes,
        "kv_bytes_per_token": _kv_bytes_per_token(config["text_config"]),
        "max_position_embeddings": config["text_config"].get("max_position_embeddings"),
        "packed_modules": len(config["modules"]),
    }


def load_text_model(pack_dir):
    """Load the text model from a pack directory. Returns (model, info)."""
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    pack_dir = Path(pack_dir)
    config_path = pack_dir / "config.json"
    if not config_path.is_file():
        _fail(f"{pack_dir}: no config.json")
    config = json.loads(config_path.read_text())
    _check_config(config)
    safetensors = pack_dir / "model.safetensors"
    if not safetensors.is_file():
        _fail(f"{pack_dir}: no model.safetensors")
    header = _safetensors_header(safetensors)

    lm_keys = {k: v for k, v in header.items() if k.startswith(LM_PREFIX)}
    if not lm_keys:
        _fail("no language_model.* tensors in checkpoint")
    excluded = {}
    for name, (_, _, nbytes) in header.items():
        if not name.startswith(LM_PREFIX):
            excluded.setdefault(name.split(".", 1)[0], 0)
            excluded[name.split(".", 1)[0]] += nbytes

    text_config = dict(config["text_config"])
    declared_types = text_config.get("layer_types")
    interval = text_config.get("full_attention_interval")
    if isinstance(declared_types, list) and isinstance(interval, int) and interval > 0:
        derived = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(len(declared_types))
        ]
        if declared_types != derived:
            _fail("text_config.layer_types contradicts full_attention_interval")
    args = TextModelArgs.from_dict(text_config)
    model = TextModel(args)

    expected = {name: value.shape for name, value in tree_flatten(model.parameters())}

    weights = mx.load(str(safetensors))
    consumed = set()
    resident_bytes = 0
    packed_count = 0
    for record in config["modules"]:
        if set(record) != set(RECORD_FIELDS):
            _fail(f"packed record keys {sorted(record)} != {sorted(RECORD_FIELDS)}")
        path, block, embedding, dtype = (record[k] for k in RECORD_FIELDS)
        if dtype != "float16":
            _fail(f"{path}: unsupported activation dtype {dtype!r}")
        if not isinstance(block, int) or isinstance(block, bool) or block < 0:
            _fail(f"{path}: invalid block {block!r}")
        parts = path.split(".")
        parent = _resolve_parent(model, parts[:-1], path)
        name = parts[-1]
        original = getattr(parent, name, None)
        if embedding != isinstance(original, nn.Embedding):
            _fail(f"{path}: packed module kind does not match the target module")
        if embedding:
            rows, width = original.weight.shape
        elif isinstance(original, nn.Linear):
            rows, width = original.weight.shape
        else:
            _fail(f"{path}: packed target is neither Linear nor Embedding")
        if width % GROUP_SIZE:
            _fail(f"{path}: width {width} is not a multiple of {GROUP_SIZE}")
        if block:
            if block not in ALLOWED_BLOCKS or width % block:
                _fail(f"{path}: block {block} invalid for width {width}")

        key = LM_PREFIX + path
        shapes = {}
        for suffix in PACKED_SUFFIXES:
            full = f"{key}.{suffix}"
            if full not in weights:
                _fail(f"{path}: missing tensor {full}")
            shapes[suffix] = weights[full]
            consumed.add(full)
            resident_bytes += header[full][2]
        weight, scales, biases = shapes["weight"], shapes["scales"], shapes["biases"]
        signs = shapes["signs"]
        if weight.dtype != mx.uint32 or tuple(weight.shape) != (rows, width // 16):
            _fail(f"{path}: packed weight {weight.shape}/{weight.dtype} != ({rows}, {width // 16}) uint32")
        for affine, label in ((scales, "scales"), (biases, "biases")):
            if affine.dtype != mx.float16 or tuple(affine.shape) != (rows, width // GROUP_SIZE):
                _fail(f"{path}: {label} {affine.shape}/{affine.dtype} must be ({rows}, {width // GROUP_SIZE}) float16")
        if block and (signs.shape != (width,)):
            _fail(f"{path}: sign vector {signs.shape} must be ({width},)")
        if not block:
            signs = None
        setattr(parent, name, Packed((weight, scales, biases), block, signs, embedding))
        packed_count += 1
        mx.eval(getattr(parent, name).parameters())

    expected = {name: value.shape for name, value in tree_flatten(model.parameters())}
    remapped = []
    for key in sorted(lm_keys):
        stripped = key[len(LM_PREFIX):]
        if stripped not in expected:
            _fail(f"unexpected language_model tensor: {key}")
        remapped.append((stripped, weights[key]))
        if key not in consumed:
            resident_bytes += header[key][2]
    missing = expected.keys() - {k for k, _ in remapped}
    if missing:
        _fail(f"checkpoint is missing text-model parameters: {sorted(missing)[:8]}")
    del weights

    model.load_weights(remapped, strict=True)
    model.eval()
    mx.eval(model.parameters())

    license_files = {
        name: (pack_dir / name).stat().st_size if (pack_dir / name).is_file() else None
        for name in ("LICENSE", "NOTICE.txt")
    }
    info = {
        "pack_dir": str(pack_dir),
        "model_type": config["model_type"],
        "base_model_type": config["base_model_type"],
        "schema_version": SCHEMA_VERSION,
        "quantization": config["quantization"],
        "hadamard_block": sorted({r["block"] for r in config["modules"]} - {0}),
        "packed_modules": packed_count,
        "resident_bytes": resident_bytes,
        "excluded_bytes": excluded,
        "license_files": license_files,
        "config_sha256": config_sha256(pack_dir),
        "source_revision": None,
        "attribution": "Created using Bonsai by Prism ML (Apache-2.0; built from Qwen3.8-27B, Apache-2.0).",
        "max_position_embeddings": args.max_position_embeddings,
        "kv_bytes_per_token": _kv_bytes_per_token(text_config),
    }
    return model, info


def load_tokenizer(pack_dir):
    """Pack tokenizer as an mlx_lm TokenizerWrapper (chat template included)."""
    from transformers import AutoTokenizer
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    return TokenizerWrapper(AutoTokenizer.from_pretrained(str(pack_dir)))
