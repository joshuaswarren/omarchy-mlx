"""Strict loader for prism_hadamard_qwen35 text packs.

Loads a Bonsai-2 style checkpoint (single model.safetensors + config.json,
schema 2) into an mlx_lm qwen3_5 TextModel with `Packed` Hadamard modules,
without importing or executing any code shipped inside the pack.

Contract enforced here (everything else is refused, never approximated):

- config.json: bounded regular-file read (16 MiB cap), model_type ==
  prism_hadamard_qwen35, schema_version == 2, base_model_type == qwen3_5,
  quantization == affine 2-bit group-128.
- model.safetensors header: bounded regular-file read (16 MiB cap) with
  validated dtypes, shapes, byte spans, and in-file data_offsets before
  memory admission or weight loading.
- admission-to-load pinning: each file is read once, admission pins the
  config bytes (sha256) and the weights file identity, and load refuses a
  pack that changed after admission or while loading.
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

import contextlib
import hashlib
import json
import os
import stat
import struct
from pathlib import Path
from typing import NamedTuple

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
# Real Bonsai-2 header is 303,102 bytes; 16 MiB also bounds JSON parser memory before admission.
_SAFETENSORS_HEADER_MAX_BYTES = 16 << 20
# Real Bonsai-2 config.json is ~58 KiB; cap reads to bound corrupt or hostile files.
_CONFIG_MAX_BYTES = 16 << 20
_READ_CHUNK_BYTES = 1 << 20
# Element size and the mlx dtype each safetensors dtype loads as (F8 loads as uint8).
_SAFETENSORS_DTYPES = {
    "BOOL": (1, "bool_"),
    "U8": (1, "uint8"),
    "I8": (1, "int8"),
    "F8_E4M3": (1, "uint8"),
    "F8_E8M0": (1, "uint8"),
    "U16": (2, "uint16"),
    "I16": (2, "int16"),
    "F16": (2, "float16"),
    "BF16": (2, "bfloat16"),
    "U32": (4, "uint32"),
    "I32": (4, "int32"),
    "F32": (4, "float32"),
    "U64": (8, "uint64"),
    "I64": (8, "int64"),
    "C64": (8, "complex64"),
}


class PackError(ValueError):
    """The checkpoint does not satisfy the pack contract."""


class PackChangedError(PackError):
    """A pack file changed after memory admission or while loading."""


class PackPin(NamedTuple):
    """Exact pack files pinned at memory admission (config sha256 + weights stat identity)."""

    config_sha256: str
    weights: tuple


def _fail(msg: str) -> None:
    raise PackError(msg)


def _file_identity(st) -> tuple:
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def _is_nonneg_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


@contextlib.contextmanager
def _open_regular(path: Path):
    """Open a regular file non-blockingly, following symlinks for HF caches."""
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOCTTY | os.O_CLOEXEC)
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            _fail(f"{path} is not a regular file")
        yield fd, st
    except OSError as exc:
        raise PackError(
            f"cannot read {path} ({exc.strerror or type(exc).__name__})"
        ) from None
    finally:
        if fd is not None:
            os.close(fd)


def _read_exact(fd: int, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = os.read(fd, min(_READ_CHUNK_BYTES, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-standard JSON constant {token!r}")


def _parse_json_object(raw: bytes, name: str, *, strict_constants: bool = False) -> dict:
    try:
        data = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_json_constant if strict_constants else None,
        )
    except (ValueError, RecursionError) as exc:
        raise PackError(f"{name} is not valid JSON ({type(exc).__name__})") from None
    if not isinstance(data, dict):
        _fail(f"{name} is not a JSON object")
    return data


def _read_config_bytes(path: Path) -> bytes:
    cap = _CONFIG_MAX_BYTES
    with _open_regular(path) as (fd, st):
        if st.st_size > cap:
            _fail(f"{path} exceeds {cap} bytes")
        raw = _read_exact(fd, cap + 1)
    if len(raw) > cap:
        _fail(f"{path} exceeds {cap} bytes")
    return raw


def config_sha256(pack_dir: Path) -> str:
    return hashlib.sha256(_read_config_bytes(Path(pack_dir) / "config.json")).hexdigest()


def _safetensors_header(path: Path) -> tuple[dict, tuple]:
    """Parse and validate the safetensors header, returning (header, identity).

    ``header`` maps name -> (dtype, shape, nbytes); ``identity`` is
    ``_file_identity`` of the descriptor the header was read from. Reads only
    the header bytes, never the tensor data; gives exact per-tensor sizes for
    residency accounting without touching weights.
    """
    with _open_regular(path) as (fd, st):
        file_size = st.st_size
        prefix = _read_exact(fd, 8)
        if len(prefix) < 8:
            _fail("model.safetensors is too short to hold a safetensors header")
        (header_len,) = struct.unpack("<Q", prefix)
        if header_len > _SAFETENSORS_HEADER_MAX_BYTES:
            _fail(
                f"model.safetensors header length {header_len} exceeds the "
                f"{_SAFETENSORS_HEADER_MAX_BYTES}-byte limit"
            )
        if 8 + header_len > file_size:
            _fail(f"model.safetensors is truncated (header length {header_len})")
        raw = _read_exact(fd, header_len)
        if len(raw) != header_len:
            _fail(f"model.safetensors is truncated (header length {header_len})")
        identity = _file_identity(st)
    header = _parse_json_object(raw, "model.safetensors header", strict_constants=True)
    metadata = header.pop("__metadata__", None)
    if metadata is not None and (
        not isinstance(metadata, dict) or not all(isinstance(v, str) for v in metadata.values())
    ):
        _fail("model.safetensors header __metadata__ must map strings to strings")
    out = {}
    for name, info in header.items():
        label = name[:80]
        if not isinstance(info, dict):
            _fail(f"tensor {label!r} has malformed metadata")
        dtype = info.get("dtype")
        shape = info.get("shape")
        offsets = info.get("data_offsets")
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or not all(_is_nonneg_int(d) for d in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(_is_nonneg_int(o) for o in offsets)
        ):
            _fail(f"tensor {label!r} has malformed metadata")
        dtype_info = _SAFETENSORS_DTYPES.get(dtype)
        if dtype_info is None:
            _fail(f"tensor {label!r} has unsupported dtype {dtype[:80]!r}")
        itemsize = dtype_info[0]
        begin, end = offsets
        if begin > end or 8 + header_len + end > file_size:
            _fail(f"tensor {label!r} has invalid data_offsets")
        span = end - begin
        if 0 in shape:
            expected_bytes = 0
        else:
            expected_bytes = itemsize
            for dim in shape:
                expected_bytes *= dim
                if expected_bytes > span:
                    break
        if span != expected_bytes:
            _fail(f"tensor {label!r} byte span does not match dtype and shape")
        out[name] = (dtype, tuple(shape), span)
    return out, identity


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


def _read_pack(pack_dir: Path) -> tuple[dict, dict, dict]:
    raw = _read_config_bytes(pack_dir / "config.json")
    config = _parse_json_object(raw, "config.json")
    _check_config(config)
    header, identity = _safetensors_header(pack_dir / "model.safetensors")
    lm_bytes = sum(n for k, (_, _, n) in header.items() if k.startswith(LM_PREFIX))
    total_header_bytes = sum(n for _, (_, _, n) in header.items())
    if not lm_bytes:
        _fail("no language_model.* tensors in checkpoint")
    facts = {
        # Conservative load-phase bound: mx.load is lazy/mmap-backed
        # (verified empirically 2026-09-20: RSS and MemAvailable stay flat
        # across mx.load of the 8.6 GB pack), but the conservative
        # admission bound assumes the whole file could materialize.
        "total_header_bytes": total_header_bytes,
        "live_weights_bytes": lm_bytes,
        "kv_bytes_per_token": _kv_bytes_per_token(config["text_config"]),
        "max_position_embeddings": config["text_config"].get("max_position_embeddings"),
        "packed_modules": len(config["modules"]),
        "pin": PackPin(hashlib.sha256(raw).hexdigest(), identity),
    }
    return config, header, facts


def _check_path(path: Path, identity: tuple) -> None:
    try:
        st = os.stat(path)
    except OSError as exc:
        raise PackChangedError(
            f"{path} changed while loading ({exc.strerror or type(exc).__name__})"
        ) from None
    if _file_identity(st) != identity:
        raise PackChangedError(f"{path} changed while loading")


def _check_loaded(path: Path, identity: tuple, header: dict, weights: dict) -> None:
    # mx.load reopens by path (kept so lazy loading and the
    # MLX_OMARCHY_VRAM_WEIGHTS directory match keep working); confirm the
    # path still names the validated file and mlx sees the validated tensor
    # names, dtypes, shapes, and byte sizes.
    _check_path(path, identity)
    if set(weights) != set(header) or any(
        weights[k].dtype != getattr(mx, _SAFETENSORS_DTYPES[header[k][0]][1])
        or tuple(weights[k].shape) != header[k][1]
        or weights[k].nbytes != header[k][2]
        for k in header
    ):
        raise PackChangedError(f"{path} changed while loading")


def pack_footprint(pack_dir):
    """Pre-load facts from config.json + the safetensors header only.

    Reads zero tensor bytes; gives the exact text-only weights size and
    KV rate so memory admission can happen BEFORE the model is loaded.
    ``facts["pin"]`` identifies the exact files measured, for
    ``load_text_model(admitted=...)``.
    """
    return _read_pack(Path(pack_dir))[2]


def load_text_model(pack_dir, *, admitted=None):
    """Load the text model from a pack directory. Returns (model, info)."""
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    pack_dir = Path(pack_dir)
    safetensors = pack_dir / "model.safetensors"
    try:
        config, header, facts = _read_pack(pack_dir)
    except PackError as exc:
        if admitted is not None:
            raise PackChangedError(
                f"{pack_dir} no longer matches memory admission ({exc})"
            ) from None
        raise
    if admitted is not None:
        if facts["pin"].config_sha256 != admitted["pin"].config_sha256:
            raise PackChangedError(f"{pack_dir / 'config.json'} changed after memory admission")
        if facts != admitted:
            raise PackChangedError(f"{safetensors} changed after memory admission")
    try:
        weights = mx.load(str(safetensors))
    except Exception:
        # mlx reopens the path, so a failure on a changed file is a change
        # during load; a failure on the unchanged file propagates as is.
        _check_path(safetensors, facts["pin"].weights)
        raise
    _check_loaded(safetensors, facts["pin"].weights, header, weights)

    lm_keys = {k: v for k, v in header.items() if k.startswith(LM_PREFIX)}
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
        "config_sha256": facts["pin"].config_sha256,
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
