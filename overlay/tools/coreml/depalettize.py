# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Frontend pre-expansion of ``constexpr_lut_to_dense`` into dense constants.

The pinned Parakeet encoder stores every ``linear`` weight palettized: the
MIL program references uint4 indices and an fp16 lookup table in the
package weight file, and ``constexpr_lut_to_dense`` materializes the dense
weight at load time. The H13 compiler input has no constexpr concept for
that op, so the mlx-omarchy Core ML frontend expands each op into a plain
``const`` whose value is the gathered dense tensor, written as a new blob.

Numerics are exact by construction: the gather copies raw fp16 payload
bytes (no float arithmetic), so the expansion is byte-identical to the
reference materialization.

Format facts this module depends on (all verified against Apple sources):

* Weight files are MIL blob storage (``MILBlob/Blob/StorageFormat.hpp``):
  a 64-byte-aligned ``storage_header`` (count u32, version u32=2, five
  reserved u64) followed by ``blob_metadata``/raw-data pairs, each entry
  aligned to 64 bytes.
* ``Value.blobFileValue.offset`` points at the ``blob_metadata`` struct:
  sentinel u32 = 0xDEADBEEF, dtype u32, sizeInBytes u64 (bit-padding
  included), data offset u64, padding bits u64, four reserved u64.
* Blob dtype codes (``BlobDataType.hpp``): Float16 = 1, Float32 = 2,
  UInt4 = 11. MIL ``DataType`` codes: FLOAT16 = 10, FLOAT32 = 11,
  UINT4 = 35.
* Sub-byte packing (``SubByteTypes.cpp`` ``PackSubByteVecImpl``): element
  ``i`` occupies bits ``nbits * (i % elements_per_byte)`` upward within
  byte ``i // elements_per_byte`` — for uint4, element ``2k`` is the low
  nibble and ``2k+1`` the high nibble (LSB-first).
* Op semantics (coremltools ``constexpr_lut_to_dense`` + ``lut_to_dense``):
  ``lut`` rank is indices rank + 2; leading lut dims tile the indices dims
  in contiguous blocks (block size = indices dim / lut dim per axis, i.e.
  ``np.repeat`` semantics); ``lut[..., palette, 0]`` gathers per element;
  the output dtype and shape are the lut dtype and the indices shape.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BLOB_SENTINEL = 0xDEADBEEF
BLOB_METADATA = struct.Struct("<IIQQQ")  # + 4 reserved u64 = 64 bytes total
BLOB_RESERVED = b"\x00" * 32
STORAGE_HEADER = struct.Struct("<II")  # count, version; + 5 reserved u64
STORAGE_VERSION = 2
ALIGNMENT = 64

BLOB_DTYPE_BY_MIL = {10: 1, 11: 2}  # FLOAT16, FLOAT32 -> blob codes
_MIL_NAME = {10: "fp16", 11: "fp32", 25: "int4", 35: "uint4", 31: "uint8",
             36: "uint2", 37: "uint1", 38: "uint6", 39: "uint3"}
_NBITS_BY_MIL = {37: 1, 36: 2, 39: 3, 35: 4, 38: 6, 31: 8}


class DepalettizeError(RuntimeError):
    """A palettized weight cannot be expanded; the reason is named."""


@dataclass(frozen=True)
class BlobMetadata:
    dtype_code: int
    size_bytes: int
    data_offset: int
    padding_bits: int


@dataclass(frozen=True)
class ExpandedConst:
    """One ``constexpr_lut_to_dense`` replaced by a dense ``const``."""

    output_name: str
    shape: tuple[int, ...]
    dtype_name: str
    element_count: int
    blob_offset: int  # metadata offset inside the expansion blob file
    byte_count: int


def read_blob_metadata(handle, offset: int) -> BlobMetadata:
    """Parse and validate one ``blob_metadata`` at ``offset``."""
    handle.seek(offset)
    raw = handle.read(BLOB_METADATA.size)
    if len(raw) != BLOB_METADATA.size:
        raise DepalettizeError(f"blob metadata at {offset} is truncated")
    sentinel, dtype_code, size_bytes, data_offset, padding_bits = (
        BLOB_METADATA.unpack(raw)
    )
    if sentinel != BLOB_SENTINEL:
        raise DepalettizeError(
            f"blob metadata at {offset} has sentinel 0x{sentinel:08x}, "
            f"expected 0x{BLOB_SENTINEL:08x}"
        )
    return BlobMetadata(dtype_code, size_bytes, data_offset, padding_bits)


def unpack_packed_uints(raw: bytes, nbits: int, element_count: int) -> np.ndarray:
    """Unpack LSB-first sub-byte integers into one ``uint8`` per element."""
    if nbits == 8:
        values = np.frombuffer(raw, dtype=np.uint8)
    else:
        bits = np.unpackbits(
            np.frombuffer(raw, dtype=np.uint8), bitorder="little"
        )
        trim = element_count * nbits
        if bits.size < trim:
            raise DepalettizeError(
                f"packed payload holds {bits.size // nbits} uint{nbits} "
                f"elements, expected {element_count}"
            )
        groups = bits[:trim].reshape(-1, nbits)
        values = np.packbits(
            np.pad(groups, ((0, 0), (0, 8 - nbits))).reshape(-1),
            bitorder="little",
        )
    return values[:element_count]


class _BlobAppender:
    """Appends blob entries to a MIL blob storage file, 64-byte aligned."""

    def __init__(self, path: Path):
        self.path = path
        if path.exists() and path.stat().st_size > 0:
            self._handle = path.open("r+b")
            header = self._handle.read(ALIGNMENT)
            count, version = STORAGE_HEADER.unpack_from(header, 0)
            if version != STORAGE_VERSION:
                raise DepalettizeError(
                    f"{path}: blob storage version {version} is not "
                    f"{STORAGE_VERSION}"
                )
            self._count = count
            self._handle.seek(0, 2)
        else:
            self._handle = path.open("wb")
            self._handle.write(_pad(STORAGE_HEADER.pack(0, STORAGE_VERSION)))
            self._count = 0

    def append(self, payload: bytes, blob_dtype_code: int) -> int:
        self._handle.seek(0, 2)
        end = self._handle.tell()
        if end % ALIGNMENT:
            self._handle.write(b"\x00" * (ALIGNMENT - end % ALIGNMENT))
        metadata_offset = self._handle.tell()
        self._handle.write(
            BLOB_METADATA.pack(
                BLOB_SENTINEL, blob_dtype_code, len(payload),
                metadata_offset + 64, 0,
            )
            + BLOB_RESERVED
        )
        self._handle.write(_pad(payload))
        self._count += 1
        self._handle.seek(0)
        self._handle.write(STORAGE_HEADER.pack(self._count, STORAGE_VERSION))
        return metadata_offset

    def close(self) -> None:
        self._handle.close()


def _pad(data: bytes) -> bytes:
    remainder = len(data) % ALIGNMENT
    return data if remainder == 0 else data + b"\x00" * (ALIGNMENT - remainder)


def _tensor_shape(tensor_type) -> tuple[int, ...]:
    shape = []
    for dim in tensor_type.dimensions:
        which = dim.WhichOneof("dimension")
        if which != "constant":
            raise DepalettizeError(
                f"palettized tensor dimension is {which or 'unset'}, "
                "not static"
            )
        shape.append(dim.constant.size)
    return tuple(shape)


def _blob_backed_tensor(binding, label: str):
    """Return ``(dtype, shape, file_name, offset)`` for one const binding."""
    which = binding.WhichOneof("binding")
    if which != "value":
        raise DepalettizeError(
            f"{label} is bound by {which or 'unset'}, not an inline value"
        )
    value = binding.value
    if value.WhichOneof("value") != "blobFileValue":
        raise DepalettizeError(
            f"{label} is not blob-backed "
            f"({value.WhichOneof('value') or 'unset'})"
        )
    if value.type.WhichOneof("type") != "tensorType":
        raise DepalettizeError(
            f"{label} type is {value.type.WhichOneof('type') or 'unset'}, "
            "not tensorType"
        )
    tensor_type = value.type.tensorType
    blob = value.blobFileValue
    return (tensor_type.dataType, _tensor_shape(tensor_type),
            blob.fileName, blob.offset)


def _read_blob(weights_dir: Path, file_name: str, offset: int, label: str) -> bytes:
    if not file_name.startswith("@model_path/weights/"):
        raise DepalettizeError(
            f"{label} blob file {file_name!r} is not under the package "
            "weights directory"
        )
    path = weights_dir / file_name.split("/")[-1]
    if not path.is_file():
        raise DepalettizeError(f"{label} blob file {path} does not exist")
    with path.open("rb") as handle:
        metadata = read_blob_metadata(handle, offset)
        handle.seek(metadata.data_offset)
        payload = handle.read(metadata.size_bytes)
    if len(payload) != metadata.size_bytes:
        raise DepalettizeError(
            f"{label} blob at {offset} is truncated: "
            f"{len(payload)} of {metadata.size_bytes} bytes"
        )
    return payload


def _gather_dense(
    indices_payload: bytes,
    indices_dtype: int,
    indices_shape: tuple[int, ...],
    lut_payload: bytes,
    lut_dtype: int,
    lut_shape: tuple[int, ...],
) -> bytes:
    """Byte-exact ``constexpr_lut_to_dense`` materialization (scalar palettes)."""
    nbits = _NBITS_BY_MIL.get(indices_dtype)
    if nbits is None:
        raise DepalettizeError(
            f"indices dtype {_MIL_NAME.get(indices_dtype, indices_dtype)} "
            "is not a packed unsigned integer type"
        )
    width = {10: 2, 11: 4}.get(lut_dtype)
    if width is None:
        raise DepalettizeError(
            f"lut dtype {_MIL_NAME.get(lut_dtype, lut_dtype)} is not fp16/fp32"
        )
    if len(lut_shape) != len(indices_shape) + 2:
        raise DepalettizeError(
            f"lut rank {len(lut_shape)} must be indices rank "
            f"{len(indices_shape)} + 2"
        )
    num_palettes = lut_shape[-2]
    if num_palettes != 1 << nbits:
        raise DepalettizeError(
            f"lut holds {num_palettes} palettes but uint{nbits} indices "
            f"address {1 << nbits}"
        )
    if lut_shape[-1] != 1:
        raise DepalettizeError(
            f"vector palettization (vector size {lut_shape[-1]}) is not "
            "supported by the frontend expansion"
        )
    element_count = int(np.prod(indices_shape)) if indices_shape else 1
    lut_count = int(np.prod(lut_shape))
    if lut_count * width != len(lut_payload):
        raise DepalettizeError(
            f"lut payload is {len(lut_payload)} bytes, shape {lut_shape} "
            f"needs {lut_count * width}"
        )

    indices = unpack_packed_uints(indices_payload, nbits, element_count)
    index_bits = element_count * nbits
    payload_bits = len(indices_payload) * 8
    if index_bits > payload_bits or payload_bits - index_bits >= nbits:
        raise DepalettizeError(
            f"indices payload is {len(indices_payload)} bytes for "
            f"{element_count} uint{nbits} elements"
        )

    dtype = "<u2" if width == 2 else "<u4"
    lut = np.frombuffer(lut_payload, dtype=dtype)
    flat = indices
    if indices_shape:
        # Contiguous per-axis palette blocks: axis a uses palette index
        # i_a // (shape[a] / lut_shape[a]) — np.repeat tile semantics.
        palette_index = np.zeros(indices_shape, dtype=np.int64)
        stride = num_palettes
        for axis in reversed(range(len(indices_shape))):
            block = indices_shape[axis] // lut_shape[axis]
            coordinates = np.arange(indices_shape[axis]) // block
            lifted = coordinates.reshape(
                [-1 if a == axis else 1 for a in range(len(indices_shape))]
            )
            palette_index += np.broadcast_to(lifted, indices_shape) * stride
            stride *= lut_shape[axis]
        flat = palette_index.reshape(-1) + indices
    return lut[flat].tobytes()


def expand_lut_constants(
    spec, weights_dir: Path, expansion_path: Path
) -> list[ExpandedConst]:
    """Replace every ``constexpr_lut_to_dense`` with a dense ``const`` op.

    ``spec`` (a loaded ``Model`` message) is mutated in place. Expanded
    dense tensors are appended to ``expansion_path`` as MIL blob storage;
    the rewritten consts reference it as
    ``@model_path/weights/<expansion_path.name>``.
    """
    weights_dir = Path(weights_dir)
    expansion_path = Path(expansion_path)
    appender = _BlobAppender(expansion_path)
    expanded: list[ExpandedConst] = []
    try:
        for block in _walk_blocks(spec):
            for op in block.operations:
                if op.type != "constexpr_lut_to_dense":
                    continue
                extra = set(op.inputs) - {"indices", "lut"}
                if extra:
                    raise DepalettizeError(
                        f"constexpr_lut_to_dense has unsupported inputs "
                        f"{sorted(extra)}"
                    )
                (indices_dtype, indices_shape, indices_file,
                 indices_offset) = _blob_backed_tensor(
                    op.inputs["indices"].arguments[0], "indices"
                )
                lut_dtype, lut_shape, lut_file, lut_offset = _blob_backed_tensor(
                    op.inputs["lut"].arguments[0], "lut"
                )
                for axis, (i_dim, l_dim) in enumerate(
                    zip(indices_shape, lut_shape)
                ):
                    if i_dim % l_dim:
                        raise DepalettizeError(
                            f"indices dim {axis} ({i_dim}) is not divisible "
                            f"by lut dim ({l_dim})"
                        )
                dense = _gather_dense(
                    _read_blob(weights_dir, indices_file, indices_offset,
                               "indices"),
                    indices_dtype,
                    indices_shape,
                    _read_blob(weights_dir, lut_file, lut_offset, "lut"),
                    lut_dtype,
                    lut_shape,
                )
                blob_offset = appender.append(
                    dense, BLOB_DTYPE_BY_MIL[lut_dtype]
                )

                output = op.outputs[0]
                op.type = "const"
                op.ClearField("inputs")
                op.ClearField("blocks")
                op.ClearField("attributes")
                value = op.attributes["val"]
                tensor_type = value.type.tensorType
                tensor_type.dataType = lut_dtype
                tensor_type.rank = len(indices_shape)
                tensor_type.ClearField("dimensions")
                for dim in indices_shape:
                    tensor_type.dimensions.add().constant.size = dim
                value.blobFileValue.fileName = (
                    f"@model_path/weights/{expansion_path.name}"
                )
                value.blobFileValue.offset = blob_offset
                expanded.append(
                    ExpandedConst(
                        output_name=output.name,
                        shape=indices_shape,
                        dtype_name=_MIL_NAME[lut_dtype],
                        element_count=(int(np.prod(indices_shape))
                                       if indices_shape else 1),
                        blob_offset=blob_offset,
                        byte_count=len(dense),
                    )
                )
    finally:
        appender.close()
    return expanded


def _walk_blocks(spec):
    for function in spec.mlProgram.functions.values():
        for block in function.block_specializations.values():
            yield from _walk_block(block)


def _walk_block(block):
    yield block
    for op in block.operations:
        for nested in op.blocks:
            yield from _walk_block(nested)
