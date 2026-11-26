"""Hadamard-packed ternary modules for Bonsai-2 packs.

Reviewed in-repo port of the pinned upstream loader
`runtime/runtime.py` in prism-ml/Ternary-Bonsai-2-27B-mlx-2bit
@ 3f926b415992eaa2ae9dd7b573706494d6bbf787 (Apache-2.0; NOTICE asks
attribution: "Created using Bonsai by Prism ML"). The pack's runtime
directory is remote code and is never imported; this port exists so the
Hadamard/packing semantics are reviewable in this repository.

Fixed semantics (do not weaken):

- Linear layers: forward Hadamard transform on the last activation
  dimension (float32, explicit +/-1 sign flip before, 1/sqrt(block)
  scale), then `mx.quantized_matmul` directly against the packed U32
  weights with F16 scales/biases (affine, bits=2, group_size=128).
  Weights are never materialized as a float matrix.
- The single inverse-transform module (token embedding): row gather,
  `mx.dequantize` per gathered row, then inverse Hadamard (sign flip
  after the transform).
- Activations are float16; Hadamard blocks are restricted to the
  pack-validated set and must divide every transformed width.
"""

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn

ALLOWED_BLOCKS = (512, 1024, 2048, 4096)
GROUP_SIZE = 128
BITS = 2


class PackedError(ValueError):
    """A packed module or its record violates the pack contract."""


def fwht(x: mx.array, block: int, signs: mx.array, inverse: bool = False) -> mx.array:
    """Sign-flip + normalized Walsh-Hadamard transform on the last dim."""
    shape, dtype = x.shape, x.dtype
    if shape[-1] % block:
        raise PackedError("Hadamard block does not divide activation width")
    x = x.astype(mx.float32)
    if not inverse:
        x = x * signs
    x = mx.hadamard_transform(x.reshape(-1, block), scale=1 / math.sqrt(block)).reshape(
        shape
    )
    if inverse:
        x = x * signs
    return x.astype(dtype)


class Packed(nn.Module):
    """A quantized Linear/Embedding with an optional Hadamard activation fold."""

    def __init__(self, arrays, block=0, signs=None, embedding=False, dtype=mx.float16):
        super().__init__()
        self.weight, self.scales, self.biases = [mx.array(a) for a in arrays]
        if block:
            if block not in ALLOWED_BLOCKS:
                raise PackedError(f"Unsupported Hadamard block size {block}")
            if signs is None:
                raise PackedError("Transformed module is missing its sign vector")
            if not mx.all((signs == 1) | (signs == -1)).item():
                raise PackedError("Sign vectors must be +/-1")
        elif signs is not None:
            raise PackedError("Untransformed module carries an unexpected sign vector")
        self.block, self.signs, self.embedding, self.dtype = (
            block,
            signs,
            embedding,
            dtype,
        )

    def __call__(self, x):
        if self.embedding:
            shape = x.shape
            indices = x.reshape(-1)
            out = (
                mx.dequantize(
                    self.weight[indices],
                    self.scales[indices],
                    self.biases[indices],
                    group_size=GROUP_SIZE,
                    bits=BITS,
                )
                .reshape(*shape, -1)
                .astype(self.dtype)
            )
            return (
                fwht(out, self.block, self.signs, inverse=True) if self.block else out
            )
        if self.block:
            x = fwht(x, self.block, self.signs)
        return mx.quantized_matmul(
            x,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=GROUP_SIZE,
            bits=BITS,
        )
