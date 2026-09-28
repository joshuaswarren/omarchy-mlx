# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Pinned Parakeet waveform-to-mel preprocessing on MLX Vulkan.

Derived from ``mweinbach/parakeet-coreml-swift`` at
``75aec2a1c991319657ff4dec5f602c12da6c5012`` (Apache-2.0), specifically
``MelFeatureExtractor.swift`` and ``MelFilterBank.swift``.

Waveform-dependent arithmetic is dispatched only through ``mx.fast.metal_kernel``.
The runtime extraction path contains no NumPy or CPU tensor arithmetic.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import importlib.metadata
import json
import sys
from array import array
from functools import cache
from pathlib import Path
from typing import NamedTuple

from vulkan_mel_constants import (
    DATA_B64,
    DATA_SHA256,
    DFT_TABLE_COUNT,
    DFT_TABLE_OFFSET,
    FILTERBANK_OFFSET,
    FLOAT_COUNT,
    HANN_OFFSET,
    UNTANGLE_COS_OFFSET,
    UNTANGLE_SIN_OFFSET,
)

CHUNK_SAMPLES = 480_000
N_FRAMES = 3_001
N_FFT = 512
N_BINS = 257
N_MELS = 128
ENCODER_FRAMES = 3_000


class _Constants(NamedTuple):
    hann: object
    dft: object
    untangle_cos: object
    untangle_sin: object
    filterbank: object


class VulkanMelResult(NamedTuple):
    mel: object
    mask: object
    encoder_features: object
    encoder_mask: object
    stages: dict[str, object]


class _TraceSnapshot(ctypes.Structure):
    _fields_ = [
        ("gpu_primitive_dispatches", ctypes.c_uint64),
        ("vk_submissions", ctypes.c_uint64),
        ("vk_buffer_copies", ctypes.c_uint64),
        ("vk_buffer_fills", ctypes.c_uint64),
        ("vk_compute_dispatches", ctypes.c_uint64),
        ("omarchy_finalize_calls", ctypes.c_uint64),
        ("commit_calls_with_work", ctypes.c_uint64),
        ("commit_calls_noop", ctypes.c_uint64),
    ]


def _mlx():
    import mlx.core as mx

    return mx


@cache
def _constant_floats() -> array:
    raw = base64.b64decode(DATA_B64, validate=True)
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise RuntimeError("pinned Parakeet Vulkan constants failed SHA-256 verification")
    values = array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    if len(values) != FLOAT_COUNT:
        raise RuntimeError("pinned Parakeet Vulkan constants have the wrong length")
    return values


@cache
def _constant_arrays(mx=None) -> _Constants:
    mx = mx or _mlx()
    values = _constant_floats()
    dft_end = DFT_TABLE_OFFSET + DFT_TABLE_COUNT
    return _Constants(
        hann=mx.array(values[HANN_OFFSET:HANN_OFFSET + 400], dtype=mx.float32),
        dft=mx.array(values[DFT_TABLE_OFFSET:dft_end], dtype=mx.float32),
        untangle_cos=mx.array(
            values[UNTANGLE_COS_OFFSET:UNTANGLE_COS_OFFSET + 128],
            dtype=mx.float32,
        ),
        untangle_sin=mx.array(
            values[UNTANGLE_SIN_OFFSET:UNTANGLE_SIN_OFFSET + 128],
            dtype=mx.float32,
        ),
        filterbank=mx.array(
            values[FILTERBANK_OFFSET:FILTERBANK_OFFSET + N_MELS * N_BINS],
            dtype=mx.float32,
        ).reshape(N_MELS, N_BINS),
    )


def _call(kernel, inputs, output_shapes, output_dtypes, grid, threadgroup):
    return kernel(
        inputs=inputs,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        grid=grid,
        threadgroup=threadgroup,
        stream=_mlx().gpu,
    )


# Evidence-only: the runtime windowing kernel applies preemphasis inline
# (fewer pipelines). This standalone kernel exists so capture-stages
# qualification can still materialize the pinned macOS "preemph" stage.
@cache
def _preemphasis_kernel(mx=None):
    mx = mx or _mlx()
    return mx.fast.metal_kernel(
        name="parakeet_preemphasis_f32",
        input_names=["waveform"],
        output_names=["preemph"],
        header=_FMA_HEADER,
        source="""
            uint i = thread_position_in_grid.x;
            uint length = uint(waveform_shape[0]);
            precise float current = i < length ? waveform[i] : 0.0f;
            precise float previous = (i > 0u && i - 1u < length)
                ? waveform[i - 1u] : 0.0f;
            precise float product = mul32(0.97f, previous);
            preemph[i] = i == 0u ? current : sub32(current, product);
        """,
        compile_options={"math_mode": "safe"},
    )


def _preemphasize(waveform):
    mx = _mlx()
    return _call(
        _preemphasis_kernel(mx),
        [waveform],
        [(CHUNK_SAMPLES,)],
        [mx.float32],
        (CHUNK_SAMPLES, 1, 1),
        (256, 1, 1),
    )[0]


@cache
def _frames_kernel(mx=None):
    mx = mx or _mlx()
    return mx.fast.metal_kernel(
        name="parakeet_frames_preemph_f32",
        input_names=["waveform", "hann"],
        output_names=["frames"],
        header=_FMA_HEADER,
        source="""
            uint index = thread_position_in_grid.x;
            uint frame = index / 512u;
            uint fft_index = index - frame * 512u;
            precise float value = 0.0f;
            if (fft_index >= 56u && fft_index < 456u) {
                uint window_index = fft_index - 56u;
                int pre_index = int(frame * 160u + window_index) - 256;
                if (pre_index >= 0 && pre_index < 480000) {
                    uint length = uint(waveform_shape[0]);
                    precise float current = uint(pre_index) < length
                        ? waveform[pre_index] : 0.0f;
                    precise float previous =
                        (pre_index > 0 && uint(pre_index - 1) < length)
                            ? waveform[uint(pre_index - 1)] : 0.0f;
                    precise float product = mul32(0.97f, previous);
                    precise float emphasized = pre_index == 0
                        ? current : sub32(current, product);
                    value = mul32(emphasized, hann[window_index]);
                }
            }
            frames[index] = value;
        """,
        compile_options={"math_mode": "safe"},
    )


def _frames(waveform, hann):
    mx = _mlx()
    return _call(
        _frames_kernel(mx),
        [waveform, hann],
        [(N_FRAMES, N_FFT)],
        [mx.float32],
        (N_FRAMES * N_FFT, 1, 1),
        (256, 1, 1),
    )[0]


_FMA_HEADER = """
struct Big96 {
    uint w0;
    uint w1;
    uint w2;
    uint sticky;
};

Big96 zero_big96() {
    Big96 value;
    value.w0 = 0u;
    value.w1 = 0u;
    value.w2 = 0u;
    value.sticky = 0u;
    return value;
}

int msb32(uint value) {
    int bit = 0;
    if (value >= 0x10000u) { value >>= 16u; bit += 16; }
    if (value >= 0x100u) { value >>= 8u; bit += 8; }
    if (value >= 0x10u) { value >>= 4u; bit += 4; }
    if (value >= 0x4u) { value >>= 2u; bit += 2; }
    if (value >= 0x2u) bit += 1;
    return bit;
}

uint low_mask(uint count) {
    return count == 0u ? 0u : 0xffffffffu >> (32u - count);
}

Big96 or_big_word(Big96 value, int word, uint bits) {
    if (word == 0) value.w0 |= bits;
    else if (word == 1) value.w1 |= bits;
    else if (word == 2) value.w2 |= bits;
    return value;
}

Big96 shifted48(uint low, uint high, int shift) {
    Big96 value = zero_big96();
    if (shift < 0) {
        int distance = -shift;
        if (distance < 32) {
            uint amount = uint(distance);
            value.sticky = (low & low_mask(amount)) != 0u ? 1u : 0u;
            low = (low >> amount) | (high << (32u - amount));
            high >>= amount;
        } else if (distance == 32) {
            value.sticky = low != 0u ? 1u : 0u;
            low = high;
            high = 0u;
        } else if (distance < 48) {
            uint amount = uint(distance - 32);
            value.sticky = (low != 0u || (high & low_mask(amount)) != 0u) ? 1u : 0u;
            low = high >> amount;
            high = 0u;
        } else {
            value.sticky = (low != 0u || high != 0u) ? 1u : 0u;
            low = 0u;
            high = 0u;
        }
        shift = 0;
    }
    int word = shift >> 5;
    uint offset = uint(shift & 31);
    value = or_big_word(value, word, low << offset);
    if (offset != 0u) value = or_big_word(value, word + 1, low >> (32u - offset));
    word = (shift + 32) >> 5;
    offset = uint((shift + 32) & 31);
    value = or_big_word(value, word, high << offset);
    if (offset != 0u) value = or_big_word(value, word + 1, high >> (32u - offset));
    return value;
}

uint big_word(Big96 value, int word) {
    if (word == 0) return value.w0;
    if (word == 1) return value.w1;
    if (word == 2) return value.w2;
    return 0u;
}

int compare_big(Big96 left, Big96 right) {
    if (left.w2 != right.w2) return left.w2 < right.w2 ? -1 : 1;
    if (left.w1 != right.w1) return left.w1 < right.w1 ? -1 : 1;
    if (left.w0 != right.w0) return left.w0 < right.w0 ? -1 : 1;
    if (left.sticky != right.sticky) return left.sticky < right.sticky ? -1 : 1;
    return 0;
}

Big96 add_big(Big96 left, Big96 right) {
    Big96 result;
    result.w0 = left.w0 + right.w0;
    uint carry = result.w0 < left.w0 ? 1u : 0u;
    uint middle = left.w1 + right.w1;
    uint middle_carry = middle < left.w1 ? 1u : 0u;
    result.w1 = middle + carry;
    carry = (middle_carry != 0u || result.w1 < middle) ? 1u : 0u;
    result.w2 = left.w2 + right.w2 + carry;
    result.sticky = left.sticky | right.sticky;
    return result;
}

Big96 subtract_big(Big96 left, Big96 right) {
    Big96 result;
    result.w0 = left.w0 - right.w0;
    uint borrow = left.w0 < right.w0 ? 1u : 0u;
    uint middle_subtrahend = right.w1 + borrow;
    uint middle_overflow = middle_subtrahend < right.w1 ? 1u : 0u;
    result.w1 = left.w1 - middle_subtrahend;
    borrow = (middle_overflow != 0u || left.w1 < middle_subtrahend) ? 1u : 0u;
    result.w2 = left.w2 - right.w2 - borrow;
    result.sticky = left.sticky | right.sticky;
    return result;
}

Big96 decrement_big(Big96 value) {
    uint previous = value.w0;
    value.w0 -= 1u;
    if (previous == 0u) {
        previous = value.w1;
        value.w1 -= 1u;
        if (previous == 0u) value.w2 -= 1u;
    }
    return value;
}

int big_msb(Big96 value) {
    if (value.w2 != 0u) return 64 + msb32(value.w2);
    if (value.w1 != 0u) return 32 + msb32(value.w1);
    if (value.w0 != 0u) return msb32(value.w0);
    return -1;
}

uint extract_big(Big96 value, int shift) {
    if (shift < 0) return value.w0 << uint(-shift);
    if (shift >= 96) return 0u;
    int word = shift >> 5;
    uint offset = uint(shift & 31);
    uint result = big_word(value, word) >> offset;
    if (offset != 0u) result |= big_word(value, word + 1) << (32u - offset);
    return result;
}

bool big_bit(Big96 value, int position) {
    if (position < 0 || position >= 96) return false;
    return ((big_word(value, position >> 5) >> uint(position & 31)) & 1u) != 0u;
}

bool any_big_below(Big96 value, int position) {
    if (value.sticky != 0u) return true;
    if (position <= 0) return false;
    if (position >= 96) return value.w0 != 0u || value.w1 != 0u || value.w2 != 0u;
    int word = position >> 5;
    uint offset = uint(position & 31);
    if (word > 0 && value.w0 != 0u) return true;
    if (word > 1 && value.w1 != 0u) return true;
    return (big_word(value, word) & low_mask(offset)) != 0u;
}

uint round_big96(Big96 magnitude, int top_exponent, uint sign) {
    int highest = big_msb(magnitude);
    if (highest < 0) return sign;
    int exponent = top_exponent - 94 + highest;
    if (exponent > 127) return sign | 0x7f800000u;
    if (exponent >= -126) {
        int cut = highest - 23;
        uint significand = extract_big(magnitude, cut) & 0x00ffffffu;
        bool guard = big_bit(magnitude, cut - 1);
        bool sticky = any_big_below(magnitude, cut - 1);
        if (guard && (sticky || (significand & 1u) != 0u)) significand += 1u;
        if (significand == 0x01000000u) {
            significand >>= 1u;
            exponent += 1;
            if (exponent > 127) return sign | 0x7f800000u;
        }
        return sign | (uint(exponent + 127) << 23u) | (significand & 0x007fffffu);
    }
    int cut = -55 - top_exponent;
    uint significand = extract_big(magnitude, cut);
    bool guard = big_bit(magnitude, cut - 1);
    bool sticky = any_big_below(magnitude, cut - 1);
    if (guard && (sticky || (significand & 1u) != 0u)) significand += 1u;
    if (significand >= 0x00800000u) return sign | 0x00800000u;
    return sign | significand;
}

uint fma32_bits(uint a_bits, uint b_bits, uint c_bits) {
uint a_exp = (a_bits >> 23u) & 0xffu;
uint b_exp = (b_bits >> 23u) & 0xffu;
uint c_exp = (c_bits >> 23u) & 0xffu;
uint a_fraction = a_bits & 0x007fffffu;
uint b_fraction = b_bits & 0x007fffffu;
uint c_fraction = c_bits & 0x007fffffu;
bool a_nan = a_exp == 0xffu && a_fraction != 0u;
bool b_nan = b_exp == 0xffu && b_fraction != 0u;
bool c_nan = c_exp == 0xffu && c_fraction != 0u;
if (a_nan || b_nan || c_nan) return 0x7fc00000u;
bool a_infinite = a_exp == 0xffu;
bool b_infinite = b_exp == 0xffu;
bool c_infinite = c_exp == 0xffu;
bool a_zero = (a_bits & 0x7fffffffu) == 0u;
bool b_zero = (b_bits & 0x7fffffffu) == 0u;
uint product_sign = (a_bits ^ b_bits) & 0x80000000u;
uint c_sign = c_bits & 0x80000000u;
if ((a_infinite && b_zero) || (b_infinite && a_zero)) {
return 0x7fc00000u;
}
if (a_infinite || b_infinite) {
if (c_infinite && product_sign != c_sign) return 0x7fc00000u;
return product_sign | 0x7f800000u;
}
if (c_infinite) return c_bits;

uint a_significand = a_exp == 0u ? a_fraction : a_fraction | 0x00800000u;
uint b_significand = b_exp == 0u ? b_fraction : b_fraction | 0x00800000u;
uint c_significand = c_exp == 0u ? c_fraction : c_fraction | 0x00800000u;
int a_lsb_exponent = a_exp == 0u ? -149 : int(a_exp) - 150;
int b_lsb_exponent = b_exp == 0u ? -149 : int(b_exp) - 150;
int c_lsb_exponent = c_exp == 0u ? -149 : int(c_exp) - 150;

uint a_low = a_significand & 0xffffu;
uint a_high = a_significand >> 16u;
uint b_low = b_significand & 0xffffu;
uint b_high = b_significand >> 16u;
uint product0 = a_low * b_low;
uint product1 = a_low * b_high + a_high * b_low;
uint product_low = product0 + (product1 << 16u);
uint carry = product_low < product0 ? 1u : 0u;
uint product_high = a_high * b_high + (product1 >> 16u) + carry;
bool product_zero = product_low == 0u && product_high == 0u;
int product_lsb_exponent = a_lsb_exponent + b_lsb_exponent;
int product_top_exponent = product_zero ? -10000 : product_lsb_exponent +
(product_high != 0u ? 32 + msb32(product_high) : msb32(product_low));
int c_top_exponent = c_significand == 0u ? -10000 :
c_lsb_exponent + msb32(c_significand);

if (product_zero && c_significand == 0u) {
return product_sign == c_sign ? product_sign : 0u;
}
int top_exponent = product_top_exponent > c_top_exponent ?
product_top_exponent : c_top_exponent;
Big96 product = product_zero ? zero_big96() : shifted48(
product_low, product_high, 94 - (top_exponent - product_lsb_exponent));
Big96 addend = c_significand == 0u ? zero_big96() : shifted48(
c_significand, 0u, 94 - (top_exponent - c_lsb_exponent));
Big96 magnitude;
uint result_sign;
if (product_sign == c_sign) {
magnitude = add_big(product, addend);
result_sign = product_sign;
} else {
int order = compare_big(product, addend);
if (order == 0) return 0u;
Big96 larger = order > 0 ? product : addend;
Big96 smaller = order > 0 ? addend : product;
result_sign = order > 0 ? product_sign : c_sign;
magnitude = subtract_big(larger, smaller);
if (smaller.sticky != 0u && larger.sticky == 0u && big_msb(magnitude) >= 0) {
magnitude = decrement_big(magnitude);
magnitude.sticky = 1u;
}
}
return round_big96(magnitude, top_exponent, result_sign);
}

uint add32_bits(uint a_bits, uint b_bits) {
return fma32_bits(a_bits, 0x3f800000u, b_bits);
}

uint sub32_bits(uint a_bits, uint b_bits) {
return add32_bits(a_bits, b_bits ^ 0x80000000u);
}

uint mul32_bits(uint a_bits, uint b_bits) {
uint a_magnitude = a_bits & 0x7fffffffu;
uint b_magnitude = b_bits & 0x7fffffffu;
if ((a_magnitude == 0u && b_magnitude < 0x7f800000u) ||
(b_magnitude == 0u && a_magnitude < 0x7f800000u)) {
return (a_bits ^ b_bits) & 0x80000000u;
}
return fma32_bits(a_bits, b_bits, 0u);
}

float fma32(float a, float b, float c) {
return uintBitsToFloat(fma32_bits(
floatBitsToUint(a), floatBitsToUint(b), floatBitsToUint(c)));
}

float add32(float a, float b) {
return uintBitsToFloat(add32_bits(floatBitsToUint(a), floatBitsToUint(b)));
}

float sub32(float a, float b) {
return uintBitsToFloat(sub32_bits(floatBitsToUint(a), floatBitsToUint(b)));
}

float mul32(float a, float b) {
    return uintBitsToFloat(mul32_bits(floatBitsToUint(a), floatBitsToUint(b)));
}

float neg32(float value) {
    return uintBitsToFloat(floatBitsToUint(value) ^ 0x80000000u);
}
"""


_LOG_HEADER = _FMA_HEADER + """
struct DoubleFloat {
    float high;
    float low;
};

DoubleFloat two_sum(float a, float b) {
    precise float sum = a + b;
    precise float recovered = sum - a;
    precise float error = (a - (sum - recovered)) + (b - recovered);
    return DoubleFloat(sum, error);
}

DoubleFloat two_product(float a, float b) {
    precise float product = a * b;
    precise float a_split = a * 4097.0f;
    precise float a_high = a_split - (a_split - a);
    precise float a_low = a - a_high;
    precise float b_split = b * 4097.0f;
    precise float b_high = b_split - (b_split - b);
    precise float b_low = b - b_high;
    precise float error =
        ((a_high * b_high - product) + a_high * b_low + a_low * b_high) +
        a_low * b_low;
    return DoubleFloat(product, error);
}

DoubleFloat add_double_float(DoubleFloat a, DoubleFloat b) {
    DoubleFloat leading = two_sum(a.high, b.high);
    precise float trailing = a.low + b.low;
    DoubleFloat combined = two_sum(leading.high, leading.low + trailing);
    return combined;
}

DoubleFloat multiply_double_float(DoubleFloat a, DoubleFloat b) {
    DoubleFloat leading = two_product(a.high, b.high);
    precise float cross = a.high * b.low + a.low * b.high;
    precise float trailing = leading.low + cross + a.low * b.low;
    return two_sum(leading.high, trailing);
}

float log32(float value) {
    uint bits = floatBitsToUint(value);
    int exponent = int((bits >> 23u) & 0xffu) - 127;
    precise float mantissa = uintBitsToFloat((bits & 0x007fffffu) | 0x3f800000u);
    if (mantissa > 1.4142135623730951f) {
        mantissa = 0.5f * mantissa;
        exponent += 1;
    }
    precise float numerator = mantissa - 1.0f;
    DoubleFloat denominator = two_sum(mantissa, 1.0f);
    precise float quotient = numerator / denominator.high;
    DoubleFloat quotient_product = multiply_double_float(
        DoubleFloat(quotient, 0.0f), denominator);
    DoubleFloat quotient_remainder = add_double_float(
        DoubleFloat(numerator, 0.0f),
        DoubleFloat(-quotient_product.high, -quotient_product.low));
    precise float remainder = quotient_remainder.high + quotient_remainder.low;
    DoubleFloat z = two_sum(quotient, remainder / denominator.high);
    DoubleFloat z_squared = multiply_double_float(z, z);
    DoubleFloat polynomial = DoubleFloat(0.05263157933950424f, -3.9213582381236733e-10f);
    polynomial = add_double_float(
        multiply_double_float(polynomial, z_squared),
        DoubleFloat(0.05882352963089943f, -2.1913472425527658e-10f));
    polynomial = add_double_float(
        multiply_double_float(polynomial, z_squared),
        DoubleFloat(0.06666667014360428f, -3.47693762670076e-09f));
    polynomial = add_double_float(
        multiply_double_float(polynomial, z_squared),
        DoubleFloat(0.07692307978868484f, -2.8656079731348427e-09f));
    polynomial = add_double_float(
        multiply_double_float(polynomial, z_squared),
        DoubleFloat(0.09090909361839294f, -2.709302115988521e-09f));
    polynomial = add_double_float(
        multiply_double_float(polynomial, z_squared),
        DoubleFloat(0.1111111119389534f, -8.278422947149977e-10f));
    polynomial = add_double_float(
        multiply_double_float(polynomial, z_squared),
        DoubleFloat(0.1428571492433548f, -6.38621200366174e-09f));
    polynomial = add_double_float(
        multiply_double_float(polynomial, z_squared),
        DoubleFloat(0.20000000298023224f, -2.9802322831784522e-09f));
    polynomial = add_double_float(
        multiply_double_float(polynomial, z_squared),
        DoubleFloat(0.3333333432674408f, -9.934107758624577e-09f));
    polynomial = add_double_float(
        multiply_double_float(polynomial, z_squared),
        DoubleFloat(1.0f, 0.0f));
    DoubleFloat reduced = multiply_double_float(
        DoubleFloat(2.0f, 0.0f), multiply_double_float(z, polynomial));
    precise float exponent_value = float(exponent);
    DoubleFloat exponent_term = two_product(exponent_value, 0.6931471824645996f);
    exponent_term = add_double_float(
        exponent_term,
        DoubleFloat(exponent_value * -1.9046542121259336e-09f, 0.0f));
    DoubleFloat result = add_double_float(exponent_term, reduced);
    precise float rounded = result.high + result.low;
    return rounded;
}
"""

_SQRT_HEADER = _LOG_HEADER + """
int compare_float_to_double(float value, DoubleFloat other) {
precise DoubleFloat delta = add_double_float(
DoubleFloat(value, 0.0f), DoubleFloat(-other.high, -other.low));
precise float difference = delta.high != 0.0f ? delta.high : delta.low;
return difference < 0.0f ? -1 : (difference > 0.0f ? 1 : 0);
}

DoubleFloat square_midpoint(DoubleFloat value) {
DoubleFloat leading = two_product(value.high, value.high);
precise float cross = (value.high * value.low) + (value.high * value.low);
DoubleFloat with_cross = add_double_float(leading, DoubleFloat(cross, 0.0f));
return add_double_float(with_cross, DoubleFloat(value.low * value.low, 0.0f));
}

float sqrt32(float value) {
uint value_bits = floatBitsToUint(value);
if ((value_bits & 0x80000000u) != 0u ||
    (value_bits & 0x7fffffffu) == 0u ||
    (value_bits & 0x7f800000u) == 0x7f800000u) return sqrt(value);
precise float scaled_value = value;
precise float result_scale = 1.0f;
if (value_bits < 0x20000000u) {
scaled_value = mul32(value, 18446744073709551616.0f);
result_scale = 0.00000000023283064365386962890625f;
} else if (value_bits > 0x60000000u) {
scaled_value = mul32(value, 5.4210108624275221700372640043497e-20f);
result_scale = 4294967296.0f;
}
precise float root = sqrt(scaled_value);
for (uint correction = 0u; correction < 8u; ++correction) {
uint root_bits = floatBitsToUint(root);
precise float lower = uintBitsToFloat(root_bits - 1u);
precise float upper = uintBitsToFloat(root_bits + 1u);
precise DoubleFloat lower_midpoint = two_sum(root, -0.5f * (root - lower));
precise DoubleFloat lower_square = square_midpoint(lower_midpoint);
int lower_comparison = compare_float_to_double(scaled_value, lower_square);
if (lower_comparison < 0 || (lower_comparison == 0 && (root_bits & 1u) != 0u)) {
root = lower;
continue;
}
precise DoubleFloat upper_midpoint = two_sum(root, 0.5f * (upper - root));
precise DoubleFloat upper_square = square_midpoint(upper_midpoint);
int upper_comparison = compare_float_to_double(scaled_value, upper_square);
if (upper_comparison > 0 || (upper_comparison == 0 && (root_bits & 1u) != 0u)) {
root = upper;
continue;
}
break;
}
return mul32(root, result_scale);
}
"""


@cache
def _dft_kernel(mx=None):
    mx = mx or _mlx()
    return mx.fast.metal_kernel(
        name="parakeet_vdsp_radix4_dif_f32",
        input_names=["frames", "tables", "untangle_cos", "untangle_sin"],
        output_names=["dft_real", "dft_imag"],
        header=_FMA_HEADER,
        source="""
            threadgroup float real0[512];
            threadgroup float imag0[512];
            threadgroup float real1[512];
            threadgroup float imag1[512];
            uint fid = thread_index_in_threadgroup / 64u;
            uint lane = thread_index_in_threadgroup % 64u;
            uint foff = fid * 256u;
            uint frame = threadgroup_position_in_grid.x * 2u + fid;
            if (frame >= frames_shape[0]) { return; }
            uint frame_base = frame * 512u;
            for (uint part = 0u; part < 4u; ++part) {
                uint packed = lane + part * 64u;
                real0[foff + packed] = frames[frame_base + packed * 2u];
                imag0[foff + packed] = frames[frame_base + packed * 2u + 1u];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stage = 0u; stage < 4u; ++stage) {
                uint prefix = stage == 0u ? 1u : (stage == 1u ? 4u : (stage == 2u ? 16u : 64u));
                uint remaining = 256u / prefix;
                uint next_remaining = remaining / 4u;
                uint group = lane / prefix;
                uint p = lane - group * prefix;
                uint source0 = foff + group * prefix + p;
                uint source1 = foff + (next_remaining + group) * prefix + p;
                uint source2 = foff + (2u * next_remaining + group) * prefix + p;
                uint source3 = foff + (3u * next_remaining + group) * prefix + p;
                precise float xr0;
                precise float xr1;
                precise float xr2;
                precise float xr3;
                precise float xi0;
                precise float xi1;
                precise float xi2;
                precise float xi3;
                if ((stage & 1u) == 0u) {
                    xr0 = real0[source0]; xr1 = real0[source1];
                    xr2 = real0[source2]; xr3 = real0[source3];
                    xi0 = imag0[source0]; xi1 = imag0[source1];
                    xi2 = imag0[source2]; xi3 = imag0[source3];
                } else {
                    xr0 = real1[source0]; xr1 = real1[source1];
                    xr2 = real1[source2]; xr3 = real1[source3];
                    xi0 = imag1[source0]; xi1 = imag1[source1];
                    xi2 = imag1[source2]; xi3 = imag1[source3];
                }

            precise float ac_real = add32(xr0, xr2);
            precise float ac_imag = add32(xi0, xi2);
            precise float ad_real = sub32(xr0, xr2);
            precise float ad_imag = sub32(xi0, xi2);
            precise float bd_real = add32(xr1, xr3);
            precise float bd_imag = add32(xi1, xi3);
            precise float bm_real = sub32(xr1, xr3);
            precise float bm_imag = sub32(xi1, xi3);
            precise float yr0;
            precise float yr1;
            precise float yr2;
            precise float yr3;
            precise float yi0;
            precise float yi1;
            precise float yi2;
            precise float yi3;

            if (stage == 0u) {
                yr0 = add32(ac_real, bd_real);
                yr1 = add32(ad_real, bm_imag);
                yr2 = sub32(ac_real, bd_real);
                yr3 = sub32(ad_real, bm_imag);
                yi0 = add32(ac_imag, bd_imag);
                yi1 = sub32(ad_imag, bm_real);
                yi2 = sub32(ac_imag, bd_imag);
                yi3 = add32(ad_imag, bm_real);
            } else {
                uint table_base = (stage - 1u) * 384u;
                precise float cos1 = tables[table_base + p];
                precise float tan1 = tables[table_base + 64u + p];
                precise float cos2 = tables[table_base + 128u + p];
                precise float tan2 = tables[table_base + 192u + p];
                precise float ratio3 = tables[table_base + 256u + p];
                precise float tan3 = tables[table_base + 320u + p];
                precise float ti1 = fma32(neg32(xr1), tan1, xi1);
                precise float tr1 = fma32(xi1, tan1, xr1);
                precise float ti2 = fma32(neg32(xr2), tan2, xi2);
                precise float tr2 = fma32(xi2, tan2, xr2);
                if (stage == 2u && p == 8u) {
                    ti2 = neg32(xr2);
                    tr2 = xi2;
                }
                precise float ti3 = fma32(neg32(xr3), tan3, xi3);
                precise float tr3 = fma32(xi3, tan3, xr3);
                precise float eip = fma32(ti2, cos2, xi0);
                precise float eim = fma32(neg32(ti2), cos2, xi0);
                precise float erp = fma32(tr2, cos2, xr0);
                precise float erm = fma32(neg32(tr2), cos2, xr0);
                if (stage == 2u && p == 8u) {
                    eip = add32(xi0, ti2); eim = sub32(xi0, ti2);
                    erp = add32(xr0, tr2); erm = sub32(xr0, tr2);
                }
                precise float oip = fma32(ti3, ratio3, ti1);
                precise float oim = fma32(neg32(ti3), ratio3, ti1);
                precise float orp = fma32(tr3, ratio3, tr1);
                precise float orm = fma32(neg32(tr3), ratio3, tr1);
                yr0 = fma32(orp, cos1, erp);
                yr1 = fma32(oim, cos1, erm);
                yr2 = fma32(neg32(orp), cos1, erp);
                yr3 = fma32(neg32(oim), cos1, erm);
                yi0 = fma32(oip, cos1, eip);
                yi1 = fma32(neg32(orm), cos1, eim);
                yi2 = fma32(neg32(oip), cos1, eip);
                yi3 = fma32(orm, cos1, eim);
            }

                uint target0 = foff + (group * 4u) * prefix + p;
                uint target1 = target0 + prefix;
                uint target2 = target1 + prefix;
                uint target3 = target2 + prefix;
                if ((stage & 1u) == 0u) {
                    real1[target0] = yr0; real1[target1] = yr1;
                    real1[target2] = yr2; real1[target3] = yr3;
                    imag1[target0] = yi0; imag1[target1] = yi1;
                    imag1[target2] = yi2; imag1[target3] = yi3;
                } else {
                    real0[target0] = yr0; real0[target1] = yr1;
                    real0[target2] = yr2; real0[target3] = yr3;
                    imag0[target0] = yi0; imag0[target1] = yi1;
                    imag0[target2] = yi2; imag0[target3] = yi3;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            uint output_base = frame * 257u;
            if (lane == 0u) {
            precise float twice_real = add32(real0[foff], real0[foff]);
            precise float twice_imag = add32(imag0[foff], imag0[foff]);
            dft_real[output_base] = mul32(0.5f, add32(twice_real, twice_imag));
            dft_real[output_base + 256u] = mul32(0.5f, sub32(twice_real, twice_imag));
            dft_imag[output_base] = 0.0f;
            dft_imag[output_base + 256u] = 0.0f;
        }
        for (uint low = lane + 1u; low <= 128u; low += 64u) {
            uint mirrored = 256u - low;
            precise float direct_real = real0[foff + low];
            precise float direct_imag = imag0[foff + low];
            precise float mirror_real = real0[foff + mirrored];
            precise float mirror_imag = imag0[foff + mirrored];
            precise float imag_sum = add32(direct_imag, mirror_imag);
            precise float real_diff = sub32(mirror_real, direct_real);
            precise float real_sum = add32(direct_real, mirror_real);
            precise float imag_diff = sub32(direct_imag, mirror_imag);
            precise float weighted_real_a = mul32(untangle_cos[low - 1u], imag_sum);
            precise float weighted_real_b = mul32(untangle_sin[low - 1u], real_diff);
            precise float weighted_real = add32(weighted_real_a, weighted_real_b);
            precise float weighted_imag_a = mul32(untangle_cos[low - 1u], real_diff);
            precise float weighted_imag_b = mul32(untangle_sin[low - 1u], imag_sum);
            precise float weighted_imag = sub32(weighted_imag_a, weighted_imag_b);
            dft_real[output_base + low] = mul32(0.5f, add32(real_sum, weighted_real));
            dft_imag[output_base + low] = mul32(0.5f, add32(weighted_imag, imag_diff));
            dft_real[output_base + mirrored] = mul32(0.5f, sub32(real_sum, weighted_real));
            dft_imag[output_base + mirrored] = mul32(0.5f, sub32(weighted_imag, imag_diff));
            }
        """,
        compile_options={"math_mode": "safe"},
    )


def _dft_frames(frames):
    mx = _mlx()
    if frames.dtype != mx.float32:
        raise ValueError("DFT frames must have float32 dtype")
    if frames.ndim != 2:
        raise ValueError("DFT frames must be two-dimensional")
    if frames.shape[1] != N_FFT:
        raise ValueError("DFT frames must have exactly 512 samples")
    constants = _constant_arrays(mx)
    return tuple(_call(
        _dft_kernel(mx),
        [frames, constants.dft, constants.untangle_cos, constants.untangle_sin],
        [(frames.shape[0], N_BINS), (frames.shape[0], N_BINS)],
        [mx.float32, mx.float32],
        ((frames.shape[0] + 1) // 2 * 128, 1, 1),
        (128, 1, 1),
    ))


@cache
def _power_kernel(mx=None):
    mx = mx or _mlx()
    return mx.fast.metal_kernel(
        name="parakeet_power_f32",
        input_names=["dft_real", "dft_imag"],
        output_names=["power"],
        header=_SQRT_HEADER,
        source="""
            uint i = thread_position_in_grid.x;
            precise float re2 = mul32(dft_real[i], dft_real[i]);
            precise float im2 = mul32(dft_imag[i], dft_imag[i]);
            precise float magnitude = sqrt32(add32(re2, im2));
            power[i] = mul32(magnitude, magnitude);
        """,
        compile_options={"math_mode": "safe"},
    )


def _power(dft_real, dft_imag):
    mx = _mlx()
    size = dft_real.shape[0] * N_BINS
    return _call(
        _power_kernel(mx),
        [dft_real, dft_imag],
        [dft_real.shape],
        [mx.float32],
        (size, 1, 1),
        (256, 1, 1),
    )[0]


_MEL_PACKED_ROW = 12


@cache
def _mel_packed_indices(mx=None):
    """Per-mel nonzero filterbank bin indices in the kernel's visit order.

    The pinned filterbank has at most 12 nonzeros per mel row (mean 3.9), so
    the dot products only need the nonzero terms. Rows are padded to a fixed
    length with zero-weight bins: the emulated fma32 contract returns the
    accumulator unchanged for a zero first operand, so padding and skipping
    exact-zero terms are both bit-identical to the dense accumulation.
    """
    mx = mx or _mlx()
    values = _constant_floats()
    fb_base = FILTERBANK_OFFSET
    rel_of_acc = (8, 12, 16, 20, 24, 28, 4, 0)
    visit_order = [
        block + rel_of_acc[accumulator] + lane
        for block in range(0, 256, 32)
        for accumulator in range(8)
        for lane in range(4)
    ]
    packed = []
    for mel in range(N_MELS):
        row_base = fb_base + mel * N_BINS
        nonzero = [s for s in visit_order if values[row_base + s] != 0.0]
        pad = next(s for s in visit_order if values[row_base + s] == 0.0)
        nonzero += [pad] * (_MEL_PACKED_ROW - len(nonzero))
        packed.extend(nonzero)
    return mx.array(packed, dtype=mx.uint32)


@cache
def _mel_kernel(mx=None):
    mx = mx or _mlx()
    return mx.fast.metal_kernel(
        name="parakeet_mel_dot_log_f32",
        input_names=["filterbank", "power", "packed"],
        output_names=["melproj", "logmel"],
        header=_LOG_HEADER,
        source="""
            uint index = thread_position_in_grid.x;
            uint frame = index / 128u;
            uint mel = index - frame * 128u;
            uint fb_base = mel * 257u;
            uint power_base = frame * 257u;
            uint packed_base = mel * 12u;
            precise float accum[32];
            for (uint i = 0u; i < 32u; ++i) accum[i] = 0.0f;
            for (uint k = 0u; k < 12u; ++k) {
                uint source = packed[packed_base + k];
                uint offset = source & 31u;
                uint relative = offset >> 2u;
                uint accumulator = relative < 2u
                    ? (7u - relative) : (relative - 2u);
                uint slot = accumulator * 4u + (offset & 3u);
                accum[slot] = fma32(
                    filterbank[fb_base + source],
                    power[power_base + source],
                    accum[slot]);
            }
            precise float first0 = add32(accum[0], accum[4]);
            precise float first1 = add32(accum[1], accum[5]);
            precise float first2 = add32(accum[2], accum[6]);
            precise float first3 = add32(accum[3], accum[7]);
            precise float second0 = add32(accum[8], accum[12]);
            precise float second1 = add32(accum[9], accum[13]);
            precise float second2 = add32(accum[10], accum[14]);
            precise float second3 = add32(accum[11], accum[15]);
            precise float third0 = add32(accum[16], accum[20]);
            precise float third1 = add32(accum[17], accum[21]);
            precise float third2 = add32(accum[18], accum[22]);
            precise float third3 = add32(accum[19], accum[23]);
            precise float fourth0 = add32(accum[24], accum[28]);
            precise float fourth1 = add32(accum[25], accum[29]);
            precise float fourth2 = add32(accum[26], accum[30]);
            precise float fourth3 = add32(accum[27], accum[31]);
            first0 = add32(first0, second0); first1 = add32(first1, second1);
            first2 = add32(first2, second2); first3 = add32(first3, second3);
            second0 = add32(third0, fourth0); second1 = add32(third1, fourth1);
            second2 = add32(third2, fourth2); second3 = add32(third3, fourth3);
            first0 = add32(first0, second0); first1 = add32(first1, second1);
            first2 = add32(first2, second2); first3 = add32(first3, second3);
            precise float pair0 = add32(first0, first1);
            precise float pair1 = add32(first2, first3);
            precise float projected = add32(pair0, pair1);
            projected = fma32(
                filterbank[fb_base + 256u],
                power[power_base + 256u],
                projected);
            melproj[index] = projected;
            precise float guarded = add32(projected, 0.000000059604644775390625f);
            logmel[index] = log32(guarded);
        """,
        compile_options={"math_mode": "safe"},
    )


def _mel_project(power, filterbank):
    mx = _mlx()
    return tuple(_call(
        _mel_kernel(mx),
        [filterbank, power, _mel_packed_indices(mx)],
        [(power.shape[0], N_MELS), (power.shape[0], N_MELS)],
        [mx.float32, mx.float32],
        (power.shape[0] * N_MELS, 1, 1),
        (256, 1, 1),
    ))


_PRECISE_DIV_HEADER = """
float precise_div(float a, float b) {
    float q = a / b;
    if (b == 0.0f || isnan(q) || isinf(q) || q == 0.0f ||
        abs(b) < 1.1754944e-38f || abs(b) > 2.8e30f ||
        abs(q) > 2.8e30f) {
        return q;
    }
    precise float p = q * b;
    if (isnan(p) || isinf(p)) {
        return q;
    }
    precise float bsplit = b * 8193.0f;
    precise float b_hi = bsplit - (bsplit - b);
    precise float b_lo = b - b_hi;
    precise float qsplit = q * 8193.0f;
    precise float q_hi = qsplit - (qsplit - q);
    precise float q_lo = q - q_hi;
    precise float err =
        ((q_hi * b_hi - p) + q_hi * b_lo + q_lo * b_hi) + q_lo * b_lo;
    precise float r = (a - p) - err;
    precise float correction = r * (1.0f / b);
    return q + correction;
}
"""


@cache
def _stats_kernel(mx=None):
    mx = mx or _mlx()
    return mx.fast.metal_kernel(
        name="parakeet_mean_std_f32",
        input_names=["logmel"],
        output_names=["mean", "std"],
        header=_SQRT_HEADER + _PRECISE_DIV_HEADER,
        source="""
            uint mel = thread_position_in_grid.x;
            precise float sum = 0.0f;
            for (uint frame = 0u; frame < 3001u; ++frame)
                sum = sum + logmel[frame * 128u + mel];
            precise float average = precise_div(sum, 3001.0f);
            precise float squared = 0.0f;
            for (uint frame = 0u; frame < 3001u; ++frame) {
                precise float deviation = logmel[frame * 128u + mel] - average;
                precise float term = deviation * deviation;
                squared = squared + term;
            }
            mean[mel] = average;
            std[mel] = sqrt32(precise_div(squared, 3000.0f));
        """,
        compile_options={"math_mode": "safe"},
    )


def _statistics(logmel):
    mx = _mlx()
    return tuple(_call(
        _stats_kernel(mx),
        [logmel],
        [(N_MELS,), (N_MELS,)],
        [mx.float32, mx.float32],
        (N_MELS, 1, 1),
        (128, 1, 1),
    ))


@cache
def _normalize_kernel(mx=None):
    mx = mx or _mlx()
    return mx.fast.metal_kernel(
        name="parakeet_normalize_mask_f32",
        input_names=["logmel", "mean", "std"],
        output_names=["mel", "mask"],
        header=_PRECISE_DIV_HEADER,
        source="""
            uint index = thread_position_in_grid.x;
            uint frame = index / 128u;
            uint bin = index - frame * 128u;
            precise float numerator = logmel[index] - mean[bin];
            precise float denominator = std[bin] + 0.00001f;
            mel[index] = precise_div(numerator, denominator);
            if (bin == 0u) mask[frame] = 1;
        """,
        compile_options={"math_mode": "safe"},
    )


def _normalize(logmel, mean, std):
    mx = _mlx()
    return tuple(_call(
        _normalize_kernel(mx),
        [logmel, mean, std],
        [(N_FRAMES, N_MELS), (N_FRAMES,)],
        [mx.float32, mx.int32],
        (N_FRAMES * N_MELS, 1, 1),
        (256, 1, 1),
    ))


def _validate_waveform(waveform, mx) -> None:
    if waveform.dtype != mx.float32:
        raise ValueError("waveform must have float32 dtype")
    if waveform.ndim != 1:
        raise ValueError("waveform must be one-dimensional")
    if waveform.size == 0:
        raise ValueError("waveform must be non-empty")
    if waveform.size > CHUNK_SAMPLES:
        raise ValueError("waveform must contain at most 480000 samples")


def extract_chunk_features(waveform, *, capture_stages: bool = False) -> VulkanMelResult:
    """Return pinned mel, masks, and the encoder slice as MLX GPU arrays."""
    mx = _mlx()
    _validate_waveform(waveform, mx)
    constants = _constant_arrays(mx)
    frames = _frames(waveform, constants.hann)
    dft_real, dft_imag = _dft_frames(frames)
    power = _power(dft_real, dft_imag)
    melproj, logmel = _mel_project(power, constants.filterbank)
    mean, std = _statistics(logmel)
    mel, mask = _normalize(logmel, mean, std)
    encoder_features = mel[:ENCODER_FRAMES].reshape(1, ENCODER_FRAMES, N_MELS)
    encoder_mask = mask[:ENCODER_FRAMES].reshape(1, ENCODER_FRAMES)
    stages = {}
    if capture_stages:
        preemph = _preemphasize(waveform)
        mx.eval(preemph)
        stages = {
            "preemph": preemph,
            "hann": constants.hann,
            "mel_fb": constants.filterbank,
            "frames": frames,
            "dft_real": dft_real,
            "dft_imag": dft_imag,
            "power": power,
            "melproj": melproj,
            "logmel": logmel,
            "mean": mean,
            "std": std,
        }
    return VulkanMelResult(mel, mask, encoder_features, encoder_mask, stages)


@cache
def _trace_function():
    distribution = importlib.metadata.distribution("mlx-omarchy")
    library_path = distribution.locate_file("mlx/lib/libmlx.so")
    library = ctypes.CDLL(str(library_path))
    function = library.mlx_omarchy_trace_snapshot
    function.argtypes = [ctypes.POINTER(_TraceSnapshot)]
    function.restype = None
    return function


def trace_snapshot() -> dict[str, int]:
    snapshot = _TraceSnapshot()
    _trace_function()(ctypes.byref(snapshot))
    return {name: int(getattr(snapshot, name)) for name, _ in snapshot._fields_}

def _comparison_stages(result, np):
    mask = np.asarray(result.mask)
    if mask.dtype != np.int32:
        raise ValueError(f"runtime mel mask must have int32 dtype, got {mask.dtype}")
    if mask.shape != (N_FRAMES,):
        raise ValueError(
            f"runtime mel mask must have shape ({N_FRAMES},), got {mask.shape}"
        )
    return {
        **result.stages,
        "mel_mask": mask.astype(np.float32),
        "mel_pinned": result.mel,
        "mel_stepwise": result.mel,
    }


def _qualification_status(comparisons, stage_names, expected_stage_names, deltas):
    stage_set_exact = set(stage_names) == set(expected_stage_names)
    all_bit_exact = len(comparisons) == len(expected_stage_names) + 4 and all(
        item["bit_exact"] for item in comparisons.values()
    )
    gpu_execution = {
        "gpu_primitive_dispatches": deltas["gpu_primitive_dispatches"] > 0,
        # 6 runtime kernels (preemph folded into frames, magnitude into
        # power) + 1 evidence-only preemph dispatch in capture mode.
        "vk_compute_dispatches": deltas["vk_compute_dispatches"] == 7,
        "vk_submissions": deltas["vk_submissions"] > 0,
    }
    return {
        "all_bit_exact": all_bit_exact,
        "stage_set_exact": stage_set_exact,
        "gpu_execution": gpu_execution,
        "qualified": all_bit_exact and stage_set_exact and all(gpu_execution.values()),
    }

def _compare_fixture(capture_dir: Path, stage_dir: Path) -> dict:
    import numpy as np

    from mel_stage_compare import STAGE_NAMES, verify_stage_evidence
    from reference import ReferenceLock

    lock = ReferenceLock.load(Path(__file__).with_name("parakeet-reference.lock"))
    verify_stage_evidence(capture_dir, stage_dir, lock)
    mx = _mlx()
    waveform = mx.load(str(capture_dir / "waveform.npy"))
    before = trace_snapshot()
    result = extract_chunk_features(waveform, capture_stages=True)
    mx.eval(result.mel, result.mask, result.encoder_features, result.encoder_mask)
    computed_stages = _comparison_stages(result, np)
    comparisons = {}
    for name in STAGE_NAMES:
        actual_mx = computed_stages.get(name)
        if actual_mx is None:
            comparisons[name] = {"missing": True, "bit_exact": False}
        else:
            actual = np.asarray(actual_mx)
            expected = np.load(stage_dir / f"{name}.npy", allow_pickle=False)
            comparisons[name] = {
                "shape": list(actual.shape),
                "dtype": str(actual.dtype),
                "bit_exact": actual.shape == expected.shape
                and actual.dtype == expected.dtype
                and actual.tobytes() == expected.tobytes(),
            }
    finals = {
        "mel": (result.mel, capture_dir / "mel.npy"),
        "mask": (result.mask, capture_dir / "mel_mask.npy"),
        "encoder_features": (
            result.encoder_features,
            capture_dir / "encoder_input_features.npy",
        ),
        "encoder_mask": (
            result.encoder_mask,
            capture_dir / "encoder_input_mask.npy",
        ),
    }
    for name, (actual_mx, path) in finals.items():
        actual = np.asarray(actual_mx)
        expected = np.load(path, allow_pickle=False)
        comparisons[name] = {
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "bit_exact": actual.shape == expected.shape
            and actual.dtype == expected.dtype
            and actual.tobytes() == expected.tobytes(),
        }
    after = trace_snapshot()
    deltas = {name: after[name] - before[name] for name in before}
    status = _qualification_status(
        comparisons, computed_stages, STAGE_NAMES, deltas
    )
    return {
        "schema": "mlx-omarchy.parakeet-vulkan-mel/1",
        "reference_model": lock.model_repo,
        "reference_revision": lock.model_revision,
        "device": str(mx.default_device()),
        "comparisons": comparisons,
        **status,
        "trace_delta": deltas,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("stage_dir", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    report = _compare_fixture(args.capture_dir, args.stage_dir)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
