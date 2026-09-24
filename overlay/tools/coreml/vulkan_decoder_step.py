# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Queued multi-dispatch GPU execution of the pinned Parakeet TDT decoder step.

The per-emission decoder step runs as SIX small custom-kernel dispatches
queued back-to-back on the Omarchy Vulkan stream with ONE host sync at the
end: layer-0 block chains, layer-0 fold+cell, layer-1 block chains,
layer-1 fold+cell, projector+relu, joint. The argmax reductions run
host-side on the returned logits. Decoder state never leaves the GPU
between calls. The joint-only callback (a frame advance without a token)
reuses the joint dispatch alone.

Exactness on this stack (receipts 2026-09-15-tdt-gpu-step, -tdt-gpu-step2):

* fp16 add/mul and bit-indexed ``float16_t`` table reads are hardware exact.
* hardware fp16 ``fma()`` is double-rounded through fp32, so the BNNS
  single-rounded fp16 FMA is reproduced by ``exact_fma16``: a TwoSum split
  of ``acc + x*y`` in fp32 (fp16 products are exact in fp32) plus an
  explicit fp16 tie correction on the residual. Validated 0/262144
  mismatches against an fp64 golden, including adversarial ties and extreme
  exponent spreads.
* every exactness-critical fp32 operation carries the GLSL ``precise``
  qualifier; without it the compiler reassociates and the TwoSum collapses.
* ``mx.matmul`` for fp16 ``[1,K]@[K,N]`` is fp32 ascending accumulation
  rounded once to fp16, then a separate fp16 bias add; the fused projector
  and joint use the same order, so the fused logits equal the landed path's
  logits bit for bit.

Why the LSTM GEMV is split this way: the BNNS contract folds the ten
128-wide k-blocks of each (lane, gate) in a fixed sequential fp16 order,
so the ten block chains are INDEPENDENT up to that fold and can run on
any thread in any group. A chain kernel computes one block chain per
thread and publishes the block sums; a fold+cell kernel folds them in
contract order (first term assigned, not added, preserving -0) and runs
the gate pairing and cell update for its lanes. Probes in
receipts/2026-09-15-tdt-gpu-step2/derivation measured the fork's queued
dispatch execution at ~0.16 ms versus multi-millisecond solo round trips,
so the six small dispatches queue ahead of one sync instead of paying the
solo latency three times.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np

from .pinned_component import load_pinned_component

_VOCAB = 8193
_JOINT_OUT = 8198

_CHAIN_THREADS = 256
_CHAIN_GROUPS = 100  # 25600 chains = 2560 lanes x 4 gates x 10 blocks
_JOINT_THREADS = int(__import__('os').environ.get('TDT_JOINT_THREADS', 256))
_EXACT_FMA16 = """
float16_t exact_fma16(float16_t acc, float16_t x, float16_t y) {
    precise float af = float(acc);
    precise float p = float(x) * float(y);
    precise float s = af + p;
    precise float bv = s - af;
    precise float l = (af - (s - bv)) + (p - bv);
    float16_t r = float16_t(s);
    if (l == 0.0f) return r;
    uint u = floatBitsToUint(s);
    uint biased = (u >> 23) & 0xFFu;
    if (biased == 0xFFu || biased < 114u) return r;
    uint mant = u & 0x7FFFFFu;
    if (((mant >> 12) & 1u) == 0u) return r;
    if ((mant & 0xFFFu) != 0u) return r;
    precise float half16 = uintBitsToFloat((biased - 11u) << 23);
    return float16_t(s + (l > 0.0f ? half16 : -half16));
}
"""

# One block chain per thread. idx = ((gate*10 + block) * 2560 + lane) keeps
# a warp on one k row of one gate/block reading consecutive lanes. The
# 128-wide block chain is the BNNS contract unit: exact_fma16 from fp16 0,
# ascending k. flags = [token_id, layer]; x_in supplies layer 1's h1.
_CHAINS_SOURCE = f"""
    uint g = threadgroup_position_in_grid.x;
    uint t = thread_index_in_threadgroup.x;
    int token_id = flags[0];
    int layer = flags[1];
    uint idx = g * {_CHAIN_THREADS}u + t;
    threadgroup float16_t sh_a[1280];
    for (uint i = t; i < 1280u; i += {_CHAIN_THREADS}u) {{
        if (i < 640u) {{
            if (layer == 0) {{
                uint tbase = uint(token_id);
                if (token_id < 0) {{ tbase = uint(token_id + 8193); }}
                sh_a[i] = embedding[tbase * 640u + i];
            }} else {{
                sh_a[i] = float16_t(x_in[i]);
            }}
        }} else {{
            sh_a[i] = float16_t(hidden_in[uint(layer) * 640u + (i - 640u)]);
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n = idx % 2560u;
    uint gb = idx / 2560u;
    uint block = gb % 10u;
    float16_t bc = float16_t(0.0f);
    uint k = block * 128u;
    for (uint j = 0u; j < 128u; ++j) {{
        bc = exact_fma16(bc, sh_a[k], weights[uint(layer) * 3276800u + k * 2560u + n]);
        ++k;
    }}
    bsum[idx] = bc;
"""

# Fold the ten block sums of every (lane, gate) in contract order, add the
# fp16 bias, then run the gate pairing and cell update for the lane. One
# group of 640 threads: thread t owns lane t and keeps all four gate
# preacts in registers, so no barrier is needed before the cell.
_FOLD_SOURCE = """
    uint t = thread_index_in_threadgroup.x;
    int layer = flags[0];
    int dbgflag = flags[1];
    uint lane = t;
    float16_t pr[4];
    for (uint gate = 0u; gate < 4u; ++gate) {
        uint n = lane + gate * 640u;
        float16_t bacc = bsum[n];
        for (uint block = 1u; block < 10u; ++block) {
            bacc = float16_t(bacc + bsum[block * 2560u + n]);
        }
        pr[gate] = float16_t(bacc + biases[uint(layer) * 2560u + n]);
    }
    uint bi = packHalf2x16(vec2(float(pr[0]), 0.0f)) & 0xFFFFu;
    uint bf = packHalf2x16(vec2(float(pr[1]), 0.0f)) & 0xFFFFu;
    uint bo = packHalf2x16(vec2(float(pr[2]), 0.0f)) & 0xFFFFu;
    uint bg = packHalf2x16(vec2(float(pr[3]), 0.0f)) & 0xFFFFu;
    float16_t si = luts[bi];
    float16_t sf = luts[bf];
    float16_t so = luts[bo];
    float16_t tg = luts[65536u + bg];
    float16_t c0 = float16_t(cell_in[uint(layer) * 640u + lane]);
    float16_t fprod = sf * c0;
    float16_t c1 = exact_fma16(fprod, si, tg);
    uint cb = packHalf2x16(vec2(float(c1), 0.0f)) & 0xFFFFu;
    float16_t tc = luts[65536u + cb];
    float16_t h1 = so * tc;
    h1_out[lane] = float(h1);
    c1_out[lane] = float(c1);
    if (dbgflag != 0) {
        dbg[uint(layer) * 640u + lane] = float(h1);
    }
"""

# Projector chain + relu for lane t (mx.matmul contract order: fp32
# ascending chain, one fp16 rounding, fp16 bias), published for the joint.
_PROJ_SOURCE = """
    uint t = thread_index_in_threadgroup.x;
    int frame = flags[0];
    int dbgflag = flags[1];
    threadgroup float16_t sh_dec[640];
    sh_dec[t] = float16_t(h1_in[t]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    precise float acc = 0.0f;
    for (uint k = 0u; k < 640u; ++k) {
        acc = acc + float(sh_dec[k]) * float(projector[k * 640u + t]);
    }
    float16_t pj = float16_t(float16_t(acc) + projector[640u * 640u + t]);
    pj_out[t] = float(pj);
    pj16_out[t] = pj;
    if (dbgflag != 0) {
        dbg[1280u + t] = float(acc);
        dbg[1920u + t] = float(pj);
    }
"""

# Joint head, lane-parallel: each thread owns one output lane j.
# jmode 1 (joint-only callback) rebuilds the relu from fp16(dec_in) + the
# encoder frame, bit-identical because dec_in already holds float(pj).
_JOINT_SOURCE = f"""
    uint g = threadgroup_position_in_grid.x;
    uint t = thread_index_in_threadgroup.x;
    uint j = g * {_JOINT_THREADS}u + t;
    int frame = flags[0];
    int jmode = flags[1];
    threadgroup float16_t sh_relu[640];
    for (uint i = t; i < 640u; i += {_JOINT_THREADS}u) {{
        float16_t pj;
        if (jmode == 0) {{
            pj = pj16_in[i];
        }} else {{
            pj = float16_t(dec_in[i]);
        }}
        float16_t rlv = float16_t(encoder[uint(frame) * 640u + i]) + pj;
        sh_relu[i] = (rlv > float16_t(0.0f)) ? rlv : float16_t(0.0f);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (j < 8198u) {{
        precise float acc = 0.0f;
        for (uint k = 0u; k < 640u; ++k) {{
            acc = acc + float(sh_relu[k]) * float(joint[k * 8198u + j]);
        }}
        logits[j] = float(float16_t(acc) + joint[640u * 8198u + j]);
    }}
"""


_WINDOW_SLOTS = 6
_WINDOW_ACC_DECLS = "\n".join(
    f"    precise float acc{slot} = 0.0f;" for slot in range(_WINDOW_SLOTS)
)
_WINDOW_ACC_UPDATES = "\n".join(
    f"            acc{slot} = acc{slot}"
    f" + float(sh_relu[{slot}u * 640u + k]) * w;"
    for slot in range(_WINDOW_SLOTS)
)
_WINDOW_WRITES = "\n".join(
    f"        if (base + {slot} < valid) {{"
    f"\n            window_out[{slot}u * 8198u + j] ="
    f" float(float16_t(acc{slot}) + joint[640u * 8198u + j]);"
    f"\n        }} else {{"
    f"\n            window_out[{slot}u * 8198u + j] = 0.0f;"
    f"\n        }}"
    for slot in range(_WINDOW_SLOTS)
)

# The same joint head evaluated for `_WINDOW_SLOTS` consecutive encoder
# frames against one decoder state (host control speculates frame-entry
# joints so blank runs resolve without a submit). The joint weight matrix
# is the bandwidth wall (640x8198 fp16 read per evaluation), so each
# thread owns one output lane j and applies every loaded weight element
# to the per-slot scalar accumulators: the weights stream once for the
# whole window, not once per slot. Per-element arithmetic stays identical
# to _JOINT_SOURCE: fp16 relu build, fp32 ascending k chain, one fp16
# rounding, fp16 bias. flags = [base_frame, jmode, valid_frames]; rows
# past valid_frames zero-fill (the control loop never queries them).
_JOINT_WINDOW_SOURCE = f"""
    uint g = threadgroup_position_in_grid.x;
    uint t = thread_index_in_threadgroup.x;
    uint j = g * {_JOINT_THREADS}u + t;
    int base = flags[0];
    int jmode = flags[1];
    int valid = flags[2];
    threadgroup float16_t sh_relu[{_WINDOW_SLOTS * 640}];
    for (uint slot = 0u; slot < {_WINDOW_SLOTS}u; ++slot) {{
        int frame = base + int(slot);
        for (uint i = t; i < 640u; i += {_JOINT_THREADS}u) {{
            float16_t pj;
            if (jmode == 0) {{
                pj = pj16_in[i];
            }} else {{
                pj = float16_t(dec_in[i]);
            }}
            if (frame < valid) {{
                float16_t ev = float16_t(encoder[uint(frame) * 640u + i]);
                float16_t rlv = ev + pj;
                sh_relu[slot * 640u + i] = (rlv > float16_t(0.0f)) ? rlv : float16_t(0.0f);
            }} else {{
                sh_relu[slot * 640u + i] = float16_t(0.0f);
            }}
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (j < 8198u) {{
{_WINDOW_ACC_DECLS}
        for (uint k = 0u; k < 640u; ++k) {{
            precise float w = float(joint[k * 8198u + j]);
{_WINDOW_ACC_UPDATES}
        }}
{_WINDOW_WRITES}
    }}
"""


@cache

def _mx():
    import mlx.core as mx

    return mx


class StepWeights:
    """Device-resident constants for the fused step dispatch."""

    def __init__(self, embedding, weights, biases, luts, projector, joint):
        self.embedding = embedding
        self.weights = weights
        self.biases = biases
        self.luts = luts
        self.projector = projector
        self.joint = joint


@cache
def _dummy_state():
    import mlx.core as mx

    state = mx.zeros((1280,), dtype=mx.float32)
    mx.eval(state)
    return state


@cache
def _chains_kernel():
    mx = _mx()
    return mx.fast.metal_kernel(
        name="parakeet_tdt_decoder_chains",
        input_names=["embedding", "weights", "hidden_in", "flags", "x_in"],
        output_names=["bsum"],
        header=_EXACT_FMA16,
        source=_CHAINS_SOURCE,
        compile_options={"math_mode": "safe"},
    )


@cache
def _fold_kernel():
    mx = _mx()
    return mx.fast.metal_kernel(
        name="parakeet_tdt_decoder_fold",
        input_names=["bsum", "biases", "luts", "cell_in", "flags"],
        output_names=["h1_out", "c1_out", "dbg"],
        header=_EXACT_FMA16,
        source=_FOLD_SOURCE,
        compile_options={"math_mode": "safe"},
    )


@cache
def _proj_kernel():
    mx = _mx()
    return mx.fast.metal_kernel(
        name="parakeet_tdt_decoder_proj",
        input_names=["h1_in", "projector", "flags"],
        output_names=["pj_out", "pj16_out", "dbg"],
        header="",
        source=_PROJ_SOURCE,
        compile_options={"math_mode": "safe"},
    )


@cache
def _joint_kernel():
    mx = _mx()
    return mx.fast.metal_kernel(
        name="parakeet_tdt_decoder_joint",
        input_names=["pj16_in", "dec_in", "encoder", "flags", "joint"],
        output_names=["logits"],
        header="",
        source=_JOINT_SOURCE,
        compile_options={"math_mode": "safe"},
    )


@cache
def _joint_window_kernel():
    mx = _mx()
    return mx.fast.metal_kernel(
        name="parakeet_tdt_decoder_joint_window",
        input_names=["pj16_in", "dec_in", "encoder", "flags", "joint"],
        output_names=["window_out"],
        header="",
        source=_JOINT_WINDOW_SOURCE,
        compile_options={"math_mode": "safe"},
    )


def pack_step_weights(decoder, joint_package: Path) -> StepWeights:
    """Pack the pinned decoder/joint constants into kernel-order buffers."""
    mx = _mx()
    weights = decoder._weights
    layer_mats = []
    biases = []
    for wih, whh, bias in (
        (weights.layer_0_ih, weights.layer_0_hh, weights.layer_0_bias),
        (weights.layer_1_ih, weights.layer_1_hh, weights.layer_1_bias),
    ):
        cat = np.concatenate(
            [np.asarray(wih, dtype=np.float16), np.asarray(whh, dtype=np.float16)],
            axis=1,
        )
        layer_mats.append(np.ascontiguousarray(cat.T, dtype=np.float16))
        biases.append(np.asarray(bias, dtype=np.float16).ravel())
    from .vulkan_decoder import _fused_luts

    sigma, tanh = _fused_luts()
    luts = np.concatenate([sigma.ravel(), tanh.ravel()]).astype(np.float16)
    projector = np.asarray(weights.projector, dtype=np.float16)
    proj_pack = np.concatenate(
        [
            np.ascontiguousarray(projector.T, dtype=np.float16).ravel(),
            np.asarray(weights.projector_bias, dtype=np.float16).ravel(),
        ]
    )
    joint_component = load_pinned_component(Path(joint_package), "joint")
    jw = np.asarray(
        joint_component.constant("head_weight_to_fp16"), dtype=np.float16
    )
    jb = np.asarray(
        joint_component.constant("head_bias_to_fp16"), dtype=np.float16
    ).ravel()
    joint_pack = np.concatenate(
        [np.ascontiguousarray(jw.T, dtype=np.float16).ravel(), jb]
    )
    with mx.stream(mx.gpu):
        packed = StepWeights(
            embedding=mx.array(np.ascontiguousarray(weights.embedding)),
            weights=mx.array(np.ascontiguousarray(np.concatenate(layer_mats))),
            biases=mx.array(np.ascontiguousarray(np.concatenate(biases))),
            luts=mx.array(np.ascontiguousarray(luts)),
            projector=mx.array(np.ascontiguousarray(proj_pack)),
            joint=mx.array(np.ascontiguousarray(joint_pack)),
        )
    mx.eval(
        packed.embedding,
        packed.weights,
        packed.biases,
        packed.luts,
        packed.projector,
        packed.joint,
    )
    return packed


def run_step(
    packed: StepWeights,
    hidden,
    cell,
    token_id: int,
    encoder,
    frame: int,
    skip_lstm: bool = False,
    dec_in=None,
    mode: int | None = None,
    spec_frames: int = 0,
    spec_valid: int = 0,
):
    """Queued dispatches on GPU buffers.

    ``hidden``/``cell`` are fp32 arrays shaped (2, 1, 640); ``encoder`` the
    fp32 encoder output (1, F, 640). Returns ``(state_out, token, duration,
    logits, dbg, window)`` with ``state_out`` the GPU fp32 array
    ``[decoder_hidden(640) | next_hidden(1280) | next_cell(1280)]``. The
    token/duration argmaxes run host-side with first-max semantics.

    With ``spec_frames`` > 0 the joint head also evaluates the
    ``spec_frames`` frames after ``frame`` against this step's decoder
    state (rows past ``spec_valid`` zero-fill), returning the
    ``(spec_frames, 8198)`` fp32 logits as ``window``; the caller resolves
    frame-entry joints host-side without another submit. The joint-only
    mode (``skip_lstm``) skips the projector dispatch entirely: its output
    is never read (jmode 1 rebuilds the relu from ``dec_in``) and
    ``state_out`` is ``dec_in`` unchanged.
    """
    mx = _mx()
    if mode is None:
        mode = 1 if skip_lstm else 0
    if hidden is None:
        hidden = _dummy_state()
        cell = hidden
    flat_hidden = hidden.reshape(1280,)
    flat_cell = cell.reshape(1280,)
    dbg_wide = mode == 2
    dbg_narrow = (2560,) if dbg_wide else (1,)
    if mode != 1:
        chains = _chains_kernel()
        fold = _fold_kernel()
        bsums = []
        h1s = []
        c1s = []
        dbgs = []
        for layer in (0, 1):
            flags_c = mx.array(
                np.array([int(token_id), layer], np.int32)
            )
            x_in = flat_hidden if layer == 0 else h1s[0]
            bsum, = chains(
                inputs=[
                    packed.embedding,
                    packed.weights,
                    flat_hidden,
                    flags_c,
                    x_in,
                ],
                output_shapes=[(25600,)],
                output_dtypes=[mx.float16],
                grid=(_CHAIN_GROUPS * _CHAIN_THREADS, 1, 1),
                threadgroup=(_CHAIN_THREADS, 1, 1),
                stream=mx.gpu,
            )
            bsums.append(bsum)
            flags_f = mx.array(
                np.array([layer, int(dbg_wide)], np.int32)
            )
            h1, c1, dbg = fold(
                inputs=[
                    bsum,
                    packed.biases,
                    packed.luts,
                    flat_cell,
                    flags_f,
                ],
                output_shapes=[(640,), (640,), dbg_narrow],
                output_dtypes=[mx.float32, mx.float32, mx.float32],
                grid=(640, 1, 1),
                threadgroup=(640, 1, 1),
                stream=mx.gpu,
            )
            h1s.append(h1)
            c1s.append(c1)
            dbgs.append(dbg)
        lstm_in = h1s[1]
    else:
        bsums = h1s = c1s = dbgs = None
        lstm_in = (
            dec_in.reshape(640,) if dec_in is not None else flat_hidden[:640]
        )
    dec16src = lstm_in
    if mode != 1:
        flags_p = mx.array(np.array([int(frame), int(dbg_wide)], np.int32))
        pj, pj16, dbg2 = _proj_kernel()(
            inputs=[
                lstm_in,
                packed.projector,
                flags_p,
            ],
            output_shapes=[(640,), (640,), dbg_narrow],
            output_dtypes=[mx.float32, mx.float16, mx.float32],
            grid=(640, 1, 1),
            threadgroup=(640, 1, 1),
            stream=mx.gpu,
        )
        dec16src = pj16
        state_out = mx.concatenate([pj, h1s[0], h1s[1], c1s[0], c1s[1]])
        dbg = dbg2
        if dbg_wide:
            dbg = mx.concatenate([dbgs[0], dbgs[1], dbg2])
    else:
        state_out = lstm_in
        dbg = None
    flagsj = mx.array(np.array([int(frame), int(mode == 1)], np.int32))
    (logits,) = _joint_kernel()(
        inputs=[
            dec16src,
            dec16src,
            encoder.reshape(-1),
            flagsj,
            packed.joint,
        ],
        output_shapes=[(_JOINT_OUT,)],
        output_dtypes=[mx.float32],
        grid=(33 * _JOINT_THREADS, 1, 1),
        threadgroup=(_JOINT_THREADS, 1, 1),
        stream=mx.gpu,
    )
    window = None
    if spec_frames > 0:
        flags_w = mx.array(
            np.array(
                [int(frame) + 1, int(mode == 1), int(spec_valid)], np.int32
            )
        )
        (window,) = _joint_window_kernel()(
            inputs=[
                dec16src,
                dec16src,
                encoder.reshape(-1),
                flags_w,
                packed.joint,
            ],
            output_shapes=[(int(spec_frames), _JOINT_OUT)],
            output_dtypes=[mx.float32],
            grid=(33 * _JOINT_THREADS, 1, 1),
            threadgroup=(_JOINT_THREADS, 1, 1),
            stream=mx.gpu,
        )
        mx.eval(state_out, logits, window)
    else:
        mx.eval(state_out, logits)
    lg = np.asarray(logits[:_JOINT_OUT])
    token = int(np.argmax(lg[:_VOCAB]))
    duration = int(np.argmax(lg[_VOCAB:_JOINT_OUT]))
    return state_out, token, duration, logits, dbg, window
