# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""GPU-resident greedy TDT decode loop (one dispatch, no per-emission sync).

The entire ``greedy_tdt_decode`` state machine from ``parakeet_tdt.py`` runs
inside a single workgroup of one custom kernel: blank-vs-emit, the duration
skip (``max(duration, 1)`` on blank, ``duration > 0`` advance), the
``max_symbols_per_step`` cap and the ``+1`` fallthrough are decided in
workgroup memory; the host reads tokens/state once at the end.  This removes
the per-emission solo-dispatch round trip that pinned step2 at 2940 ms.

Bit-exactness reuses the landed step2 contract per output lane:

* ``exact_fma16`` and every exactness-critical fp32 op keep the ``precise``
  qualifier and the step2 op order (block chains from fp16 0 ascending k,
  fold b0 assigned then +b1..+b9, bias, gate LUT pairing, cell, projector
  fp32 ascending chain with one fp16 rounding, joint fp32 ascending chain).
* The in-kernel argmax reproduces ``np.argmax`` first-max semantics: each
  thread scans its ascending strided subset with strict ``>`` (NaN wins
  like ``np.argmax``), the cross-thread reduce prefers the smaller index
  on ties and the first NaN.
* The control decisions replicate ``decode_tdt``/``greedy_tdt_decode``
  branch for branch; a duration index outside the class count raises on
  the host, like ``TdtControlError``.

Decoder/joint math provenance: receipts 2026-09-15-tdt-gpu-step and
-tdt-gpu-step2 (branch agent/tdt-gpu-step2, commit d5d250a2).  The GLSL is
restructured only in thread assignment; per-output arithmetic is identical,
so results are bit-identical to the landed per-emission path.

Windowed readback: ``frame_stop`` bounds one launch; the full loop state
(hidden, cell, decoder state, input token, validity, frame) chains into the
next launch through ``initial_*`` and ``frame_start`` so a windowed decode
produces the identical emission stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np

from .vulkan_decoder_step import _EXACT_FMA16  # noqa: F401

_LOOP_THREADS = 1024


# Uniform layout (int32): [0] valid_frames, [1] frame_stop (kernel loops
# until frame >= stop), [2] blank_token_id, [3] max_symbols_per_step,
# [4] n_dur, [5..9] durations, [10] emission capacity, [11] initial
# decoder-state-valid flag, [12] initial input token, [13] frame start.
_CFG_VALID = 0
_CFG_STOP = 1
_CFG_BLANK = 2
_CFG_MAXSYM = 3
_CFG_NDUR = 4
_CFG_DUR0 = 5
_CFG_CAP = 10
_CFG_IVALID = 11
_CFG_ITOKEN = 12
_CFG_FSTART = 13

# s_ctl indices (workgroup int[16]).
_CTL_FRAME = 0
_CTL_INPUT = 1
_CTL_VALID = 2
_CTL_COUNT = 3
_CTL_OP = 4
_CTL_ERR = 5
_CTL_TOK = 6
_CTL_DURI = 7
_CTL_RUN = 8
_CTL_SYM = 9

_LT = _LOOP_THREADS


def _loop_glsl() -> str:
    """Build the fused loop kernel source (f-string interpolation only)."""
    return f"""
    uint t = thread_index_in_threadgroup.x;

    threadgroup float s_hidden[1280];
    threadgroup float s_cell[1280];
    threadgroup float s_h1[640];
    threadgroup float16_t s_a[1280];
    threadgroup float16_t s_pj[640];
    threadgroup float16_t s_relu[640];
    threadgroup float s_bval[{_LT}];
    threadgroup uint s_bidx[{_LT}];
    threadgroup float s_dval[8];
    threadgroup int s_ctl[16];

    for (uint i = t; i < 1280u; i += {_LT}u) {{
        s_hidden[i] = hidden_in[i];
        s_cell[i] = cell_in[i];
        s_pj[i] = float16_t(dec_in[i]);
    }}
    if (t == 0u) {{
        s_ctl[{_CTL_FRAME}] = cfg[{_CFG_FSTART}];
        s_ctl[{_CTL_INPUT}] = cfg[{_CFG_ITOKEN}];
        s_ctl[{_CTL_VALID}] = cfg[{_CFG_IVALID}];
        s_ctl[{_CTL_COUNT}] = 0;
        s_ctl[{_CTL_OP}] = 0;
        s_ctl[{_CTL_ERR}] = 0;
        s_ctl[{_CTL_TOK}] = 0;
        s_ctl[{_CTL_DURI}] = 0;
        s_ctl[{_CTL_RUN}] = 1;
        s_ctl[{_CTL_SYM}] = 0;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    while (true) {{
        if (t == 0u) {{ s_ctl[{_CTL_SYM}] = 0; }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        while (true) {{
            bool need = (s_ctl[{_CTL_VALID}] == 0) ||
                        (s_ctl[{_CTL_INPUT}] != cfg[{_CFG_BLANK}]);
            if (need) {{
                int tok = s_ctl[{_CTL_INPUT}];
                for (uint layer = 0u; layer < 2u; ++layer) {{
                    for (uint k = t; k < 640u; k += {_LT}u) {{
                        if (layer == 0u) {{
                            uint tbase = uint(tok);
                            if (tok < 0) {{ tbase = uint(tok + 8193); }}
                            s_a[k] = embedding[tbase * 640u + k];
                        }} else {{
                            s_a[k] = float16_t(s_h1[k]);
                        }}
                        s_a[640u + k] =
                            float16_t(s_hidden[layer * 640u + k]);
                    }}
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint p = t; p < 2560u; p += {_LT}u) {{
                        uint gate = p / 640u;
                        uint lane = p % 640u;
                        uint n = gate * 640u + lane;
                        float16_t bacc = float16_t(0.0f);
                        for (uint block = 0u; block < 10u; ++block) {{
                            float16_t bc = float16_t(0.0f);
                            uint k = block * 128u;
                            for (uint j = 0u; j < 128u; ++j) {{
                                bc = exact_fma16(
                                    bc, s_a[k],
                                    weights[layer * 3276800u
                                            + k * 2560u + n]);
                                ++k;
                            }}
                            if (block == 0u) {{
                                bacc = bc;
                            }} else {{
                                bacc = float16_t(bacc + bc);
                            }}
                        }}
                        float16_t pr = float16_t(
                            bacc + biases[layer * 2560u + n]);
                        if (gate == 0u) {{
                            s_relu[lane] = pr;
                        }} else if (gate == 1u) {{
                            s_bval[lane] = float(pr);
                        }} else if (gate == 2u) {{
                            s_bidx[lane] = floatBitsToUint(float(pr));
                        }} else {{
                            s_h1[lane] = float(pr);
                        }}
                    }}
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint lane = t; lane < 640u; lane += {_LT}u) {{
                        float16_t pr0 = s_relu[lane];
                        float16_t pr1 = float16_t(s_bval[lane]);
                        float16_t pr2 = float16_t(
                            uintBitsToFloat(s_bidx[lane]));
                        float16_t pr3 = float16_t(s_h1[lane]);
                        uint bi = packHalf2x16(vec2(float(pr0), 0.0f))
                                  & 0xFFFFu;
                        uint bf = packHalf2x16(vec2(float(pr1), 0.0f))
                                  & 0xFFFFu;
                        uint bo = packHalf2x16(vec2(float(pr2), 0.0f))
                                  & 0xFFFFu;
                        uint bg = packHalf2x16(vec2(float(pr3), 0.0f))
                                  & 0xFFFFu;
                        float16_t si = luts[bi];
                        float16_t sf = luts[bf];
                        float16_t so = luts[bo];
                        float16_t tg = luts[65536u + bg];
                        float16_t c0 = float16_t(
                            s_cell[layer * 640u + lane]);
                        float16_t fprod = sf * c0;
                        float16_t c1 = exact_fma16(fprod, si, tg);
                        uint cb = packHalf2x16(vec2(float(c1), 0.0f))
                                  & 0xFFFFu;
                        float16_t tc = luts[65536u + cb];
                        float16_t h1 = so * tc;
                        s_hidden[layer * 640u + lane] = float(h1);
                        s_cell[layer * 640u + lane] = float(c1);
                        s_h1[lane] = float(h1);
                    }}
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }}
                for (uint k = t; k < 640u; k += {_LT}u) {{
                    s_pj[k] = float16_t(s_h1[k]);
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint lane = t; lane < 640u; lane += {_LT}u) {{
                    precise float acc = 0.0f;
                    for (uint k = 0u; k < 640u; ++k) {{
                        acc = acc + float(s_pj[k])
                                  * float(projector[k * 640u + lane]);
                    }}
                    s_relu[lane] = float16_t(
                        float16_t(acc) + projector[640u * 640u + lane]);
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint i = t; i < 640u; i += {_LT}u) {{
                    s_pj[i] = s_relu[i];
                }}
                if (t == 0u) {{ s_ctl[{_CTL_VALID}] = 1; }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint frame = uint(s_ctl[{_CTL_FRAME}]);
            for (uint i = t; i < 640u; i += {_LT}u) {{
                float16_t rlv = float16_t(encoder[frame * 640u + i])
                                + s_pj[i];
                s_relu[i] = (rlv > float16_t(0.0f)) ? rlv : float16_t(0.0f);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float best_v = 0.0f;
            uint best_i = 0u;
            for (uint j = t; j < 8198u; j += {_LT}u) {{
                precise float acc = 0.0f;
                for (uint k = 0u; k < 640u; ++k) {{
                    acc = acc + float(s_relu[k])
                              * float(joint[k * 8198u + j]);
                }}
                float lg = float(float16_t(acc)
                                 + joint[640u * 8198u + j]);
                if (j < 8193u) {{
                    if ((j == t) || ((lg != lg) && (best_v == best_v))
                        || (lg > best_v)) {{
                        best_v = lg;
                        best_i = j;
                    }}
                }} else {{
                    s_dval[j - 8193u] = lg;
                }}
            }}
            s_bval[t] = best_v;
            s_bidx[t] = best_i;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (t == 0u) {{
                float bv = s_bval[0];
                uint bix = s_bidx[0];
                for (uint i = 1u; i < {_LT}u; ++i) {{
                    float v = s_bval[i];
                    uint ix = s_bidx[i];
                    bool take;
                    if (v != v) {{
                        take = (bv == bv) || (ix < bix);
                    }} else if (bv != bv) {{
                        take = false;
                    }} else {{
                        take = (v > bv) || (v == bv && ix < bix);
                    }}
                    if (take) {{ bv = v; bix = ix; }}
                }}
                float dv = s_dval[0];
                uint dix = 0u;
                for (uint i = 1u; i < uint(cfg[{_CFG_NDUR}]); ++i) {{
                    float v = s_dval[i];
                    bool take;
                    if (v != v) {{
                        take = (dv == dv) || (i < dix);
                    }} else if (dv != dv) {{
                        take = false;
                    }} else {{
                        take = (v > dv) || (v == dv && i < dix);
                    }}
                    if (take) {{ dv = v; dix = i; }}
                }}
                s_ctl[{_CTL_TOK}] = int(bix);
                s_ctl[{_CTL_DURI}] = int(dix);
                int op = 0;
                if (dix >= uint(cfg[{_CFG_NDUR}])) {{
                    s_ctl[{_CTL_ERR}] = 1;
                    op = 2;
                }} else {{
                    int d = cfg[{_CFG_DUR0} + dix];
                    int blank = cfg[{_CFG_BLANK}];
                    int tokd = int(bix);
                    if (tokd == blank) {{
                        int adv = (d > 1) ? d : 1;
                        s_ctl[{_CTL_FRAME}] = s_ctl[{_CTL_FRAME}] + adv;
                        op = 1;
                    }} else {{
                        int c = s_ctl[{_CTL_COUNT}];
                        if (c < cfg[{_CFG_CAP}]) {{
                            emissions[c * 3 + 0] = tokd;
                            emissions[c * 3 + 1] = s_ctl[{_CTL_FRAME}];
                            emissions[c * 3 + 2] = d;
                            s_ctl[{_CTL_COUNT}] = c + 1;
                        }} else {{
                            s_ctl[{_CTL_ERR}] = 2;
                            op = 2;
                        }}
                        s_ctl[{_CTL_INPUT}] = tokd;
                        s_ctl[{_CTL_SYM}] = s_ctl[{_CTL_SYM}] + 1;
                        if (d > 0) {{
                            s_ctl[{_CTL_FRAME}] =
                                s_ctl[{_CTL_FRAME}] + d;
                            op = 1;
                        }}
                    }}
                    if (op == 0 && s_ctl[{_CTL_SYM}] >= cfg[{_CFG_MAXSYM}]) {{
                        s_ctl[{_CTL_FRAME}] = s_ctl[{_CTL_FRAME}] + 1;
                        op = 1;
                    }}
                }}
                s_ctl[{_CTL_OP}] = op;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (s_ctl[{_CTL_OP}] != 0) {{ break; }}
        }}

        if (t == 0u) {{
            s_ctl[{_CTL_RUN}] =
                (s_ctl[{_CTL_ERR}] == 0
                 && s_ctl[{_CTL_FRAME}] < cfg[{_CFG_VALID}]
                 && s_ctl[{_CTL_FRAME}] < cfg[{_CFG_STOP}]) ? 1 : 0;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (s_ctl[{_CTL_RUN}] == 0) {{ break; }}
    }}

    for (uint i = t; i < 1280u; i += {_LT}u) {{
        state[640u + i] = s_hidden[i];
        state[1920u + i] = s_cell[i];
    }}
    for (uint i = t; i < 640u; i += {_LT}u) {{
        state[i] = float(s_pj[i]);
    }}
    if (t == 0u) {{
        ctl[0] = s_ctl[{_CTL_COUNT}];
        ctl[1] = s_ctl[{_CTL_ERR}];
        ctl[2] = s_ctl[{_CTL_FRAME}];
        ctl[3] = s_ctl[{_CTL_VALID}];
        ctl[4] = s_ctl[{_CTL_INPUT}];
        ctl[5] = s_ctl[{_CTL_RUN}];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""


@dataclass
class TdtLoopOutput:
    """Emissions and recurrent state produced by the GPU-resident loop."""

    token_ids: list[int]
    frame_indices: list[int]
    durations: list[int]
    decoder_state: Any
    hidden: Any
    cell: Any
    final_frame: int
    input_token: int
    decoder_state_valid: bool


def _mx():
    import mlx.core as mx

    return mx


@cache
def _loop_kernel():
    mx = _mx()
    return mx.fast.metal_kernel(
        name="parakeet_tdt_loop",
        input_names=[
            "embedding", "weights", "biases", "luts", "projector",
            "joint", "encoder", "hidden_in", "cell_in", "dec_in", "cfg",
        ],
        output_names=["emissions", "state", "ctl"],
        header=_EXACT_FMA16,
        source=_loop_glsl(),
        compile_options={"math_mode": "safe"},
    )


# Workgroup memory the loop kernel declares: s_hidden/s_cell (2x1280 f32),
# s_h1 (640 f32), s_a/s_pj/s_relu (1280/640/640 f16), s_bval/s_bidx
# (2x1024 u32/f32), s_dval (8 f32), s_ctl (16 i32).
LOOP_WORKGROUP_MEMORY_BYTES = 28768


def device_blockers() -> list[str]:
    """Capabilities the loop kernel needs that this device lacks, named.

    The kernel is one 1024-invocation workgroup with ~28 KB of workgroup
    memory and fp16 threadgroup/storage access, so the device must report
    the omarchy Vulkan limit axes at or above those requirements.  Software
    rasterizers (llvmpipe/lavapipe) and capability-simulation profiles never
    back the loop: the host control path decodes instead.
    """

    import mlx.core as mx

    info = mx.device_info()
    if not info:
        return ["no GPU device reported by mx.device_info()"]
    blockers: list[str] = []
    device = str(info.get("device_name", "")).lower()
    driver = str(info.get("driver", "")).lower()
    if int(info.get("simulated", 0)) == 1:
        blockers.append(
            "capability simulation active "
            f"(profile {info.get('simulation_profile')!r})"
        )
    if "llvmpipe" in device or "lavapipe" in device or "llvmpipe" in driver or "lavapipe" in driver:
        blockers.append(f"software rasterizer {info.get('device_name')!r}")
    for key, needed in (
        ("max_compute_work_group_invocations", _LOOP_THREADS),
        ("max_compute_work_group_size_x", _LOOP_THREADS),
        ("max_compute_shared_memory_size", LOOP_WORKGROUP_MEMORY_BYTES),
    ):
        have = info.get(key)
        if have is None:
            blockers.append(f"device_info lacks {key} (omarchy Vulkan runtime required)")
        elif int(have) < needed:
            blockers.append(f"{key} {int(have)} < {needed}")
    for key in ("shader_float16", "storage_buffer_16bit_access"):
        if int(info.get(key, 0)) != 1:
            blockers.append(f"{key} not reported by the device")
    return blockers


def run_tdt_loop(
    packed,
    encoder,
    valid_frames: int,
    config,
    initial_hidden=None,
    initial_cell=None,
    frame_stop: int | None = None,
    initial_decoder_state=None,
    initial_input_token: int | None = None,
    initial_valid: bool = False,
    frame_start: int = 0,
) -> TdtLoopOutput:
    """Run the greedy TDT decode in one kernel dispatch.

    ``packed`` is the step2 ``StepWeights`` (``pack_step_weights``),
    ``encoder`` the fp32 encoder output (1, F, 640) and ``config`` the
    ``reference.TdtConfig`` from the reference lock.  Emissions and the
    recurrent state are read back once; no per-emission host sync exists.

    ``frame_stop`` bounds the launch for windowed readbacks; chain windows
    by passing the previous ``TdtLoopOutput``'s ``hidden``/``cell``/
    ``decoder_state``/``input_token``/``decoder_state_valid`` back through
    the matching ``initial_*`` arguments plus ``frame_start``.
    """

    import mlx.core as mx

    blank = int(config.blank_token_id)
    durs = [int(d) for d in config.durations]
    maxsym = int(config.max_symbols_per_step)
    stop = valid_frames if frame_stop is None else min(int(frame_stop), valid_frames)
    cap = max(1, int(valid_frames) * maxsym)
    if initial_valid:
        ivalid = 1
        itoken = blank if initial_input_token is None else int(initial_input_token)
        dec0 = np.asarray(initial_decoder_state, dtype=np.float32).reshape(640,)
    else:
        ivalid = 0
        itoken = blank
        dec0 = np.zeros((640,), dtype=np.float32)
    cfg = np.array(
        [
            int(valid_frames), int(stop), blank, maxsym, len(durs), *durs,
            cap, ivalid, itoken, int(frame_start),
        ],
        dtype=np.int32,
    )
    if initial_hidden is None:
        hidden0 = np.zeros((1280,), dtype=np.float32)
    else:
        hidden0 = np.asarray(initial_hidden, dtype=np.float32).reshape(1280,)
    if initial_cell is None:
        cell0 = np.zeros((1280,), dtype=np.float32)
    else:
        cell0 = np.asarray(initial_cell, dtype=np.float32).reshape(1280,)
    enc = mx.array(encoder) if isinstance(encoder, np.ndarray) else encoder
    with mx.stream(mx.gpu):
        (emissions, state, ctl) = _loop_kernel()(
            inputs=[
                packed.embedding,
                packed.weights,
                packed.biases,
                packed.luts,
                packed.projector,
                packed.joint,
                enc.reshape(-1),
                mx.array(hidden0),
                mx.array(cell0),
                mx.array(dec0),
                mx.array(cfg),
            ],
            output_shapes=[(cap * 3,), (3200,), (8,)],
            output_dtypes=[mx.int32, mx.float32, mx.int32],
            grid=(_LOOP_THREADS, 1, 1),
            threadgroup=(_LOOP_THREADS, 1, 1),
            stream=mx.gpu,
        )
        mx.eval(emissions, state, ctl)
    ctl_h = np.asarray(ctl)
    err = int(ctl_h[1])
    if err != 0:
        reason = (
            "duration index outside class count"
            if err == 1
            else "emission capacity exceeded"
        )
        raise ValueError(f"[omarchy-parakeet] GPU TDT loop failed: {reason}")
    count = int(ctl_h[0])
    em = np.asarray(emissions)[: count * 3].reshape(count, 3)
    st = np.asarray(state)
    return TdtLoopOutput(
        token_ids=[int(x) for x in em[:, 0]],
        frame_indices=[int(x) for x in em[:, 1]],
        durations=[int(x) for x in em[:, 2]],
        decoder_state=st[0:640].reshape(1, 640).copy(),
        hidden=st[640:1920].reshape(2, 1, 640).copy(),
        cell=st[1920:3200].reshape(2, 1, 640).copy(),
        final_frame=int(ctl_h[2]),
        input_token=int(ctl_h[4]),
        decoder_state_valid=bool(ctl_h[3]),
    )
