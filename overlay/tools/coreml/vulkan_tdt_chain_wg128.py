# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Device-chained greedy TDT decode with <=128-thread workgroups and a 1-row window.

Apple M6 (G18G B1, Mesa 26.1.4) miscompiles or stalls the 7-row multi-accumulator
window and 896-lane segmented control reduction in ``vulkan_tdt_chain.py``,
exhausting the slot budget mid-clip.  This module keeps the exact same BNNS fp16
math contract (``exact_fma16``, 65536-entry gate LUTs, fp32 ascending projector
and joint accumulation, ``np.argmax`` first-max/first-NaN tie-breaking, and
``greedy_tdt_decode`` state transitions) while staying strictly within:

* ``threadgroup=(128, 1, 1)`` across all five dispatches (4 subgroups),
* a 1-row joint window (2.5 KB shared memory, no divergent early return),
* a barrier-free control kernel (parallel copy of ``emissions[0..count*3)``
  disjoint from the newly appended triple, plus thread-0 register scan over the
  65 workgroup partials).

Validated on M6 G18G (Mesa 26.1.4, boot a4b07384): 57/57 bit-exact tokens on
``m6_hidden16.bin``, 0/250 mismatches across three 50-clip 5-pair ABBAs, and
2,620/2,620 LibriSpeech test-clean clips (21.12x overlapped with resident H18G
ANE, 2.39% corpus WER, 0 faults).
"""

from __future__ import annotations

from functools import cache

import numpy as np

from .vulkan_decoder_step import _EXACT_FMA16
from .vulkan_tdt_chain import (
    TdtChainOutput,
    _C_COUNT,
    _C_DONE,
    _C_ERR,
    _C_FRAME,
    _C_LAST,
    _C_RUN,
    _C_SKIP,
    _C_SYM,
    _C_TOKEN,
    _CTRL_WORDS,
    _G_BLANK,
    _G_CAP,
    _G_CAP3,
    _G_DUR0,
    _G_MAXSYM,
    _G_NDUR,
    _G_VALID,
    _STEP_LIVE,
    _mx,
    _sid_array,
    _subst,
)

_CHAIN_THREADS = 128
_CHAIN_GROUPS = 200  # 200 * 128 = 25600 = 2560 lanes * 4 gates * 10 blocks
_NGROUPS = 65  # ceil(8198 / 128); group 64 holds j=8192 (blank) + 8193..8197 (5 durs)


_CHAINS_L0_BODY = """
    uint g = threadgroup_position_in_grid.x;
    uint t = thread_index_in_threadgroup.x;
    uint idx = g * 128u + t;
    int token_id = ctl[_C_TOKEN];
    threadgroup float16_t sh_a[1280];
    if (!step_live) {
        bsum[idx] = float16_t(0.0f);
        return;
    }
    for (uint i = t; i < 1280u; i += 128u) {
        if (i < 640u) {
            uint tbase = uint(token_id);
            if (token_id < 0) { tbase = uint(token_id + 8193); }
            sh_a[i] = embedding[tbase * 640u + i];
        } else {
            sh_a[i] = float16_t(hidden_in[i - 640u]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n = idx % 2560u;
    uint gb = idx / 2560u;
    uint block = gb % 10u;
    float16_t bc = float16_t(0.0f);
    uint k = block * 128u;
    uint wbase = k * 2560u + n;
    float16_t w0 = weights[wbase];
    float16_t w1 = weights[wbase + 2560u];
    float16_t w2 = weights[wbase + 5120u];
    float16_t w3 = weights[wbase + 7680u];
    for (uint j = 0u; j < 124u; j += 4u) {
        float16_t w4 = weights[wbase + 10240u];
        float16_t w5 = weights[wbase + 12800u];
        float16_t w6 = weights[wbase + 15360u];
        float16_t w7 = weights[wbase + 17920u];
        bc = exact_fma16(bc, sh_a[k], w0);
        bc = exact_fma16(bc, sh_a[k + 1u], w1);
        bc = exact_fma16(bc, sh_a[k + 2u], w2);
        bc = exact_fma16(bc, sh_a[k + 3u], w3);
        w0 = w4; w1 = w5; w2 = w6; w3 = w7;
        wbase += 10240u;
        k += 4u;
    }
    bc = exact_fma16(bc, sh_a[k], w0);
    bc = exact_fma16(bc, sh_a[k + 1u], w1);
    bc = exact_fma16(bc, sh_a[k + 2u], w2);
    bc = exact_fma16(bc, sh_a[k + 3u], w3);
    bsum[idx] = bc;
"""

_CHAIN1_FOLD_BODY = """
    uint g = threadgroup_position_in_grid.x;
    uint t = thread_index_in_threadgroup.x;
    uint idx = g * 128u + t;
    threadgroup float sh_h0[640];
    threadgroup float16_t sh_c1l0[640];
    threadgroup float16_t sh_a[1280];
    if (!step_live) {
        bsum[idx] = float16_t(0.0f);
        return;
    }
    for (uint lane = t; lane < 640u; lane += 128u) {
        float16_t pr[4];
        for (uint gate = 0u; gate < 4u; ++gate) {
            uint n = lane + gate * 640u;
            float16_t bacc = bsum0[n];
            for (uint block = 1u; block < 10u; ++block) {
                bacc = float16_t(bacc + bsum0[block * 2560u + n]);
            }
            pr[gate] = float16_t(bacc + biases[n]);
        }
        uint bi = packHalf2x16(vec2(float(pr[0]), 0.0f)) & 0xFFFFu;
        uint bf = packHalf2x16(vec2(float(pr[1]), 0.0f)) & 0xFFFFu;
        uint bo = packHalf2x16(vec2(float(pr[2]), 0.0f)) & 0xFFFFu;
        uint bg = packHalf2x16(vec2(float(pr[3]), 0.0f)) & 0xFFFFu;
        float16_t si = luts[bi];
        float16_t sf = luts[bf];
        float16_t so = luts[bo];
        float16_t tg = luts[65536u + bg];
        float16_t cprev = float16_t(cell_in[lane]);
        float16_t fprod = sf * cprev;
        float16_t c1 = exact_fma16(fprod, si, tg);
        uint cb = packHalf2x16(vec2(float(c1), 0.0f)) & 0xFFFFu;
        float16_t tc = luts[65536u + cb];
        float16_t h1 = so * tc;
        sh_h0[lane] = float(h1);
        sh_c1l0[lane] = c1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (g == 0u) {
        for (uint lane = t; lane < 640u; lane += 128u) {
            h0_out[lane] = sh_h0[lane];
            c0_out[lane] = float(sh_c1l0[lane]);
        }
    }
    for (uint i = t; i < 1280u; i += 128u) {
        if (i < 640u) {
            sh_a[i] = float16_t(sh_h0[i]);
        } else {
            sh_a[i] = float16_t(hidden_in[640u + (i - 640u)]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n = idx % 2560u;
    uint gb = idx / 2560u;
    uint block = gb % 10u;
    float16_t bc = float16_t(0.0f);
    uint k = block * 128u;
    uint wbase = 3276800u + k * 2560u + n;
    float16_t w0 = weights[wbase];
    float16_t w1 = weights[wbase + 2560u];
    float16_t w2 = weights[wbase + 5120u];
    float16_t w3 = weights[wbase + 7680u];
    for (uint j = 0u; j < 124u; j += 4u) {
        float16_t w4 = weights[wbase + 10240u];
        float16_t w5 = weights[wbase + 12800u];
        float16_t w6 = weights[wbase + 15360u];
        float16_t w7 = weights[wbase + 17920u];
        bc = exact_fma16(bc, sh_a[k], w0);
        bc = exact_fma16(bc, sh_a[k + 1u], w1);
        bc = exact_fma16(bc, sh_a[k + 2u], w2);
        bc = exact_fma16(bc, sh_a[k + 3u], w3);
        w0 = w4; w1 = w5; w2 = w6; w3 = w7;
        wbase += 10240u;
        k += 4u;
    }
    bc = exact_fma16(bc, sh_a[k], w0);
    bc = exact_fma16(bc, sh_a[k + 1u], w1);
    bc = exact_fma16(bc, sh_a[k + 2u], w2);
    bc = exact_fma16(bc, sh_a[k + 3u], w3);
    bsum[idx] = bc;
"""

_FOLD_PROJ_BODY = """
    uint t = thread_index_in_threadgroup.x;
    threadgroup float sh_h1[640];
    if (!step_live) {
        for (uint lane = t; lane < 640u; lane += 128u) {
            h_state[lane] = h_state_in[lane];
            h_state[640u + lane] = h_state_in[640u + lane];
            c_state[lane] = c_state_in[lane];
            c_state[640u + lane] = c_state_in[640u + lane];
            pj16[lane] = pj16_in[lane];
            pj[lane] = float(pj16_in[lane]);
        }
        return;
    }
    for (uint lane = t; lane < 640u; lane += 128u) {
        float16_t pr[4];
        for (uint gate = 0u; gate < 4u; ++gate) {
            uint n = lane + gate * 640u;
            float16_t bacc = bsum[n];
            for (uint block = 1u; block < 10u; ++block) {
                bacc = float16_t(bacc + bsum[block * 2560u + n]);
            }
            pr[gate] = float16_t(bacc + biases[2560u + n]);
        }
        uint bi = packHalf2x16(vec2(float(pr[0]), 0.0f)) & 0xFFFFu;
        uint bf = packHalf2x16(vec2(float(pr[1]), 0.0f)) & 0xFFFFu;
        uint bo = packHalf2x16(vec2(float(pr[2]), 0.0f)) & 0xFFFFu;
        uint bg = packHalf2x16(vec2(float(pr[3]), 0.0f)) & 0xFFFFu;
        float16_t si = luts[bi];
        float16_t sf = luts[bf];
        float16_t so = luts[bo];
        float16_t tg = luts[65536u + bg];
        float16_t c0 = float16_t(cell_in[640u + lane]);
        float16_t fprod = sf * c0;
        float16_t c1 = exact_fma16(fprod, si, tg);
        uint cb = packHalf2x16(vec2(float(c1), 0.0f)) & 0xFFFFu;
        float16_t tc = luts[65536u + cb];
        float16_t h1 = so * tc;
        h_state[lane] = float(h0_in[lane]);
        h_state[640u + lane] = float(h1);
        c_state[lane] = c0_in[lane];
        c_state[640u + lane] = float(c1);
        sh_h1[lane] = float(h1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint lane = t; lane < 640u; lane += 128u) {
        precise float acc = 0.0f;
        float16_t pw0 = projector[0u * 640u + lane];
        float16_t pw1 = projector[1u * 640u + lane];
        float16_t pw2 = projector[2u * 640u + lane];
        float16_t pw3 = projector[3u * 640u + lane];
        float16_t pw4 = projector[4u * 640u + lane];
        float16_t pw5 = projector[5u * 640u + lane];
        float16_t pw6 = projector[6u * 640u + lane];
        float16_t pw7 = projector[7u * 640u + lane];
        for (uint k = 0u; k < 632u; k += 8u) {
            float16_t pn0 = projector[(k + 8u) * 640u + lane];
            float16_t pn1 = projector[(k + 9u) * 640u + lane];
            float16_t pn2 = projector[(k + 10u) * 640u + lane];
            float16_t pn3 = projector[(k + 11u) * 640u + lane];
            float16_t pn4 = projector[(k + 12u) * 640u + lane];
            float16_t pn5 = projector[(k + 13u) * 640u + lane];
            float16_t pn6 = projector[(k + 14u) * 640u + lane];
            float16_t pn7 = projector[(k + 15u) * 640u + lane];
            acc = acc + float(float16_t(sh_h1[k + 0u])) * float(pw0);
            acc = acc + float(float16_t(sh_h1[k + 1u])) * float(pw1);
            acc = acc + float(float16_t(sh_h1[k + 2u])) * float(pw2);
            acc = acc + float(float16_t(sh_h1[k + 3u])) * float(pw3);
            acc = acc + float(float16_t(sh_h1[k + 4u])) * float(pw4);
            acc = acc + float(float16_t(sh_h1[k + 5u])) * float(pw5);
            acc = acc + float(float16_t(sh_h1[k + 6u])) * float(pw6);
            acc = acc + float(float16_t(sh_h1[k + 7u])) * float(pw7);
            pw0 = pn0; pw1 = pn1; pw2 = pn2; pw3 = pn3;
            pw4 = pn4; pw5 = pn5; pw6 = pn6; pw7 = pn7;
        }
        acc = acc + float(float16_t(sh_h1[632u + 0u])) * float(pw0);
        acc = acc + float(float16_t(sh_h1[632u + 1u])) * float(pw1);
        acc = acc + float(float16_t(sh_h1[632u + 2u])) * float(pw2);
        acc = acc + float(float16_t(sh_h1[632u + 3u])) * float(pw3);
        acc = acc + float(float16_t(sh_h1[632u + 4u])) * float(pw4);
        acc = acc + float(float16_t(sh_h1[632u + 5u])) * float(pw5);
        acc = acc + float(float16_t(sh_h1[632u + 6u])) * float(pw6);
        acc = acc + float(float16_t(sh_h1[632u + 7u])) * float(pw7);
        float16_t pjv = float16_t(float16_t(acc) + projector[640u * 640u + lane]);
        pj16[lane] = pjv;
        pj[lane] = float(pjv);
    }
"""

_WINDOW_1ROW_BODY = """
    uint g = threadgroup_position_in_grid.x;
    uint t = thread_index_in_threadgroup.x;
    uint j = g * 128u + t;
    threadgroup float s_val[128];
    threadgroup uint s_idx[128];
    threadgroup float16_t sh_relu[640];
    int frame = ctl[_C_FRAME];
    int valid = cfg[_G_VALID];
    for (uint i = t; i < 640u; i += 128u) {
        if (frame >= 0 && frame < valid) {
            float16_t rlv = float16_t(encoder[uint(frame) * 640u + i]) + pj16_in[i];
            sh_relu[i] = (rlv > float16_t(0.0f)) ? rlv : float16_t(0.0f);
        } else {
            sh_relu[i] = float16_t(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    precise float acc = 0.0f;
    if (j < 8198u) {
        for (uint k = 0u; k < 640u; k += 4u) {
            float w0 = float(joint[(k + 0u) * 8198u + j]);
            float w1 = float(joint[(k + 1u) * 8198u + j]);
            float w2 = float(joint[(k + 2u) * 8198u + j]);
            float w3 = float(joint[(k + 3u) * 8198u + j]);
            acc = acc + float(sh_relu[k + 0u]) * w0;
            acc = acc + float(sh_relu[k + 1u]) * w1;
            acc = acc + float(sh_relu[k + 2u]) * w2;
            acc = acc + float(sh_relu[k + 3u]) * w3;
        }
        acc = float(float16_t(acc) + joint[640u * 8198u + j]);
    }
    // Map NaN to +3.4e38 so the first (smallest-index) NaN wins ties under
    // strict >, matching np.argmax semantics without divergent NaN branches.
    float v = (j < 8193u) ? ((acc == acc) ? acc : 3.4e38f) : -3.4e38f;
    uint ix = j;
    s_val[t] = v;
    s_idx[t] = ix;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 64u; stride >= 1u; stride >>= 1u) {
        if (t < stride) {
            float v2 = s_val[t + stride];
            uint ix2 = s_idx[t + stride];
            if (v2 > v || (v2 == v && ix2 < ix)) { v = v2; ix = ix2; }
            s_val[t] = v;
            s_idx[t] = ix;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0u) {
        pval_t[g] = v;
        pidx_t[g] = int(ix);
    }
    if (g == 64u && t >= 1u && t <= 5u) {
        s_val[120u + t] = (acc == acc) ? acc : 3.4e38f;
        s_idx[120u + t] = j;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (g == 64u && t == 0u) {
        float dv = s_val[121u];
        uint di = s_idx[121u];
        for (uint i = 2u; i <= 5u; ++i) {
            float v2 = s_val[120u + i];
            uint ix2 = s_idx[120u + i];
            if (v2 > dv || (v2 == dv && ix2 < di)) { dv = v2; di = ix2; }
        }
        pval_d[0] = dv;
        pidx_d[0] = int(di) - 8193;
    }
"""

_CONTROL_1ROW_BODY = """
    uint t = thread_index_in_threadgroup.x;
    int frame = ctl[_C_FRAME];
    int sym = ctl[_C_SYM];
    int count = ctl[_C_COUNT];
    int err = ctl[_C_ERR];
    int token = ctl[_C_TOKEN];
    int done = ctl[_C_DONE];
    int run = ctl[_C_RUN];
    int need = 0;
    int last = ctl[_C_LAST];
    if (ctl[_C_SKIP] == 0 && run != 0 && done == 0) { last = sid[0]; }

    // Copy already-emitted triples [0, count*3); strictly disjoint from the
    // new triple written at [count*3, count*3+3) by thread 0, so no barrier
    // is needed inside this kernel.
    uint copy_n = (count > 0) ? uint(count * 3) : 0u;
    for (uint i = t; i < copy_n; i += 128u) {
        emissions_next[i] = emissions_in[i];
    }

    if (t == 0u) {
        if (run != 0 && done == 0 && err == 0) {
            if (frame >= cfg[_G_VALID]) {
                done = 1;
            } else {
                float p0 = pval_t[0];
                float bv = (p0 == p0) ? p0 : 3.4e38f;
                int bi = pidx_t[0];
                for (uint g = 1u; g < 65u; ++g) {
                    float pg = pval_t[g];
                    float v2 = (pg == pg) ? pg : 3.4e38f;
                    int i2 = pidx_t[g];
                    if (v2 > bv || (v2 == bv && i2 < bi)) {
                        bv = v2;
                        bi = i2;
                    }
                }
                int tok = bi;
                int duri = pidx_d[0];
                if (duri < 0 || duri >= cfg[_G_NDUR]) {
                    err = 1;
                } else {
                    int d = cfg[_G_DUR0 + duri];
                    if (tok == cfg[_G_BLANK]) {
                        int hop = (d > 1) ? d : 1;
                        frame += hop;
                        sym = 0;
                        // Match greedy_tdt_decode: while count == 0,
                        // input_token == blank so decoder state is reused;
                        // once a token has been emitted, input_token != blank
                        // so every subsequent frame re-steps the decoder.
                        need = (count == 0) ? 1 : 0;
                    } else if (count >= cfg[_G_CAP]) {
                        err = 2;
                    } else {
                        emissions_next[count * 3 + 0] = tok;
                        emissions_next[count * 3 + 1] = frame;
                        emissions_next[count * 3 + 2] = d;
                        count++;
                        token = tok;
                        sym++;
                        need = 0;
                        if (d > 0) {
                            frame += d;
                            sym = 0;
                        } else if (sym >= cfg[_G_MAXSYM]) {
                            frame += 1;
                            sym = 0;
                        }
                    }
                }
            }
            if (frame >= cfg[_G_VALID]) { done = 1; }
        }
        ctl_next[_C_FRAME] = frame;
        ctl_next[_C_SYM] = sym;
        ctl_next[_C_COUNT] = count;
        ctl_next[_C_ERR] = err;
        ctl_next[_C_SKIP] = need;
        ctl_next[_C_TOKEN] = token;
        ctl_next[_C_DONE] = done;
        ctl_next[_C_RUN] = (err == 0 && done == 0) ? 1 : 0;
        ctl_next[_C_LAST] = last;
    }
"""


def _build_chains_l0() -> str:
    return _subst(
        _STEP_LIVE + _CHAINS_L0_BODY,
        _C_TOKEN=str(_C_TOKEN),
        _C_RUN=str(_C_RUN),
        _C_DONE=str(_C_DONE),
        _C_SKIP=str(_C_SKIP),
    )


def _build_chain1_fold() -> str:
    return _subst(
        _STEP_LIVE + _CHAIN1_FOLD_BODY,
        _C_RUN=str(_C_RUN),
        _C_DONE=str(_C_DONE),
        _C_SKIP=str(_C_SKIP),
    )


def _build_fold_proj() -> str:
    return _subst(
        _STEP_LIVE + _FOLD_PROJ_BODY,
        _C_RUN=str(_C_RUN),
        _C_DONE=str(_C_DONE),
        _C_SKIP=str(_C_SKIP),
    )


def _build_window() -> str:
    return _subst(
        _WINDOW_1ROW_BODY,
        _C_FRAME=str(_C_FRAME),
        _G_VALID=str(_G_VALID),
    )


def _build_control() -> str:
    return _subst(
        _CONTROL_1ROW_BODY,
        _C_FRAME=str(_C_FRAME),
        _C_SYM=str(_C_SYM),
        _C_COUNT=str(_C_COUNT),
        _C_ERR=str(_C_ERR),
        _C_SKIP=str(_C_SKIP),
        _C_TOKEN=str(_C_TOKEN),
        _C_DONE=str(_C_DONE),
        _C_RUN=str(_C_RUN),
        _C_LAST=str(_C_LAST),
        _G_VALID=str(_G_VALID),
        _G_BLANK=str(_G_BLANK),
        _G_MAXSYM=str(_G_MAXSYM),
        _G_NDUR=str(_G_NDUR),
        _G_DUR0=str(_G_DUR0),
        _G_CAP3=str(_G_CAP3),
        _G_CAP=str(_G_CAP),
    )


@cache
def _kernels():
    mx = _mx()
    c0 = mx.fast.metal_kernel(
        name="parakeet_tdt_chain_wg128_c0",
        input_names=["embedding", "weights", "hidden_in", "ctl"],
        output_names=["bsum"],
        header=_EXACT_FMA16,
        source=_build_chains_l0(),
        compile_options={"math_mode": "safe"},
    )
    c1 = mx.fast.metal_kernel(
        name="parakeet_tdt_chain_wg128_c1_fold",
        input_names=["bsum0", "weights", "hidden_in", "cell_in", "biases",
                     "luts", "ctl"],
        output_names=["bsum", "h0_out", "c0_out"],
        header=_EXACT_FMA16,
        source=_build_chain1_fold(),
        compile_options={"math_mode": "safe"},
    )
    fp = mx.fast.metal_kernel(
        name="parakeet_tdt_chain_wg128_fold_proj",
        input_names=["bsum", "biases", "luts", "cell_in", "h0_in", "c0_in",
                     "h_state_in", "c_state_in", "pj16_in", "projector", "ctl"],
        output_names=["h_state", "c_state", "pj", "pj16"],
        header=_EXACT_FMA16,
        source=_build_fold_proj(),
        compile_options={"math_mode": "safe"},
    )
    win = mx.fast.metal_kernel(
        name="parakeet_tdt_chain_wg128_window",
        input_names=["pj16_in", "encoder", "joint", "ctl", "cfg"],
        output_names=["pval_t", "pidx_t", "pval_d", "pidx_d"],
        header="",
        source=_build_window(),
        compile_options={"math_mode": "safe"},
    )
    ctl = mx.fast.metal_kernel(
        name="parakeet_tdt_chain_wg128_control",
        input_names=["pval_t", "pidx_t", "pidx_d", "emissions_in", "ctl",
                     "cfg", "sid"],
        output_names=["ctl_next", "emissions_next"],
        header="",
        source=_build_control(),
        compile_options={"math_mode": "safe"},
    )
    return c0, c1, fp, win, ctl


def run_tdt_chain_wg128(
    packed,
    encoder,
    valid_frames: int,
    config,
    initial_hidden,
    initial_cell,
    slots_per_chunk: int = 64,
    max_chunks: int = 12,
) -> TdtChainOutput:
    """Device-chained greedy TDT decode using <=128-thread workgroups."""
    mx = _mx()
    blank = int(config.blank_token_id)
    durs = [int(d) for d in config.durations]
    maxsym = int(config.max_symbols_per_step)
    cap = max(1, int(valid_frames) * maxsym)
    total_slots = int(valid_frames) + cap + 1
    cfg = np.zeros((_G_CAP3 + 1,), np.int32)
    cfg[_G_VALID] = int(valid_frames)
    cfg[_G_BLANK] = blank
    cfg[_G_MAXSYM] = maxsym
    cfg[_G_NDUR] = len(durs)
    for i, d in enumerate(durs):
        cfg[_G_DUR0 + i] = d
    cfg[_G_CAP] = cap
    cfg[_G_CAP3] = cap * 3
    cfg_dev = mx.array(cfg)

    ctl0 = np.zeros((_CTRL_WORDS,), np.int32)
    ctl0[_C_FRAME] = 0
    ctl0[_C_SYM] = 0
    ctl0[_C_COUNT] = 0
    ctl0[_C_ERR] = 0
    ctl0[_C_SKIP] = 1 if valid_frames == 0 else 0
    ctl0[_C_TOKEN] = blank
    ctl0[_C_DONE] = 0
    ctl0[_C_RUN] = 1
    ctl0[_C_LAST] = -1

    emissions = mx.array(np.zeros((cap * 3,), np.int32))
    enc_flat = encoder.reshape(-1)
    cap3 = cap * 3

    h_state = initial_hidden.reshape(1280,)
    c_state = initial_cell.reshape(1280,)
    pj16_prev = mx.array(np.zeros((640,), np.float16))
    folds = {}
    ctl_rows = [mx.array(ctl0)]
    slot = 0
    chunks = 0

    c0_k, c1_k, fp_k, win_k, ctl_k = _kernels()

    with mx.stream(mx.gpu):
        while chunks < max_chunks and slot < total_slots:
            end = min(slot + slots_per_chunk, total_slots)
            for i in range(slot, end):
                ctl_i = ctl_rows[i]
                sid = _sid_array(i)
                bsum0, = c0_k(
                    inputs=[packed.embedding, packed.weights, h_state, ctl_i],
                    output_shapes=[(25600,)],
                    output_dtypes=[mx.float16],
                    grid=(_CHAIN_GROUPS * _CHAIN_THREADS, 1, 1),
                    threadgroup=(_CHAIN_THREADS, 1, 1),
                    stream=mx.gpu,
                )
                bsum1, h0, c0 = c1_k(
                    inputs=[bsum0, packed.weights, h_state, c_state,
                            packed.biases, packed.luts, ctl_i],
                    output_shapes=[(25600,), (640,), (640,)],
                    output_dtypes=[mx.float16, mx.float32, mx.float32],
                    grid=(_CHAIN_GROUPS * _CHAIN_THREADS, 1, 1),
                    threadgroup=(_CHAIN_THREADS, 1, 1),
                    stream=mx.gpu,
                )
                h_s, c_s, pj, pj16 = fp_k(
                    inputs=[bsum1, packed.biases, packed.luts, c_state, h0, c0,
                            h_state, c_state, pj16_prev, packed.projector,
                            ctl_i],
                    output_shapes=[(1280,), (1280,), (640,), (640,)],
                    output_dtypes=[mx.float32, mx.float32, mx.float32,
                                   mx.float16],
                    grid=(_CHAIN_THREADS, 1, 1),
                    threadgroup=(_CHAIN_THREADS, 1, 1),
                    stream=mx.gpu,
                )
                pval_t, pidx_t, pval_d, pidx_d = win_k(
                    inputs=[pj16, enc_flat, packed.joint, ctl_i, cfg_dev],
                    output_shapes=[(_NGROUPS,), (_NGROUPS,), (1,), (1,)],
                    output_dtypes=[mx.float32, mx.int32, mx.float32, mx.int32],
                    grid=(_NGROUPS * _CHAIN_THREADS, 1, 1),
                    threadgroup=(_CHAIN_THREADS, 1, 1),
                    stream=mx.gpu,
                )
                ctl_next, emissions_next = ctl_k(
                    inputs=[pval_t, pidx_t, pidx_d, emissions, ctl_i, cfg_dev,
                            sid],
                    output_shapes=[(_CTRL_WORDS,), (cap3,)],
                    output_dtypes=[mx.int32, mx.int32],
                    grid=(_CHAIN_THREADS, 1, 1),
                    threadgroup=(_CHAIN_THREADS, 1, 1),
                    stream=mx.gpu,
                )
                ctl_rows.append(ctl_next)
                emissions = emissions_next
                h_state, c_state = h_s, c_s
                pj16_prev = pj16
                folds[i] = (h_s, c_s, pj)
            mx.eval(ctl_rows[end])
            row = np.asarray(ctl_rows[end])
            slot = end
            chunks += 1
            if int(row[_C_DONE]) or not int(row[_C_RUN]):
                break

    row = np.asarray(ctl_rows[slot])
    err = int(row[_C_ERR])
    if err != 0:
        reason = ("duration index outside class count" if err == 1
                  else "emission capacity exceeded")
        raise ValueError(f"[omarchy-parakeet] TDT chain failed: {reason}")
    if not int(row[_C_DONE]):
        raise ValueError(
            "[omarchy-parakeet] TDT chain slot budget exhausted "
            f"({slot} slots, frame {int(row[_C_FRAME])} of {valid_frames})"
        )
    count = int(row[_C_COUNT])
    em = np.asarray(emissions)[: count * 3].reshape(count, 3)
    last = int(row[_C_LAST])
    if last >= 0:
        h_s, c_s, _pj = folds[last]
        hidden = np.asarray(h_s).reshape(2, 1, 640)
        cell = np.asarray(c_s).reshape(2, 1, 640)
    else:
        hidden = np.asarray(initial_hidden).reshape(2, 1, 640)
        cell = np.asarray(initial_cell).reshape(2, 1, 640)
    return TdtChainOutput(
        token_ids=[int(x) for x in em[:, 0]],
        frame_indices=[int(x) for x in em[:, 1]],
        durations=[int(x) for x in em[:, 2]],
        hidden=hidden,
        cell=cell,
        final_frame=int(row[_C_FRAME]),
        slots_used=slot,
    )
