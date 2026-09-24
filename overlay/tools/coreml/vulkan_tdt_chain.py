# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Device-chained greedy TDT decode: per-emission slots, no per-slot sync.

The host control loop (``parakeet_tdt.greedy_tdt_decode``) pays one
submission round trip per emission plus the fused step's dispatch queue;
the speculative joint window (``vulkan_decoder_step`` ``spec_frames``)
removed the frame-entry submits but the ~104 emission submits remain the
TDT floor.  This module removes them: the decode is enqueued as a fixed
schedule of per-emission slots whose control state lives in DEVICE
buffers, so a whole chunk of slots becomes one submission (one
``mx.eval`` per chunk instead of per emission).

Per active slot (six queued dispatches):

1. ``chains`` layer 0 (embedding row looked up from the device token),
2. ``fold`` layer 0 (contract-order block fold + LUT gates + cell),
3. ``chains`` layer 1,
4. ``fold_proj`` layer 1 fused with the projector (h1 staged in
   threadgroup memory; assembles the whole decoder state and passes it
   through untouched on inactive slots),
5. ``window`` joint head: seven consecutive encoder frames against this
   slot's decoder state (row 0 is the emission joint; rows past
   ``valid_frames`` zero-fill), weights streamed with the exact
   ``_JOINT_SOURCE`` element order,
6. ``control``: in-kernel first-max argmax per row (``np.argmax``
   semantics: strict ``>`` on an ascending scan, smaller index wins
   ties, first NaN wins) plus the duration-head argmax, then the greedy
   TDT walk appending emissions and writing the next slot's control
   word.

Every slot reads its control word from the previous slot's control
output, so the chain runs on the device.  A decode longer than one
chunk — or a blank run longer than six frames exhausting the window —
ends the chunk cleanly (``run=0`` or ``skip=1``) and the host re-enqueues
from the device control word: the chunk boundary costs one round trip,
not a decision.

Bit-exactness contract (identical element order to the landed path):

* LSTM: ``exact_fma16`` block chains and the fold+cell of
  ``vulkan_decoder_step`` verbatim (fp16 0 start, ascending k, ten block
  sums folded first-term-assigned, bit-indexed LUT gate pairing, cell).
* Projector: fp32 ascending chain, one fp16 rounding, fp16 bias — the
  ``mx.matmul`` contract; the fused fold stages h1 in threadgroup
  memory and the projector consumes exactly those values.
* Joint rows: fp16 relu build, fp32 ascending k chain, one fp16
  rounding, fp16 bias; row 0 reproduces the single-frame joint logits
  bit for bit.  A joint-only slot (window exhausted) reuses the stored
  ``pj16`` — the same ``float16_t(pj)`` the landed jmode-1 rebuild
  round-trips through — so its rows match the fallback bit for bit.
* Control: the ``greedy_tdt_decode`` state machine including the
  max-symbol roll-over (``frame += 1`` when the inner loop exhausts
  without advancing) and the emission/frame/duration recording order.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from .vulkan_decoder_step import _CHAIN_GROUPS, _CHAIN_THREADS, _EXACT_FMA16

_WINDOW_ROWS = 7  # row 0 = emission joint, rows 1..6 = frame-entry spec
_JOINT_OUT = 8198
_NGROUPS = 33  # window workgroups: ceil(8198 / 256)

# Control word layout (int32, one row per slot).
_C_FRAME = 0
_C_SYM = 1
_C_COUNT = 2
_C_ERR = 3
_C_SKIP = 4  # this slot is joint-only (previous window exhausted)
_C_TOKEN = 5
_C_DONE = 6
_C_RUN = 7
_C_LAST = 8  # slot index of the last executed decoder step (-1: none)
_CTRL_WORDS = 9

# Config layout (int32).
_G_VALID = 0
_G_BLANK = 1
_G_MAXSYM = 2
_G_NDUR = 3
_G_DUR0 = 4  # five duration classes
_G_CAP = 9
_G_CAP3 = 10

# Workgroup-uniform guards, inlined (the translator emits the source into
# one function, so helper definitions are a syntax error).  ``step_live``
# gates the LSTM/projector step (skipped on joint-only and finished
# slots); ``loop_live`` gates the joint window and control walk (they run
# on joint-only slots too).
_STEP_LIVE = "bool step_live = (ctl[_C_RUN] != 0) && (ctl[_C_DONE] == 0) && (ctl[_C_SKIP] == 0);"
_LOOP_LIVE = "bool loop_live = (ctl[_C_RUN] != 0) && (ctl[_C_DONE] == 0);"

_CHAINS_BODY = """
    uint g = threadgroup_position_in_grid.x;
    uint t = thread_index_in_threadgroup.x;
    uint idx = g * _CHANTHREADSu + t;
    int token_id = ctl[_C_TOKEN];
    threadgroup float16_t sh_a[1280];
    if (!step_live) {
        bsum[idx] = float16_t(0.0f);
        return;
    }
    for (uint i = t; i < 1280u; i += _CHANTHREADSu) {
        if (i < 640u) {
            if (_LAYERu == 0u) {
                uint tbase = uint(token_id);
                if (token_id < 0) { tbase = uint(token_id + 8193); }
                sh_a[i] = embedding[tbase * 640u + i];
            } else {
                sh_a[i] = float16_t(x_in[i]);
            }
        } else {
            sh_a[i] = float16_t(hidden_in[uint(_LAYER) * 640u + (i - 640u)]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n = idx % 2560u;
    uint gb = idx / 2560u;
    uint block = gb % 10u;
    float16_t bc = float16_t(0.0f);
    uint k = block * 128u;
    uint wbase = uint(_LAYER) * 3276800u + k * 2560u + n;
    // Loads are independent of the bc chain: prefetch four weight elements
    // ahead so the DRAM stream stays in flight.  The exact_fma16 adds keep
    // their ascending-k order, so the result is bit-identical.
    float16_t w0 = weights[wbase];
    float16_t w1 = weights[wbase + 2560u];
    float16_t w2 = weights[wbase + 5120u];
    float16_t w3 = weights[wbase + 7680u];
    for (uint j = 0u; j < 128u; j += 4u) {
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
    bsum[idx] = bc;
"""

_FOLD_BODY = """
    uint t = thread_index_in_threadgroup.x;
    float16_t pr[4];
    if (!step_live) {
        h1_out[t] = 0.0f;
        c1_out[t] = 0.0f;
        return;
    }
    uint lane = t;
    for (uint gate = 0u; gate < 4u; ++gate) {
        uint n = lane + gate * 640u;
        float16_t bacc = bsum[n];
        for (uint block = 1u; block < 10u; ++block) {
            bacc = float16_t(bacc + bsum[block * 2560u + n]);
        }
        pr[gate] = float16_t(bacc + biases[0u * 2560u + n]);
    }
    uint bi = packHalf2x16(vec2(float(pr[0]), 0.0f)) & 0xFFFFu;
    uint bf = packHalf2x16(vec2(float(pr[1]), 0.0f)) & 0xFFFFu;
    uint bo = packHalf2x16(vec2(float(pr[2]), 0.0f)) & 0xFFFFu;
    uint bg = packHalf2x16(vec2(float(pr[3]), 0.0f)) & 0xFFFFu;
    float16_t si = luts[bi];
    float16_t sf = luts[bf];
    float16_t so = luts[bo];
    float16_t tg = luts[65536u + bg];
    float16_t c0 = float16_t(cell_in[0u * 640u + lane]);
    float16_t fprod = sf * c0;
    float16_t c1 = exact_fma16(fprod, si, tg);
    uint cb = packHalf2x16(vec2(float(c1), 0.0f)) & 0xFFFFu;
    float16_t tc = luts[65536u + cb];
    float16_t h1 = so * tc;
    h1_out[lane] = float(h1);
    c1_out[lane] = float(c1);
"""

_FOLD_PROJ_BODY = """
    uint t = thread_index_in_threadgroup.x;
    uint lane = t;
    float16_t pr[4];
    threadgroup float sh_h1[640];
    if (!step_live) {
        h_state[lane] = h_state_in[lane];
        h_state[640u + lane] = h_state_in[640u + lane];
        c_state[lane] = c_state_in[lane];
        c_state[640u + lane] = c_state_in[640u + lane];
        pj16[lane] = pj16_in[lane];
        pj[lane] = float(pj16_in[lane]);
        return;
    }
    for (uint gate = 0u; gate < 4u; ++gate) {
        uint n = lane + gate * 640u;
        float16_t bacc = bsum[n];
        for (uint block = 1u; block < 10u; ++block) {
            bacc = float16_t(bacc + bsum[block * 2560u + n]);
        }
        pr[gate] = float16_t(bacc + biases[1u * 2560u + n]);
    }
    uint bi = packHalf2x16(vec2(float(pr[0]), 0.0f)) & 0xFFFFu;
    uint bf = packHalf2x16(vec2(float(pr[1]), 0.0f)) & 0xFFFFu;
    uint bo = packHalf2x16(vec2(float(pr[2]), 0.0f)) & 0xFFFFu;
    uint bg = packHalf2x16(vec2(float(pr[3]), 0.0f)) & 0xFFFFu;
    float16_t si = luts[bi];
    float16_t sf = luts[bf];
    float16_t so = luts[bo];
    float16_t tg = luts[65536u + bg];
    float16_t c0 = float16_t(cell_in[1u * 640u + lane]);
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
    precise float acc = 0.0f;
    for (uint k = 0u; k < 640u; ++k) {
        acc = acc + float(float16_t(sh_h1[k]))
              * float(projector[k * 640u + lane]);
    }
    float16_t pjv = float16_t(float16_t(acc) + projector[640u * 640u + lane]);
    pj16[lane] = pjv;
    pj[lane] = float(pjv);
"""

_WINDOW_BODY = """
    uint g = threadgroup_position_in_grid.x;
    uint t = thread_index_in_threadgroup.x;
    uint j = g * 256u + t;
    threadgroup float s_val[256];
    threadgroup uint s_idx[256];
    threadgroup float16_t sh_relu[ROWS6];
    if (!loop_live) {
        return;
    }
    int base = ctl[_C_FRAME];
    int valid = cfg[_G_VALID];
    for (uint slot = 0u; slot < ROWSU; ++slot) {
        int frame = base + int(slot);
        for (uint i = t; i < 640u; i += 256u) {
            if (frame < valid) {
                float16_t rlv = float16_t(encoder[uint(frame) * 640u + i])
                                + pj16_in[i];
                sh_relu[slot * 640u + i] =
                    (rlv > float16_t(0.0f)) ? rlv : float16_t(0.0f);
            } else {
                sh_relu[slot * 640u + i] = float16_t(0.0f);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
_ACC_DECLS
    uint nrows = (ctl[_C_COUNT] == 0u) ? ROWSU : 1u;
    if (j < 8198u) {
        if (nrows == ROWSU) {
            for (uint k = 0u; k < 640u; ++k) {
                precise float w = float(joint[k * 8198u + j]);
_ACC_UPDATES
            }
        } else {
_ACC_UPDATES_1
        }
    }
    // Per-row argmax partials over the FINAL logits (one fp16 rounding +
    // fp16 bias per element, exactly the landed joint output): token
    // columns j < 8193 reduce inside the workgroup (sentinel for threads
    // without a token column); the five duration columns live in
    // workgroup 32 and reduce there.  The comparator is np.argmax's:
    // strict >, smaller index wins ties, first NaN wins.
_AV_ARRAY
    for (uint row = 0u; row < nrows; ++row) {
        float v = (j < 8193u)
                  ? float(float16_t(av[row]) + joint[640u * 8198u + j])
                  : -3.0e38f;
        uint ix = j;
        s_val[t] = v; s_idx[t] = ix;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128u; stride >= 1u; stride >>= 1u) {
            if (t < stride) {
                float v2 = s_val[t + stride];
                uint ix2 = s_idx[t + stride];
                bool take;
                if (v2 != v2) { take = (v != v) ? (ix2 < ix) : true; }
                else if (v != v) { take = false; }
                else { take = (v2 > v) || (v2 == v && ix2 < ix); }
                if (take) { v = v2; ix = ix2; }
                s_val[t] = v; s_idx[t] = ix;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (t == 0u) {
            pval_t[row * NGRPS + g] = v;
            pidx_t[row * NGRPS + g] = int(ix);
        }
        if (g == 32u && t >= 1u && t <= 5u) {
            s_val[256u - 8u + t] =
                float(float16_t(av[row]) + joint[640u * 8198u + j]);
            s_idx[256u - 8u + t] = j;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (g == 32u && t == 0u) {
            float dv = s_val[248u + 1u];
            uint di = s_idx[248u + 1u];
            for (uint i = 2u; i <= 5u; ++i) {
                float v2 = s_val[248u + i];
                uint ix2 = s_idx[248u + i];
                bool take;
                if (v2 != v2) { take = (dv == dv) ? false : (ix2 < di); }
                else if (dv != dv) { take = true; }
                else { take = (v2 > dv) || (v2 == dv && ix2 < di); }
                if (take) { dv = v2; di = ix2; }
            }
            pval_d[row] = dv;
            pidx_d[row] = int(di) - 8193;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""


_REDUCE = """
        s_val[t] = bv; s_idx[t] = bi;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0u) {
            float v0 = s_val[0]; uint i0 = s_idx[0];
            for (uint i = 1u; i < 1024u; ++i) {
                float v = s_val[i]; uint ix = s_idx[i];
                bool take;
                if (v != v) { take = (v0 == v0) || (ix < i0); }
                else if (v0 != v0) { take = false; }
                else { take = (v > v0) || (v == v0 && ix < i0); }
                if (take) { v0 = v; i0 = ix; }
            }
            s_tok[row] = int(i0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t < 5u) {
            s_val[t] = logits[row * 8198u + 8193u + t];
            s_idx[t] = t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0u) {
            float v0 = s_val[0]; uint i0 = 0u;
            for (uint i = 1u; i < 5u; ++i) {
                float v = s_val[i]; uint ix = s_idx[i];
                bool take;
                if (v != v) { take = (v0 == v0) || (ix < i0); }
                else if (v0 != v0) { take = false; }
                else { take = (v > v0) || (v == v0 && ix < i0); }
                if (take) { v0 = v; i0 = ix; }
            }
            s_duri[row] = int(i0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
"""

_CONTROL_BODY = """
    uint t = thread_index_in_threadgroup.x;
    int frame = ctl[_C_FRAME];
    int sym = ctl[_C_SYM];
    int count = ctl[_C_COUNT];
    int err = ctl[_C_ERR];
    int token = ctl[_C_TOKEN];
    int done = ctl[_C_DONE];
    int run = ctl[_C_RUN];
    int need = 0;
    int adv = 0;
    int last = ctl[_C_LAST];
    if (ctl[_C_SKIP] == 0 && run != 0 && done == 0) { last = sid[0]; }

    bool loop_live = (ctl[_C_RUN] != 0) && (ctl[_C_DONE] == 0);
    threadgroup float s_val[64];
    threadgroup uint s_idx[64];
    threadgroup int s_tok[ROWSC];
    threadgroup int s_duri[ROWSC];
    for (uint i = t; i < cfg[_G_CAP3]; i += 1024u) {
        emissions_next[i] = emissions_in[i];
    }
    if (loop_live) {
        // reduce the window's 33 per-workgroup token partials per row,
        // then carry the duration partial (workgroup 32) unchanged
        for (uint row = 0u; row < ROWSU; ++row) {
            float v = -3.0e38f;
            uint ix = 0u;
            if (t < NGRPSu) {
                v = pval_t[row * NGRPS + t];
                ix = uint(pidx_t[row * NGRPS + t]);
            }
            if (t < 64u) { s_val[t] = v; s_idx[t] = ix; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (t == 0u) {
                float v0 = -3.0e38f; uint i0 = 0u;
                bool have = false;
                for (uint i = 0u; i < NGRPSu; ++i) {
                    float v2 = s_val[i]; uint ix2 = s_idx[i];
                    bool take;
                    if (v2 != v2) { take = (!have) ? true : (ix2 < i0); }
                    else if (!have) { take = true; }
                    else { take = (v2 > v0) || (v2 == v0 && ix2 < i0); }
                    if (take) { v0 = v2; i0 = ix2; have = true; }
                }
                s_tok[row] = int(i0);
                s_duri[row] = int(pidx_d[row]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        if (t == 0u) {
            for (uint row = 0u; row < ROWSU; ++row) {
                s_tok[row] = 0;
                s_duri[row] = 0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (t == 0u) {
        if (run != 0 && done == 0 && err == 0) {
            int r = 0;
            while (sym < cfg[_G_MAXSYM]) {
                if (frame >= cfg[_G_VALID]) { done = 1; break; }
                if (r >= ROWSC) { need = 1; break; }
                int tok = s_tok[r];
                int duri = s_duri[r];
                if (duri < 0 || duri >= cfg[_G_NDUR]) { err = 1; break; }
                int d = cfg[_G_DUR0 + duri];
                if (tok == cfg[_G_BLANK]) {
                    int hop = (d > 1) ? d : 1;
                    frame += hop;
                    if (count != 0) {
                        // state changes at every frame entry once a token
                        // has been emitted (input_token != blank): the
                        // next slot must re-step, rows are stale
                        sym = 0;
                        break;
                    }
                    r += hop;
                    adv = 1;
                    sym = 0;
                    continue;
                }
                if (count >= cfg[_G_CAP]) { err = 2; break; }
                emissions_next[count * 3 + 0] = tok;
                emissions_next[count * 3 + 1] = frame;
                emissions_next[count * 3 + 2] = d;
                count++;
                token = tok;
                sym++;
                if (d > 0) {
                    frame += d;
                    sym = 0;
                    break;
                }
                break;  // d == 0: next slot re-runs the decoder at this frame
            }
            if (err == 0 && done == 0 && adv == 0 && sym >= cfg[_G_MAXSYM]) {
                frame += 1;
                sym = 0;
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


def _subst(template: str, **repl: str) -> str:
    out = template
    for key, value in repl.items():
        out = out.replace(key, value)
    return out


def _build_chains(layer: int) -> str:
    return _subst(
        _STEP_LIVE + _CHAINS_BODY,
        _CHANTHREADS=str(_CHAIN_THREADS),
        _LAYER=str(layer),
        _C_TOKEN=str(_C_TOKEN),
        _C_RUN=str(_C_RUN),
        _C_DONE=str(_C_DONE),
        _C_SKIP=str(_C_SKIP),
    )


def _build_fold() -> str:
    return _subst(
        _STEP_LIVE + _FOLD_BODY,
        _C_RUN=str(_C_RUN), _C_DONE=str(_C_DONE), _C_SKIP=str(_C_SKIP),
    )


def _build_fold_proj() -> str:
    return _subst(
        _STEP_LIVE + _FOLD_PROJ_BODY,
        _C_RUN=str(_C_RUN), _C_DONE=str(_C_DONE), _C_SKIP=str(_C_SKIP),
    )


def _build_window() -> str:
    acc_decls = "\n".join(
        f"        precise float acc{slot} = 0.0f;" for slot in range(_WINDOW_ROWS)
    )
    acc_updates = "\n".join(
        f"            acc{slot} = acc{slot}"
        f" + float(sh_relu[{slot}u * 640u + k]) * w;"
        for slot in range(_WINDOW_ROWS)
    )
    acc_updates_1 = (
        "            for (uint k = 0u; k < 640u; ++k) {\n"
        "                precise float w1r = float(joint[k * 8198u + j]);\n"
        "                acc0 = acc0 + float(sh_relu[k]) * w1r;\n"
        "            }"
    )
    av_array = "    float av[" + str(_WINDOW_ROWS) + "];\n" + "\n".join(
        f"    av[{slot}] = acc{slot};" for slot in range(_WINDOW_ROWS)
    )
    return _subst(
        _LOOP_LIVE + _WINDOW_BODY,
        ROWS6=str(_WINDOW_ROWS * 640),
        ROWSU=f"{_WINDOW_ROWS}u",
        AVN=str(_WINDOW_ROWS),
        ROWSC=str(_WINDOW_ROWS),
        NGRPS=str(_NGROUPS),
        NGRPSu=f"{_NGROUPS}u",
        _ACC_DECLS=acc_decls,
        _ACC_UPDATES_1=acc_updates_1,
        _ACC_UPDATES=acc_updates,
        _AV_ARRAY=av_array,
        _C_FRAME=str(_C_FRAME),
        _G_VALID=str(_G_VALID),
        _C_RUN=str(_C_RUN), _C_DONE=str(_C_DONE), _C_SKIP=str(_C_SKIP),
        _C_COUNT=str(_C_COUNT),
    )


def _build_control() -> str:
    return _subst(
        _CONTROL_BODY,
        _REDUCE=_REDUCE,
        ROWS6=str(_WINDOW_ROWS * 640),
        ROWSU=f"{_WINDOW_ROWS}u",
        AVN=str(_WINDOW_ROWS),
        ROWSC=str(_WINDOW_ROWS),
        NGRPS=str(_NGROUPS),
        NGRPSu=f"{_NGROUPS}u",
        _C_FRAME=str(_C_FRAME), _C_SYM=str(_C_SYM), _C_COUNT=str(_C_COUNT),
        _C_ERR=str(_C_ERR), _C_SKIP=str(_C_SKIP), _C_TOKEN=str(_C_TOKEN),
        _C_DONE=str(_C_DONE), _C_RUN=str(_C_RUN), _C_LAST=str(_C_LAST),
        _G_VALID=str(_G_VALID), _G_BLANK=str(_G_BLANK),
        _G_MAXSYM=str(_G_MAXSYM), _G_NDUR=str(_G_NDUR),
        _G_DUR0=str(_G_DUR0), _G_CAP3=str(_G_CAP3), _G_CAP=str(_G_CAP),
    )


@dataclass
class TdtChainOutput:
    """Emissions and recurrent state produced by the device-chained loop."""

    token_ids: list[int]
    frame_indices: list[int]
    durations: list[int]
    hidden: object
    cell: object
    final_frame: int
    slots_used: int = 0
    decode_path: str = "gpu-chain"


@cache
def _mx():
    import mlx.core as mx

    return mx


@cache
def _chains_kernel(layer: int):
    mx = _mx()
    return mx.fast.metal_kernel(
        name=f"parakeet_tdt_chain_chains_l{layer}",
        input_names=["embedding", "weights", "hidden_in", "x_in", "ctl"],
        output_names=["bsum"],
        header=_EXACT_FMA16,
        source=_build_chains(layer),
        compile_options={"math_mode": "safe"},
    )


@cache
def _fold_kernel():
    mx = _mx()
    return mx.fast.metal_kernel(
        name="parakeet_tdt_chain_fold_l0",
        input_names=["bsum", "biases", "luts", "cell_in", "ctl"],
        output_names=["h1_out", "c1_out"],
        header=_EXACT_FMA16,
        source=_build_fold(),
        compile_options={"math_mode": "safe"},
    )


@cache
def _fold_proj_kernel():
    mx = _mx()
    return mx.fast.metal_kernel(
        name="parakeet_tdt_chain_fold_proj_l1",
        input_names=["bsum", "biases", "luts", "cell_in", "h0_in", "c0_in",
                     "h_state_in", "c_state_in", "pj16_in", "projector",
                     "ctl"],
        output_names=["h_state", "c_state", "pj", "pj16"],
        header=_EXACT_FMA16,
        source=_build_fold_proj(),
        compile_options={"math_mode": "safe"},
    )


@cache
def _window_kernel():
    mx = _mx()
    return mx.fast.metal_kernel(
        name="parakeet_tdt_chain_window",
        input_names=["pj16_in", "encoder", "joint", "ctl", "cfg"],
        output_names=["pval_t", "pidx_t", "pval_d", "pidx_d"],
        header="",
        source=_build_window(),
        compile_options={"math_mode": "safe"},
    )


@cache
def _sid_array(slot: int):
    import mlx.core as mx

    return mx.array(np.array([slot], np.int32))


@cache
def _control_kernel():
    mx = _mx()
    return mx.fast.metal_kernel(
        name="parakeet_tdt_chain_control",
        input_names=["pval_t", "pidx_t", "pidx_d", "emissions_in", "ctl",
                     "cfg", "sid"],
        output_names=["ctl_next", "emissions_next"],
        header="",
        source=_build_control(),
        compile_options={"math_mode": "safe"},
    )


def run_tdt_chain(
    packed,
    encoder,
    valid_frames: int,
    config,
    initial_hidden,
    initial_cell,
    slots_per_chunk: int = 64,
    max_chunks: int = 12,
) -> TdtChainOutput:
    """Greedy TDT decode as a device-chained slot schedule.

    ``packed`` is the shared ``StepWeights``; ``encoder`` the fp32 encoder
    output (1, F, 640); ``config`` the pinned ``TdtConfig``.  Control
    state lives in device buffers; the host syncs once per chunk and only
    to decide whether another chunk is needed.
    """
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
    folds = {}  # slot -> fold_proj outputs, for the final state readout
    ctl_rows = [mx.array(ctl0)]
    slot = 0
    chunks = 0

    with mx.stream(mx.gpu):
        while chunks < max_chunks and slot < total_slots:
            end = min(slot + slots_per_chunk, total_slots)
            for i in range(slot, end):
                ctl_i = ctl_rows[i]
                sid = _sid_array(i)
                bsum0, = _chains_kernel(0)(
                    inputs=[packed.embedding, packed.weights, h_state,
                            h_state, ctl_i],
                    output_shapes=[(25600,)],
                    output_dtypes=[mx.float16],
                    grid=(_CHAIN_GROUPS * _CHAIN_THREADS, 1, 1),
                    threadgroup=(_CHAIN_THREADS, 1, 1),
                    stream=mx.gpu,
                )
                h0, c0 = _fold_kernel()(
                    inputs=[bsum0, packed.biases, packed.luts, c_state, ctl_i],
                    output_shapes=[(640,), (640,)],
                    output_dtypes=[mx.float32, mx.float32],
                    grid=(640, 1, 1), threadgroup=(640, 1, 1),
                    stream=mx.gpu,
                )
                bsum1, = _chains_kernel(1)(
                    inputs=[packed.embedding, packed.weights, h_state, h0,
                            ctl_i],
                    output_shapes=[(25600,)],
                    output_dtypes=[mx.float16],
                    grid=(_CHAIN_GROUPS * _CHAIN_THREADS, 1, 1),
                    threadgroup=(_CHAIN_THREADS, 1, 1),
                    stream=mx.gpu,
                )
                h_s, c_s, pj, pj16 = _fold_proj_kernel()(
                    inputs=[bsum1, packed.biases, packed.luts, c_state, h0,
                            c0, h_state, c_state, pj16_prev,
                            packed.projector, ctl_i],
                    output_shapes=[(1280,), (1280,), (640,), (640,)],
                    output_dtypes=[mx.float32, mx.float32, mx.float32,
                                   mx.float16],
                    grid=(640, 1, 1), threadgroup=(640, 1, 1),
                    stream=mx.gpu,
                )
                pval_t, pidx_t, pval_d, pidx_d = _window_kernel()(
                    inputs=[pj16, enc_flat, packed.joint, ctl_i, cfg_dev],
                    output_shapes=[(_WINDOW_ROWS * _NGROUPS,),
                                   (_WINDOW_ROWS * _NGROUPS,),
                                   (_WINDOW_ROWS,), (_WINDOW_ROWS,)],
                    output_dtypes=[mx.float32, mx.int32, mx.float32,
                                   mx.int32],
                    grid=(_NGROUPS * 256, 1, 1), threadgroup=(256, 1, 1),
                    stream=mx.gpu,
                )
                ctl_next, emissions_next = _control_kernel()(
                    inputs=[pval_t, pidx_t, pidx_d, emissions, ctl_i,
                            cfg_dev, sid],
                    output_shapes=[(_CTRL_WORDS,), (cap3,)],
                    output_dtypes=[mx.int32, mx.int32],
                    grid=(1024, 1, 1), threadgroup=(1024, 1, 1),
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
