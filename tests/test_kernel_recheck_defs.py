#!/usr/bin/env python3
"""Issue #206 item 2: tests for tools/kernel_recheck/defs.py.

numpy only. defs.py imports no mlx by design, and run.py defers its mlx
import to the GPU child, so the --reference-only dry run is exercised here
without mlx. No test in this file needs mlx.

Every ref_* check recomputes a small hand-built case with independent code
written in this file (plain loops / literal numbers) -- it never calls the
function under test to build its own expectation. defs.bf16 is used as a
rounding primitive only where the documented kernel contract says "round to
the bf16 grid"; bf16 itself is verified independently against a from-scratch
round-to-nearest-even construction in test_bf16_*.

Shrunken configs: refs that read module-level size constants (FLUX, CBQ,
GLUE_*) are called with those constants monkeypatched to the smallest shape
that still satisfies their internal layout constraints, so no test builds
the full cbq/flux/glue sizes.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.kernel_recheck import defs  # noqa: E402

SOURCES_DIR = REPO_ROOT / "tools" / "kernel_recheck" / "sources"


def all_builders():
    return {name: getattr(defs, name) for name in dir(defs)
            if name.startswith("build_") and callable(getattr(defs, name))}


def bf16_indep(x):
    """Independent bf16 round-to-nearest-even: compare the low 16 bits to the
    midpoint, break exact ties to even, and repack the (possibly carried)
    upper 16 bits. Deliberately a different construction from defs.bf16."""
    arr = np.asarray(x, dtype=np.float32)
    bits = arr.view(np.uint32)
    hi = bits >> np.uint32(16)
    lo = bits & np.uint32(0xFFFF)
    bump = (lo > np.uint32(0x8000)) | (
        (lo == np.uint32(0x8000)) & ((hi & np.uint32(1)) == np.uint32(1)))
    out = ((hi + bump.astype(np.uint32)) << np.uint32(16)).view(np.float32)
    return out.reshape(arr.shape)


def bits_of(a):
    return np.asarray(a, dtype=np.float32).view(np.uint32)


# ---------------------------------------------------------------- builders

def named(out):
    """Builders return dicts; build_cbq_stack returns a tuple."""
    if isinstance(out, dict):
        return {k: v for k, v in out.items()}
    return {f"_{i}": v for i, v in enumerate(out)}


def test_builders_deterministic_seed_sensitive_documented_dtypes():
    # One test builds every real-size builder output (the cbq/flux shapes are
    # the expensive ones) so they are constructed exactly once per run.
    spec_by_builder = {}
    for spec in defs.SPECS:
        spec_by_builder.setdefault(spec["builder"].__name__, []).append(spec)
    np_dtype = {"bfloat16": np.float32, "float16": np.float16,
                "float32": np.float32, "uint32": np.uint32, "int32": np.int32,
                "uint8": np.uint8, "uint16": np.uint16, "int8": np.int8}
    for name, fn in sorted(all_builders().items()):
        a1 = named(fn())
        a2 = named(fn())
        assert set(a1) == set(a2), name
        for key in a1:
            assert np.array_equal(a1[key], a2[key]), f"{name}.{key} not deterministic"
        b = named(fn(12345))
        assert set(b) == set(a1), name
        differing = [k for k in a1 if not np.array_equal(a1[k], b[k])]
        assert differing, f"{name} identical for a different seed"
        for spec in spec_by_builder.get(name, []):
            assert set(spec["inputs"]) <= set(a1), spec["name"]
            for arg, dt_name in zip(spec["inputs"], defs.IN_DTYPES[spec["name"]]):
                arr = np.asarray(a1[arg])
                assert arr.dtype == np_dtype[dt_name], \
                    f"{spec['name']}.{arg}: {arr.dtype} != {dt_name}"
                assert np.isfinite(arr.astype(np.float64)).all(), \
                    f"{spec['name']}.{arg} not finite"
                if dt_name == "bfloat16":
                    assert np.array_equal(bits_of(arr), bits_of(defs.bf16(arr))), \
                        f"{spec['name']}.{arg} not on the bf16 grid"
    assert {"build_cbq_stack", "build_cbq_glu", "build_cbq_situ",
            "build_cbq_gd"} <= set(all_builders())  # helper builders exist


# ---------------------------------------------------------------- rounding

def test_bf16_rounds_to_nearest_even_independent_grid():
    # hand-picked bit patterns: exact ties (even/odd upper), -0.0, +-inf,
    # qNaN/sNaN, and the max finite that carries up to infinity
    patterns = [
        0x3F80_0000,  # 1.0 (no bump)
        0x3F80_8000,  # exact tie, even upper 0x3F80 -> stays 1.0
        0x3F81_8000,  # exact tie, odd upper 0x3F81 -> carries to 0x3F82
        0x3F80_8001,  # above midpoint -> carries
        0x3F80_7FFF,  # below midpoint -> stays
        0x8000_0000,  # -0.0 stays -0.0
        0x7F80_0000,  # +inf
        0xFF80_0000,  # -inf
        0x7FC0_1234,  # quiet NaN payload
        0xFFC0_1234,  # negative NaN payload
        0x7F7F_FFFF,  # max finite, low bits set -> carries to +inf
        0x7F7F_8000,  # exact tie at max finite -> carries to +inf
        0x0000_8000,  # tie between the two smallest subnormals -> 0.0
        0x8080_8000,  # negative tie with odd upper -> carries (more negative)
    ]
    x = np.array(patterns, dtype=np.uint32).view(np.float32)
    got = bits_of(defs.bf16(x))
    want = bits_of(bf16_indep(x))
    assert np.array_equal(got, want), [(hex(a), hex(b), hex(c)) for a, b, c
                                       in zip(patterns, got, want)]
    assert defs.bf16(np.float32(1.3)).shape == ()       # 0-d stays 0-d
    neg_zero = defs.bf16(np.float32(-0.0))
    assert bits_of(neg_zero) == np.uint32(0x80000000)   # sign of -0.0 kept

    # random bit sweep: defs.bf16 must equal the independent RNE everywhere
    rnd = np.random.default_rng(206)
    sweep = rnd.integers(0, 2 ** 32, (20000,), dtype=np.uint64).astype(np.uint32)
    xs = sweep.view(np.float32)
    assert np.array_equal(bits_of(defs.bf16(xs)), bits_of(bf16_indep(xs)))

    # NaN must survive as NaN (dry run greps for it)
    assert np.isnan(defs.bf16(np.float32(np.nan)))


def test_f16_rounds_onto_the_fp16_grid():
    vals = [0.0, -0.0, 1.0, 1.0009765625, 1.0004883,  # 1.0004883 is a tie
            65504.0, 65520.0, np.inf, -np.inf, np.nan, 2.0 ** -14, 2.0 ** -15]
    x = np.array(vals, dtype=np.float32)
    with np.errstate(over="ignore"):    # 65520 -> inf is the expected cast
        got = defs.f16(x)
        want = np.asarray(x, dtype=np.float16)
    assert got.dtype == np.float16
    assert np.array_equal(bits_of_f16(got), bits_of_f16(want))  # NaN-safe
    scalar = defs.f16(np.float32(0.5))
    assert scalar.dtype == np.float16 and float(scalar) == 0.5


def bits_of_f16(a):
    return np.asarray(a, dtype=np.float16).view(np.int16)


# ---------------------------------------------------------------- tolerances

def test_tolerance_constants_documented():
    assert defs.REL == ("rel", 2.0 ** -7, 5e-3)     # 2 bf16 ULP + abs floor
    assert defs.REL16 == ("rel", 2.0 ** -9, 1e-3)   # 2 fp16 ULP + abs floor
    assert defs.EXACT == ("exact",)
    assert defs.REL != defs.REL16


def test_config_tables_internally_consistent():
    spec_names = [s["name"] for s in defs.SPECS]
    assert len(spec_names) == len(set(spec_names)), "duplicate spec names"
    assert set(defs.UPSTREAM) == set(defs.PREDICT) == set(defs.IN_DTYPES) \
        == set(spec_names)
    assert set(defs.PREDICT.values()) <= {"pass", "crash", "refused"}
    for key, msl in defs.UPSTREAM.items():
        assert msl == key + ".msl", key
        assert (SOURCES_DIR / msl).is_file(), msl
    for spec in defs.SPECS:
        n_out = len(spec["outputs"])
        assert len(spec["out_shapes"]) == n_out, spec["name"]
        assert len(spec["out_dtypes"]) == n_out, spec["name"]
        assert len(spec["tol"]) == n_out, spec["name"]
        for tol in spec["tol"]:
            if tol[0] == "exact":
                assert len(tol) == 1, tol
            else:
                assert tol[0] == "rel" and len(tol) in (2, 3), tol
                assert all(isinstance(v, float) for v in tol[1:]), tol
        assert len(defs.IN_DTYPES[spec["name"]]) == len(spec["inputs"])
        if spec.get("header"):
            assert (SOURCES_DIR / spec["header"]).is_file()
        assert callable(spec["builder"]) and callable(spec["ref"])
        assert defs.get_spec(spec["name"]) is spec
    with pytest.raises(KeyError):
        defs.get_spec("no_such_kernel")


# ------------------------------------------------------- independent refs

def test_ref_bitlinear_small_case():
    # K=4, one packed column (O4=1): byte 0b11100100 gives lane q the code
    # in bits 2q..2q+1 -> weights (-1, 0, +1, +2) for lanes 0..3 at i=0;
    # columns 1..3 use uniform bytes (all code 1 / 2 / 3).
    x = defs.f16([[0.5, -1.0, 2.0, 0.25]])
    pw = np.array([[0b11100100, 0b01010101, 0b10101010, 0b11111111]],
                  dtype=np.uint8)
    ws = defs.f16([0.5])
    out = defs.ref_bitlinear(
        {"x": x, "packed_weights": pw, "weight_scale": ws})[0]
    # per lane q: 0.5 * (wq0*0.5 + wq1*(-1) + wq2*2 + wq3*0.25)
    expect = np.array([[1.0, 1.25, 1.5, 1.75]])
    assert out.shape == (1, 4)
    assert np.allclose(out, expect, rtol=0, atol=1e-12)


def _mask_expected(rel, proj, q_offset, sliding, rel_extent, S):
    """Plain-loop reimplementation of the banded-mask contract."""
    rel = rel.astype(np.float64)
    proj = proj.astype(np.float64)
    B, LQ, H, d_rel = rel.shape
    val = np.empty((B, H, LQ, S))
    for b in range(B):
        for h in range(H):
            for i in range(LQ):
                for j in range(S):
                    dist = i + q_offset - j
                    if 0 <= dist < rel_extent:
                        val[b, h, i, j] = sum(
                            rel[b, i, h, d] * proj[d, dist]
                            for d in range(d_rel))
                    elif dist < 0 or dist >= sliding:
                        val[b, h, i, j] = -1e30
                    else:
                        val[b, h, i, j] = 0.0
    return val


def test_ref_mask_banded_window_and_v2_offset():
    # ref_mask: fixed q_offset=16, S=32, sliding=32, extent=16. LQ=17 puts a
    # query exactly ON the sliding boundary (dist == 32) and just inside it.
    rel = np.zeros((1, 17, 1, 2))
    for i in range(17):
        rel[0, i, 0, :] = [1.0, 0.0] if i % 2 == 0 else [0.0, 1.0]
    proj = np.stack([0.5 * np.arange(16), 100.0 + np.arange(16)]).astype(
        np.float32)  # (2,16): acc = 0.5*dist (even i) or 100+dist (odd i)
    rel_bf = defs.bf16(rel)
    got = defs.ref_mask({"rel": rel_bf, "proj": proj,
                         "kshape": np.zeros((1, 32, 4), np.float32)})[0]
    want = _mask_expected(rel_bf, proj, q_offset=16, sliding=32,
                          rel_extent=16, S=32)
    assert got.shape == (1, 1, 17, 32)
    assert np.array_equal(got, want)
    assert got[0, 0, 16, 0] == -1e30        # dist == 32 == sliding: masked
    assert got[0, 0, 15, 0] == 0.0          # dist == 31: inside window, no band
    assert got[0, 0, 0, 1] == 0.5 * 15      # dist == 15: last banded column
    assert got[0, 0, 0, 17] == -1e30        # dist == -1

    # ref_mask_v2: S comes from kshape, q_offset = S - LQ
    kshape = np.zeros((1, 8, 4), dtype=np.float32)
    rel2 = defs.bf16(np.array([[[[1.0, 2.0]], [[3.0, 4.0]],
                                [[5.0, 6.0]], [[7.0, 8.0]]]]))  # (1,4,1,2)
    got2 = defs.ref_mask_v2({"rel": rel2, "proj": proj, "kshape": kshape})[0]
    want2 = _mask_expected(rel2, proj, q_offset=8 - 4, sliding=32,
                           rel_extent=16, S=8)
    assert got2.shape == (1, 1, 4, 8)
    assert np.array_equal(got2, want2)


def test_ref_sconv_decode_small_case():
    x = defs.bf16(np.array([[[1.0, 2.0], [3.0, 4.0]]]))            # (1,2,2)
    state = np.array([[[0.5, 0.5], [1.0, 1.0], [2.0, 2.0]]], np.float32)
    w = np.array([0.0, 1.0, 0.0, 0.0,   0.0, 0.0, 0.0, 1.0], np.float32)
    res = defs.bf16(np.zeros((1, 2, 2), np.float32))
    out, nstate = defs.ref_sconv({"x": x, "state": state, "w": w, "res": res})
    # ch0 taps [0,1,0,0]: acc = xp[:, i+1]; ch1 taps [0,0,0,1]: acc = xp[:, i+3]
    # xp rows t=0..4: (0.5,.5) (1,1) (2,2) (1,2) (3,4)
    # i=0: ch0 acc=(1,1) -> 1+1, 1+2 ; ch1 acc=(1,2) -> 1+1, 2+2
    # i=1: ch0 acc=(2,2) -> 2+3, 2+4 ; ch1 acc=(3,4) -> 3+3, 4+4
    assert out.shape == (1, 2, 2)
    assert np.array_equal(out, np.array([[[2.0, 4.0], [5.0, 8.0]]]))
    assert nstate.shape == (1, 3, 2)
    assert np.allclose(nstate, [[[2.0, 2.0], [1.0, 2.0], [3.0, 4.0]]], atol=0)

    # variant whose final inner+res sum needs bf16 rounding: res = bf16(0.1)
    # = 0.10009765625, so the sums 2.10009765625 / 4.10009765625 /
    # 5.10009765625 / 8.10009765625 land OFF the bf16 grid and must round
    # to 2.09375 / 4.09375 / 5.09375 / 8.125
    res2 = defs.bf16(np.full((1, 2, 2), 0.1, dtype=np.float32))
    out2, _ = defs.ref_sconv({"x": x, "state": state, "w": w, "res": res2})
    want2 = np.array([[[2.09375, 4.09375], [5.09375, 8.125]]])
    assert np.array_equal(out2, want2)
    for s, want_v in zip((2.0, 4.0, 5.0, 8.0),
                         (2.09375, 4.09375, 5.09375, 8.125)):
        assert float(defs.bf16(np.float32(s + 0.10009765625))) == want_v


def test_ref_route_topk_lowest_index_tie_break():
    # R=8 routed + SH=1 shared; two exact ties (j0/j1 at 0.5, and the picked
    # set exercises "lowest j wins") -- with >= the later index would steal.
    logits = defs.bf16(np.array([[0.5, 0.5, 0.25, -0.25, 1.0, -1.0, 0.125,
                                  0.0, 3.0]], dtype=np.float32))
    corr = defs.bf16(np.zeros(8, np.float32))
    wscale = np.array([0.001], dtype=np.float32)
    idx, wk, gamma = defs.ref_route({"logits": logits, "corr": corr,
                                     "wscale": wscale})
    assert np.array_equal(idx, np.array([[4, 0, 1, 2, 6, 7, 3, 5]],
                                        dtype=np.uint32))
    # independent softmax over log-sigmoid of (picked logits + shared logit)
    picked = [1.0, 0.5, 0.5, 0.25, 0.125, 0.0, -0.25, -1.0]
    vals = picked + [3.0]
    lp = [math.log(1.0 / (1.0 + math.exp(-v))) for v in vals]
    m = max(lp)
    lse = m + math.log(sum(math.exp(v - m) for v in lp))
    want_wk = np.array([[math.exp(v - lse) * 0.001 for v in lp[:8]]])
    assert np.allclose(wk, want_wk, rtol=1e-6, atol=0)
    assert np.allclose(gamma[0, :256], math.exp(lp[8] - lse) * 0.001,
                       rtol=1e-6, atol=0)


def test_ref_down_combine_nibble_groups():
    # Every 4-bit nibble of wq is 1, act is all ones: accq_l = 64 ones,
    # actx_l = 64. With sc=0, bi=0.5: dot = 0.5*64*32 = 1024 per sg.
    E, OUT, IN, K, N = 1, 1, 2048, 8, 1
    inp = {
        "xin": defs.bf16(np.ones((N * K, IN), np.float32)),
        "wq": np.full((E, OUT, IN // 8), 0x11111111, dtype=np.uint32),
        "sc": defs.bf16(np.zeros((E, OUT, 32), np.float32)),
        "bi": defs.bf16(np.full((E, OUT, 32), 0.5, np.float32)),
        "idx": np.zeros((N, K), dtype=np.uint32),
        "wk": defs.bf16(np.ones((N, K), np.float32)),
    }
    inp["wk"][0, 3] = 2.0                       # one partial doubles
    out = defs.ref_down_combine(inp)[0]
    # partials: 7*1024 + 1*2048 = 9216, all bf16-exact
    assert out.shape == (1, OUT)
    assert np.array_equal(out, np.array([[9216.0]]))

    inp2 = dict(inp)
    inp2["sc"] = defs.bf16(np.ones((E, OUT, 32), np.float32))
    inp2["bi"] = defs.bf16(np.zeros((E, OUT, 32), np.float32))
    inp2["wk"] = defs.bf16(np.ones((N, K), np.float32))
    out2 = defs.ref_down_combine(inp2)[0]
    assert np.array_equal(out2, np.array([[8 * 2048.0]]))   # 16384


def test_ref_llg_bitmask_exact():
    logits = defs.bf16(np.arange(64, dtype=np.float32) * 0.25 - 8.0).reshape(1, 64)
    mask = np.array([[0x21, 0x00000002]], dtype=np.int32)  # tokens 0, 5, 33
    out = defs.ref_llg({"logits": logits, "mask": mask})[0]
    want = np.full((1, 64), -np.inf)
    want[0, 0] = logits[0, 0]
    want[0, 5] = logits[0, 5]
    want[0, 33] = logits[0, 33]
    assert out.shape == (1, 64)
    assert np.array_equal(out, want)


def test_ref_depthwise_conv1d_small_case():
    x = np.array([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]],
                 dtype=np.float32)                     # (1,4,2)
    w = np.array([[[1.0], [2.0], [4.0]], [[0.0], [1.0], [0.0]]],
                 dtype=np.float32)                     # (2,3,1)
    params = np.array([1, 4, 2, 3, 1, 4], dtype=np.int32)
    out = defs.ref_depthwise({"inp": x, "weight": w, "params": params})[0]
    # ch0 taps [1,2,4], pad 1: t0=2*1+4*2=10, t1=1+4+12=17, t2=2+6+16=24,
    # t3=3+8=11 ; ch1 taps [0,1,0]: passthrough
    want = np.array([[[10.0, 10.0], [17.0, 20.0], [24.0, 30.0], [11.0, 40.0]]])
    assert out.shape == (1, 4, 2)
    assert np.allclose(out, want, rtol=0, atol=1e-12)


def test_ref_qk_relu_squared_double_rounding():
    q = defs.f16(np.array([[[[1.0, 2.0], [-3.0, 4.0]]]]))       # (1,1,2,2)
    k = defs.f16(np.array([[[[1.0, 0.0], [0.0, 1.0]]]]))
    scale = defs.f16([0.5])
    out = defs.ref_qk({"q": q, "k": k, "scale": scale})[0]
    # dots [[1,2],[-3,4]] -> *0.5 -> relu [[0.5,1],[0,2]] -> square
    want = np.array([[[[0.25, 1.0], [0.0, 4.0]]]], dtype=np.float16)
    assert out.dtype == np.float16 and out.shape == (1, 1, 2, 2)
    assert np.array_equal(out, want)

    # non-exact path: scale cast happens BEFORE the multiply (fp16 boundary),
    # then the square rounds again
    q2 = defs.f16(np.array([[[[1.1]]]]))                        # f16(1.1)
    k2 = defs.f16(np.array([[[[1.0]]]]))
    scale2 = defs.f16([1.0])
    out2 = defs.ref_qk({"q": q2, "k": k2, "scale": scale2})[0]
    want2 = np.float16(np.float64(np.float16(np.float64(np.float16(1.1))
                                            * np.float64(np.float16(1.0))))
                       ** 2)
    assert out2[0, 0, 0, 0] == want2
    assert out2[0, 0, 0, 0] != np.float16(0.0)


def test_ref_phonon_base5_unpack():
    # 16 symbols -> one 16-bit output word; two 10-symbol base-5 groups,
    # each packed little-endian into 3 bytes. Row 1 maxes the low six digits
    # (low value == 5^6 - 1), so a wrong radix (5^k vs 5^k - 1) provably
    # shifts the digit-6 quotient and is caught.
    rows_syms = [
        [0, 1, 2, 3, 4, 1, 3, 0, 4, 2, 2, 2, 2, 2, 2, 2],
        [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 2, 2, 2, 2, 2, 2],
    ]
    quint = np.zeros((2, 6), dtype=np.uint8)
    for row, syms in enumerate(rows_syms):
        for g in range(2):
            payload = sum(syms[g * 10 + off] * 5 ** off
                          for off in range(10) if g * 10 + off < 16)
            assert payload < 2 ** 24
            quint[row, 3 * g] = payload & 0xFF
            quint[row, 3 * g + 1] = (payload >> 8) & 0xFF
            quint[row, 3 * g + 2] = (payload >> 16) & 0xFF
    table = {0: (0, 0), 1: (0, 2), 2: (1, 1), 3: (2, 0), 4: (2, 2)}
    exp = []
    for syms in rows_syms:
        base_exp = 0
        resid_exp = 0
        for j, s in enumerate(syms):
            b_code, r_code = table[s]
            base_exp |= b_code << (2 * j)
            resid_exp |= r_code << (2 * j)
        exp.append((base_exp, resid_exp))
    assert exp[0][0] != exp[0][1] != 0   # row 0 codes are asymmetric
    base, resid = defs.ref_phonon(
        {"quint5_q": quint, "in_features": np.array([16], dtype=np.uint32)})
    assert base.shape == (2, 1) and resid.shape == (2, 1)
    assert base.dtype == np.uint32 and resid.dtype == np.uint32
    for row, (base_exp, resid_exp) in enumerate(exp):
        assert int(base[row, 0]) == base_exp, f"base row {row}"
        assert int(resid[row, 0]) == resid_exp, f"resid row {row}"


# ------------------------------------------------------------ cbq (small)

CBQ_SMALL = dict(E=2, O=4, K=256, R=2, N=1)


def _cbq_inputs():
    # qh mostly 0 (hi nibbles 0, mult 1, delta +0.125); one nonzero word:
    # 0x9203 = 0b1001_0010_0000_0011 -> hi=[3,0,0,1], mult=(2*1+1)=3,
    # delta=-0.125 -- all hand-checked from the bit layout.
    qs = np.zeros((2, 4, 32), dtype=np.uint8)
    qh = np.zeros((2, 4, 8), dtype=np.uint16)
    d = np.zeros((2, 4, 1), dtype=np.float32)
    for e in range(2):
        for o in range(4):
            for s in range(32):
                qs[e, o, s] = (e * 2 + o + s) % 8
    qh[0, 0, 0] = 0x9203
    d[0, :, 0] = 0.5
    d[1, :, 0] = 0.25
    return qs, qh, defs.bf16(d)


def _cbq_grid(monkeypatch):
    grid = np.zeros((2048, 8), dtype=np.int8)
    for g in range(8):
        grid[g] = [g, -g, g - 4, 4 - g, 1, -1, 0, 2]
    for r in range(4):                  # hi nibble 3 sends group-0 bytes here
        grid[768 + r] = [r + 5, -(r + 5), r, -r, 1, -1, 0, 2]
    for base, off in ((256, 9), (512, 13)):   # hi3==1 vs hi3==2 must differ
        for r in range(4):
            grid[base + r] = [r + off, -(r + off), 1, -1, 0, 2, r, -r]
    monkeypatch.setattr(defs, "_GRID_CACHE", grid)
    monkeypatch.setattr(defs, "CBQ", CBQ_SMALL)
    return grid


def _dequant_mm(grid, qs, qh, d, e):
    """Independent dequant for the mm path: the grid LUT enters SIGNED;
    one qh word governs each 32-element group (4 bytes of 8)."""
    W = np.zeros((4, 256))
    for o in range(4):
        for k in range(256):
            group, r = divmod(k, 32)
            l, b = divmod(r, 8)
            word = int(qh[e, o, group])
            hi = (word >> (3 * l)) & 7
            gi = int(qs[e, o, group * 4 + l]) | (hi << 8)
            g = int(grid[gi, b])                        # signed int8
            dl = float(d[e, o, 0]) * (2 * ((word >> 12) & 7) + 1)
            delta = -0.125 if (word >> 15) & 1 else 0.125
            W[o, k] = (g + delta) * dl
    return W


def test_ref_cbq_mm_and_glu_small_case(monkeypatch):
    grid = _cbq_grid(monkeypatch)
    qs, qh, d = _cbq_inputs()
    x = defs.bf16((np.arange(256, dtype=np.float32) % 9 - 4) * 0.5
                  ).reshape(1, 256)
    inp = {"x": x, "eidx": np.array([0, 1], np.uint32),
           "tokidx": np.zeros(2, np.uint32),
           "cb_qs": qs, "cb_qh": qh, "cb_d": d, "grid": grid.reshape(-1)}
    y = defs.ref_cbq_mm(inp)[0]
    assert y.shape == (2, 4)
    for r, e in enumerate((0, 1)):
        W = _dequant_mm(grid, qs, qh, d, e)
        want = defs.bf16(np.float32(x[0] @ W.T))
        assert np.array_equal(y[r], want), f"row {r}"

    # glu: two stacks through the same math, concatenated outputs
    qs2, qh2, d2 = qs.copy(), qh.copy(), d * 2.0
    inp_g = dict(inp)
    inp_g.update({"a_qs": qs, "a_qh": qh, "a_d": d,
                  "b_qs": qs2, "b_qh": qh2, "b_d": d2})
    yg, yu = defs.ref_cbq_glu(inp_g)
    assert yg.shape == (2, 4) and yu.shape == (2, 4)
    assert np.array_equal(yg, y)                    # stack a == cb_* stack
    for r, e in enumerate((0, 1)):
        W2 = _dequant_mm(grid, qs2, qh2, d2, e)
        want = defs.bf16(np.float32(x[0] @ W2.T))
        assert np.array_equal(yu[r], want), f"glu row {r}"


def test_ref_cbq_situ_small_case(monkeypatch):
    grid = _cbq_grid(monkeypatch)
    qs, qh, d = _cbq_inputs()
    x = defs.bf16(np.ones((1, 256), np.float32) * 0.5)
    gbuf = defs.bf16(np.array([[0.5, -1.0, 2.0, 0.25],
                               [1.0, -0.5, 0.75, 1.5]], np.float32))
    y = defs.ref_cbq_situ({"x": x, "eidx": np.array([0, 1], np.uint32),
                           "tokidx": np.zeros(2, np.uint32), "cb_qs": qs,
                           "cb_qh": qh, "cb_d": d, "grid": grid.reshape(-1),
                           "gbuf": gbuf})[0]
    assert y.shape == (2, 4)
    for r, e in enumerate((0, 1)):
        W = _dequant_mm(grid, qs, qh, d, e)
        for o in range(4):
            yv = float((x[0] @ W.T)[o])
            gg = float(gbuf[r, o])
            uu = float(defs.bf16(np.float32(yv)))
            want = float(defs.bf16(np.float32(
                (4.0 * np.tanh(gg / 4.0)) * (1.0 / (1.0 + np.exp(-gg)))
                * (25.0 * np.tanh(uu / 25.0)))))
            assert y[r, o] == want, f"situ[{r},{o}]"


def test_ref_cbq_grad_d_and_x_small_case(monkeypatch):
    grid = _cbq_grid(monkeypatch)
    qs, qh, d = _cbq_inputs()
    x = defs.bf16(((np.arange(256, dtype=np.float32) % 7) - 3) * 0.5
                  ).reshape(1, 256)
    cot = defs.bf16(np.array([[0.5, -0.25, 0.125, 1.0],
                              [0.0, 0.5, -0.5, 0.25]], np.float32))
    inp = {"x": x, "cot": cot, "eidx": np.array([0, 1], np.uint32),
           "tokidx": np.zeros(2, np.uint32), "cb_qs": qs, "cb_qh": qh,
           "cb_d": d, "grid": grid.reshape(-1)}
    gd = defs.ref_cbq_gd(inp)[0]
    gx = defs.ref_cbq_gx(inp)[0]
    assert gd.shape == (2, 4, 1) and gx.shape == (1, 256)

    # independent: grad paths read the grid bytes UNSIGNED (unlike mm)
    gd_want = np.zeros((2, 4, 1))
    gx_want = np.zeros((1, 256))
    for r in range(2):
        e, tok = r, 0
        for o in range(4):
            ct = float(cot[r, o])
            if ct == 0.0:
                continue
            for sb in range(8):
                word = int(qh[e, o, sb])
                mult = 2 * ((word >> 12) & 7) + 1
                delta = -0.125 if (word >> 15) & 1 else 0.125
                p32 = 0.0
                for l in range(4):
                    gi = int(qs[e, o, sb * 4 + l]) | (((word >> (3 * l)) & 7) << 8)
                    for b in range(8):
                        gbyte = int(grid[gi, b]) & 0xFF
                        xk = float(x[tok, sb * 32 + l * 8 + b])
                        p32 += (gbyte + delta) * xk
                        gx_want[tok, (sb * 4 + l) * 8 + b] += \
                            ct * float(d[e, o, 0]) * mult * (gbyte + delta)
                gd_want[e, o, 0] += ct * mult * p32
    assert np.array_equal(gd, gd_want)
    assert np.array_equal(gx, gx_want)
    assert gx_want[0, :].any() and gd_want.any()


# ------------------------------------------------------- kda glue (small)

def _stable_sig_bf16(v):
    """The documented mlx stable sigmoid, written out (rounds each step)."""
    e = float(defs.bf16(np.float32(math.exp(abs(v)))))
    den = float(defs.bf16(np.float32(1.0 + e)))
    y = float(defs.bf16(np.float32(1.0 / den)))
    return y if v < 0 else float(defs.bf16(np.float32(1.0 - y)))


def test_ref_glue_pre_and_post_small_case(monkeypatch):
    monkeypatch.setattr(defs, "GLUE_H", 2)          # GLUE_DIM stays 128:
    monkeypatch.setattr(defs, "GLUE_C", 256)        # ref reshapes (4,32)
    H, DIM, C = 2, 128, 256
    rng = np.random.default_rng(7)
    bf = lambda *s: defs.bf16(rng.standard_normal(s).astype(np.float32) * 0.5)
    inp = {"hp": bf(49376), "z": bf(C), "s0": bf(3 * C), "s1": bf(3 * C),
           "s2": bf(3 * C), "wq": bf(4 * C), "wk": bf(4 * C), "wv": bf(4 * C),
           "dtb": rng.standard_normal(C).astype(np.float32),
           "alog": rng.standard_normal(DIM).astype(np.float32),
           "cons": np.array([-2.0, 0.7, 0.49, np.float32(1e-6 / DIM)],
                            dtype=np.float32)}
    q, k, v, g, beta, ns0, ns1, ns2 = defs.ref_glue_pre(inp)
    assert q.shape == (1, 1, H, DIM) and beta.shape == (1, 1, H)
    assert ns0.shape == (1, 3, C)

    f = lambda key: inp[key].astype(np.float64)
    hp, z, dtb, alog = f("hp"), f("z"), f("dtb"), f("alog")
    conv = {"q": np.zeros(DIM), "k": np.zeros(DIM), "v": np.zeros(DIM)}
    for h in range(H):
        for d in range(DIM):
            ci = h * DIM + d
            for name, s_ in (("q", "s0"), ("k", "s1"), ("v", "s2")):
                taps = [f(s_)[ci], f(s_)[C + ci], f(s_)[2 * C + ci],
                        hp[ci if name == "q" else C + ci if name == "k"
                           else 2 * C + ci]]
                w_ = f("w" + name)
                a = taps[0] * w_[ci * 4] + taps[1] * w_[ci * 4 + 1] \
                    + taps[2] * w_[ci * 4 + 2] + taps[3] * w_[ci * 4 + 3]
                c_ = float(defs.bf16(np.float32(a)))
                conv[name][d] = float(defs.bf16(np.float32(c_ * _stable_sig_bf16(c_))))
        mq = float((conv["q"] * conv["q"]).sum()) / DIM + float(inp["cons"][3])
        mk = float((conv["k"] * conv["k"]).sum()) / DIM + float(inp["cons"][3])
        nq = defs.bf16(conv["q"] * (1.0 / np.sqrt(mq)))
        nk = defs.bf16(conv["k"] * (1.0 / np.sqrt(mk)))
        want_q = defs.bf16(nq.astype(np.float32) * np.float32(inp["cons"][2]))
        want_k = defs.bf16(nk.astype(np.float32) * np.float32(inp["cons"][1]))
        want_v = defs.bf16(conv["v"].astype(np.float32))
        assert np.array_equal(q[0, 0, h], want_q), f"q head {h}"
        assert np.array_equal(k[0, 0, h], want_k), f"k head {h}"
        assert np.array_equal(v[0, 0, h], want_v), f"v head {h}"
        assert beta[0, 0, h] == _stable_sig_bf16(hp[49280 + h]), f"beta {h}"
        for d in range(DIM):
            ci = h * DIM + d
            zf = z[ci] + dtb[ci]
            s = 1.0 / (1.0 + np.exp(-(np.exp(alog[d]) * zf)))
            assert g[0, 0, h, d] == np.exp(float(inp["cons"][0]) * s)
            assert ns0[0, 0, ci] == f("s0")[C + ci]
            assert ns2[0, 2, ci] == hp[2 * C + ci]

    # post: rmsnorm -> weight -> gated by sigmoid(hp[36864+ci])
    inp_p = {"inp": bf(C), "hp": bf(49376), "w": bf(DIM),
             "cons": np.array([1e-6], dtype=np.float32)}
    y = defs.ref_glue_post(inp_p)[0]
    assert y.shape == (1, 1, C)
    ip = inp_p["inp"].astype(np.float64)
    wt = inp_p["w"].astype(np.float64)
    for h in range(H):
        xs = ip[h * DIM:(h + 1) * DIM]
        inv = 1.0 / np.sqrt((xs * xs).sum() / DIM + 1e-6)
        for d in range(DIM):
            ci = h * DIM + d
            nx0 = float(defs.bf16(np.float32(xs[d] * inv)))
            nx = float(defs.bf16(np.float32(nx0 * float(wt[d]))))
            want = float(defs.bf16(np.float32(nx * _stable_sig_bf16(
                inp_p["hp"].astype(np.float64)[36864 + ci]))))
            assert y[0, 0, ci] == want, f"post ci={ci}"


# ------------------------------------------------------- route_fused/situ

def test_ref_route_fused_ties():
    # gates tied at the top (argmax must keep the LOWEST index) and equal
    # pool weights for the hi slots (lowest kk wins)
    gate = defs.bf16(np.array([2.0, 2.0, 1.0, 0.5, -1.0, 0.25], np.float32))
    bias = defs.bf16(np.zeros(6, np.float32))
    lo_map = np.array([7, 7, 3, 0, 0, 0], dtype=np.int32)
    hi_map = np.array([1, 1, -1, 0, 0, 0], dtype=np.int32)
    meta = np.array([3, 2], dtype=np.int32)          # K=3, KHI=2
    lo_idx, w_lo, hi_idx, w_hi = defs.ref_route_fused(
        {"gate_out": gate, "bias": bias, "lo_map": lo_map, "hi_map": hi_map,
         "meta": meta})
    assert np.array_equal(lo_idx, np.array([[[7, 7, 3]]], dtype=np.uint32))
    assert np.array_equal(hi_idx, np.array([[[2, 2]]], dtype=np.uint32))
    sc = [1.0 / (1.0 + math.exp(-float(v))) for v in gate[:3]]
    tot = sum(sc) + 1e-20
    wn = [s / tot for s in sc]
    assert w_lo[0, 0, 0] == 0.0 and w_lo[0, 0, 1] == 0.0   # experts with hi
    assert w_lo[0, 0, 2] == float(defs.bf16(np.float32(wn[2])))
    assert w_hi[0, 0, 0] == float(defs.bf16(np.float32(wn[0])))
    assert w_hi[0, 0, 1] == float(defs.bf16(np.float32(wn[1])))


def _situ_expected(g, u, b1, b2):
    uu = float(defs.bf16(np.float32(u)))
    return float(defs.bf16(np.float32(
        (b1 * np.tanh(g / b1)) * (1.0 / (1.0 + np.exp(-g)))
        * (b2 * np.tanh(uu / b2)))))


def test_ref_situ_and_pair_small_case():
    gu = defs.bf16(np.array([0.5, -1.0, 2.0, 0.25,
                             1.0, -0.5, 0.75, 1.5], np.float32))
    cons = np.array([4.0, 25.0, 4.0], dtype=np.float32)
    out = defs.ref_situ({"gu": gu, "cons": cons})[0]
    assert out.shape == (1, 1, 4)
    for i in range(4):
        want = _situ_expected(float(gu[i]), float(gu[4 + i]), 4.0, 25.0)
        assert out[0, 0, i] == want, f"situ[{i}]"

    ga = defs.bf16(np.array([0.5, -1.0, 2.0], np.float32))
    ub = defs.bf16(np.array([1.0, -0.5, 0.75], np.float32))
    out2 = defs.ref_situ_pair({"ga": ga, "ub": ub,
                               "cons": np.array([4.0, 25.0], np.float32)})[0]
    assert out2.shape == (3,)
    for i in range(3):
        assert out2[i] == _situ_expected(float(ga[i]), float(ub[i]), 4.0, 25.0)


# ---------------------------------------------------------------- flux2

def _flux_expected(rows, nq, nk, cos, sin, hd, eps):
    """Independent fused rms-norm + rope for the shrunken config
    (heads=1, dim=hd): q/k cols 0..hd, hd..2hd; v passes through raw."""
    s_tot = rows.shape[0]
    out_q = np.zeros((1, 1, s_tot, hd))
    out_k = np.zeros_like(out_q)
    out_v = np.zeros_like(out_q)
    for s in range(s_tot):
        row = rows[s].astype(np.float64)
        for name, src, nvec, dest in (("q", row[0:hd], nq[s], out_q),
                                      ("k", row[hd:2 * hd], nk[s], out_k)):
            inv = 1.0 / np.sqrt((src * src).sum() / hd + eps)
            vec = src * inv * nvec
            for i in range(hd // 2):
                d = 2 * i
                a0, a1 = vec[d], vec[d + 1]
                c, sn = cos[s, i], sin[s, i]
                vec[d] = a0 * c - a1 * sn
                vec[d + 1] = a1 * c + a0 * sn
            dest[0, 0, s] = vec
        out_v[0, 0, s] = row[2 * hd:3 * hd]
    return [out_q, out_k, out_v]


def test_ref_flux_norm_rope_small_case(monkeypatch):
    monkeypatch.setattr(defs, "FLUX", dict(dim=64, heads=1, hd=64, eps=1e-6,
                                           img=1, txt=1))
    r = defs.rng(42)
    bf = lambda *s: defs.bf16(r.standard_normal(s).astype(np.float32) * 0.4)
    nq, nk = bf(64), bf(64)
    naq, nak = bf(64), bf(64)
    cos, sin = bf(2, 32), bf(2, 32)
    img = bf(1, 1 * 3 * 64)
    txt = bf(1, 1 * 3 * 64)

    got = defs.ref_flux_double({"img_qkv": img, "txt_qkv": txt,
                                "norm_q": nq, "norm_k": nk,
                                "norm_added_q": naq, "norm_added_k": nak,
                                "cos_vals": cos, "sin_vals": sin})
    rows = np.stack([txt[0], img[0]], axis=0)           # txt rows first
    nq2 = np.concatenate([np.repeat(naq[None], 1, axis=0), nq[None]], axis=0)
    nk2 = np.concatenate([np.repeat(nak[None], 1, axis=0), nk[None]], axis=0)
    want = _flux_expected(rows, nq2, nk2, cos.astype(np.float64),
                          sin.astype(np.float64), 64, 1e-6)
    for got_o, want_o in zip(got, want):
        assert got_o.shape == (1, 1, 2, 64)
        assert np.allclose(got_o, want_o, rtol=1e-12, atol=0)

    fused = bf(1, 2, 3 * 64)
    got1 = defs.ref_flux_single({"fused": fused, "norm_q": nq, "norm_k": nk,
                                 "cos_vals": cos, "sin_vals": sin})
    nq3 = np.repeat(nq[None], 2, axis=0)
    nk3 = np.repeat(nk[None], 2, axis=0)
    want1 = _flux_expected(fused[0], nq3, nk3, cos.astype(np.float64),
                           sin.astype(np.float64), 64, 1e-6)
    for got_o, want_o in zip(got1, want1):
        assert np.allclose(got_o, want_o, rtol=1e-12, atol=0)


# ---------------------------------------------------------------- dry run

def test_reference_only_dry_run_structure(capsys):
    run = pytest.importorskip("tools.kernel_recheck.run")
    for name in ("qk_relu_squared", "custom_depthwise_conv1d"):
        rc = run.reference_only(name)
        out = capsys.readouterr().out
        assert rc == 0, out
        line = out.strip().splitlines()[-1]
        spec = defs.get_spec(name)
        shapes = ";".join("x".join(map(str, s)) for s in spec["out_shapes"])
        assert line.startswith(f"KERNEL {name} dryrun ref-ok"), line
        assert f"predict={defs.PREDICT[name]}" in line
        assert f"outs=({shapes})" in line
        assert "kinds=(f64)" in line
        assert line.endswith("maxdiff 0")
    assert "mlx.core" not in sys.modules    # the dry run never imports mlx
