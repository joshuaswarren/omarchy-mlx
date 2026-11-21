#!/usr/bin/env python3
"""Issue #564: atomic_outputs kernels are judged at upstream's rel-L2 bar.

Upstream's own self-test gates fp32 kernels at relative-L2 <= 1e-5
(`k3_cbq.py` __main__ block, HF avlp12/Kimi-K3-Alis-MLX-Dynamic-2.10bpw,
"fp32 기대 ≤1e-5"). Kernels declared `atomic_outputs=True` accumulate
through fp32 atomics, so a few near-cancellation elements exceed the
per-element ("rel", 1e-4) rule while the aggregate error is ~3e-8 — the
cbq_grad_d/cbq_grad_x outputs dumped on M1 (T8103) and M1 Max (G13C) are
bit-identical, so the miss is a pass-rule artifact, not a device result.
`check()` accepts an atomic_outputs output at rel-L2 <= 1e-5 and keeps the
per-element count as information in the detail string.

numpy only. The replay cases load the committed M1 Max output npys and
compare them against the fp64 references built from the same deterministic
input builders the GPU run used.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.kernel_recheck import defs, run  # noqa: E402

RECEIPTS = REPO_ROOT / "receipts" / "2026-11-21-cbq-applegpu"


def _spec(atomic, tol=("rel", 1e-4)):
    s = {"tol": [tol]}
    if atomic:
        s["atomic_outputs"] = True
    return s


def test_atomic_outputs_pass_when_only_per_element_misses():
    """A 25% relative error on a near-cancellation element fails
    ("rel", 1e-4) but the rel-L2 is ~2e-7, inside upstream's 1e-5 bar,
    so an atomic_outputs spec passes."""
    ref = np.linspace(-3.0, 3.0, 4096)
    ref[7] = 1e-4
    out = ref.copy()
    out[7] *= 1.25
    ok, _, detail = run.check("synthetic", _spec(atomic=True), {}, [ref], [out])
    assert ok, detail
    assert "rel-L2" in detail
    assert "per-element info" in detail
    assert "bad 1/4096" in detail


def test_atomic_outputs_fail_when_rel_l2_exceeds_bar():
    """A uniform 0.1% scaling gives rel-L2 = 1e-3 > 1e-5: still wrong."""
    ref = np.linspace(-3.0, 3.0, 4096)
    out = ref * 1.001
    ok, _, detail = run.check("synthetic", _spec(atomic=True), {}, [ref], [out])
    assert not ok
    assert "rel-L2" in detail


def test_non_atomic_outputs_keep_per_element_gate():
    """The same one-element miss still fails a non-atomic spec."""
    ref = np.linspace(-3.0, 3.0, 4096)
    out = ref.copy()
    out[7] *= 1.25
    ok, _, detail = run.check("synthetic", _spec(atomic=False), {}, [ref], [out])
    assert not ok
    assert "rel-L2" not in detail


@pytest.mark.parametrize("name,bad_count", [("cbq_grad_d", 33), ("cbq_grad_x", 1)])
def test_committed_gpu_outputs_pass_at_rel_l2(name, bad_count):
    """Replay the M1 Max receipt npys (bit-identical to the M1 dumps) against
    the fp64 references: pass at rel-L2 <= 1e-5 with the per-element miss
    counts (33/65536 and 1/4096) carried as information."""
    spec = defs.get_spec(name)
    inputs = run.build_inputs(spec)
    ref = spec["ref"](inputs)[0]
    out = np.load(RECEIPTS / f"{name}.out0.npy")
    ok, _, detail = run.check(name, spec, inputs, [ref], [out])
    assert ok, detail
    assert "rel-L2" in detail
    assert f"bad {bad_count}/" in detail
