"""Admission-math tests (Main review round 3, failing-first).

These pin the dtype-aware footprint estimate and the no-growth relabel
contract BEFORE the implementation lands in server.py:

  - parameter bytes scale with run dtype (fp32 = 2x fp16 parameter bytes)
  - workspace is derived from max_questions x seq x hidden x dtype incl the
    fp32 softmax upcast, and scales with the batch bound (not a flat 512 MiB)
  - the admitted total NEVER grows at relabel: a measured floor larger than
    the admitted total is a mismatch and is refused
  - the resident floor is the parameter nbytes of the run dtype, not a
    global allocator counter
"""

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "serve"))

import pytest  # noqa: E402

pytest.importorskip("mlx.core")  # the laya server imports mlx
from mlx_omarchy_laya import server as laya_server  # noqa: E402
from mlx_omarchy_laya.model import load_encoder_config  # noqa: E402

CONVERTED = None  # set by conftest-less env probe below
for _c in ("/tmp/laya-work/converted/laya",):
    if Path(_c).exists():
        CONVERTED = _c


def _manifest():
    return json.loads((Path(CONVERTED) / "manifest.json").read_text())


def _enc_cfg():
    return load_encoder_config(Path(CONVERTED) / "encoder")


@unittest.skipUnless(CONVERTED, "needs the converted checkpoint's manifest/config shapes")
class FootprintEstimateTests(unittest.TestCase):
    def test_parameter_bytes_double_under_fp32(self):
        fp16 = laya_server.parameter_bytes(_manifest(), "float16")
        fp32 = laya_server.parameter_bytes(_manifest(), "float32")
        bf16 = laya_server.parameter_bytes(_manifest(), "bfloat16")
        self.assertEqual(fp16, bf16)
        self.assertEqual(fp32, fp16 * 2)
        self.assertGreater(fp16, 0)

    def test_workspace_matches_documented_formula_and_scales(self):
        m, e = _manifest(), _enc_cfg()
        B, T = 64, m["context_max_tokens"]
        D, H, I = e.hidden_size, e.num_attention_heads, e.intermediate_size
        logits_const = 2 * B * laya_server._MAX_OPTIONS * 4  # fp32 logits, dtype-free
        for dtype, sz in (("float16", 2), ("float32", 4)):
            scores = B * H * T * T * (sz + 4)  # run-dtype scores + fp32 softmax output
            residual = 4 * B * T * D * sz
            ffn = B * T * (3 * D + 2 * I + 4 * D) * sz
            expected = laya_server._WORKSPACE_SAFETY_FACTOR * (scores + residual + ffn + logits_const)
            self.assertEqual(laya_server.workspace_bytes(m, e, dtype, max_questions=B), expected)
        fp16 = laya_server.workspace_bytes(m, e, "float16", max_questions=B)
        fp32 = laya_server.workspace_bytes(m, e, "float32", max_questions=B)
        self.assertGreater(fp32, fp16)
        # measured CPU fp16 reference (16 x 512): peak growth 1,287,019,628 B
        # must sit under the bound with the safety factor applied
        ref = laya_server.workspace_bytes(m, e, "float16", max_questions=16,
                                          context_override=512)
        self.assertGreaterEqual(ref, 1287019628)
        # scores term dominates: B*H*T^2 bytes scale ~quadratic in context
        long_ctx = laya_server.workspace_bytes(m, e, "float16", max_questions=8, context_override=1024)
        self.assertGreater(long_ctx, laya_server.workspace_bytes(m, e, "float16", max_questions=8))

    def test_total_estimate_is_params_plus_workspace(self):
        m, e = _manifest(), _enc_cfg()
        total = laya_server.total_estimate_bytes(m, e, "float16", max_questions=64)
        self.assertEqual(total,
                         laya_server.parameter_bytes(m, "float16")
                         + laya_server.workspace_bytes(m, e, "float16", max_questions=64))


class RelabelContractTests(unittest.TestCase):
    def test_floor_may_not_exceed_admitted_total(self):
        # mismatch is a refusal: the admitted total is a commitment, never grown
        self.assertTrue(laya_server.floor_matches_admission(1000, 900))
        self.assertTrue(laya_server.floor_matches_admission(1000, 1000))
        self.assertFalse(laya_server.floor_matches_admission(1000, 1001))

    def test_floor_is_parameter_bytes_not_allocator_global(self):
        m = _manifest() if CONVERTED else {"weights_header": {"weight_bytes": 842609210}}
        header = m["weights_header"]
        elements = header.get("param_elements")
        if elements is None:
            elements = header["weight_bytes"] // 2
        self.assertEqual(laya_server.resident_floor_bytes(m, "float16"), elements * 2)
        self.assertEqual(laya_server.resident_floor_bytes(m, "float32"), elements * 4)
        self.assertGreater(laya_server.resident_floor_bytes(m, "float16"), 0)

    def test_param_elements_preferred_over_weight_bytes(self):
        # nit from MCQ review: a future non-fp16 pack must not silently halve
        # the floor via the weight_bytes//2 fallback
        m = {"weights_header": {"weight_bytes": 1000, "param_elements": 3000}}
        self.assertEqual(laya_server.parameter_bytes(m, "float16"), 6000)
        self.assertEqual(laya_server.parameter_bytes(m, "float32"), 12000)

    def test_relabel_drift_guard_refuses_over_requirement(self):
        # Main round-3 condition: params + recomputed workspace must fit the
        # admitted total; otherwise relabel is inadmissible
        self.assertTrue(laya_server.relabel_admissible(2000, 900, 200))
        self.assertTrue(laya_server.relabel_admissible(1100, 900, 200))
        self.assertFalse(laya_server.relabel_admissible(1000, 900, 200))
        self.assertFalse(laya_server.relabel_admissible(1099, 900, 200))


if __name__ == "__main__":
    unittest.main()
