"""Generate tests/fixtures/laya/laya_reference.json.

Runs BOTH the pinned upstream torch implementation and the mlx port on the
battery inputs (CPU, float32) and stores the full expected envelopes in the
committed fixture. The fixture is what hardware qualification compares
against, so the GPU gate never needs torch or the 843 MB download.

Environment (see tests/test_laya_reference.py):
  LAYA_UPSTREAM_DIR, LAYA_REF_CKPT_DIR, LAYA_MLX_CKPT

Run:  python tests/fixtures/laya/generate_fixture.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "serve"))
sys.path.insert(0, str(REPO / "tests"))

from test_laya_reference import BATTERY  # noqa: E402  (shared battery definition)

OUT = Path(__file__).resolve().parent / "laya_reference.json"


def main():
    upstream_dir = os.environ["LAYA_UPSTREAM_DIR"]
    ref_ckpt = os.environ["LAYA_REF_CKPT_DIR"]
    converted = os.environ["LAYA_MLX_CKPT"]

    from mlx_omarchy_laya.api import LayaEngine
    import test_laya_reference as refmod

    engine = LayaEngine(converted, dtype="float32", require_gpu=False)
    cases = []
    for name, state, questions in BATTERY:
        ref = refmod._upstream_envelope(upstream_dir, ref_ckpt, state, questions)
        got = engine.system_one(state, questions)
        # enforce agreement at generation time; the fixture never stores a disagreement
        refmod._assert_envelopes_close(_Silent(), ref, got, name)
        for envelope in (ref, got):
            envelope.pop("timings", None)  # wall times are machine-specific, never expectations
        cases.append({
            "name": name,
            "state": state,
            "questions": questions,
            "expected_upstream_fp32_cpu": ref,
            "expected_mlx_fp32_cpu": got,
        })
    fixture = {
        "schema": "laya-reference-fixture/1",
        "source_repo": "convaiinnovations/laya",
        "source_revision": "1c5edc17a7acd8701df6fc341c0d179f1c62c982",
        "pack_repo": "aac6fef/laya-mlx",
        "pack_revision": "20aed815fc6acde75733882e7ec0e3f28aeb9717",
        "reference_env": {
            "mlx_engine": engine.model_id,
            "max_len": engine.max_len,
            "head_max_len": engine.head_max_len,
        },
        "cases": cases,
    }
    OUT.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print("wrote %s (%d cases)" % (OUT, len(cases)))


class _Silent:
    """assert helper sink: failures still raise AssertionError with context."""

    def assertEqual(self, a, b, msg=""):
        assert a == b, msg

    def assertAlmostEqual(self, a, b, delta=None, msg=""):
        assert abs(a - b) <= delta, msg

    def assertLessEqual(self, a, b, msg=""):
        assert a <= b, msg

    def fail(self, msg=""):
        raise AssertionError(msg)


if __name__ == "__main__":
    main()
