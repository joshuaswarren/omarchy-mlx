"""Tests for the mlx_lm patch-marker gate in score_bench.py.

A Linux score run on an unpatched mlx_lm silently takes the eager f32
fallback paths and produces inflated ms/token rows - this happened on a
nightly scorecard and went unnoticed. The runner must refuse unless
STOCK_MLX_LM=1 and must record the marker count in the receipt env.

Stdlib unittest only; score_bench.py runs as a subprocess against stub
mlx/mlx_lm packages so no real mlx install is needed.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCORE_BENCH = REPO_ROOT / "receipts" / "2026-10-09-linux-vs-macos" / "score_bench.py"

# Minimal stubs that carry score_bench.py through warmup, decode and prefill
# to the receipt write: FakeArr tolerates the [None] and [:n] indexing the
# script applies to mx.array results.
FAKE_ARR = """class FakeArr(list):
    def __getitem__(self, i):
        return FakeArr()
def array(x, *a, **k):
    return FakeArr()
def eval(*a, **k):
    pass
def get_peak_memory():
    return 0
def default_device():
    return 'gpu'
"""

STUBS = {
    "mlx/__init__.py": "",
    "mlx/core.py": FAKE_ARR,
    "mlx_lm/__init__.py":
        "__version__ = '0.0.0-stub'\n"
        "class _M:\n"
        "    def __call__(self, *a, **k):\n"
        "        return None\n"
        "class _T:\n"
        "    def encode(self, t):\n"
        "        return [1, 2, 3]\n"
        "def load(m):\n"
        "    return _M(), _T()\n"
        "def generate(*a, **k):\n"
        "    pass\n",
    "mlx_lm/generate.py":
        "def generate_step(*a, **k):\n"
        "    return iter([(1, None), (2, None)])\n",
    "mlx_lm/models/__init__.py": "",
    "mlx_lm/models/cache.py": "def make_prompt_cache(*a, **k):\n    return None\n",
}
MARKER_FILE = ("mlx_lm/models/gated_delta.py",
               "# omarchy fused route\nx = 'mx.fast.gated_delta_update'\n")


def _venv(root, marker=False):
    for rel, body in STUBS.items():
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body)
    if marker:
        f = root / MARKER_FILE[0]
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(MARKER_FILE[1])
    return str(root)


def _run(marker=False, env_extra=None):
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    env = dict(os.environ)
    env["PYTHONPATH"] = _venv(root, marker)
    env.pop("STOCK_MLX_LM", None)
    env.update(env_extra or {})
    prompts = root / "prompts.jsonl"
    prompts.write_text('{"text": "hi"}\n')
    out = root / "o.json"
    r = subprocess.run(
        [sys.executable, str(SCORE_BENCH), "--model", "m",
         "--prompts", str(prompts), "--out", str(out),
         "--warmup", "0", "--pf-reps", "1", "--decode-passes", "1",
         "--decode-limit", "1", "--pf-tokens", "2"],
        capture_output=True, text=True, env=env)
    return r, out, tmp


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux-only gate")
class ScoreBenchMarkerGateTests(unittest.TestCase):
    def test_refuses_unpatched_mlx_lm(self):
        r, out, tmp = _run()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("0 omarchy patch markers", r.stderr + r.stdout)
        self.assertIn("install.sh --patch-venv", r.stderr + r.stdout)
        tmp.cleanup()

    def test_stock_env_bypasses_gate_and_records_zero(self):
        r, out, tmp = _run(env_extra={"STOCK_MLX_LM": "1"})
        self.assertNotIn("0 omarchy patch markers", r.stderr + r.stdout)
        res = json.loads(out.read_text())
        self.assertEqual(res["env"]["mlx_lm_patch_markers"], 0)
        tmp.cleanup()

    def test_patched_tree_passes_gate_and_records_count(self):
        r, out, tmp = _run(marker=True)
        self.assertNotIn("0 omarchy patch markers", r.stderr + r.stdout)
        res = json.loads(out.read_text())
        self.assertGreater(res["env"]["mlx_lm_patch_markers"], 0)
        tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
