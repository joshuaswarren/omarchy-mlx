#!/usr/bin/env python3
"""SUP-121: token-parity pins across the kernel toggle, per shape.

One test per (gpu family, model, ctx): greedy 64 ids through the shipped
default kernel route, then the same prompt with the PFW GDN decode route
toggled off (MLX_OMARCHY_GDN_PF_WIDE=0, the plain prefetch route) - both
must equal the golden digest recorded on the shape's machine
(tests/omarchy/token_parity_goldens.json).

Runs only where the shape can run: the machine's GPU family must match a
golden entry and the model snapshot must be in the HF cache (every shape
skips elsewhere, so CI without the Apple GPU or the models is a clean
skip). Reverting a kernel change that shifts ids turns the test red on
the machine the golden came from.

Stdlib unittest; runs the generator as a subprocess (scripts/token_parity_digest.py)
so each shape is a fresh process, exactly how the goldens were recorded.
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "token_parity_digest.py"
GOLDENS = REPO_ROOT / "tests" / "omarchy" / "token_parity_goldens.json"
HF = Path(os.environ.get("HF_HUB_CACHE", "")) if os.environ.get("HF_HUB_CACHE") \
    else Path.home() / ".cache" / "huggingface" / "hub"

MODELS = {
    "2B": "models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e7c97b86b34b381",
    "4B": "models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b",
}
CONTEXTS = ("512", "2048", "2304", "2305")


def gpu_family():
    """The GPU part name this machine's mlx reports, or None without a GPU."""
    try:
        r = subprocess.run(
            [sys.executable, "-c",
             "import mlx.core as mx; print(str(mx.default_device()))"],
            capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    line = r.stdout.strip()
    if r.returncode != 0 or "gpu" not in line:
        return None
    for part in ("G13G", "G13C", "G14C"):
        if part in line:
            return part
    return line or None


def digest(model_dir, ctx, extra_env=None):
    env = {**os.environ}
    for key in list(env):
        if key.startswith("MLX_OMARCHY_"):
            env.pop(key)
    env.update(extra_env or {})
    r = subprocess.run(
        [sys.executable, str(SCRIPT), model_dir, str(ctx)],
        capture_output=True, text=True, timeout=900, env=env, check=False)
    out = r.stdout.strip().splitlines()
    assert r.returncode == 0 and out, f"digest run failed rc={r.returncode}: {r.stderr[-400:]}"
    return out[-1]


def model_snapshot(name):
    snap = HF / MODELS[name]
    if not (snap / "config.json").is_file():
        for candidate in HF.glob(MODELS[name].split("/snapshots")[0] + "/snapshots/*"):
            if (candidate / "config.json").is_file():
                return str(candidate)
        return None
    return str(snap)


@unittest.skipUnless(GOLDENS.exists(), "no golden table")
class TokenParityTests(unittest.TestCase):
    def test_the_golden_table_parses(self):
        table = json.loads(GOLDENS.read_text())
        self.assertIn("G13G", table)

    def test_every_context_matches_its_golden_and_the_kernel_toggle(self):
        table = json.loads(GOLDENS.read_text())
        family = gpu_family()
        if family not in table:
            self.skipTest(f"no goldens for this GPU family ({family})")
        machine = table[family]
        tried = 0
        for model_name, contexts in machine.items():
            if model_name.startswith("_"):
                continue
            model_dir = model_snapshot(model_name)
            if model_dir is None:
                continue
            for ctx in contexts:
                if ctx not in CONTEXTS:
                    continue
                with self.subTest(model=model_name, ctx=ctx):
                    got = digest(model_dir, ctx)
                    self.assertEqual(got, contexts[ctx],
                                     f"{model_name} ctx{ctx} drifted from the golden")
                    tried += 1
                    # the kernel toggle: the same prompt through the plain
                    # prefetch route must land on the same ids
                    got_toggled = digest(model_dir, ctx,
                                         extra_env={"MLX_OMARCHY_GDN_PF_WIDE": "0"})
                    self.assertEqual(got_toggled, contexts[ctx],
                                     f"{model_name} ctx{ctx} drifted with PF_WIDE=0")
        if tried == 0:
            self.skipTest("no model snapshots present for this family's shapes")

    def test_the_boundary_contexts_are_pinned(self):
        """wB4's ask: the sdpa_online_default k<=2304 split (merging #997) gets its
        2304/2305 rows so the shipped default is what the goldens pin."""
        table = json.loads(GOLDENS.read_text())
        for family, machine in table.items():
            if not isinstance(machine, dict):
                continue
            for model_name, contexts in machine.items():
                if model_name.startswith("_") or not isinstance(contexts, dict):
                    continue
                self.assertIn("2304", contexts, f"{family}/{model_name}: 2304 missing")
                self.assertIn("2305", contexts, f"{family}/{model_name}: 2305 missing")


if __name__ == "__main__":
    unittest.main()
