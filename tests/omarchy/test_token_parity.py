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
skip). A device name carrying an Apple part code with no golden rows is a
failure, not a skip (#1134): the coverage gap must stay loud. Reverting a
kernel change that shifts ids turns the test red on the machine the
golden came from.

Stdlib unittest; runs the generator as a subprocess (scripts/token_parity_digest.py)
so each shape is a fresh process, exactly how the goldens were recorded.
"""
import json
import os
import re
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
    "Qwen3.8-2B-mlx-4Bit": "models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e7c97b86b34b381",
    "Qwen3-4B-Instruct-2507-4bit": "models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b",
}
CONTEXTS = ("512", "2048", "2304", "2305")


# The Apple GPU part code as the Vulkan device name carries it, e.g.
# "Apple M1 (G13G B1)" -> G13G, "Apple M6 (G18G B1, Mesa 26.1.4)" -> G18G.
# Two digits plus one letter; the revision letter after it never parses.
PART_CODE = re.compile(r"\bG\d{2}[A-Z]\b")


def gpu_family_from_name(device_name):
    """The Apple GPU part code in a device-name string, or None.

    NVIDIA/AMD/Intel names and the bare marketing names ("Apple M1 Max")
    carry no part code; neither does the device repr "Device(gpu, 0)"
    that made #1134's probe never match.
    """
    if not device_name:
        return None
    m = PART_CODE.search(device_name)
    return m.group(0) if m else None


def _probe_device_name():
    """The GPU device name the installed mlx reports, or None without one."""
    try:
        r = subprocess.run(
            [sys.executable, "-c",
             "import mlx.core as mx\n"
             "fn = getattr(mx, 'device_info', None)\n"
             "if fn is None:\n"
             "    fn = getattr(mx.metal, 'device_info', None)\n"
             "if fn is None:\n"
             "    raise SystemExit(1)\n"
             "info = fn()\n"
             "print(info.get('device_name', '') if hasattr(info, 'get') "
             "else info)"],
            capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def gpu_family():
    """This machine's GPU family key, or None without a GPU.

    The Apple part code when the device name carries one; the raw device
    name otherwise (a non-Apple GPU still keys no golden entry and skips).
    """
    name = _probe_device_name()
    if name is None:
        return None
    return gpu_family_from_name(name) or name


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


class GpuFamilyParsingTests(unittest.TestCase):
    """ISS-01/#1134: the part code comes from the Vulkan device name, never
    from str(mx.default_device()) ("Device(gpu, 0)" names no part)."""

    APPLE_NAMES = {
        "Apple M1 (G13G B1)": "G13G",
        "Apple M1 Max (G13C C0)": "G13C",
        "Apple M2 Max (G14C C0)": "G14C",
        "Apple M3 (G15G B1)": "G15G",
        "Apple M4 (G16G B0)": "G16G",
        "Apple M6 (G18G B1)": "G18G",
        "Apple M6 (G18G B1, Mesa 26.1.4)": "G18G",
    }

    NON_APPLE_NAMES = (
        "NVIDIA GeForce RTX 5060 Ti",
        "AMD Radeon RX 7900 XTX",
        "Intel Arc B60 Graphics",
        "Apple M1 Max",          # marketing name without a part code
        "Device(gpu, 0)",        # the #1134 input: the repr, not a name
        "",
    )

    def test_apple_device_names_parse_their_part_code(self):
        for name, want in self.APPLE_NAMES.items():
            with self.subTest(name=name):
                self.assertEqual(gpu_family_from_name(name), want)

    def test_non_apple_and_partless_names_parse_no_part_code(self):
        for name in self.NON_APPLE_NAMES:
            with self.subTest(name=name):
                self.assertIsNone(gpu_family_from_name(name))

    def test_a_host_with_an_unmatched_part_code_fails_not_skips(self):
        # G15G is a parsed Apple family with no golden rows: the parity
        # test must surface it instead of silently skipping (#1134).
        self.assertTrue(PART_CODE.fullmatch("G15G"))
        self.assertNotIn("G15G", json.loads(GOLDENS.read_text()))
        self.assertFalse(PART_CODE.fullmatch("NVIDIA GeForce RTX 5060 Ti"))
        self.assertFalse(PART_CODE.fullmatch("Device(gpu, 0)"))


@unittest.skipUnless(GOLDENS.exists(), "no golden table")
class TokenParityTests(unittest.TestCase):
    def test_the_golden_table_parses(self):
        table = json.loads(GOLDENS.read_text())
        self.assertIn("G13G", table)

    def test_every_context_matches_its_golden_and_the_kernel_toggle(self):
        table = json.loads(GOLDENS.read_text())
        family = gpu_family()
        if family is None:
            self.skipTest("no GPU device name reported")
        if family not in table:
            if PART_CODE.fullmatch(family):
                self.fail(f"known Apple GPU {family} has no golden rows; "
                          f"this test must compare, not skip (#1134)")
            self.skipTest(f"no goldens for this GPU family ({family})")
        machine = table[family]
        tried = 0
        for model_name, contexts in machine.items():
            if model_name.startswith("_") or not isinstance(contexts, dict):
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
