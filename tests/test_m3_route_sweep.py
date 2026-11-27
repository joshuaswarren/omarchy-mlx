"""Tests for scripts/m3_route_sweep.py (card M3-03).

The sweep is a zero-build runner: every check here runs without mlx, a
GPU, or a model. The one end-to-end contract is ``run --dry-run`` -
the card names it - which must emit the card row with all legs
``status="planned"`` and exit 0 on any host.
"""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "m3_route_sweep.py"


def load_module():
    spec = importlib.util.spec_from_file_location("m3_route_sweep", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LeverTableTests(unittest.TestCase):
    """The 8 levers must name real env vars in the backend source."""

    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()
        sources = [
            REPO_ROOT / "overlay/mlx/backend/omarchy" / name
            for name in ("encoder.h", "device.h", "primitives.cpp",
                         "fused_chain.cpp", "device_local_activations.h")
        ]
        cls.overlay_text = "".join(
            p.read_text(encoding="utf-8") for p in sources if p.is_file())

    def test_card_order_and_count(self):
        self.assertEqual(
            [lever.arm for lever in self.mod.LEVERS],
            ["batch_work", "qmm_geom", "sdpa_online_rope",
             "qmm_fast_only", "resnorm_vec", "norm_apple_128",
             "fused_gemv", "agx_simdmat"])

    def test_every_env_name_is_real(self):
        for lever in self.mod.LEVERS:
            for var, _ in lever.env:
                self.assertIn(var, self.overlay_text,
                              f"{lever.arm}: {var} not in omarchy source")

    def test_lever_values(self):
        env = {v: val for lever in self.mod.LEVERS for v, val in lever.env}
        self.assertEqual(env["MLX_OMARCHY_BATCH_WORK"], "100000")
        self.assertEqual(env["MLX_OMARCHY_QMM_GEOM"], "2x8w2")
        self.assertEqual(env["MLX_OMARCHY_SDPA_ONLINE_ROPE_MAX_KEYS"],
                         "2304")
        self.assertEqual(env["MLX_OMARCHY_SDPA_G13C_FAST"], "1")
        self.assertEqual(env["MLX_OMARCHY_FUSED_GEMV"], "0")
        self.assertEqual(env["MLX_OMARCHY_GEMV_BALANCED_QKV"], "1")
        self.assertEqual(env["AGX_SIMDMAT"], "1")


class ArmEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()

    def test_baseline_has_no_lever_env(self):
        self.assertEqual(self.mod.arm_env("baseline"), {})

    def test_combined_sets_every_lever_var(self):
        combined = self.mod.arm_env("combined")
        self.assertEqual(sorted(combined), sorted(self.mod.LEVER_VARS))

    def test_leg_env_scrubs_inherited_lever_vars(self):
        inherited = {v: "9" for v in self.mod.LEVER_VARS}
        inherited["PATH"] = "/usr/bin"
        leg = self.mod.leg_env(inherited, "baseline")
        self.assertNotIn("MLX_OMARCHY_BATCH_WORK", leg)
        self.assertNotIn("AGX_SIMDMAT", leg)
        self.assertEqual(leg["PATH"], "/usr/bin")
        armed = self.mod.leg_env(inherited, "qmm_fast_only")
        self.assertEqual(armed["MLX_OMARCHY_QMM_FAST_ONLY"], "1")
        self.assertNotIn("AGX_SIMDMAT", armed)

    def test_select_arms(self):
        self.assertEqual(self.mod.select_arms(None),
                         list(self.mod.ARM_IDS))
        self.assertEqual(self.mod.select_arms(["levers"]),
                         [l.arm for l in self.mod.LEVERS])
        self.assertEqual(
            self.mod.select_arms(["baseline", "levers", "combined"]),
            ["baseline"] + [l.arm for l in self.mod.LEVERS] + ["combined"])
        with self.assertRaises(KeyError):
            self.mod.select_arms(["bogus"])


class ParseAndGreedyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()

    def test_parse_bench_output(self):
        stdout = ("uclamp: min=0\n"
                  "provenance: verified=match wheel=w\n"
                  "prompt_tokens 12\n"
                  "decode 43.38 tok/s over 63 tokens (64 requested, EOS "
                  "suppressed)\n"
                  "prefill 1.502s (reported separately, excluded from "
                  "decode)\n"
                  "generated_ids sha256:294463eb6dba3a58 n=64 first=1,2,3"
                  " last=4,5,6\n")
        got = self.mod.parse_bench_output(stdout)
        self.assertEqual(got["decode_tps"], 43.38)
        self.assertEqual(got["prefill_s"], 1.502)
        self.assertEqual(got["ids_sha256_16"], "294463eb6dba3a58")
        self.assertEqual(got["generated_ids_n"], 64)
        self.assertEqual(got["prompt_tokens"], 12)

    def test_greedy_check(self):
        legs = [{"arm": "baseline", "status": "measured",
                 "ids_sha256_16": "aa"},
                {"arm": "x", "status": "measured",
                 "ids_sha256_16": "aa"}]
        self.assertEqual(self.mod.greedy_check(legs)["verdict"],
                         "identical")
        legs[1]["ids_sha256_16"] = "bb"
        got = self.mod.greedy_check(legs)
        self.assertEqual(got["verdict"], "mismatch")
        self.assertEqual(got["mismatched"], ["x"])
        self.assertEqual(
            self.mod.greedy_check([legs[1]])["verdict"], "no-baseline")

    def test_verify_maps(self):
        self.assertIsNotNone(self.mod.verify_maps({}))
        self.assertIsNotNone(
            self.mod.verify_maps({"/x/mlx.so (deleted)": "a" * 64}))
        self.assertIsNotNone(
            self.mod.verify_maps({"/x/mlx.so": "unreadable: EACCES"}))
        with tempfile.NamedTemporaryFile(suffix="_mlx.so") as fh:
            fh.write(b"mapped-bytes")
            fh.flush()
            self.assertIsNone(self.mod.verify_maps(
                {fh.name: self.mod.sha256_file(fh.name)}))
            self.assertIsNotNone(self.mod.verify_maps(
                {fh.name: "0" * 64}))


class CliTests(unittest.TestCase):
    def run_script(self, *argv):
        return subprocess.run([sys.executable, str(SCRIPT), *argv],
                              capture_output=True, text=True, timeout=60)

    def test_self_test(self):
        proc = self.run_script("--self-test")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("self-test: OK", proc.stdout)

    def test_plan_emits_plan_and_card_row(self):
        proc = self.run_script("plan")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lines = [json.loads(l) for l in proc.stdout.strip().splitlines()]
        self.assertEqual(lines[0]["schema"], "m3-route-sweep/1")
        self.assertEqual(lines[0]["mode"], "plan")
        arms = [p["arm"] for p in lines[0]["plan"]]
        self.assertEqual(arms[0], "baseline")
        self.assertEqual(len(arms), 10)  # baseline + 8 levers + combined
        self.assertEqual(lines[1]["card"], "M3-03")
        self.assertTrue(all(l["status"] == "planned"
                            for l in lines[1]["legs"]))

    def test_dry_run_verified(self):
        """The card's --dry-run contract: full plan + card row, no mlx."""
        proc = self.run_script("run", "--dry-run",
                               "--model", "/nonexistent/model",
                               "--prompt", "hi")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lines = [json.loads(l) for l in proc.stdout.strip().splitlines()]
        self.assertEqual(lines[0]["mode"], "run")
        row = lines[1]
        self.assertEqual(row["schema"], "m3-route-sweep/1")
        self.assertEqual(row["greedy"]["verdict"], "not-run")
        self.assertTrue(all(l["status"] == "planned"
                            for l in row["legs"]))
        self.assertNotIn("REFUSING", proc.stderr)

    def test_dry_run_arms_subset(self):
        proc = self.run_script("run", "--dry-run",
                               "--arms", "baseline,agx_simdmat")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        row = json.loads(proc.stdout.strip().splitlines()[1])
        self.assertEqual([l["arm"] for l in row["legs"]],
                         ["baseline", "agx_simdmat"])
        self.assertEqual(row["legs"][1]["env"], {"AGX_SIMDMAT": "1"})

    def test_run_requires_model_and_prompt(self):
        proc = self.run_script("run")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--model is required", proc.stderr)

    def test_unknown_arm_refused(self):
        proc = self.run_script("run", "--dry-run", "--arms", "nope")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("unknown arm", proc.stderr)


if __name__ == "__main__":
    unittest.main()
