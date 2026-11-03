"""GPU profiler command-buffer-span mode (MLX_OMARCHY_GPU_PROFILE_MODE=cb), tested on the host against a fake Vulkan device table.

tests/profiler_cb_span/harness.cpp drives overlay/mlx/backend/omarchy/gpu_profiler.h through the hook sequence encoder.cpp uses and prints how many timestamp
writes, pipeline barriers and query resets the profiler recorded. What this proves: the default per-dispatch profile is unchanged (two timestamps and one barrier per
dispatch, one "d" record per dispatch), and the cb mode records exactly two timestamps per submission, no per-dispatch timestamp or barrier, and one "c" record per
submission. What it does not prove: anything about real hardware, or that the generated tokens are identical (that needs a model; the jw16 row compares digests).
"""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
HARNESS = HERE / "profiler_cb_span" / "harness.cpp"
STUBS = HERE / "profiler_cb_span" / "stubs"
OVERLAY = REPO / "overlay"


@unittest.skipUnless(shutil.which("g++") and Path("/usr/include/vulkan/vulkan.h").exists(), "needs g++ and the Vulkan headers")
class CommandBufferSpanMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="profcb-")
        cls.exe = os.path.join(cls.tmp, "harness")
        r = subprocess.run(["g++", "-std=c++20", "-O1", "-DMLX_OMARCHY_GPU_PROFILING", "-I", str(STUBS), "-I", str(OVERLAY), "-o", cls.exe, str(HARNESS)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError("harness did not compile:\n" + r.stderr[-2500:])

    def run_harness(self, mode, submits=1, dispatches=3, rejoin=False):
        prof = os.path.join(self.tmp, f"prof-{mode}-{submits}-{dispatches}-{rejoin}.jsonl")
        env = dict(os.environ, MLX_OMARCHY_GPU_PROFILE=prof)
        env.pop("MLX_OMARCHY_GPU_PROFILE_MODE", None)
        if mode is not None:
            env["MLX_OMARCHY_GPU_PROFILE_MODE"] = mode
        r = subprocess.run([self.exe, str(submits), str(dispatches)] + (["rejoin"] if rejoin else []), capture_output=True, text=True, env=env, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        counts = json.loads(r.stdout.strip().splitlines()[-1])
        recs = [json.loads(l) for l in open(prof) if l.strip()]
        return counts, recs, r.stderr

    def test_default_mode_is_the_unchanged_per_dispatch_profile(self):
        counts, recs, _ = self.run_harness(None, submits=1, dispatches=3)
        self.assertEqual(counts["timestamps"], 6, "two timestamps per dispatch")
        self.assertEqual(counts["barriers"], 3, "one barrier per dispatch")
        d = [r for r in recs if r["k"] == "d"]
        self.assertEqual(len(d), 3)
        self.assertEqual([(r["t0"], r["t1"]) for r in d], [(1000, 1100), (1200, 1300), (1400, 1500)])
        self.assertEqual([r for r in recs if r["k"] == "c"], [])
        self.assertEqual([r for r in recs if r["k"] == "meta"][0].get("mode"), "dispatch")

    def test_cb_mode_records_two_timestamps_per_submission_and_nothing_per_dispatch(self):
        counts, recs, _ = self.run_harness("cb", submits=1, dispatches=5)
        self.assertEqual(counts["timestamps"], 2, "start and end of the command buffer")
        self.assertEqual(sorted(counts["ts_queries"]), [0, 1])
        self.assertEqual(counts["barriers"], 0, "no per-dispatch barrier")
        self.assertEqual([r for r in recs if r["k"] == "d"], [], "no per-dispatch record")
        c = [r for r in recs if r["k"] == "c"]
        self.assertEqual(len(c), 1)
        self.assertEqual((c[0]["s"], c[0]["t0"], c[0]["t1"], c[0]["nd"]), (1, 1000, 1100, 5))
        self.assertEqual([r for r in recs if r["k"] == "meta"][0].get("mode"), "cb")

    def test_cb_mode_keeps_the_host_side_records(self):
        _, recs, _ = self.run_harness("cb", submits=2, dispatches=2)
        kinds = [r["k"] for r in recs]
        for k in ("meta", "b", "s", "j", "end"):
            self.assertIn(k, kinds, k)
        end = [r for r in recs if r["k"] == "end"][0]
        self.assertEqual((end["dispatches"], end["submissions"]), (4, 2), "dispatches are still counted")

    def test_cb_mode_two_submissions_in_one_ring_slot_each_get_their_own_record(self):
        counts, recs, _ = self.run_harness("cb", submits=2, dispatches=1)
        self.assertEqual(counts["timestamps"], 4)
        c = [r for r in recs if r["k"] == "c"]
        self.assertEqual([r["s"] for r in c], [1, 2])
        self.assertTrue(all(r["t0"] == 1000 and r["t1"] == 1100 for r in c))

    def test_a_second_join_does_not_emit_the_records_again(self):
        _, recs, _ = self.run_harness("cb", submits=2, dispatches=1, rejoin=True)
        self.assertEqual([r["s"] for r in recs if r["k"] == "c"], [1, 2])
        _, recs, _ = self.run_harness(None, submits=1, dispatches=2, rejoin=True)
        self.assertEqual(len([r for r in recs if r["k"] == "d"]), 2, "the default mode does not duplicate either")

    def test_an_unknown_mode_falls_back_to_the_default_and_says_so(self):
        counts, recs, err = self.run_harness("bogus", submits=1, dispatches=2)
        self.assertEqual(counts["timestamps"], 4)
        self.assertIn("MLX_OMARCHY_GPU_PROFILE_MODE", err)
        self.assertEqual([r for r in recs if r["k"] == "meta"][0].get("mode"), "dispatch")

    def test_the_compiled_out_mirror_has_the_new_hook(self):
        src = HARNESS.read_text()
        r = subprocess.run(["g++", "-std=c++20", "-fsyntax-only", "-I", str(STUBS), "-I", str(OVERLAY), "-x", "c++", "-"],
                           input=('#include <span>\n#include <vulkan/vulkan.h>\n#include "mlx/backend/omarchy/compute.h"\n#include "mlx/backend/omarchy/device.h"\n'
                                  '#include "mlx/backend/omarchy/gpu_profiler.h"\nusing namespace mlx::core::omarchy;\n'
                                  'void f(void* o, VkCommandBuffer c){ prof::get().before_end(o, 0, c); }\n'),   # the mirror relies on the includer, as encoder.cpp does
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        self.assertIn("before_end", src)


if __name__ == "__main__":
    unittest.main()
