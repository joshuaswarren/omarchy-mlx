"""The opt-in GPU base-pstate overlay compiles, is opt-in only, and sets the one property on the stock node."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DTS = ROOT / "packaging/dt/t8103-gpu-pstate.dts"
BUILD = ROOT / "packaging/build-dtbo.sh"

BASE = """/dts-v1/;
/ {
	compatible = "apple,j293", "apple,t8103";
	#address-cells = <2>;
	#size-cells = <2>;
	soc {
		#address-cells = <2>;
		#size-cells = <2>;
		gpu@206400000 {
			apple,perf-base-pstate = <1>;
			apple,perf-tgt-utilization = <85>;
		};
	};
};
"""


def have_dtc():
    return all(shutil.which(t) for t in ("dtc", "fdtget", "fdtoverlay"))


@unittest.skipUnless(have_dtc(), "dtc, fdtget and fdtoverlay are not installed")
class GpuPstateOverlayTests(unittest.TestCase):
    def test_build_installs_opt_in_overlay_and_overlay_changes_only_the_pstate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dest = root / "pkg"
            subprocess.run(["bash", str(BUILD), str(dest)], check=True, capture_output=True)
            dtbo = dest / "usr/share/omarchy-platform/dtb-overlays/t8103/omarchy-gpu-pstate.dtbo"
            self.assertTrue(dtbo.is_file())
            key = subprocess.run(["fdtget", "-t", "s", str(dtbo), "/", "omarchy,opt-in"],
                                 check=True, capture_output=True, text=True).stdout.strip()
            self.assertEqual(key, "gpu-pstate-t8103")

            (root / "base.dts").write_text(BASE)
            subprocess.run(["dtc", "-q", "-I", "dts", "-O", "dtb", "-o", str(root / "base.dtb"), str(root / "base.dts")],
                           check=True)
            subprocess.run(["fdtoverlay", "-i", str(root / "base.dtb"), "-o", str(root / "merged.dtb"), str(dtbo)],
                           check=True, capture_output=True)

            def get(dtb, prop):
                return subprocess.run(["fdtget", "-t", "u", str(dtb), "/soc/gpu@206400000", prop],
                                      check=True, capture_output=True, text=True).stdout.strip()

            self.assertEqual(get(root / "base.dtb", "apple,perf-base-pstate"), "1")
            self.assertEqual(get(root / "merged.dtb", "apple,perf-base-pstate"), "6")
            self.assertEqual(get(root / "merged.dtb", "apple,perf-tgt-utilization"), "85")

    def test_source_is_opt_in_and_targets_only_the_gpu_node(self):
        text = DTS.read_text()
        self.assertIn('omarchy,opt-in = "gpu-pstate-t8103"', text)
        self.assertEqual(text.count("target-path"), 1)
        self.assertIn('target-path = "/soc/gpu@206400000"', text)


if __name__ == "__main__":
    unittest.main()
