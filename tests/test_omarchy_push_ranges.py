"""The Vulkan spec allows each shader stage in at most one push-constant range
(VUID-VkPipelineLayoutCreateInfo-pPushConstantRanges-00292). Two compute ranges
make the Honeykrisp common runtime assert in debug builds
(vk_pipeline.c get_push_range_for_stage: push_range == NULL) and are undefined
behavior elsewhere. This source check keeps the omarchy backend at one compute
range per pipeline layout."""
import re
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "overlay/mlx/backend/omarchy"


class PushRangeTest(unittest.TestCase):
    def test_one_compute_range_per_layout(self):
        for src in sorted(BACKEND.glob("*.cpp")):
            text = src.read_text()
            for m in re.finditer(r"VkPushConstantRange\s+\w+\[(\d+)\]", text):
                self.assertEqual(
                    int(m.group(1)), 1,
                    f"{src.name}: a push-constant range array of {m.group(1)}"
                    " entries; one range per stage only",
                )
            for m in re.finditer(r"pushConstantRangeCount\s*=\s*(\d+)", text):
                self.assertLessEqual(
                    int(m.group(1)), 1,
                    f"{src.name}: pushConstantRangeCount = {m.group(1)}",
                )

    def test_runtime_layout_range_covers_scalar_tail(self):
        text = (BACKEND / "compute.cpp").read_text()
        m = re.search(r"push_range\.offset = 0;\s*push_range\.size = (\d+);", text)
        self.assertIsNotNone(m, "the runtime layout must declare one range at 0")
        self.assertGreaterEqual(int(m.group(1)), 160)


if __name__ == "__main__":
    unittest.main()
