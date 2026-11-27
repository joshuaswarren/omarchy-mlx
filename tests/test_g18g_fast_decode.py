# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""G18G (Apple M6) fast-decode gate coverage: the DEC-12 audit's per-flag map."""

import pathlib
import re
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
DEVICE_H = (REPO / "overlay" / "mlx" / "backend" / "omarchy" / "device.h").read_text(encoding="utf-8")
PRIMITIVES = (REPO / "overlay" / "mlx" / "backend" / "omarchy" / "primitives.cpp").read_text(encoding="utf-8")

G18G = "Apple M6 (G18G)"


def _gate_body(name: str) -> str:
    m = re.search(rf"inline \w+ {name}\([^)]*\) \{{(.*?)\n\}}", DEVICE_H, re.S)
    assert m, f"{name} not found in device.h"
    return m.group(1)


def _default_on(name: str, device: str) -> bool:
    """Eval a no-env part-list gate: true iff the device's family token appears in the return chain."""
    body = _gate_body(name)
    if "getenv" in body and "return true" in body.split("return")[-1]:
        return True  # no-gating default
    m = re.search(r"device_name\.find\(\"(G\w+)\"\)", body)
    families = set(re.findall(r'device_name\.find\("(\w+)"\)', body))
    return any(f in device for f in families) and not re.search(r"==\s*std::string::npos;\s*$", body.strip())


class CorrectnessExclusionsHold(unittest.TestCase):
    """mesa#737 miscompute family stays off on G18G."""

    def test_gather_qmm_sub_still_off(self):
        body = _gate_body("gather_qmm_sub_default_part")
        self.assertIn('find("G18G") == std::string::npos', body)

    def test_gdn_conv_delta_norm_still_off(self):
        body = _gate_body("gdn_conv_delta_norm_default_part")
        self.assertIn('find("G18G") != std::string::npos', body)
        self.assertIn("return false", body)

    def test_gdn_composed_still_forced(self):
        body = _gate_body("gdn_g18g_composed_default")
        self.assertIn('find("G18G") == std::string::npos', body)

    def test_forward_and_vjp_compose_on_g18g(self):
        self.assertGreaterEqual(PRIMITIVES.count("gdn_g18g_composed_default"), 2)


class EnabledGates(unittest.TestCase):
    def test_qmm_word_r2s4_lists_g18g(self):
        self.assertIn('find("G18G")', _gate_body("qmm_word_r2s4_default"))

    def test_norm_apple_128_lists_g18g(self):
        self.assertIn('find("G18G")', _gate_body("norm_apple_128_default_part"))

    def test_wave_sched_default_on(self):
        body = _gate_body("wave_sched_default_part")
        self.assertTrue(body.rstrip().endswith("return true;"))  # no-gating default covers G18G

    def test_sdpa_split_threshold_default_on(self):
        body = _gate_body("sdpa_split_min_keys")
        self.assertIn("return 320", body)  # no-gating default covers unlisted G18G


class SuspectsStayOff(unittest.TestCase):
    """512/1024-lane workgroups span the mesa#737 lanes-128-255 window."""

    def test_resnorm_vec4_not_listed(self):
        self.assertNotIn("G18G", _gate_body("qmm_resnorm_vec4_default"))

    def test_multi_r2s16_not_listed(self):
        self.assertNotIn("G18G", _gate_body("qmm_g14c_multi_r2s16_default_part"))


if __name__ == "__main__":
    unittest.main()
