# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Unit tests for the M6 G18G <=128-thread device-chained TDT path."""

import types
import unittest
from unittest import mock

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml import vulkan_tdt_chain
from coreml import vulkan_tdt_chain_wg128 as wg128


class TdtChainWg128RoutingTest(unittest.TestCase):
    def _fake_mx(self, device_name: str):
        return types.SimpleNamespace(
            device_info=lambda: {"device_name": device_name}
        )

    def test_auto_selects_wg128_on_g18g_and_m6(self):
        for name in ("Apple M6 (G18G B1)", "Apple M6", "apple g18g"):
            with mock.patch.dict("os.environ", {}, clear=True):
                with mock.patch.object(
                    vulkan_tdt_chain, "_mx", return_value=self._fake_mx(name)
                ):
                    self.assertTrue(vulkan_tdt_chain._use_wg128_chain(), name)

    def test_keeps_default_chain_on_older_apple_gpus(self):
        for name in (
            "Apple M1 Max (G13C C0)",
            "Apple M2 Max (G14C B0)",
            "Apple M5 Max (G17X)",
            "",
        ):
            with mock.patch.dict("os.environ", {}, clear=True):
                with mock.patch.object(
                    vulkan_tdt_chain, "_mx", return_value=self._fake_mx(name)
                ):
                    self.assertFalse(vulkan_tdt_chain._use_wg128_chain(), name)

    def test_env_override_takes_precedence(self):
        with mock.patch.object(
            vulkan_tdt_chain,
            "_mx",
            return_value=self._fake_mx("Apple M1 Max (G13C C0)"),
        ):
            with mock.patch.dict(
                "os.environ", {"MLX_OMARCHY_TDT_CHAIN_WG128": "1"}
            ):
                self.assertTrue(vulkan_tdt_chain._use_wg128_chain())
        with mock.patch.object(
            vulkan_tdt_chain,
            "_mx",
            return_value=self._fake_mx("Apple M6 (G18G B1)"),
        ):
            with mock.patch.dict(
                "os.environ", {"MLX_OMARCHY_TDT_CHAIN_WG128": "0"}
            ):
                self.assertFalse(vulkan_tdt_chain._use_wg128_chain())

    def test_run_tdt_chain_delegates_when_wg128_selected(self):
        sentinel = object()
        seen = {}

        def fake_wg128(**kwargs):
            seen.update(kwargs)
            return sentinel

        with mock.patch.object(
            vulkan_tdt_chain, "_use_wg128_chain", return_value=True
        ):
            with mock.patch.object(
                wg128, "run_tdt_chain_wg128", side_effect=fake_wg128
            ):
                out = vulkan_tdt_chain.run_tdt_chain(
                    packed="p",
                    encoder="e",
                    valid_frames=10,
                    config="cfg",
                    initial_hidden="h",
                    initial_cell="c",
                    slots_per_chunk=32,
                    max_chunks=4,
                )
        self.assertIs(out, sentinel)
        self.assertEqual(seen["valid_frames"], 10)
        self.assertEqual(seen["slots_per_chunk"], 32)
        self.assertEqual(seen["max_chunks"], 4)


class TdtChainWg128KernelInvariantsTest(unittest.TestCase):
    def test_grid_and_workgroup_bounds(self):
        self.assertEqual(wg128._CHAIN_THREADS, 128)
        self.assertEqual(wg128._CHAIN_GROUPS * wg128._CHAIN_THREADS, 25600)
        self.assertEqual(wg128._NGROUPS, 65)
        self.assertGreaterEqual(wg128._NGROUPS * wg128._CHAIN_THREADS, 8198)
        self.assertLess((wg128._NGROUPS - 1) * wg128._CHAIN_THREADS, 8193)

    def test_generated_sources_have_no_unsubstituted_placeholders(self):
        sources = {
            "c0": wg128._build_chains_l0(),
            "c1": wg128._build_chain1_fold(),
            "fp": wg128._build_fold_proj(),
            "win": wg128._build_window(),
            "ctl": wg128._build_control(),
        }
        for name, src in sources.items():
            for token in (
                "_C_FRAME", "_C_SYM", "_C_COUNT", "_C_ERR", "_C_SKIP",
                "_C_TOKEN", "_C_DONE", "_C_RUN", "_C_LAST",
                "_G_VALID", "_G_BLANK", "_G_MAXSYM", "_G_NDUR", "_G_DUR0",
                "_G_CAP", "_G_CAP3",
            ):
                self.assertNotIn(token, src, f"{name} left placeholder {token}")

    def test_window_has_no_early_return_and_control_is_barrier_free(self):
        win_src = wg128._build_window()
        ctl_src = wg128._build_control()
        self.assertNotIn("return;", win_src)
        self.assertIn("s_val[128]", win_src)
        self.assertIn("precise float acc", win_src)
        self.assertNotIn("threadgroup_barrier", ctl_src)
        self.assertIn("g < 65u", ctl_src)
        self.assertIn("need = (count == 0) ? 1 : 0;", ctl_src)

    def test_lstm_kernels_preserve_exact_fma16_contract(self):
        self.assertIn("exact_fma16(", wg128._build_chains_l0())
        self.assertIn("exact_fma16(", wg128._build_chain1_fold())
        self.assertIn("exact_fma16(", wg128._build_fold_proj())


if __name__ == "__main__":
    unittest.main()
