#!/usr/bin/env python3
"""Tests for scripts/ane_soc_from_collect.py and the Linux SET-candidate
labeling in collect_quick/collect_common.

Fixture rows are shrunk-but-faithful shapes of the published community
rows (t6020 Linux 029744bb…, t6020 macOS 4e8224a39f40…, t6021 macOS
2b83108f…/a3e974f8…, t6030 macOS a4abb1b1…).
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import ane_soc_from_collect as gen  # noqa: E402
from collect_common import _cap_port_detail  # noqa: E402


def macos_row(soc="t6020", model="Mac14,9", set_base="0x8e08c000",
              pmgr="0x8e080000", confirms=True, with_platform=True):
    macos = {
        "available": True,
        "set_base_candidate": {
            "pmgr_block": pmgr, "offset": "0xc000", "base": set_base,
            "driver_window_confirms": confirms,
        },
        "ane_nodes": [{
            "name": "ane0",
            "IOInterruptSpecifiers": "t\x03",
            "range_kinds": ["ane_mmio", "pmgr_block", "pmgr_plus_c000"],
            "reg_ranges": ["0x84000000/0x2000000", "0x8e080000/0x4034",
                           "0x8e08c000/0x4000"],
        }],
        "dart_nodes": [{
            "name": "dart-ane0",
            "IOInterruptSpecifiers": "u\x03",
            "reg_ranges": ["0x85800000/0x4000", "0x85810000/0x4000",
                           "0x85820000/0x4000", "0x85804000/0x4000"],
        }],
    }
    if with_platform:
        macos["platform"] = {"soc_id": soc, "model": model}
    return {"content_sha256": "a" * 64, "chip": "Apple M2 Pro",
            "model": model, "summary": {"ane_port_detail": {"macos": macos}}}


def linux_row(soc="t6020", pmgr_base="0x28e080000"):
    return {
        "content_sha256": "b" * 64, "chip": f"apple,{soc}",
        "model": "Apple MacBook Pro (14-inch, M2 Pro, 2023)",
        "summary": {"ane_port_detail": {"devicetree": {
            "ane_node_present": False,
            "pmgr_blocks": [{
                "path": f"soc/power-management@{pmgr_base}",
                "reg": [f"{pmgr_base}/0x8000"],
                "children": [
                    {"name": "power-controller@2e0", "label": "ane_cpu"},
                    {"name": "power-controller@4018", "label": "ane_set1"},
                    {"name": "power-controller@4020", "label": "ane_set2"},
                ],
            }],
            "aic": {"path": "soc/interrupt-controller@28e100000"},
            "darts": {}, "boot": {"compatible": ["apple,j414s"]},
        }}},
    }


def derive(*rows):
    pairs = [(r, r["content_sha256"][:12]) for r in rows]
    return gen.derive(gen.group_rows(pairs))


class TranslationTest(unittest.TestCase):
    def test_t6020_pair_yields_proven_constants(self):
        out = derive(linux_row(), macos_row())["t6020"]
        self.assertEqual(out["ps_base"], 0x28E08C000)
        self.assertEqual(out["pmgr_phys"], 0x28E080000)
        self.assertEqual(out["ane_reg"], (0x284000000, 0x2000000))
        self.assertEqual(out["irq"], 0x374)
        self.assertEqual(out["compatible"], "apple,t6020-ane")

    def test_high_bits_cross_soc_only_on_low32_match(self):
        # t6021 has no Linux row; the t6020 Linux row's ANE pmgr block
        # (low-32 0x8e080000) sources the high bits, recorded in
        # ps_source.
        out = derive(linux_row(), macos_row(soc="t6021", model="Mac14,5"),
                     macos_row(soc="t6021", model="Mac14,5"))["t6021"]
        self.assertEqual(out["ps_base"], 0x28E08C000)
        self.assertIn("soc t6020", out["ps_source"])

    def test_low32_mismatch_never_translates(self):
        # A Linux pmgr block whose low-32 does not match the macOS
        # pmgr_block must not be used for high bits.
        out = derive(linux_row(pmgr_base="0x23b700000"),
                     macos_row())["t6020"]
        self.assertIn("refused", out)
        self.assertIn("low-32", out["refused"])


class RefusalTest(unittest.TestCase):
    def test_no_driver_window_refuses_without_blacklist(self):
        # t6030 shape: driver_window_confirms=false.
        out = derive(macos_row(soc="t6030", model="Mac15,7",
                               set_base="0x14070c000", pmgr="0x140700000",
                               confirms=False))
        self.assertIn("set_base omitted", out["t6030"]["refused"])
        self.assertIn("not blacklisted", out["t6030"]["refused"])

    def test_linux_only_soc_never_yields_set_base(self):
        # The ane_set* pwrstate cells exist but no macOS row: no SET.
        row = linux_row()
        out = derive(row)["t6020"]
        self.assertIn("refused", out)
        self.assertIn("macOS", out["refused"])

    def test_missing_c000_window_refuses(self):
        row = macos_row()
        node = row["summary"]["ane_port_detail"]["macos"]["ane_nodes"][0]
        node["range_kinds"] = ["ane_mmio", "pmgr_block"]
        node["reg_ranges"] = node["reg_ranges"][:2]
        out = derive(row)["t6020"]
        self.assertIn("refused", out)
        self.assertIn("0xc000 window", out["refused"])


class EmissionTest(unittest.TestCase):
    def test_overlay_mirrors_proven_t6001_shape(self):
        out = derive(linux_row(), macos_row())["t6020"]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t6020-ane-overlay.dts"
            err = gen.emit_overlay(out, p)
            self.assertIsNone(err)
            text = p.read_text()
        # Three dart instances from the 64K-aligned ranges, engine node,
        # IRQs from the macOS specifiers.
        self.assertIn("iommu@285800000", text)
        self.assertIn("iommu@285810000", text)
        self.assertIn("iommu@285820000", text)
        self.assertIn("ane@284000000", text)
        self.assertIn("interrupts = <0 0 884 4>;", text)
        self.assertIn("iommus = <&ane_dart0 0>, <&ane_dart1 0>,"
                      " <&ane_dart2 0>;", text)
        # pwrstate ane_set* nodes are referenced as genpd suppliers via
        # target-path aliases -- empty __overlay__ bodies, never a reg
        # of their own, and never the SET base.
        self.assertIn("power-controller@4018\";\n"
                      "\t\tps_ane_set1: __overlay__ {\n"
                      "\t\t};", text)

    def test_c_row_carries_recognized_row(self):
        out = derive(linux_row(), macos_row())["t6020"]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t6020-ane-soc.c"
            gen.emit_c_row(out, p)
            text = p.read_text()
        self.assertIn(".ps_base = 0x28e08c000ULL,", text)
        self.assertIn(".qual = ANE_RECOGNIZED,", text)
        self.assertIn('"apple,t6020-ane"', text)

    def test_grouping_merges_v064_row_by_model(self):
        # A v0.6.4 row (no platform.soc_id, macOS chip string) must join
        # the soc group that shares its board model, not split off.
        old = {"content_sha256": "c" * 64, "chip": "Apple M2 Max",
               "model": "Mac14,5",
               "summary": {"ane_port_detail": {"macos": {
                   "available": True,
                   "ane_nodes": [{"name": "ane0",
                                  "reg": "00000084000000000000000200"}],
               }}}}
        socs = gen.group_rows([(macos_row(soc="t6021", model="Mac14,5"),
                                "x"), (old, "y")])
        self.assertNotIn("Apple M2 Max", socs)
        self.assertIn("t6021", socs)
        self.assertEqual(len(socs["t6021"].macos_shas), 2)


class CollectorLabelingTest(unittest.TestCase):
    def test_linux_set_base_candidate_labels_pwrstate_cells(self):
        # The Linux collector must ship a set_base_candidate that marks
        # status not_available and keeps the pwrstate cells separate.
        detail = _cap_port_detail({"devicetree": {
            "ane_node_present": False, "ane_nodes": {}, "ane_reg": None,
            "darts": {}, "aic": None, "phandles": {}, "boot": None,
            "pmgr_domains": [
                {"path": "soc/power-management@28e080000/"
                         "power-controller@4018", "label": "ane_set1",
                 "compatible": ["apple,pmgr-pwrstate"]},
            ],
            "pmgr_blocks": [],
            "set_base_candidate": {
                "status": "not_available_from_device_tree",
                "ane_pwrstate_cells": [{"label": "ane_set1",
                                        "offset": 0x4018}],
                "note": "not the SET MMIO window",
            },
        }}, None)
        sbc = detail["devicetree"]["set_base_candidate"]
        self.assertEqual(sbc["status"], "not_available_from_device_tree")
        self.assertEqual(sbc["ane_pwrstate_cells"][0]["offset"], 0x4018)


if __name__ == "__main__":
    unittest.main()


class IrqEncodings(unittest.TestCase):
    """IOInterruptSpecifiers arrive in four encodings across published
    rows; all decode to the same AIC number (t6021 ane: 884), and the
    lossy form must refuse rather than guess."""

    def test_canonical_hex(self):
        self.assertEqual(gen._irq_number("0x74030000"), 884)

    def test_python_bytes_repr(self):
        # The v0.6.4 bug read this as 0x7427625b.
        self.assertEqual(gen._irq_number("b't\\x03\\x00\\x00'"), 884)

    def test_latin1_text(self):
        self.assertEqual(gen._irq_number("t\x03\x00\x00"), 884)

    def test_bytes_list(self):
        self.assertEqual(gen._irq_number([b"t\x03\x00\x00"]), 884)

    def test_lossy_is_none_not_a_guess(self):
        self.assertIsNone(gen._irq_number("\ufffd\x03\x00\x00"))
        self.assertIsNone(gen._irq_number("b'\ufffd\x03\x00\x00'"))


class AneNodeChoice(unittest.TestCase):
    """The generator must pick ane0/die 0, not the first list entry
    (T6022 row 90c6b9bf3e03 carries ane0+ane1)."""

    def test_ane0_wins_over_ane1(self):
        nodes = [{"name": "ane1"}, {"name": "ane0"}]
        self.assertEqual(gen._ane_node_of({"ane_nodes": nodes})["name"],
                         "ane0")

    def test_bare_ane_wins_over_higher_die(self):
        nodes = [{"name": "ane3"}, {"name": "ane"}]
        self.assertEqual(gen._ane_node_of({"ane_nodes": nodes})["name"],
                         "ane")

    def test_lowest_die_number_wins(self):
        nodes = [{"name": "ane10"}, {"name": "ane2"}]
        self.assertEqual(gen._ane_node_of({"ane_nodes": nodes})["name"],
                         "ane2")
