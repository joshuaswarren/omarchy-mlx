"""Host tests for the §73 ANE/Core ML installed-state probe.

Uses a fake /sys tree, matching scripts/test_collect.py AneDevicetreeProbe.
No Apple hardware, no device ioctls.
"""
from __future__ import annotations

import io
import os
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import installed_state as state  # noqa: E402


def _write(path: Path, data: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data)


def _fake_tree(ane: bool = False, module: bool = False, accel: bool = False) -> Path:
    root = Path(tempfile.mkdtemp(prefix="mlx-omarchy-state-"))
    dt = root / "sys" / "firmware" / "devicetree" / "base"
    dt.mkdir(parents=True)
    _write(dt / "compatible", b"apple,t8103\x00apple,arm-platform\x00")
    if ane:
        node = dt / "ane@26a000000"
        _write(node / "compatible", b"apple,t8103-ane\x00apple,ane\x00")
    if module:
        _write(root / "sys" / "module" / "ane" / "version", "f2a3e5e+lifecycle6\n")
        _write(root / "sys" / "module" / "ane" / "srcversion", "deadbeef\n")
    if accel:
        node = root / "dev" / "accel" / "accel0"
        node.parent.mkdir(parents=True, exist_ok=True)
        node.write_bytes(b"")
    return root


class FakeSysTreeTests(unittest.TestCase):
    def test_stock_tree_has_no_ane(self) -> None:
        root = _fake_tree()
        report = state.collect(sysroot=root, home=root / "home", frontend_root=root)
        self.assertFalse(report["ane"]["fdt_node"])
        self.assertIsNone(report["ane"]["fdt_compatible"])
        self.assertFalse(report["ane"]["accel0"])
        self.assertFalse(report["ane"]["module_present"])
        self.assertFalse(report["ane"]["available"])

    def test_ane_compatible_is_found(self) -> None:
        root = _fake_tree(ane=True)
        report = state.probe_ane(sysroot=root)
        self.assertTrue(report["fdt_node"])
        self.assertEqual(report["fdt_compatible"], ["apple,ane", "apple,t8103-ane"])
        self.assertFalse(report["available"])

    def test_regular_accel_file_is_not_a_character_device(self) -> None:
        root = _fake_tree(ane=True, module=True, accel=True)
        report = state.probe_ane(sysroot=root)
        self.assertTrue(report["accel0"])
        self.assertFalse(report["accel0_character_device"])
        self.assertTrue(report["module_present"])
        self.assertEqual(report["module_version"], "f2a3e5e+lifecycle6")
        self.assertFalse(report["available"])

    def test_null_character_device_plus_fdt_and_module_is_available(self) -> None:
        """/dev/null is a host char device used only as a S_ISCHR positive control."""
        self.assertTrue(stat.S_ISCHR(os.stat("/dev/null", follow_symlinks=False).st_mode))
        root = _fake_tree(ane=True, module=True)
        report = state.probe_ane(sysroot=root, accel="/dev/null")
        self.assertTrue(report["accel0_character_device"])
        self.assertTrue(report["available"])

    def test_require_ane_refuses_missing_module(self) -> None:
        root = _fake_tree(ane=True)
        report = state.collect(
            sysroot=root, accel="/dev/null", home=root / "home", frontend_root=root
        )
        with self.assertRaises(SystemExit) as raised:
            state.require_ane(report)
        self.assertIn("missing FDT node or ane module", str(raised.exception))

    def test_require_ane_accepts_composed_fake_tree(self) -> None:
        root = _fake_tree(ane=True, module=True)
        report = state.collect(
            sysroot=root, accel="/dev/null", home=root / "home", frontend_root=root
        )
        state.require_ane(report)

    def test_coreml_cache_and_frontend(self) -> None:
        root = _fake_tree()
        home = root / "home"
        frontend = root / "coreml"
        _write(frontend / "__init__.py", "# probe\n")
        cache = home / ".cache" / "mlx-omarchy" / "coreml"
        (cache / "abc123").mkdir(parents=True)
        (cache / ".hidden").mkdir()
        report = state.probe_coreml(home=home, frontend_root=frontend)
        self.assertTrue(report["frontend_present"])
        self.assertFalse(report["command_present"])
        self.assertTrue(report["cache_present"])
        self.assertEqual(report["cache_entries"], 1)
        self.assertEqual(report["cache_root"], str(cache))

    def test_repo_frontend_is_visible_without_override(self) -> None:
        report = state.probe_coreml(home=Path("/tmp/no-such-mlx-home"))
        self.assertTrue(report["frontend_present"])
        self.assertFalse(report["cache_present"])
        self.assertEqual(report["cache_entries"], 0)


class CliTests(unittest.TestCase):
    def test_json_round_trip(self) -> None:
        root = _fake_tree(ane=True, module=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = state.main(
                [
                    "--json",
                    "--sysroot",
                    str(root),
                    "--accel",
                    "/dev/null",
                    "--home",
                    str(root / "home"),
                    "--frontend-root",
                    str(root),
                ]
            )
        self.assertEqual(code, 0)
        self.assertIn('"available": true', buf.getvalue())


if __name__ == "__main__":
    unittest.main()
