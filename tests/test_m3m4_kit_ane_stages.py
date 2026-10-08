"""Refusal-path and mapping tests for the m3m4 kit's opt-in ANE stage.

Every run is MKIT_DRY_RUN=1 against a fake MKIT_ANE_ROOT: nothing is
insmodded and no hardware is touched. Module names, stage names/numbers and
parameter names mirror omarchy-ane ane/h15 and ane/h16 (see
scripts/m3m4_kit.sh). Mutation controls for every asserted refusal live in
the private lab notebook artifacts (tests must fail when a refusal line is
deleted from the kit); MKIT_KIT_PATH points the suite at a mutated copy.
"""

import os
import pathlib
import platform
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
KIT = pathlib.Path(os.environ.get("MKIT_KIT_PATH", str(ROOT / "scripts" / "m3m4_kit.sh")))


class M3m4KitAneStageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = pathlib.Path(self.tmp.name)
        self.rechip("t8122")
        stubs = base / "stubs"
        stubs.mkdir()
        self.modinfo = stubs / "modinfo"
        self.set_vermagic(platform.release())
        self.fwfetch = stubs / "fwfetch"
        self.fwfetch.write_text("#!/bin/sh\nexit 0\n")
        self.modinfo.chmod(0o755)
        self.fwfetch.chmod(0o755)
        self.ko_h15 = base / "ane_h15.ko"
        self.ko_h15.write_bytes(b"\x7fELFkit-test")
        self.ko_h16 = base / "ane_h16.ko"
        self.ko_h16.write_bytes(b"\x7fELFkit-test")
        self.base_env = {
            "OUT": str(base / "out"),
            "MKIT_DRY_RUN": "1",
            "MKIT_ANE_ROOT": str(base / "fakeroot"),
            "MKIT_MODINFO": str(self.modinfo),
            "MKIT_FW_FETCH": str(self.fwfetch),
            "MKIT_ANE_KO": str(self.ko_h15),
        }

    def rechip(self, soc):
        base = pathlib.Path(self.tmp.name)
        root = base / "fakeroot"
        (root / "proc" / "device-tree").mkdir(parents=True, exist_ok=True)
        (root / "proc" / "device-tree" / "compatible").write_bytes(
            f"apple,{soc}\0apple,{soc}\0".encode())
        optin = root / "etc" / "omarchy-mac-boot"
        optin.mkdir(parents=True, exist_ok=True)
        (optin / "dtb-overlays.opt-in").write_text(
            "ane-h15-experimental\nane-h16-experimental\n")
        (root / "sys" / "module").mkdir(parents=True, exist_ok=True)

    def set_vermagic(self, release):
        self.modinfo.write_text(
            f"#!/bin/sh\necho '{release} SMP preempt mod_unload aarch64'\n")

    def run_kit(self, stage=None, **over):
        env = dict(os.environ)
        env.update(self.base_env)
        env.update({k: str(v) for k, v in over.items()})
        if stage is None:
            env.pop("MKIT_ANE_STAGE", None)
        else:
            env["MKIT_ANE_STAGE"] = stage
        return subprocess.run(
            ["bash", str(KIT)], env=env, capture_output=True, text=True,
            timeout=120)

    def optin_path(self):
        return (pathlib.Path(self.base_env["MKIT_ANE_ROOT"]) /
                "etc/omarchy-mac-boot/dtb-overlays.opt-in")

    def test_h15_stage_commands_numeric_and_named(self):
        for stage_token, param in (("1", "stage=1"), ("status", "stage=status"),
                                   ("0", "stage=0")):
            with self.subTest(stage=stage_token):
                result = self.run_kit(f"t8122:{stage_token}")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("ane_h15.ko optin=t8122", result.stdout)
                self.assertIn(f"optin=t8122 {param}", result.stdout)
                self.assertNotIn("REFUSED", result.stdout)
        self.rechip("t6034")
        result = self.run_kit("t6034:status")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("insmod", result.stdout)
        self.assertIn("optin=t6034 stage=status", result.stdout)

    def test_h16_stage_commands_and_hello_param(self):
        self.rechip("t8132")
        result = self.run_kit(
            "t8132:boot", MKIT_ANE_CONFIRM_BOOT=1, MKIT_ANE_HELLO_WAIT_MS=1000,
            MKIT_ANE_KO=self.ko_h16)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ane_h16.ko optin=t8132 stage=boot", result.stdout)
        self.assertIn("hello_wait_ms=1000", result.stdout)
        self.assertNotIn("confirm_boot", result.stdout)
        self.rechip("t6041")
        result = self.run_kit("t6041:3", MKIT_ANE_CONFIRM_BOOT=1,
                              MKIT_ANE_KO=self.ko_h16)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ane_h16.ko optin=t6041 stage=3", result.stdout)

    def test_h15_boot_refused_without_confirm(self):
        result = self.run_kit("t8122:3")
        self.assertEqual(result.returncode, 1)
        self.assertIn("boot-class-confirm: REFUSED", result.stdout)
        self.assertIn("MKIT_ANE_CONFIRM_BOOT", result.stdout)

    def test_h16_boot_refused_without_confirm(self):
        self.rechip("t8132")
        result = self.run_kit("t8132:boot", MKIT_ANE_KO=self.ko_h16)
        self.assertEqual(result.returncode, 1)
        self.assertIn("boot-class-confirm: REFUSED", result.stdout)

    def test_h15_boot_refused_without_fw_path(self):
        result = self.run_kit("t8122:3", MKIT_ANE_CONFIRM_BOOT=1)
        self.assertEqual(result.returncode, 1)
        self.assertIn("h15-fw-path: REFUSED", result.stdout)
        self.assertIn("MKIT_ANE_FW_PATH", result.stdout)

    def test_h15_wrapper_stage_refused(self):
        for stage_token in ("wrapper", "2"):
            with self.subTest(stage=stage_token):
                result = self.run_kit(f"t8122:{stage_token}")
                self.assertEqual(result.returncode, 1)
                self.assertIn("stage-valid: REFUSED", result.stdout)
                self.assertIn("the module refuses it", result.stdout)

    def test_h16_stage2_refused(self):
        self.rechip("t8132")
        result = self.run_kit("t8132:2", MKIT_ANE_KO=self.ko_h16)
        self.assertEqual(result.returncode, 1)
        self.assertIn("stage-valid: REFUSED", result.stdout)
        self.assertIn("not an h16 module stage", result.stdout)

    def test_unknown_chip_refused(self):
        result = self.run_kit("t6001:1")
        self.assertEqual(result.returncode, 1)
        self.assertIn("chip-map: REFUSED", result.stdout)
        self.assertIn("unknown chip 't6001'", result.stdout)

    def test_chip_mismatch_refused(self):
        result = self.run_kit("t6030:1")
        self.assertEqual(result.returncode, 1)
        self.assertIn("soc-match: REFUSED", result.stdout)
        self.assertIn("reports 't8122'", result.stdout)

    def test_missing_optin_key_refused(self):
        self.optin_path().write_text("some-other-overlay\n")
        result = self.run_kit("t8122:1")
        self.assertEqual(result.returncode, 1)
        self.assertIn("opt-in-key: REFUSED", result.stdout)
        self.assertIn("ane-h15-experimental", result.stdout)

    def test_vermagic_mismatch_refused(self):
        self.set_vermagic("1.2.3-other")
        result = self.run_kit("t8122:1")
        self.assertEqual(result.returncode, 1)
        self.assertIn("vermagic: REFUSED", result.stdout)

    def test_missing_ko_refused(self):
        result = self.run_kit("t8122:1", MKIT_ANE_KO="")
        self.assertEqual(result.returncode, 1)
        self.assertIn("ko-file: REFUSED", result.stdout)
        self.assertIn("MKIT_ANE_KO", result.stdout)

    def test_wrong_ko_basename_refused(self):
        result = self.run_kit("t8122:1", MKIT_ANE_KO=self.ko_h16)
        self.assertEqual(result.returncode, 1)
        self.assertIn("ko-file: REFUSED", result.stdout)
        self.assertIn("ane_h15.ko", result.stdout)

    def test_loaded_ane_module_refused(self):
        loaded = (pathlib.Path(self.base_env["MKIT_ANE_ROOT"]) /
                  "sys/module/ane_t6021")
        loaded.mkdir()
        result = self.run_kit("t8122:1")
        self.assertEqual(result.returncode, 1)
        self.assertIn("no-ane-module-loaded: REFUSED", result.stdout)
        self.assertIn("ane_t6021", result.stdout)

    def test_h16_pinned_firmware_refused(self):
        self.rechip("t8132")
        self.fwfetch.write_text("#!/bin/sh\necho 'sha mismatch' >&2\nexit 1\n")
        result = self.run_kit("t8132:status", MKIT_ANE_KO=self.ko_h16)
        self.assertEqual(result.returncode, 1)
        self.assertIn("pinned-firmware: REFUSED", result.stdout)

    def test_malformed_stage_flag_refused(self):
        result = self.run_kit("t8122")
        self.assertEqual(result.returncode, 1)
        self.assertIn("<chip>:<stage>", result.stderr)

    def test_dry_run_prints_every_result_and_no_execution(self):
        result = self.run_kit("t8122:1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ANE insmod command: insmod", result.stdout)
        self.assertIn("MKIT_DRY_RUN=1: every check passed; nothing was executed",
                      result.stdout)
        self.assertNotIn("REFUSED", result.stdout)
        self.optin_path().write_text("some-other-overlay\n")
        self.set_vermagic("1.2.3-other")
        result = self.run_kit("t8122:1")
        self.assertEqual(result.returncode, 1)
        self.assertIn("opt-in-key: REFUSED", result.stdout)
        self.assertIn("vermagic: REFUSED", result.stdout)
        self.assertIn("MKIT_DRY_RUN=1: 2 refusal(s)", result.stderr)

    def test_no_flag_keeps_non_m3_output_unchanged(self):
        self.rechip("t6001")
        result = self.run_kit()
        self.assertEqual(result.returncode, 1)
        self.assertIn("This is not an M3/M4 Mac (detected: apple,t6001)",
                      result.stdout)
        self.assertNotIn("ANE stage", result.stdout)
        self.assertNotIn("ANE check", result.stdout)

    def test_m3_family_ids_reach_the_default_kit_banner(self):
        """No ANE flag: each M3 id gets its marketing name. M3 Max is t6031 and
        t6034 (binned); t6030 is the M3 Pro. curl and python3 are stubbed to fail
        so the run stops at the banner without network or collection."""
        stubs = pathlib.Path(self.tmp.name) / "failstubs"
        stubs.mkdir()
        for name in ("curl", "python3"):
            (stubs / name).write_text("#!/bin/sh\nexit 1\n")
            (stubs / name).chmod(0o755)
        path = f"{stubs}:{os.environ.get('PATH', '')}"
        for soc, name in (("t8122", "Apple M3"), ("t6030", "Apple M3 Pro"),
                          ("t6031", "Apple M3 Max"), ("t6034", "Apple M3 Max")):
            with self.subTest(soc=soc):
                self.rechip(soc)
                result = self.run_kit(MKIT_DRY_RUN=0, PATH=path)
                self.assertIn(f"M3/M4 kit: {name} (apple,{soc}) on Linux", result.stdout,
                              result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
