#!/usr/bin/env python3
"""Offline macOS collector contracts, runnable on Linux without MLX."""

import base64
import contextlib
import hashlib
from importlib.metadata import FileHash, PackagePath
import io
import json
import plistlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect_common as cc
import collect_deep as cd
import collect_macos as cm
import collect_quick as cq
import mlx_provenance as prov
import test_collect as legacy_tests


class MacHostTests(unittest.TestCase):
    def test_hardware_does_not_require_mlx_and_keeps_only_selected_fields(self):
        facts = {"machine": "arm64", "chip": "Apple M3", "cores": "8",
                 "memsize_bytes": 24 * 1024**3, "os": "macOS 15.0 (24A335)",
                 "gpu": {"chipset": "Apple M3", "gpu_cores": "10", "metal": "Metal 3"},
                 "serial_number": "DO-NOT-KEEP"}
        with patch.object(cm.bench_matrix, "host_facts", return_value=facts), \
                patch.object(cm, "run_tool", side_effect=[
                    {"stdout": "Mac15,12", "exit_code": 0},
                    {"stdout": "8", "exit_code": 0}]):
            host = cm.probe_host(cc.Redactor())
        self.assertEqual(host["model"], "Mac15,12")
        self.assertEqual(host["memory_total_mib"], 24576)
        self.assertEqual(host["cpu_online"], 8)
        self.assertEqual(host["cpu"]["present"], 8)
        self.assertEqual(host["gpu"]["gpu_cores"], "10")
        self.assertNotIn("DO-NOT-KEEP", json.dumps(host))
        self.assertNotIn("devicetree", host)

    def test_missing_system_tools_leave_unknown_values(self):
        with patch.object(cm.bench_matrix, "host_facts", return_value={"machine": "arm64"}), \
                patch.object(cm, "run_tool", return_value={"stdout": "", "exit_code": None}):
            host = cm.probe_host(cc.Redactor())
        self.assertTrue(host["available"])
        for key in ("model", "chip", "memory_total_mib", "cpu_online", "gpu"):
            self.assertIsNone(host[key])

    def test_linux_diagnostics_are_not_run_on_mac(self):
        with patch("platform.system", return_value="Darwin"), \
                patch.object(cq, "run_tool", side_effect=AssertionError("Linux command")):
            for probe in (cq.probe_mesa, cq.probe_mesa_package, cq.probe_ane):
                self.assertEqual(probe(cc.Redactor()), cm.not_applicable())
            self.assertEqual(cd.section_profile(cc.Redactor(), "/missing", "/missing"),
                             cm.not_applicable())

    def test_context_excludes_process_names_ids_and_raw_power_output(self):
        with patch.object(cm.bench_matrix, "power_state", return_value={
                "raw": "private battery identity", "source": "Battery", "percent": 76,
                "charging": False}), patch.object(cm.bench_matrix, "clean_check", return_value={
                    "status": "contended", "scanned": 400,
                    "matched": [{"pid": "123", "comm": "private-service"}]}):
            context = cm.measurement_context()
        self.assertEqual(context["model_processes"]["matched_count"], 1)
        self.assertEqual(context["power"]["percent"], 76)
        for secret in ("private", "123", "comm", "raw"):
            self.assertNotIn(secret, json.dumps(context))

    def test_native_payload_uses_existing_schema_and_null_linux_fields(self):
        quick = {"host": {"system": "Darwin", "model": "Mac14,2", "chip": "Apple M2",
                          "kernel_release": "25.6.0", "os": "macOS 26.6.2 (25G83)",
                          "cpu_online": 8, "cpu": {"present": 8}},
                 "mlx": {"mlx_version": "0.32.1", "metal_available": True,
                         "default_device": "Device(gpu, 0)"}}
        payload = cc.build_payload("deep", quick, {}, benchmark=[{"n": 256, "median_ms": 1.2}])
        schema = json.loads((Path(cd.REPO) / "services/community-data/schema/payload-v1.schema.json").read_text())
        self.assertEqual(set(payload), set(schema["properties"]))
        self.assertEqual(payload["model"], "Mac14,2")
        self.assertEqual(payload["chip"], "Apple M2")
        self.assertIn("macOS", payload["kernel"])
        self.assertIn("Metal", payload["mlx_device"])
        self.assertEqual(payload["benchmark"][0]["median_ms"], 1.2)
        for key in ("mesa_driver", "mesa_device", "ane_dt_node", "ane_dt_compatible",
                    "boot_chain", "cmdline", "hotplug_control"):
            self.assertIsNone(payload[key], key)
        text = cd.build_submission({}, {"quick.json": cc.json_bytes(quick)}, "mac.tar.gz")
        self.assertIn("Native MLX: 0.32.1", text)
        self.assertIn("does not prove Linux support", text)
        self.assertNotIn("mlx-omarchy: not installed", text)

    def test_nested_observations_are_redacted_before_writing(self):
        red = cc.Redactor(hostname="private-host", username="private-user", home="/Users/private-user")
        with tempfile.TemporaryDirectory() as ws, patch.object(cd, "Redactor", return_value=red), \
                patch.object(cd, "section_environment", return_value={
                    "available": True, "nested": [{"path": "/Users/private-user/data", "host": "private-host"}]}):
            cd.section_child("environment", ws, cd.REPO)
            output = (Path(ws) / "environment.json").read_text()
        self.assertNotIn("private-user", output)
        self.assertNotIn("private-host", output)
        self.assertIn("[home]/data", output)

    def test_host_alias_inside_macos_capture_paths_is_redacted(self):
        # macOS captures quote paths too (kextstat/ioreg output); the
        # derived short name must not survive there either.
        red = cc.Redactor(hostname="privhost-200", username="private-user",
                          home="/Users/private-user")
        line = ("kextstat: /Library/Extensions/privhost-ane.kext "
                "loaded by privhost-200")
        out = red.apply(line)
        self.assertIn("[host]-ane.kext", out)
        self.assertNotIn("privhost-200", out)
        self.assertEqual(red.counts.get("hostname_alias"), 1)


class ReviewRegressionTests(unittest.TestCase):
    def test_structured_redaction_preserves_schema_keys(self):
        red = cc.Redactor(username="cpu", hostname="model", home="/Users/cpu")
        facts = {"cpu": {"present": 8}, "model": "Mac14,2",
                 "notes": ["cpu on model", "/Users/cpu/data"]}
        result = red.apply_value(facts)
        self.assertEqual(result["cpu"]["present"], 8)
        self.assertEqual(result["model"], "Mac14,2")
        self.assertEqual(result["notes"], ["[user] on [host]", "[home]/data"])

    def test_linux_collection_preserves_vulkan_versions(self):
        probes = {name: lambda red: {"available": False} for name in cq.DEFAULT_PROBES}
        probes["mesa"] = cq.probe_mesa
        with patch("platform.system", return_value="Linux"), \
                patch.object(cq, "run_tool", return_value={
                    "available": True, "exit_code": 0,
                    "stdout": legacy_tests.PrimaryGpuSelection.SUMMARY}):
            result = cq.collect(probes)
        self.assertEqual(result["mesa"]["gpu"]["conformanceVersion"], "1.4.0.0")
        self.assertEqual(result["mesa"]["devices"][0]["conformanceVersion"], "1.4.0.0")

    def test_version_context_does_not_exempt_other_addresses(self):
        result = cc.Redactor().apply_value({
            "conformanceVersion": "1.4.0.0 contact 198.51.100.7",
            "address": "198.51.100.7",
        })
        self.assertEqual(result["conformanceVersion"], "1.4.0.0 contact [redacted-ip4]")
        self.assertEqual(result["address"], "[redacted-ip4]")

    def test_power_status_matches_the_complete_state(self):
        for state, charging in (("charging", True), ("discharging", False),
                                ("not charging", False), ("charged", False),
                                ("unknown", None)):
            with self.subTest(state=state), patch("platform.system", return_value="Darwin"), \
                    patch.object(cm.bench_matrix.subprocess, "run", return_value=types.SimpleNamespace(
                        stdout=f"Now drawing from 'AC Power'\n -InternalBattery-0 80%; {state}; 0:00 remaining")):
                result = cm.measurement_context()
            self.assertIs(result["power"]["charging"], charging)

    def test_missing_battery_status_is_unknown(self):
        with patch("platform.system", return_value="Darwin"), \
                patch.object(cm.bench_matrix.subprocess, "run", return_value=types.SimpleNamespace(stdout="")):
            result = cm.measurement_context()
        self.assertIsNone(result["power"]["charging"])

    def test_skipped_quick_section_keeps_native_labels(self):
        files = {"quick.json": cc.json_bytes({"available": False}),
                 "correctness.json": cc.json_bytes({"available": True, "mlx_version": "0.32.1",
                                                    "device": "Device(gpu, 0)"}),
                 "benchmark.json": cc.json_bytes({"python": {"matmul": [{"n": 256, "median_ms": 1.0}]}})}
        with patch("platform.system", return_value="Darwin"):
            manifest, archive, payload = cd.finalize(files, ["quick"], {}, "mac.tar.gz", cd.REPO)
        self.assertEqual(manifest["system"], "Darwin")
        self.assertIn("macOS", payload["kernel"])
        self.assertEqual(payload["benchmark"][0]["median_ms"], 1.0)
        self.assertEqual(payload["mlx_version"], "0.32.1")
        self.assertEqual(payload["mlx_device"], "Metal GPU (native macOS MLX, Device(gpu, 0))")
        cover = files["submission.md"].decode()
        self.assertIn("Native macOS reference", cover)
        self.assertIn("Native MLX: 0.32.1", cover)
        self.assertIn("Metal available: True", cover)
        self.assertNotIn("Vulkan:", cover)
        self.assertTrue(archive)

    def test_benchmark_recovers_native_metadata_without_quick_or_correctness(self):
        for quick in (None, {"available": False}, {"mlx": {"mlx_version": None,
                                                         "default_device": None,
                                                         "metal_available": None}}):
            with self.subTest(quick=quick), patch("platform.system", return_value="Darwin"):
                files = {"benchmark.json": cc.json_bytes({"python": {
                    "available": True, "device": "Device(gpu, 0)",
                    "provenance": {"mx_version": "0.32.1", "verified": "unverified"},
                    "matmul": [{"n": 256, "median_ms": 1.0}],
                }})}
                if quick is not None:
                    files["quick.json"] = cc.json_bytes(quick)
                original = dict(files)
                _, _, payload = cd.finalize(files, ["quick", "correctness"], {}, "mac.tar.gz", cd.REPO)
                self.assertEqual(payload["mlx_version"], "0.32.1")
                self.assertIn("Metal GPU", payload["mlx_device"])
                self.assertIn("Native MLX: 0.32.1, Metal available: True", files["submission.md"].decode())
                for name, data in original.items():
                    self.assertEqual(files[name], data)

    def test_failed_native_probes_do_not_claim_metal_available(self):
        files = {name: cc.json_bytes(data) for name, data in {
            "correctness.json": {"available": False, "mlx_version": "0.32.1", "device": "Device(cpu, 0)"},
            "benchmark.json": {"python": {"available": False, "device": "Device(cpu, 0)",
                                          "provenance": {"mx_version": "0.32.1", "verified": "mismatch"}}},
        }.items()}
        with patch("platform.system", return_value="Darwin"):
            _, _, payload = cd.finalize(files, ["quick", "correctness", "benchmark"], {}, "mac.tar.gz", cd.REPO)
        self.assertIsNone(payload["mlx_version"])
        self.assertIsNone(payload["mlx_device"])
        self.assertIn("Metal available: unknown", files["submission.md"].decode())

    def test_quick_metadata_takes_priority_over_probe_metadata(self):
        quick = {"host": {"system": "Darwin"}, "mlx": {
            "mlx_version": "0.32.2", "metal_available": False, "default_device": "Device(cpu, 0)"}}
        files = {"quick.json": cc.json_bytes(quick), "correctness.json": cc.json_bytes({
            "available": True, "mlx_version": "0.32.1", "device": "Device(gpu, 0)"})}
        with patch("platform.system", return_value="Darwin"):
            _, _, payload = cd.finalize(files, [], {}, "mac.tar.gz", cd.REPO)
        self.assertEqual(payload["mlx_version"], "0.32.2")
        self.assertEqual(payload["mlx_device"], "Device(cpu, 0)")
        self.assertIn("Native MLX: 0.32.2, Metal available: False", files["submission.md"].decode())

    def test_macos_thermal_unavailability_reaches_manifest_and_cover(self):
        with tempfile.TemporaryDirectory() as ws, patch("platform.system", return_value="Darwin"):
            files, unavailable, redaction = cd.assemble_files(ws, cd.REPO, [])
            manifest, _, _ = cd.finalize(files, unavailable, redaction, "mac.tar.gz", cd.REPO)
        self.assertFalse(json.loads(files["thermal.json"])["available"])
        self.assertEqual(manifest["sections_unavailable"].count("thermal"), 1)
        cover = files["submission.md"].decode()
        self.assertIn("Not available on this machine: quick, environment, correctness, benchmark, profile, ane, thermal", cover)


class NativeProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.site = Path(self.tmp.name).resolve()
        self.extension = self.site / "core.cpython-314-darwin.so"
        self.library = self.site / "libmlx.dylib"
        self.extension.write_bytes(b"extension")
        self.library.write_bytes(b"library")
        self.mx = types.ModuleType("mlx.core")
        self.mx.__version__ = "0.32.1"
        self.mx.metal = Mock()
        self.mx.metal.is_available.return_value = True
        self.mx.gpu = "gpu"
        self.mx.set_default_device = Mock()
        mlx = types.ModuleType("mlx")
        mlx.core = self.mx
        patcher = patch.dict(sys.modules, {"mlx": mlx, "mlx.core": self.mx})
        patcher.start()
        self.addCleanup(patcher.stop)
        binaries = patch.object(prov, "_loaded_macos_binaries", return_value=[self.extension, self.library])
        binaries.start()
        self.addCleanup(binaries.stop)

    def distribution(self, files, version="0.32.1"):
        entries = []
        for path in files:
            # PackagePath carries the same RECORD metadata as importlib.metadata.
            entry = PackagePath(path.name)
            digest = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).decode().rstrip("=")
            entry.hash = FileHash("sha256=" + digest)
            entries.append(entry)
        return types.SimpleNamespace(version=version, files=entries,
                                     locate_file=lambda name: self.site / name,
                                     read_text=lambda name: "pip" if name == "INSTALLER" else "record")

    def inspect(self, distributions):
        def lookup(name):
            if name not in distributions:
                raise prov.importlib.metadata.PackageNotFoundError(name)
            return distributions[name]
        with patch.object(prov.importlib.metadata, "distribution", side_effect=lookup):
            return prov.native_provenance()

    def test_split_mlx_and_metal_wheels_verify_loaded_binaries(self):
        result = self.inspect({"mlx": self.distribution([self.extension]),
                               "mlx-metal": self.distribution([self.library])})
        self.assertEqual(result["verified"], "match")
        self.assertEqual(len(result["files"]), 2)
        self.assertNotIn(str(self.site), json.dumps(result))

    def test_homebrew_without_record_keeps_fingerprints_unverified(self):
        dist = self.distribution([])
        dist.read_text = lambda name: "brew" if name == "INSTALLER" else None
        result = self.inspect({"mlx": dist})
        self.assertEqual(result["verified"], "unverified")
        self.assertEqual(result["packages"]["mlx"]["installer"], "brew")
        self.assertFalse(result["packages"]["mlx"]["record_available"])
        self.assertEqual(len(result["files"]), 2)

    def test_hash_mismatch_cannot_be_hidden_by_later_unverified_file(self):
        dist = self.distribution([self.extension])
        self.extension.write_bytes(b"changed extension")
        result = self.inspect({"mlx": dist})
        self.assertEqual(result["verified"], "mismatch")
        self.assertIn(self.extension.name, result["mismatch"])

    def test_compiled_version_mismatch_refuses(self):
        result = self.inspect({"mlx": self.distribution([], version="0.30.0")})
        self.assertEqual(result["verified"], "mismatch")

    def test_stale_backend_package_version_refuses(self):
        # Each binary matches its own RECORD, yet mlx-metal lags mlx.
        result = self.inspect({"mlx": self.distribution([self.extension]),
                               "mlx-metal": self.distribution([self.library], version="0.31.0")})
        self.assertEqual(result["verified"], "mismatch")
        self.assertIn("mlx-metal==0.31.0", result["mismatch"])
        self.assertTrue(all(entry["match"] for entry in result["files"]))

    def test_source_install_has_hashes_but_no_verification_claim(self):
        result = self.inspect({})
        self.assertEqual(result["verified"], "unverified")
        self.assertEqual(len(result["files"]), 2)

    def test_probe_requires_metal_and_selects_gpu(self):
        with patch("platform.system", return_value="Darwin"), \
                patch.object(prov, "native_provenance", return_value={"verified": "unverified"}), \
                patch.object(prov, "provenance_line", return_value="provenance: test"), \
                patch("sys.stderr", new=io.StringIO()):
            prov.prepare_probe(self.mx)
            self.mx.set_default_device.assert_called_once_with("gpu")
            self.mx.metal.is_available.return_value = False
            with self.assertRaisesRegex(RuntimeError, "Metal GPU unavailable"):
                prov.prepare_probe(self.mx)

    def test_both_probes_refuse_mismatch_before_tensor_work(self):
        self.mx.default_device = lambda: "Device(gpu, 0)"
        for source, result_key in ((cd.CORRECTNESS_PROBE, "ops"), (cd.BENCH_PROBE, "matmul")):
            with self.subTest(probe=result_key), \
                    patch.object(prov, "prepare_probe", return_value={
                        "verified": "mismatch", "mismatch": "changed library"}), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(SystemExit):
                    exec(source.replace("__SCRIPTS_DIR__", repr(cd.SCRIPTS_DIR)), {})
                result = json.loads(output.getvalue())
            self.assertFalse(result["available"])
            self.assertEqual(result[result_key], [])
            self.assertIn("changed library", result["error"])

    def test_linux_provenance_gate_is_unchanged(self):
        with patch("platform.system", return_value="Linux"), \
                patch.object(prov, "installed_provenance", return_value={"verified": "mismatch"}) as installed, \
                patch.object(prov, "native_provenance") as native, \
                patch.object(prov, "provenance_line", return_value="provenance: test"), \
                patch("sys.stderr", new=io.StringIO()):
            self.assertEqual(prov.prepare_probe(self.mx)["verified"], "mismatch")
        installed.assert_called_once_with()
        native.assert_not_called()
        self.mx.set_default_device.assert_not_called()


class AnePortProbeTests(unittest.TestCase):
    """The macOS ANE port capture: identity, DT-shaped nodes, CoreML,
    powermetrics, and mandatory identity redaction."""

    @staticmethod
    def _probe_output():
        return json.dumps({
            "available": True,
            "instances": [{"name": "ane,t8020", "matched": "ane,t8020",
                           "firmware_loaded": True, "cores": 16,
                           "version": 96, "hw_board_type": 96,
                           "arch": "h13g"}],
            "ane_nodes": [{"name": "ane0",
                           "compatible": ["ane,t8020"],
                           "reg": "0200" * 8,
                           "IOInterruptControllers": "aic",
                           "IOInterruptSpecifiers": "02000000",
                           "IOClass": None, "phandle": 4097}],
            "dart_nodes": [{"name": "dart-ane0",
                            "compatible": ["dart,t6000"],
                            "reg": "0300" * 8,
                            "IOInterruptControllers": "aic",
                            "IOInterruptSpecifiers": "03000000",
                            "IOClass": "AppleT6000DART",
                            "phandle": 4113}],
            "coreml": {"available": False, "compute_units": None,
                       "error": "ModuleNotFoundError"},
            "powermetrics": {"available": False, "power_mw": None,
                             "error": "powermetrics must be invoked as "
                                      "the superuser"},
            "truncated": [],
        })

    def probe(self, red):
        with patch.object(cm, "run_python_probe", return_value={
                "available": True, "exit_code": 0, "error": None,
                "stderr": "", "stdout": self._probe_output()}) as run:
            result = cm.probe_ane_port(red)
        return result, run

    def test_capture_is_structured_and_bounded(self):
        result, _ = self.probe(cc.Redactor())
        self.assertTrue(result["available"])
        macos = result["macos"]
        self.assertEqual(macos["instances"][0]["cores"], 16)
        self.assertEqual(macos["dart_nodes"][0]["IOClass"],
                         "AppleT6000DART")
        self.assertFalse(macos["powermetrics"]["available"])
        self.assertEqual(macos["coreml"]["error"], "ModuleNotFoundError")

    def test_probe_failure_is_recorded_not_raised(self):
        with patch.object(cm, "run_python_probe", return_value={
                "available": False, "exit_code": None, "error": "not-found",
                "stderr": "", "stdout": ""}):
            result = cm.probe_ane_port(cc.Redactor())
        self.assertFalse(result["available"])
        self.assertEqual(result["error"], "not-found")

    def test_synthetic_serial_and_uuid_do_not_survive(self):
        out = self._probe_output().replace("h13g", "C02XY9876543")
        with patch.object(cm, "run_python_probe", return_value={
                "available": True, "exit_code": 0, "error": None,
                "stderr": "", "stdout": out}):
            blob = json.dumps(cm.probe_ane_port(cc.Redactor()))
        self.assertNotIn("C02XY9876543", blob)

    def test_quick_dispatch_uses_macos_probe_on_darwin(self):
        with patch("platform.system", return_value="Darwin"), \
                patch.object(cm, "run_python_probe", return_value={
                    "available": False, "exit_code": None,
                    "error": "not-found", "stderr": "", "stdout": ""}):
            result = cq.probe_ane_port(cc.Redactor())
        self.assertFalse(result["available"])


class AneProbeCodeTests(unittest.TestCase):
    """Executes the real ANE_PROBE_CODE string against a fixture rebuilt
    from the t6021-test-host T6021 capture (ane-linux-experiments
    receipts/2026-09-17-t6021-test-host-t6021-macos-capture/), whose raw ioreg
    bytes are the oracle for the byte-order fix and every new field.
    """

    @staticmethod
    def _fake_run_factory(service, h11, platform):
        import subprocess as _sp

        def fake_run(argv, capture_output=None, timeout=None):
            class P:
                returncode = 0
                stderr = b""
                stdout = b""
            p = P()
            a = list(argv)
            if a[:2] == ["ioreg", "-a"] and a[2] == "-rc":
                if a[3] in ("AppleH13ANEInterface",
                            "AppleH16ANEInterface"):
                    # Real zero-match shape: ioreg -a exits 0 and prints
                    # nothing at all (not an empty plist).
                    p.stdout = b""
                    return p
                data = {"H11ANEIn": h11,
                        "IOPlatformExpertDevice": platform}.get(a[3])
                assert data is not None, a
                p.stdout = plistlib.dumps(data)
            elif a[:3] == ["ioreg", "-a", "-p"]:
                p.stdout = plistlib.dumps(service)
            elif a[:2] == ["plutil", "-convert"]:
                p.stdout = json.dumps({"CFBundleVersion": "9.512.0",
                    "CFBundleShortVersionString": "9.509.0"}).encode()
            elif a[0] == "test":
                ok = a[1] == "-f" and a[2] in (
                    "/usr/libexec/aned", "/usr/libexec/aneuserd")
                p.returncode = 0 if ok else 1
            elif a[0] == "powermetrics":
                p.returncode = 1
                p.stderr = b"powermetrics must be invoked as root"
            else:
                raise AssertionError(a)
            return p
        return fake_run

    @classmethod
    def _run_probe(cls):
        service = {
            "IORegistryEntryChildren": [{
                "name": b"soc",
                "IORegistryEntryChildren": [
                    {  # ane0 - oracle: node-ane0.txt
                        "name": b"ane0",
                        "AAPL,phandle": bytes.fromhex("69010000"),
                        "reg": bytes.fromhex(
                "000000840000000000000002000000000000088e00000000344000000000000000c0088e000000000040000000000000",
                        ),
                        "compatible": b"ane,t8020\x00",
                        "IOInterruptControllers": [
                            "IOInterruptController000000A4"],
                        "IOInterruptSpecifiers": [b"t\x03\x00\x00"],
                        "IOClass": "AppleARMIODevice",
                    },
                    {  # dart-ane0 - oracle: node-dart-ane0.txt
                        "name": b"dart-ane0",
                        "AAPL,phandle": bytes.fromhex("6a010000"),
                        "reg": bytes.fromhex(
                "000080850000000000400000000000000000818500000000004000000000000000008285000000000040000000000000"
                "00408085000000000040000000000000",
                        ),
                        "dart-id": bytes.fromhex("25000000"),
                        "compatible": b"dart,t8110\x00",
                    },
                    {  # mapper-ane0: 4-byte reg, the u32 straggler
                        "name": b"mapper-ane0",
                        "AAPL,phandle": bytes.fromhex("6b010000"),
                        "reg": bytes.fromhex("00000000"),
                        "compatible": b"iommu-mapper\x00",
                    },
                    {  # pmgr - 1168 bytes / 73 ranges, real blob
                        "name": b"pmgr",
                        "IORegistryEntryLocation": "8E080000",
                        "reg": bytes.fromhex(
                "0000088e0000000000000800000000000000289e00000000000010000000000000002890000000000000100000000000"
                "0000688e0000000000c00200000000000000008e0000000000000800000000000000209e000000000000080000000000"
                "0000608e0000000000800700000000000040009e0000000000100000000000000000e210000000000050010000000000"
                "0000f0100000000000000500000000000000e4100000000000000100000000000000e510000000000010000000000000"
                "0000e7100000000000000900000000000000051000000000000001000000000000001510000000000000010000000000"
                "00002510000000000000010000000000000035100000000000000100000000000000e211000000000050010000000000"
                "0000f0110000000000000500000000000000e4110000000000000100000000000000e511000000000010000000000000"
                "0000e7110000000000000900000000000000051100000000000001000000000000001511000000000000010000000000"
                "00002511000000000000010000000000000035110000000000000100000000000000e212000000000050010000000000"
                "0000f0120000000000000500000000000000e4120000000000000100000000000000e512000000000010000000000000"
                "0000e7120000000000000900000000000000051200000000000001000000000000001512000000000000010000000000"
                "000025120000000000000100000000000000351200000000000001000000000000c0c28e000000000000040000000000"
                "0000408e0000000000000900000000000000509e00000000008009000000000000005c9e000000000040010000000000"
                "0000218e0000000000c000000000000000c0208e00000000004000000000000000808685000000000040000000000000"
                "0080ea040200000000400000000000000000208e0000000000800000000000000000e804020000000080020000000000"
                "00003c8e0000000000000200000000000040228e0000000000400000000000000090078e000000000040000000000000"
                "0000000a0200000000000001000000000000008c0100000000000001000000000000000c020000000000000100000000"
                "000000860000000000000001000000000000008400000000000000010000000000000084010000000000000100000000"
                "0000008a0100000000000001000000000000009c0300000000000001000000000000008a000000000000000100000000"
                "000000160100000000000001000000000000001001000000000000010000000000000000070000000000000100000000"
                "000000000b0000000000000100000000000000000f000000000000010000000000000000130000000000000100000000"
                "000000490100000000000001000000000000009a01000000000000010000000000000090020000000000000100000000"
                "000000060300000000000001000000000000008e0200000000000001000000000000000e030000000000000100000000"
                "0000029b0100000000400000000000000040029b01000000004000000000000000000097010000000000000100000000"
                "000000080200000000c0ff0100000000",
                        ),
                        "IORegistryEntryChildren": [],
                    },
                ],
            }],
        }
        h11 = [{"IOClass": "H11ANEIn",
                "IONameMatched": "ane,t8020",
                "CFBundleIdentifier":
                    "com.apple.driver.AppleH11ANEInterface",
                "FirmwareLoaded": True,
                "DeviceProperties": {
                    "ANEDevicePropertyNumANECores": 16,
                    "ANEDevicePropertyANEVersion": 128,
                    "ANEDevicePropertyANEMinorVersion": 17,
                    "ANEDevicePropertyANEHWBoardType": 160,
                    "ANEDevicePropertyTypeANEArchitectureTypeStr":
                        "h14g"}}]
        platform = [{"target-type": b"J414c",
                     "platform-name": b"t6021" + b"\x00" * 59,
                     "compatible": [b"Mac14,5\x00"],
                     "model": b"Mac14,5\x00"}]
        ns = {}
        with patch("subprocess.run", side_effect=cls._fake_run_factory(
                service, h11, platform)):
            exec(compile(cm.ANE_PROBE_CODE, "<probe>", "exec"), ns)
        return ns["out"]

    def test_phandles_are_host_endian_not_swapped(self):
        out = self._run_probe()
        ph = {n["name"]: n["phandle"]
              for g in ("ane_nodes", "dart_nodes") for n in out[g]}
        # v0.6.4 shipped 0x69010000 / 0x6a010000 / 0x6b010000 here.
        self.assertEqual(ph["ane0"], 0x169)
        self.assertEqual(ph["dart-ane0"], 0x16a)
        self.assertEqual(ph["mapper-ane0"], 0x16b)

    def test_ane_reg_decodes_all_three_ranges_with_kinds(self):
        out = self._run_probe()
        ane = out["ane_nodes"][0]
        self.assertEqual(ane["reg_ranges"],
                         ["0x84000000/0x2000000", "0x8e080000/0x4034",
                          "0x8e08c000/0x4000"])
        self.assertEqual(ane["range_kinds"],
                         ["ane_mmio", "pmgr_block", "pmgr_plus_c000"])
        self.assertEqual(len(ane["reg"]), 96)

    def test_set_base_candidate_is_confirmed_by_driver_window(self):
        out = self._run_probe()
        self.assertEqual(out["set_base_candidate"], {
            "pmgr_block": "0x8e080000", "offset": "0xc000",
            "base": "0x8e08c000", "driver_window_confirms": True})

    def test_dart_topology_dart_id_and_explicit_iommu_miss(self):
        out = self._run_probe()
        dart = [n for n in out["dart_nodes"]
                if n["name"] == "dart-ane0"][0]
        self.assertEqual(dart["reg_ranges"],
                         ["0x85800000/0x4000", "0x85810000/0x4000",
                          "0x85820000/0x4000", "0x85804000/0x4000"])
        self.assertEqual(dart["dart_id"], 0x25)
        self.assertIsNone(dart["iommu_cells"])
        self.assertIn("iommu_cells:unavailable_on_macos",
                      out["truncated"])

    def test_mapper_u32_straggler_is_marked_not_misparsed(self):
        out = self._run_probe()
        mapper = [n for n in out["dart_nodes"]
                  if n["name"] == "mapper-ane0"][0]
        self.assertIsNone(mapper["reg_ranges"])
        self.assertIn("reg_ranges:mapper-ane0", out["truncated"])

    def test_pmgr_reg_carries_all_73_ranges(self):
        out = self._run_probe()
        pmgr = out["pmgr_nodes"][0]
        self.assertEqual(pmgr["reg_ranges_total"], 73)
        self.assertEqual(len(pmgr["reg_ranges"]), 73)
        self.assertEqual(pmgr["reg_ranges"][0], "0x8e080000/0x80000")
        self.assertEqual(pmgr["reg_ranges"][-1], "0x208000000/0x1ffc000")
        self.assertNotIn("reg_bytes:pmgr", out["truncated"])

    def test_driver_identity_records_h14g_and_empty_classes(self):
        out = self._run_probe()
        d = out["driver"]
        self.assertEqual(d["matched_class"], "H11ANEIn")
        self.assertEqual(d["arch"], "h14g")
        self.assertEqual(d["cores"], 16)
        self.assertEqual(d["ane_version"], 128)
        self.assertEqual(d["ane_minor_version"], 17)
        self.assertEqual(d["kext_version"], "9.512.0")
        self.assertEqual(d["instance_count"], 1)
        self.assertEqual(d["classes_empty"],
                         ["AppleH13ANEInterface",
                          "AppleH16ANEInterface"])

    def test_platform_identity_decodes_soc_id(self):
        out = self._run_probe()
        self.assertEqual(out["platform"], {
            "target_type": "J414c", "soc_id": "t6021",
            "compatible": ["Mac14,5"], "model": "Mac14,5"})

    def test_compiler_provenance_records_versions_and_conditions(self):
        out = self._run_probe()
        c = out["compiler"]
        self.assertEqual(c["framework_version"], "9.509.0")
        self.assertEqual(c["kext_version"], "9.512.0")
        self.assertEqual(c["daemons"],
                         ["/usr/libexec/aned", "/usr/libexec/aneuserd"])
        self.assertFalse(c["compiler_service_present"])
        self.assertTrue(c["binaries_cache_resident"])

    def test_oversized_reg_and_range_cap_stay_bounded_and_honest(self):
        # Synthetic: a 2049-byte reg and a 129-range reg must be capped
        # with explicit markers, never silently dropped or unbounded.
        pmgr = {"name": b"pmgr",
                "reg": bytes(range(256)) * 16 + b"\x00",  # 4097 bytes
                "IORegistryEntryChildren": []}
        ane = {"name": b"ane0", "reg": b"".join(
                   (i & 0xffffffff).to_bytes(4, "little")
                   for i in range(257 * 4)),  # 257 ranges x 16 bytes
               "IORegistryEntryChildren": []}
        service = {"IORegistryEntryChildren": [pmgr, ane]}

        def fake_run(argv, capture_output=None, timeout=None):
            class P:
                returncode = 0
                stderr = b""
                stdout = b""
            p = P()
            a = list(argv)
            if a[:3] == ["ioreg", "-a", "-p"]:
                p.stdout = plistlib.dumps(service)
            elif a[:2] == ["ioreg", "-a"] and a[2] == "-rc":
                p.stdout = plistlib.dumps([])
            else:
                raise AssertionError(a)
            return p

        ns = {}
        with patch("subprocess.run", side_effect=fake_run):
            exec(compile(cm.ANE_PROBE_CODE, "<probe>", "exec"), ns)
        out = ns["out"]
        self.assertEqual(len(out["pmgr_nodes"][0]["reg"]), 8192)
        self.assertIn("reg_bytes:pmgr", out["truncated"])
        self.assertIsNone(out["pmgr_nodes"][0]["reg_ranges_total"])
        self.assertEqual(len(out["ane_nodes"][0]["reg_ranges"]), 256)
        self.assertIn("reg_ranges:ane0:257", out["truncated"])

    def test_probe_ane_port_passes_new_blocks_through(self):
        payload = json.dumps({
            "available": True, "instances": [], "ane_nodes": [],
            "dart_nodes": [], "pmgr_nodes": [],
            "coreml": {"available": False, "compute_units": None,
                       "error": None},
            "powermetrics": {"available": False, "power_mw": None,
                              "error": None},
            "driver": {"matched_class": "H11ANEIn"},
            "platform": {"soc_id": "t6021"},
            "compiler": {"framework_version": "9.509.0"},
            "set_base_candidate": {"base": "0x8e08c000"},
            "truncated": [],
        })
        with patch.object(cm, "run_python_probe", return_value={
                "available": True, "exit_code": 0, "error": None,
                "stderr": "", "stdout": payload}):
            result = cm.probe_ane_port(cc.Redactor())
        macos = result["macos"]
        self.assertEqual(macos["driver"]["matched_class"], "H11ANEIn")
        self.assertEqual(macos["platform"]["soc_id"], "t6021")
        self.assertEqual(macos["compiler"]["framework_version"],
                         "9.509.0")
        self.assertEqual(macos["set_base_candidate"]["base"],
                         "0x8e08c000")


if __name__ == "__main__":
    unittest.main(verbosity=2)
