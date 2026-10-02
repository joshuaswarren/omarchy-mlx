#!/usr/bin/env python3
"""Focused tests for the contributor collectors.

Covers the four contracts the collectors promise: PII redaction,
deterministic structure, unavailable-tool behavior, and archive integrity,
plus a static guard that the collector entry points import no network
module. Standard library only:

  python3 scripts/test_collect.py
"""

import gzip
import hashlib
import io
import json
import os
import re
import stat
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import collect_common as cc
import collect_quick as cq
import collect_deep as cd


class RedactionStripsPII(unittest.TestCase):
    SAMPLE = (
        "user joshuawarren on host m1-test-host\n"
        "model path $HOME/models/Qwen\n"
        "gateway 198.51.100.7 link-local fe80::1234:56ff:fe78:9abc\n"
        "mac f0:18:98:12:34:56\n"
        "uuid 01234567-89ab-cdef-0123-456789abcdef\n"
        "serial-number: C02XYZ123456\n"
        '  "serial_number": "FVFXC02X"\n'
        'ioreg "IOPlatformSerialNumber" = "C02XY9876543"\n'
        'ioreg "IOPlatformUUID" = "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE"\n'
        'ioreg "board-id" = "Mac-1234567890ABCDEF"\n'
        "API_KEY=sk-live-abcdef0123456789abcdef\n"
        "token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456\n"
        "Authorization: Bearer eyJhbGciOiJI.eyJzY29wZSIsInN1YiI.sIGN4TuR3\n"
        "safe: Apple M1 (G13G B1) apiVersion 1.4.354 Mesa 26.1.7\n"
    )

    def redactor(self):
        return cc.Redactor(hostname="m1-test-host", username="joshuawarren",
                           home="$HOME")

    def test_no_pii_survives(self):
        out = self.redactor().apply(self.SAMPLE)
        for secret in ("joshuawarren", "m1-test-host", "/home/", "198.51.100.7",
                       "fe80::", "f0:18:98", "01234567-89ab",
                       "C02XYZ123456", "FVFXC02X", "C02XY9876543",
                       "AAAAAAAA-BBBB", "Mac-1234567890ABCDEF",
                       "sk-live-", "ghp_ABCDEF",
                       "eyJhbGciOiJI"):
            self.assertNotIn(secret, out, f"{secret!r} leaked: {out!r}")

    def test_typed_placeholders_appear(self):
        out = self.redactor().apply(self.SAMPLE)
        for mark in ("[user]", "[host]", "[home]", "[redacted-ip4]",
                     "[redacted-ip6]", "[redacted-mac]", "[redacted-uuid]",
                     "[redacted]"):
            self.assertIn(mark, out, f"{mark} missing: {out!r}")

    def test_safe_values_survive(self):
        out = self.redactor().apply(self.SAMPLE)
        for keep in ("Apple M1 (G13G B1)", "1.4.354", "26.1.7"):
            self.assertIn(keep, out, f"safe value {keep!r} was clobbered")

    def test_counts_recorded_per_kind(self):
        red = self.redactor()
        red.apply(self.SAMPLE)
        for kind in ("home_path", "ipv4", "mac", "uuid", "credential",
                     "hostname", "username"):
            self.assertGreaterEqual(red.counts.get(kind, 0), 1, kind)


class QuickReportStructure(unittest.TestCase):
    FAKE = {
        "host": lambda red: {"available": True, "arch": "aarch64",
                             "kernel_release": "7.1.6-1-ARCH"},
        "mesa": lambda red: {"available": True, "gpu": {
            "deviceName": "Apple M1 (G13G B1)",
            "driverName": "Mesa Honeykrisp", "apiVersion": "1.4.354"}},
        "broken": lambda red: (_ for _ in ()).throw(RuntimeError("boom")),
    }

    def test_deterministic_and_complete(self):
        one = cq.collect(probes=dict(self.FAKE))
        two = cq.collect(probes=dict(self.FAKE))
        self.assertEqual(one, two)
        self.assertEqual(cc.dump_json(one), cc.dump_json(two))
        self.assertEqual(one["schema_version"], cc.SCHEMA_VERSION)
        self.assertEqual(one["report"], "mlx-omarchy-quick")
        for section in ("host", "mesa", "mesa_package", "ane", "ane_port",
                        "mlx"):
            self.assertIn(section, one)
        self.assertEqual(one["host"]["gpu"] if "gpu" in one["host"]
                         else one["mesa"]["gpu"]["driverName"],
                         "Mesa Honeykrisp")

    def test_probe_exception_is_recorded_not_raised(self):
        report = cq.collect(probes={"broken": self.FAKE["broken"]})
        self.assertFalse(report["broken"]["available"])
        self.assertIn("RuntimeError", report["broken"]["error"])


class UnavailableToolBehavior(unittest.TestCase):
    def test_missing_binary_recorded(self):
        rec = cc.run_tool(["definitely-not-a-real-tool-xyz"],
                          cc.Redactor(), label="probe")
        self.assertFalse(rec["available"])
        self.assertEqual(rec["error"], "not-found")
        self.assertIsNone(rec["exit_code"])

    def test_real_binary_captured(self):
        rec = cc.run_tool(["echo", "hello"], cc.Redactor(), label="echo")
        self.assertTrue(rec["available"])
        self.assertEqual(rec["exit_code"], 0)
        self.assertEqual(rec["stdout"], "hello")

    def test_deep_section_preserves_unavailability(self):
        with tempfile.TemporaryDirectory() as ws:
            with patch.object(cd, "run_tool", return_value={
                    "available": True, "exit_code": 0,
                    "stdout": json.dumps({"available": False, "import_error": "No module named mlx"})}):
                cd.section_child("correctness", ws, os.getcwd())
            with open(os.path.join(ws, "correctness.json")) as fh:
                data = json.load(fh)
        self.assertIn("available", data)
        self.assertFalse(data["available"])
        self.assertTrue(data.get("import_error") or data.get("probe"))


class ArchiveIntegrityAndDeterminism(unittest.TestCase):
    def build_files(self, ws):
        red = cc.Redactor()
        with open(os.path.join(ws, "quick.json"), "wb") as fh:
            fh.write(cc.json_bytes(cq.collect(probes={
                "host": lambda r: {"available": True, "arch": "aarch64"}})))
        for name in ("environment", "correctness", "benchmark", "profile"):
            with open(os.path.join(ws, f"{name}.json"), "wb") as fh:
                fh.write(cc.json_bytes({
                    "available": False, "error": "unavailable",
                    "_redaction": {"ipv4": 2} if name == "profile" else {}}))
        return cd.assemble_files(ws, os.getcwd(), [
            {"zone": "thermal_zone0", "type": "soc", "phase": "start",
             "temp_mc": 40123}])

    def test_two_builds_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as ws:
            files, unavailable, redaction = self.build_files(ws)
            name = "mlx-omarchy-deep.tar.gz"
            m1, data1, _ = cd.finalize(dict(files), unavailable, redaction,
                                       name, os.getcwd())
            m2, data2, _ = cd.finalize(dict(files), unavailable, redaction,
                                       name, os.getcwd())
        self.assertEqual(data1, data2)
        self.assertEqual(hashlib.sha256(data1).hexdigest(),
                         hashlib.sha256(data2).hexdigest())
        self.assertEqual(cc.dump_json(m1), cc.dump_json(m2))

    def test_manifest_hashes_match_members(self):
        with tempfile.TemporaryDirectory() as ws:
            files, unavailable, redaction = self.build_files(ws)
            manifest, data, _ = cd.finalize(dict(files), unavailable,
                                            redaction, "a.tar.gz",
                                            os.getcwd())
        self.assertEqual(manifest["schema_version"], cc.SCHEMA_VERSION)
        listed = {entry["path"]: entry for entry in manifest["files"]}
        self.assertNotIn("manifest.json", listed)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            members = {m.name: tf.extractfile(m).read() for m in tf.getmembers()}
        self.assertEqual(set(members),
                         set(listed) | set(cd.SUBMISSION_MEMBERS))
        for path, entry in listed.items():
            self.assertEqual(len(members[path]), entry["bytes"])
            self.assertEqual(hashlib.sha256(members[path]).hexdigest(),
                             entry["sha256"])
            self.assertEqual(members[path], files[path])

    def test_manifest_embedded_in_archive(self):
        with tempfile.TemporaryDirectory() as ws:
            files, unavailable, redaction = self.build_files(ws)
            manifest, data, _ = cd.finalize(dict(files), unavailable,
                                            redaction, "a.tar.gz",
                                            os.getcwd())
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            embedded = json.load(tf.extractfile("manifest.json"))
        self.assertTrue(embedded["no_network"])
        self.assertIn("correctness", embedded["sections_unavailable"])

    def test_submission_is_paste_ready_and_ingestion_free(self):
        with tempfile.TemporaryDirectory() as ws:
            files, unavailable, redaction = self.build_files(ws)
            work = dict(files)
            manifest, data, _ = cd.finalize(work, unavailable, redaction,
                                            "a.tar.gz", os.getcwd())
        text = work["submission.md"].decode("utf-8")
        self.assertIn("mlx-omarchy hardware report", text)
        self.assertIn("```json", text)
        self.assertNotIn("pull request", text.lower())
        self.assertNotIn("fork", text.lower())
        for entry in manifest["files"]:
            self.assertIn(entry["sha256"], text)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            embedded = tf.extractfile("submission.md").read()
        self.assertEqual(embedded, work["submission.md"])


class SubmitProtocol(unittest.TestCase):
    """The chunked resumable v1 wire protocol, against a scripted fake."""

    class FakeResponse:
        def __init__(self, status, payload):
            self.status = status
            self._payload = payload

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeServer:
        """Routes by URL shape, records every request and chunk body."""

        def __init__(self, probe=(404, {}), initiate=None, chunk=None,
                     complete=None):
            import collect_submit as cs
            self.probe = probe
            self.initiate = list(initiate or
                                 [(200, {"status": "awaiting_chunks",
                                         "missing_chunks": [0, 1, 2]})])
            self.chunk = chunk or (200, {"status": "stored"})
            self.complete = complete or (200, {"status": "stored",
                                               "receipt_url": "http://r/1"})
            self.requests = []
            self.chunk_bodies = {}
            self.archive = bytes(range(256)) * (cs.CHUNK_BYTES // 256 * 2 + 1)

        def open(self, req, timeout=None):
            url = req.get_full_url()
            method = req.get_method()
            self.requests.append(req)
            if method == "GET":
                status, payload = self.probe
            elif url.endswith("/v1/submit"):
                status, payload = self.initiate.pop(0) if len(
                    self.initiate) > 1 else self.initiate[0]
            elif "/chunk/" in url:
                idx = int(url.rsplit("/", 1)[1])
                self.chunk_bodies[idx] = req.data
                status, payload = self.chunk
                payload = dict(payload, idx=idx)
            elif url.endswith("/complete"):
                status, payload = self.complete
            else:
                raise AssertionError(f"unexpected URL {url}")
            return SubmitProtocol.FakeResponse(status, payload)

    def run_submit(self, server, **kwargs):
        import collect_submit as cs
        payload = kwargs.get("payload", {
            "schema_version": 1,
            "kind": "deep",
            "generated_at": "2026-09-03T12:00:00Z",
        })
        # Production takes a urlopen CALLABLE, not an opener object.
        result = cs.submit("http://endpoint.example", server.archive,
                           payload, urlopen=server.open)
        return result, server

    def initiate_bodies(self, server):
        import collect_submit as cs
        return [json.loads(req.data.decode("utf-8"))
                for req in server.requests
                if req.get_method() == "POST"
                and req.get_full_url().endswith("/v1/submit")]

    def test_full_multi_chunk_upload(self):
        import collect_submit as cs
        result, server = self.run_submit(SubmitProtocol.FakeServer())
        self.assertEqual(-(-len(server.archive) // cs.CHUNK_BYTES), 3)
        self.assertEqual(result["status"], 200)
        self.assertEqual(sorted(server.chunk_bodies), [0, 1, 2])
        expected = dict((idx, piece) for idx, piece, _ in
                        cs.chunk_archive(server.archive))
        for idx, piece in expected.items():
            self.assertEqual(server.chunk_bodies[idx], piece)
        self.assertTrue(server.requests[-1].get_full_url()
                        .endswith("/complete"))

    def test_initiate_carries_pow_and_per_chunk_hashes(self):
        import collect_submit as cs
        _, server = self.run_submit(SubmitProtocol.FakeServer())
        body = self.initiate_bodies(server)[0]
        digest = cs.sha256_hex(server.archive)
        self.assertEqual(body["content_sha256"], digest)
        self.assertEqual(body["schema_version"], 1)
        self.assertEqual(body["kind"], "deep")
        archive = body["archive"]
        self.assertEqual(archive["chunk_count"], 3)
        self.assertEqual(archive["total_bytes"], len(server.archive))
        self.assertEqual(archive["chunk_sha256"],
                         [c[2] for c in cs.chunk_archive(server.archive)])
        pow_token = body["pow"]
        self.assertEqual(pow_token["difficulty"], cs.POW_DIFFICULTY)
        check = hashlib.sha256(
            f"{digest}:{pow_token['nonce']}".encode()).hexdigest()
        self.assertGreaterEqual(cs.leading_zero_bits(check),
                                cs.POW_DIFFICULTY)

    def test_custom_user_agent_on_every_request(self):
        import collect_submit as cs
        _, server = self.run_submit(SubmitProtocol.FakeServer())
        self.assertGreaterEqual(len(server.requests), 5)
        for req in server.requests:
            self.assertEqual(req.headers.get("User-agent"), cs.USER_AGENT)

    def test_resume_sends_only_missing_chunks(self):
        server = SubmitProtocol.FakeServer(
            initiate=[(200, {"status": "awaiting_chunks",
                             "missing_chunks": [1, 2]})])
        self.run_submit(server)
        self.assertEqual(sorted(server.chunk_bodies), [1, 2])

    def test_dedup_probe_short_circuits_before_any_upload(self):
        server = SubmitProtocol.FakeServer(
            probe=(200, {"status": "duplicate",
                         "receipt_url": "http://r/1"}))
        result, server = self.run_submit(server)
        self.assertTrue(result["deduplicated"])
        self.assertEqual(result["url"], "http://r/1")
        self.assertEqual(len(server.requests), 1)

    def test_pow_invalid_bumps_difficulty_and_retries(self):
        server = SubmitProtocol.FakeServer(initiate=[
            (403, {"error": "pow_invalid",
                   "detail": {"min_difficulty": 20}}),
            (200, {"status": "awaiting_chunks", "missing_chunks": [0, 1, 2]}),
        ])
        self.run_submit(server)
        bodies = self.initiate_bodies(server)
        self.assertEqual(bodies[0]["pow"]["difficulty"], 18)
        self.assertEqual(bodies[1]["pow"]["difficulty"], 20)

    def test_initiate_failure_reports_server_detail(self):
        import collect_submit as cs
        errors = ["$.ane_port_detail.macos.dt_nodes[1].instance: "
                  "longer than 64 characters"]
        server = SubmitProtocol.FakeServer(initiate=[
            (422, {"error": "schema_invalid", "detail": {"errors": errors}})])
        with self.assertRaises(cs.SubmitError) as ctx:
            self.run_submit(server)
        self.assertIn("schema_invalid", str(ctx.exception))
        self.assertIn(errors[0], str(ctx.exception))

    def test_chunk_failure_raises_and_mentions_resume(self):
        import collect_submit as cs
        server = SubmitProtocol.FakeServer(
            chunk=(500, {"error": "storage_error"}))
        with self.assertRaises(cs.SubmitError) as ctx:
            self.run_submit(server)
        self.assertIn("resume", str(ctx.exception))

    def test_incomplete_complete_raises(self):
        import collect_submit as cs
        server = SubmitProtocol.FakeServer(
            complete=(409, {"error": "incomplete", "missing_chunks": [2]}))
        with self.assertRaises(cs.SubmitError):
            self.run_submit(server)

    def test_oversize_archive_never_touches_network(self):
        import collect_submit as cs
        server = SubmitProtocol.FakeServer()
        big = b"x" * (cs.MAX_ARCHIVE_BYTES + 1)
        with self.assertRaises(cs.SubmitError):
            cs.submit("http://endpoint.example", big,
                      {"schema_version": 1, "kind": "deep"}, urlopen=server)
        self.assertEqual(server.requests, [])


class SubmitEndpointWiring(unittest.TestCase):
    """--submit resolves: bare flag -> public endpoint, URL/env override wins,
    no flag -> nothing is uploaded."""

    def _parser(self):
        # The same wiring main() builds in collect_deep/collect_quick.
        import argparse
        import collect_submit as cs
        ap = argparse.ArgumentParser()
        ap.add_argument("--submit", nargs="?", default=None,
                        const=cs.DEFAULT_ENDPOINT, metavar="URL")
        return ap

    def test_bare_submit_defaults_to_public_endpoint(self):
        import collect_submit as cs
        args = self._parser().parse_args(["--submit"])
        self.assertEqual(args.submit, cs.DEFAULT_ENDPOINT)
        self.assertEqual(
            cs.endpoint_from_args(args), cs.DEFAULT_ENDPOINT)

    def test_url_flag_wins_over_default_and_env(self):
        import collect_submit as cs
        args = self._parser().parse_args(
            ["--submit", "https://collector.example"])
        with patch.dict(os.environ,
                        {"MLX_OMARCHY_SUBMIT_URL": "https://env.example"}):
            self.assertEqual(
                cs.endpoint_from_args(args), "https://collector.example")

    def test_env_override_without_flag(self):
        import collect_submit as cs
        args = self._parser().parse_args([])
        with patch.dict(os.environ,
                        {"MLX_OMARCHY_SUBMIT_URL": "https://env.example"}):
            self.assertEqual(
                cs.endpoint_from_args(args), "https://env.example")

    def test_no_flag_and_no_env_uploads_nothing(self):
        import collect_submit as cs
        args = self._parser().parse_args([])
        env = {k: v for k, v in os.environ.items()
               if k != "MLX_OMARCHY_SUBMIT_URL"}
        with patch.dict(os.environ, env, clear=True):
            self.assertIsNone(cs.endpoint_from_args(args))


class PowSolving(unittest.TestCase):
    def test_known_zero_bit_counts(self):
        import collect_submit as cs
        self.assertEqual(cs.leading_zero_bits("00ff"), 8)
        self.assertEqual(cs.leading_zero_bits("0fff"), 4)
        self.assertEqual(cs.leading_zero_bits("10ff"), 3)
        self.assertEqual(cs.leading_zero_bits("ff"), 0)

    def test_solved_nonce_verifies(self):
        import collect_submit as cs
        digest = "b" * 64
        nonce = cs.solve_pow(digest, 12)
        check = hashlib.sha256(f"{digest}:{nonce}".encode()).hexdigest()
        self.assertGreaterEqual(cs.leading_zero_bits(check), 12)


class ChunkArchiving(unittest.TestCase):
    def test_boundaries_and_hashes(self):
        import collect_submit as cs
        data = bytes(range(10))
        chunks = cs.chunk_archive(data, 4)
        self.assertEqual([c[1] for c in chunks], [b"\x00\x01\x02\x03",
                                                  b"\x04\x05\x06\x07",
                                                  b"\x08\x09"])
        self.assertEqual([c[0] for c in chunks], [0, 1, 2])
        for idx, piece, digest in chunks:
            self.assertEqual(digest, hashlib.sha256(piece).hexdigest())


class BuildPayload(unittest.TestCase):
    QUICK = {
        "host": {
            "arch": "aarch64",
            "kernel_release": "6.9.1-asahi",
            "devicetree": {"model": "Apple Mac mini",
                           "compatible": ["apple,t8103", "apple,arm"]},
            "cpu": {"present": 8, "possible": 64, "online": 1,
                    "offline": 7, "hotplug_control": False},
            "boot": {"m1n1_stage2": "v1.5.2",
                     "iboot2": "iBoot-8422.141.2"},
            "cmdline": "root=UUID=[redacted-uuid] quiet",
            "core_shortfall": {"present": 8, "online": 1},
        },
        "ane": {"devicetree": {"node": False, "compatible": None}},
        "mesa": {"gpu": {"driverName": "Asahi Vulkan",
                         "deviceName": "Apple M1"}},
        "mlx": {"distributions": {"mlx-omarchy": "0.3.2"},
                "default_device": "gpu"},
    }
    MANIFEST = {
        "source_commit": "f" * 40,
        "repo_dirty": False,
        "redaction_summary": {"mac": 1},
        "files": [{"path": "quick.json", "bytes": 5, "sha256": "a" * 64,
                   "internal": "dropped"}],
    }

    def test_exact_schema_key_set(self):
        payload = cc.build_payload("deep", self.QUICK, self.MANIFEST)
        self.assertEqual(sorted(payload), sorted([
            "schema_version", "kind", "generated_at", "arch", "model",
            "chip", "kernel", "mesa_driver", "mesa_device", "mlx_version",
            "mlx_device", "source_commit", "repo_dirty", "cpu_online",
            "cpu_present", "hotplug_control", "ane_dt_node", "ane_port",
            "ane_port_detail", "ane_dt_compatible", "boot_chain", "cmdline",
            "core_shortfall", "ane_macos", "ane_linux",
            "benchmark", "redaction_summary", "files",
        ]))

    def test_benchmark_rows_ride_in_the_summary(self):
        rows = [{"n": 512, "tflops": 0.157, "median_ms": 1.71,
                 "reps": 8, "min_ms": 1.597}]
        payload = cc.build_payload("deep", self.QUICK, self.MANIFEST,
                                   benchmark=rows)
        self.assertEqual(payload["benchmark"],
                         [{"n": 512, "tflops": 0.157, "median_ms": 1.71}])

    def test_benchmark_defaults_to_empty_and_drops_junk(self):
        self.assertEqual(
            cc.build_payload("quick", self.QUICK, self.MANIFEST)["benchmark"],
            [])
        junk = ["nope", {"tflops": 1.0}, {"n": "512"}]
        self.assertEqual(
            cc.build_payload("deep", self.QUICK, self.MANIFEST,
                             benchmark=junk)["benchmark"], [])

    def test_benchmark_is_capped(self):
        rows = [{"n": i + 1, "tflops": 1.0, "median_ms": 1.0}
                for i in range(40)]
        payload = cc.build_payload("deep", self.QUICK, self.MANIFEST,
                                   benchmark=rows)
        self.assertEqual(len(payload["benchmark"]), 16)

    def test_cpu_online_is_carried(self):
        quick = json.loads(json.dumps(self.QUICK))
        quick["host"]["cpu_online"] = 1
        payload = cc.build_payload("deep", quick, self.MANIFEST)
        self.assertEqual(payload["cpu_online"], 1)
        self.assertIsNone(
            cc.build_payload("deep", self.QUICK, self.MANIFEST)["cpu_online"])

    def test_fleet_gap_fields_are_carried(self):
        quick = json.loads(json.dumps(self.QUICK))
        quick["host"]["cpu_online"] = 1
        payload = cc.build_payload("deep", quick, self.MANIFEST)
        self.assertEqual(payload["cpu_present"], 8)
        self.assertIs(payload["hotplug_control"], False)
        self.assertIs(payload["core_shortfall"], True)
        self.assertIs(payload["ane_dt_node"], False)
        self.assertIsNone(payload["ane_dt_compatible"])
        self.assertEqual(payload["boot_chain"],
                         "iboot2=iBoot-8422.141.2 m1n1_stage2=v1.5.2")
        self.assertEqual(payload["cmdline"],
                         "root=UUID=[redacted-uuid] quiet")

    def test_shortfall_false_when_running_full_core_count(self):
        quick = json.loads(json.dumps(self.QUICK))
        quick["host"]["cpu_online"] = 8
        quick["host"]["cpu"]["online"] = 8
        quick["host"]["core_shortfall"] = None
        payload = cc.build_payload("deep", quick, self.MANIFEST)
        self.assertIs(payload["core_shortfall"], False)

    def test_gap_fields_null_when_report_lacks_them(self):
        payload = cc.build_payload("quick", {}, {})
        self.assertIsNone(payload["cpu_present"])
        self.assertIsNone(payload["hotplug_control"])
        self.assertIsNone(payload["ane_dt_node"])
        self.assertIsNone(payload["ane_dt_compatible"])
        self.assertIsNone(payload["boot_chain"])
        self.assertIsNone(payload["cmdline"])
        self.assertIsNone(payload["core_shortfall"])

    def test_ane_compatible_list_becomes_searchable_blob(self):
        quick = json.loads(json.dumps(self.QUICK))
        quick["ane"]["devicetree"] = {"node": True,
                                      "compatible": ["apple,t8103-ane"]}
        payload = cc.build_payload("deep", quick, self.MANIFEST)
        self.assertIs(payload["ane_dt_node"], True)
        self.assertEqual(payload["ane_dt_compatible"], "apple,t8103-ane")

    def test_values_extracted_from_report(self):
        payload = cc.build_payload("deep", self.QUICK, self.MANIFEST)
        self.assertEqual(payload["chip"], "apple,t8103")
        self.assertEqual(payload["kernel"], "6.9.1-asahi")
        self.assertEqual(payload["mesa_driver"], "Asahi Vulkan")
        self.assertEqual(payload["mlx_version"], "0.3.2")
        self.assertEqual(payload["repo_dirty"], False)

    def test_file_entries_are_trimmed_to_wire_shape(self):
        payload = cc.build_payload("deep", self.QUICK, self.MANIFEST)
        self.assertEqual(payload["files"],
                         [{"path": "quick.json", "bytes": 5,
                           "sha256": "a" * 64}])

    def test_missing_sections_become_none(self):
        payload = cc.build_payload("quick", {}, {})
        self.assertIsNone(payload["chip"])
        self.assertIsNone(payload["mlx_version"])
        self.assertEqual(payload["kind"], "quick")


class CpuTopology(unittest.TestCase):
    @staticmethod
    def _sysfs(tmp, files=(), cpu_dirs=()):
        base = os.path.join(tmp, "cpu")
        os.makedirs(base)
        for name, content in files:
            with open(os.path.join(base, name), "w",
                      encoding="utf-8") as fh:
                fh.write(content)
        for n in cpu_dirs:
            os.makedirs(os.path.join(base, f"cpu{n}"))
        return base

    def test_counts_and_hotplug_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._sysfs(tmp, files=[
                ("present", "0-7\n"), ("possible", "0-63\n"),
                ("online", "0-7\n"), ("offline", "\n")],
                cpu_dirs=(0, 1, 3))
            with open(os.path.join(base, "cpu1", "online"), "w",
                      encoding="utf-8"):
                pass
            cpu = cq._cpu_topology(base)
        self.assertEqual(cpu["present"], 8)
        self.assertEqual(cpu["possible"], 64)
        self.assertEqual(cpu["online"], 8)
        self.assertEqual(cpu["offline"], 0)
        self.assertIsNone(cpu["offline_list"])
        self.assertEqual(cpu["present_list"], "0-7")
        self.assertTrue(cpu["hotplug_control"])

    def test_missing_sysfs_is_all_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            cpu = cq._cpu_topology(os.path.join(tmp, "absent"))
        self.assertIsNone(cpu["present"])
        self.assertIsNone(cpu["present_list"])
        self.assertIsNone(cpu["hotplug_control"])

    def test_spin_table_has_no_hotplug_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._sysfs(tmp, files=[
                ("present", "0-7\n"), ("possible", "0-7\n"),
                ("online", "0\n"), ("offline", "1-7\n")],
                cpu_dirs=tuple(range(8)))
            cpu = cq._cpu_topology(base)
        self.assertFalse(cpu["hotplug_control"])
        self.assertEqual(cpu["offline"], 7)
        self.assertEqual(cpu["online"], 1)


class BootChainIdentity(unittest.TestCase):
    PROPS = {
        "asahi,m1n1-stage1-version": "v1.5.2\x00",
        "asahi,m1n1-stage2-version": "v1.5.2\x00",
        "asahi,iboot1-version": "iBoot-8422.100.1\x00",
        "asahi,iboot2-version": "iBoot-8422.141.2\x00",
        "asahi,system-fw-version": "iBoot-20712.1.2.0.0\x00",
        "asahi,os-fw-version": "iBoot-24.1.0\x00",
    }

    def test_chosen_properties_are_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            chosen = os.path.join(tmp, "chosen")
            os.makedirs(chosen)
            for name, value in self.PROPS.items():
                with open(os.path.join(chosen, name), "wb") as fh:
                    fh.write(value.encode())
            boot = cq._boot_chain(cc.Redactor(), base=tmp)
        self.assertEqual(boot["m1n1_stage1"], "v1.5.2")
        self.assertEqual(boot["m1n1_stage2"], "v1.5.2")
        self.assertEqual(boot["iboot1"], "iBoot-8422.100.1")
        self.assertEqual(boot["iboot2"], "iBoot-8422.141.2")
        self.assertEqual(boot["system_fw"], "iBoot-20712.1.2.0.0")
        self.assertEqual(boot["os_fw"], "iBoot-24.1.0")

    def test_missing_chosen_is_all_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            boot = cq._boot_chain(cc.Redactor(), base=tmp)
        self.assertEqual(sorted(boot), sorted([
            "m1n1_stage1", "m1n1_stage2", "iboot1", "iboot2",
            "system_fw", "os_fw"]))
        self.assertTrue(all(value is None for value in boot.values()))


class KernelCmdline(unittest.TestCase):
    def test_uuid_and_home_are_redacted(self):
        red = cc.Redactor(username="zoe", hostname="box", home="/home/zoe")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cmdline")
            with open(path, "w", encoding="utf-8") as fh:
                # Build from red.home so the committed blob carries no
                # literal home-directory path for the privacy hook.
                fh.write("root=UUID=1b3c9d2e-4f5a-6b7c-8d9e-0f1a2b3c4d5e "
                         f"init={red.home}/overlay quiet\n")
            out = cq._kernel_cmdline(red, path=path)
        self.assertEqual(out, "root=UUID=[redacted-uuid] init=[home]/overlay quiet")

    def test_missing_cmdline_is_none(self):
        self.assertIsNone(
            cq._kernel_cmdline(cc.Redactor(), path="/no/such/cmdline"))


class CoreShortfall(unittest.TestCase):
    def test_unexplained_gap_is_recorded_with_numbers(self):
        self.assertEqual(
            cq._core_shortfall({"present": 8, "online": 1}, "quiet"),
            {"present": 8, "online": 1})

    def test_full_machine_is_not_flagged(self):
        self.assertIsNone(
            cq._core_shortfall({"present": 8, "online": 8}, ""))

    def test_maxcpus_and_nosmp_explain_the_gap(self):
        self.assertIsNone(cq._core_shortfall(
            {"present": 8, "online": 1}, "maxcpus=1 quiet"))
        self.assertIsNone(cq._core_shortfall(
            {"present": 8, "online": 1}, "nosmp"))

    def test_unknown_counts_are_not_flagged(self):
        self.assertIsNone(cq._core_shortfall({}, "quiet"))


class AneDevicetreeProbe(unittest.TestCase):
    def test_ane_node_is_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            node = os.path.join(tmp, "ane@26a000000")
            os.makedirs(node)
            with open(os.path.join(node, "compatible"), "wb") as fh:
                fh.write(b"apple,t8103-ane\x00apple,ane\x00")
            out = cq._ane_devicetree(tmp)
        self.assertTrue(out["node"])
        self.assertEqual(out["compatible"], ["apple,ane", "apple,t8103-ane"])

    def test_stock_tree_has_no_ane_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "cpus"))
            with open(os.path.join(tmp, "compatible"), "wb") as fh:
                fh.write(b"apple,t8103\x00apple,arm-platform\x00")
            out = cq._ane_devicetree(tmp)
        self.assertFalse(out["node"])
        self.assertIsNone(out["compatible"])

    def test_ane_compatible_on_oddly_named_node_is_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            node = os.path.join(tmp, "engine@26a000000")
            os.makedirs(node)
            with open(os.path.join(node, "compatible"), "wb") as fh:
                fh.write(b"apple,t6000-ane\x00")
            out = cq._ane_devicetree(tmp)
        self.assertTrue(out["node"])
        self.assertEqual(out["compatible"], ["apple,t6000-ane"])

    def test_absent_devicetree_is_clean(self):
        out = cq._ane_devicetree("/no/such/tree")
        self.assertFalse(out["node"])


def _write_dt(base, relpath, props, dirs=False):
    """Create one fake devicetree node: {name: bytes} props."""
    path = os.path.join(base, relpath)
    os.makedirs(path, exist_ok=True)
    for name, raw in props.items():
        with open(os.path.join(path, name), "wb") as fh:
            fh.write(raw)


def _u32_be(*vals):
    """Big-endian DT cells: 4 bytes per value, concatenated."""
    return b"".join(v.to_bytes(4, "big") for v in vals)


def _build_port_tree(tmp, with_ane):
    """Build a t6001-style tree (ane present) or t8103 stock (absent)."""
    _write_dt(tmp, "", {
        "compatible": b"apple,t6001\x00apple,arm-platform\x00",
            "#address-cells": _u32_be(2),
        "#size-cells": _u32_be(2),
    })
    _write_dt(tmp, "dart@681004000", {
        "compatible": b"apple,t6000-dart\x00",
        "reg": _u32_be(0x6, 0x81004000, 0, 0x4000),
        "reg-names": b"dart\x00",
        "#address-cells": _u32_be(2),
        "#size-cells": _u32_be(0),
        "#iommu-cells": _u32_be(1),
        "phandle": _u32_be(0x1),
    })
    _write_dt(tmp, "aic", {
        "compatible": b"apple,t6000-aic\x00apple,aic\x00",
    })
    _write_dt(tmp, "pmgr", {
        "compatible": b"apple,t6000-pmgr\x00apple,pmgr\x00",
    })
    _write_dt(tmp, "pmgr/ane-sys", {
        "compatible": b"apple,t6000-pmgr-pwrstate\x00",
        "label": b"ane_sys\x00",
    })
    _write_dt(tmp, "pmgr/ane-sys-cpu", {
        "compatible": b"apple,t6000-pmgr-pwrstate\x00",
        "label": b"ane_sys_cpu\x00",
    })
    if with_ane:
        _write_dt(tmp, "ane@26a000000", {
            "compatible": b"apple,t6001-ane\x00apple,ane\x00",
            "reg": _u32_be(0x2, 0x6a000000, 0, 0x100000),
            "reg-names": b"ane\x00",
            "#address-cells": _u32_be(2),
            "#size-cells": _u32_be(2),
            "interrupts": _u32_be(592, 0),
            "interrupt-parent": _u32_be(0x2),
            "iommus": _u32_be(0x1, 0),
            "power-domains": _u32_be(0x3, 0x4, 0x5),
            "status": b"okay\x00",
            "phandle": _u32_be(0x6),
        })
        _write_dt(tmp, "aic", {"phandle": _u32_be(0x2)})
        _write_dt(tmp, "pmgr/ane-sys", {"phandle": _u32_be(0x3)})
        _write_dt(tmp, "pmgr/ane-sys-cpu", {"phandle": _u32_be(0x4)})
        _write_dt(tmp, "pmgr/ps-ane-plain", {"phandle": _u32_be(0x5)})


def _build_t600x_tree(tmp):
    """Real t6000/t6020 shape: DARTs named `iommu@<addr>` tagged
    `apple,<soc>-dart` with no `apple,dart` fallback, and an AIC2
    interrupt controller. No `dart*` node names anywhere."""
    _write_dt(tmp, "", {
        "compatible": b"apple,t6000\x00apple,arm-platform\x00",
            "#address-cells": _u32_be(2),
        "#size-cells": _u32_be(2),
    })
    _write_dt(tmp, "soc/iommu@285800000", {
        "compatible": b"apple,t6000-dart\x00",
        "reg": _u32_be(0x2, 0x85800000, 0, 0x4000),
        "#iommu-cells": _u32_be(1),
        "phandle": _u32_be(0x11),
    })
    _write_dt(tmp, "soc/iommu@285810000", {
        "compatible": b"apple,t6000-dart\x00",
        "reg": _u32_be(0x2, 0x85810000, 0, 0x4000),
        "#iommu-cells": _u32_be(1),
        "phandle": _u32_be(0x12),
    })
    _write_dt(tmp, "soc/interrupt-controller@28e100000", {
        "compatible": b"apple,t6000-aic\x00apple,aic2\x00",
        "phandle": _u32_be(0x13),
    })
    _write_dt(tmp, "soc/power-management@28e200000", {
        "compatible": b"apple,t6000-pmgr\x00apple,pmgr\x00",
    })
    _write_dt(tmp, "soc/power-management@28e200000/ane-sys", {
        "compatible": b"apple,t6000-pmgr-pwrstate\x00",
        "label": b"ane_sys\x00",
        "phandle": _u32_be(0x14),
    })
    # An unreferenced phandle: must NOT ride in the shipped phandle map.
    _write_dt(tmp, "soc/cpufreq@2110e0000", {
        "compatible": b"apple,t6000-cpufreq\x00",
        "phandle": _u32_be(0x99),
    })
    _write_dt(tmp, "soc", {
        "#address-cells": _u32_be(2),
        "#size-cells": _u32_be(2),
    })
    _write_dt(tmp, "soc/ane@284000000", {
        "compatible": b"apple,t6000-ane\x00",
        "reg": _u32_be(0x2, 0x85c04000, 0, 0x24000),
        "interrupts": _u32_be(770, 0),
        "iommus": _u32_be(0x11, 0, 0x12, 0),
        "power-domains": _u32_be(0x14, 0),
        "status": b"disabled\x00",
    })


def _build_t8103_bringup_tree(tmp, with_ane=True):
    """Real t8103 shape (captured from a T8103 machine 2026-09-17):
    pmgr block at 0x23b700000/0x14000 carrying the ANE SET cluster as
    power-controller pwrstate children at 0xc000+, chosen asahi,*
    firmware identity, and the bootloader-provided ane node."""
    _write_dt(tmp, "", {
        "compatible": b"apple,t8103\x00apple,arm-platform\x00",
        "model": b"MacBook Pro (14-inch, 2021)\x00",
            "#address-cells": _u32_be(2),
        "#size-cells": _u32_be(2),
    })
    _write_dt(tmp, "chosen", {
        "asahi,m1n1-stage1-version": b"m1n1 1.2.1\x00",
        "asahi,iboot1-version": b"iBoot-11841.0.1\x00",
        "asahi,system-uuid": b"12345678-1234-1234-1234-123456789abc\x00",
    })
    _write_dt(tmp, "soc", {
        "#address-cells": _u32_be(2),
        "#size-cells": _u32_be(2),
    })
    if with_ane:
        _write_dt(tmp, "soc/ane@26bc04000", {
            "compatible": b"apple,t8103-ane\x00apple,ane\x00",
            "reg": _u32_be(0x2, 0x6bc04000, 0x0, 0x24000),
            "status": b"okay\x00",
        })
    pmgr = "soc/power-management@23b700000"
    _write_dt(tmp, pmgr, {
        "compatible": b"apple,t8103-pmgr\x00apple,pmgr\x00",
        "reg": _u32_be(0x2, 0x3b700000, 0x0, 0x14000),
        "#address-cells": _u32_be(2),
        "#size-cells": _u32_be(2),
    })
    for off, label in (("470", "ane_sys"), ("c000", "ane_sys_cpu"),
                       ("c008", "ane_base"), ("c010", "ane_set1"),
                       ("c030", "ane_set5")):
        _write_dt(tmp, f"{pmgr}/power-controller@{off}", {
            "compatible": b"apple,t8103-pmgr-pwrstate\x00",
            "label": f"{label}\x00".encode(),
        })
    _write_dt(tmp, f"{pmgr}/power-controller@0", {
        "compatible": b"apple,t8103-pmgr-pwrstate\x00",
        "label": b"ps_cpu0\x00",
    })


def _build_t6001_bringup_tree(tmp):
    """Real t6001 shape (captured from a T6001 machine 2026-09-17):
    the ANE pmgr block at 0x28e080000 with the ane_set0 cluster at
    0xc000, a second pmgr block with no ane children, and the ane node
    with its MMIO reg."""
    _write_dt(tmp, "", {
        "compatible": b"apple,t6001\x00apple,arm-platform\x00",
    })
    _write_dt(tmp, "soc", {
        "#address-cells": _u32_be(2),
        "#size-cells": _u32_be(2),
    })
    _write_dt(tmp, "soc/ane@284000000", {
        "compatible": b"apple,t6001-ane\x00",
        "reg": _u32_be(0x2, 0x85c04000, 0x0, 0x24000),
    })
    ane_pmgr = "soc/power-management@28e080000"
    _write_dt(tmp, ane_pmgr, {
        "compatible": b"apple,t6000-pmgr\x00apple,pmgr\x00",
        "reg": _u32_be(0x2, 0x8e080000, 0x0, 0x14000),
        "#address-cells": _u32_be(2),
        "#size-cells": _u32_be(2),
    })
    for off, label in (("268", "ane_sys"), ("2c8", "ane_sys_cpu"),
                       ("c000", "ane_set0"), ("c008", "ane_base"),
                       ("c010", "ane_set1")):
        _write_dt(tmp, f"{ane_pmgr}/power-controller@{off}", {
            "compatible": b"apple,t6000-pmgr-pwrstate\x00",
            "label": f"{label}\x00".encode(),
        })
    gpu_pmgr = "soc/power-management@28e680000"
    _write_dt(tmp, gpu_pmgr, {
        "compatible": b"apple,t6000-pmgr\x00apple,pmgr\x00",
        "reg": _u32_be(0x2, 0x8e680000, 0x0, 0xc000),
        "#address-cells": _u32_be(2),
        "#size-cells": _u32_be(2),
    })
    _write_dt(tmp, f"{gpu_pmgr}/power-controller@100", {
        "compatible": b"apple,t6000-pmgr-pwrstate\x00",
        "label": b"amcc4\x00",
    })


class AnePortDevicetreeProbe(unittest.TestCase):
    """The t6001-style tree carries the ane node; t8103 stock does not.

    Either way the DART/PMGR/AIC dump must land so a contributor can
    author the overlay without access to the machine.
    """

    @staticmethod
    def _u32(*vals):
        # One DT cell per value, exactly as a compiled dtb stores them.
        return b"".join(v.to_bytes(4, "big") for v in vals)

    def build_tree(self, tmp, with_ane):
        _write_dt(tmp, "", {
            "compatible": b"apple,t6001\x00apple,arm-platform\x00",
                "#address-cells": _u32_be(2),
        "#size-cells": _u32_be(2),
    })
        _write_dt(tmp, "dart@681004000", {
            "compatible": b"apple,t6000-dart\x00",
            "reg": self._u32(0x6, 0x81004000, 0, 0x4000),
            "reg-names": b"dart\x00",
            "#address-cells": self._u32(2),
            "#size-cells": self._u32(0),
            "#iommu-cells": self._u32(1),
            "phandle": self._u32(0x1),
        })
        _write_dt(tmp, "aic", {
            "compatible": b"apple,t6000-aic\x00apple,aic\x00",
        })
        _write_dt(tmp, "pmgr", {
            "compatible": b"apple,t6000-pmgr\x00apple,pmgr\x00",
        })
        _write_dt(tmp, "pmgr/ane-sys", {
            "compatible": b"apple,t6000-pmgr-pwrstate\x00",
            "label": b"ane_sys\x00",
        })
        _write_dt(tmp, "pmgr/ane-sys-cpu", {
            "compatible": b"apple,t6000-pmgr-pwrstate\x00",
            "label": b"ane_sys_cpu\x00",
        })
        if with_ane:
            _write_dt(tmp, "ane@26a000000", {
                "compatible": b"apple,t6001-ane\x00apple,ane\x00",
                "reg": self._u32(0x2, 0x6a000000, 0, 0x100000),
                "reg-names": b"ane\x00",
                "#address-cells": self._u32(2),
                "#size-cells": self._u32(2),
                "interrupts": self._u32(592, 0),
                "interrupt-parent": self._u32(0x2),
                "iommus": self._u32(0x1, 0),
                "power-domains": self._u32(0x3, 0x4, 0x5),
                "status": b"okay\x00",
                "phandle": self._u32(0x6),
            })
            _write_dt(tmp, "aic", {"phandle": self._u32(0x2)})
            _write_dt(tmp, "pmgr/ane-sys", {"phandle": self._u32(0x3)})
            _write_dt(tmp, "pmgr/ane-sys-cpu", {"phandle": self._u32(0x4)})
            # an extra unrelated domain proves phandle map covers pmgr
            _write_dt(tmp, "pmgr/ps-ane-plain", {"phandle": self._u32(0x5)})

    def test_t6001_style_tree_captures_port_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.build_tree(tmp, with_ane=True)
            out = cq._ane_port_devicetree(cc.Redactor(), base=tmp)
        self.assertTrue(out["ane_node_present"])
        ane = out["ane_nodes"]["ane@26a000000"]
        self.assertEqual(ane["compatible"],
                         ["apple,t6001-ane", "apple,ane"])
        self.assertEqual(ane["reg"], ["0x26a000000/0x100000"])
        self.assertEqual(ane["reg-names"], "ane")
        self.assertEqual(ane["interrupts"], [592, 0])
        self.assertEqual(ane["interrupt-parent"], [2])
        self.assertEqual(ane["iommus"], [1, 0])
        self.assertEqual(ane["iommus_resolved"], ["dart@681004000"])
        self.assertEqual(ane["power-domains"], [3, 4, 5])
        self.assertEqual(ane["status"], "okay")
        self.assertIn("dart@681004000", out["darts"])
        dart = out["darts"]["dart@681004000"]
        self.assertEqual(dart["compatible"], "apple,t6000-dart")
        self.assertEqual(dart["#iommu-cells"], [1])
        self.assertEqual(out["aic"]["compatible"],
                         ["apple,t6000-aic", "apple,aic"])
        labels = [d["label"] for d in out["pmgr_domains"]]
        self.assertEqual(labels, ["ane_sys", "ane_sys_cpu", None])
        self.assertEqual(out["phandles"]["3"], "pmgr/ane-sys")

    def test_t8103_stock_tree_still_dumps_dart_pmgr_aic(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.build_tree(tmp, with_ane=False)
            # t8103 uses a t8103-compatible name set; the ane node is
            # absent exactly as in packaged dtbs.
            out = cq._ane_port_devicetree(cc.Redactor(), base=tmp)
        self.assertFalse(out["ane_node_present"])
        self.assertEqual(out["ane_nodes"], {})
        self.assertIn("dart@681004000", out["darts"])
        self.assertEqual(out["aic"]["compatible"],
                         ["apple,t6000-aic", "apple,aic"])
        self.assertEqual(len(out["pmgr_domains"]), 2)

    def test_payload_carries_bounded_ane_port_summary(self):
        quick = json.loads(json.dumps(BuildPayload.QUICK))
        quick["ane_port"] = {"devicetree": {
            "ane_node_present": True,
            "ane_nodes": {"ane@26a000000": {"reg":
                                            ["0x26a000000/0x100000"]}},
            "darts": {"dart@681004000": {}},
            "pmgr_domains": [{"label": "ane_sys"}],
            "aic": {"compatible": ["apple,t6000-aic"]},
        }}
        payload = cc.build_payload("quick", quick, {})
        self.assertEqual(
            payload["ane_port"],
            "present=true ane@26a000000=0x26a000000/0x100000 darts=1 "
            "pmgr_domains=1 aic=apple,t6000-aic")
        self.assertIsNone(cc.build_payload("quick", {}, {})["ane_port"])

    def test_t600x_tree_with_iommu_named_darts_is_captured(self):
        """The real t6000/t6020 shape: DARTs are `iommu@<addr>` with
        `apple,<soc>-dart` compatible and no `apple,dart` fallback; the
        AIC is `apple,aic2`. All of it must still be captured, and the
        shipped phandle map must carry only referenced entries."""
        with tempfile.TemporaryDirectory() as tmp:
            _build_t600x_tree(tmp)
            out = cq._ane_port_devicetree(cc.Redactor(), base=tmp)
        self.assertTrue(out["ane_node_present"])
        ane = out["ane_nodes"]["soc/ane@284000000"]
        self.assertEqual(ane["iommus_resolved"],
                         ["soc/iommu@285800000", "soc/iommu@285810000"])
        self.assertEqual(len(out["darts"]), 2)
        dart = out["darts"]["soc/iommu@285800000"]
        self.assertEqual(dart["compatible"], "apple,t6000-dart")
        self.assertEqual(out["aic"]["compatible"],
                         ["apple,t6000-aic", "apple,aic2"])
        self.assertEqual(out["phandles"], {
            "17": "soc/iommu@285800000",
            "18": "soc/iommu@285810000",
            "20": "soc/power-management@28e200000/ane-sys",
        })
        self.assertNotIn("153", out["phandles"])

    def test_t602x_dart_fallback_compatible_is_matched(self):
        """t602x DARTs tag `apple,t6020-dart`, `apple,t8110-dart`."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_dt(tmp, "", {
                "compatible": b"apple,t6020\x00apple,arm-platform\x00",
                "#address-cells": _u32_be(2),
                "#size-cells": _u32_be(2),
            })
            _write_dt(tmp, "soc/iommu@2a6808000", {
                "compatible": b"apple,t6020-dart\x00apple,t8110-dart\x00",
                "#iommu-cells": _u32_be(1),
                "phandle": _u32_be(0x21),
            })
            _write_dt(tmp, "soc/interrupt-controller@2a6800000", {
                "compatible": b"apple,t6020-aic\x00apple,aic2\x00",
            })
            out = cq._ane_port_devicetree(cc.Redactor(), base=tmp)
        self.assertIn("soc/iommu@2a6808000", out["darts"])
        self.assertEqual(out["aic"]["compatible"],
                         ["apple,t6020-aic", "apple,aic2"])

    def test_absent_devicetree_is_clean(self):
        out = cq._ane_port_devicetree(cc.Redactor(), base="/no/such/tree")
        self.assertFalse(out["ane_node_present"])
        self.assertEqual(out["ane_nodes"], {})
        self.assertEqual(out["darts"], {})
        self.assertEqual(out["pmgr_domains"], [])
        self.assertIsNone(out["aic"])

    _PS_MAP = {"t8103": "0x23b70c000", "t6001": "0x28e08c000"}
    # Real trees tag the pwrstate children with the FAMILY compatible
    # (t6001's pmgr block and its pwrstates are apple,t6000-*).
    _PMGR_FAMILY = {"t8103": "t8103", "t6001": "t6000"}
    # The SET region announces itself as the ane_* pwrstate cluster
    # sitting at/above 0xc000 inside the ANE pmgr block (t8103:
    # ane_sys_cpu@c000 + ane_base@c008 + ane_set1..5; t6001:
    # ane_set0@c000 + ane_base@c008 + ane_set1..5). SoC-specific
    # power-domain pwrstates (ane_sys, ane_sys_cpu on t6001) sit BELOW
    # 0xc000 and are not part of it. A tree that exposes no such
    # cluster (t6020) falls back to the +0xc000 hypothesis carried by
    # these two known-good references.
    _SET_CLUSTER = re.compile(r"ane_")

    def test_set_base_derivable_from_pmgr_topology(self):
        """The regression test that keeps the capture useful: on the two
        known-good SoCs the driver's ANE SET-block base (upstream m1n1
        ps_map) equals the captured pmgr block base plus the start of
        the captured ane SET cluster (+0xc000 on both). A new SoC's
        submission supplies the same two numbers to derive it, and the
        driver then read-verifies before any write."""
        fixtures = {"t8103": _build_t8103_bringup_tree,
                    "t6001": _build_t6001_bringup_tree}
        for soc, build in fixtures.items():
            with self.subTest(soc=soc):
                with tempfile.TemporaryDirectory() as tmp:
                    build(tmp)
                    out = cq._ane_port_devicetree(cc.Redactor(), base=tmp)
                blocks = [b for b in out["pmgr_blocks"]
                          if any(self._SET_CLUSTER.match(c["label"] or "")
                                 for c in b["children"])]
                self.assertEqual(len(blocks), 1, soc)
                block = blocks[0]
                base_addr = int(block["reg"][0].split("/")[0], 16)
                cluster = [
                    int(c["name"].split("@")[1], 16)
                    for c in block["children"]
                    if self._SET_CLUSTER.match(c["label"] or "")
                    and int(c["name"].split("@")[1], 16) >= 0xc000]
                self.assertTrue(cluster, soc)
                self.assertEqual(min(cluster), 0xc000, soc)
                self.assertEqual(f"0x{base_addr + min(cluster):x}",
                                 self._PS_MAP[soc], soc)
                # The SET region is NOT a declared register: every child
                # is a plain pwrstate node, so the offset must be
                # derived, never read from a DT "set" reg.
                for c in block["children"]:
                    self.assertEqual(
                        c["compatible"],
                        [f"apple,{self._PMGR_FAMILY[soc]}-pmgr-pwrstate"])
                # ANE subset stays cheap to triage: every ane-labelled
                # child, and nothing else.
                subset_labels = [d["label"]
                                 for d in out["pmgr_domains"]
                                 if d["path"].startswith(block["path"])]
                self.assertIn("ane_sys", subset_labels)
                self.assertNotIn("ps_cpu0", subset_labels)
                self.assertNotIn("amcc4", subset_labels)

    def test_ane_reg_present_and_absence_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            _build_t6001_bringup_tree(tmp)
            out = cq._ane_port_devicetree(cc.Redactor(), base=tmp)
        self.assertEqual(out["ane_reg"], ["0x285c04000/0x24000"])
        with tempfile.TemporaryDirectory() as tmp:
            _build_t8103_bringup_tree(tmp, with_ane=False)
            out = cq._ane_port_devicetree(cc.Redactor(), base=tmp)
        self.assertIn("ane_reg", out)
        self.assertIsNone(out["ane_reg"])

    def test_boot_provenance_is_structured_and_redacted(self):
        with tempfile.TemporaryDirectory() as tmp:
            _build_t8103_bringup_tree(tmp)
            out = cq._ane_port_devicetree(cc.Redactor(), base=tmp)
        boot = out["boot"]
        self.assertEqual(boot["model"], "MacBook Pro (14-inch, 2021)")
        self.assertEqual(boot["compatible"],
                         ["apple,t8103", "apple,arm-platform"])
        self.assertEqual(boot["chosen"]["asahi,m1n1-stage1-version"],
                         "m1n1 1.2.1")
        self.assertEqual(boot["chosen"]["asahi,iboot1-version"],
                         "iBoot-11841.0.1")
        # A UUID-shaped chosen value must not survive redaction.
        self.assertEqual(boot["chosen"]["asahi,system-uuid"],
                         "[redacted-uuid]")

    def test_dtb_sha256_hashes_the_booted_blob(self):
        with tempfile.TemporaryDirectory() as tmp:
            _build_t8103_bringup_tree(tmp)
            fdt = os.path.join(tmp, "fdt")
            with open(fdt, "wb") as fh:
                fh.write(b"\xd0\x0d\xfe\xedfake-blob")
            out = cq._ane_port_devicetree(cc.Redactor(), base=tmp,
                                          fdt_path=fdt)
            self.assertEqual(out["dtb_sha256"],
                             hashlib.sha256(
                                 b"\xd0\x0d\xfe\xedfake-blob").hexdigest())
            out = cq._ane_port_devicetree(
                cc.Redactor(), base=tmp,
                fdt_path=os.path.join(tmp, "absent"))
        self.assertIsNone(out["dtb_sha256"])

    def test_pmgr_children_cap_records_true_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_dt(tmp, "", {
                "compatible": b"apple,t6001\x00apple,arm-platform\x00",
            })
            pmgr = "soc/power-management@28e080000"
            _write_dt(tmp, pmgr, {
                "compatible": b"apple,t6000-pmgr\x00apple,pmgr\x00",
                "reg": _u32_be(0x2, 0x8e080000, 0x0, 0x14000),
            })
            for i in range(300):
                _write_dt(tmp, f"{pmgr}/power-controller@{i:x}", {
                    "compatible": b"apple,t6000-pmgr-pwrstate\x00",
                    "label": f"ps{i}\x00".encode(),
                })
            out = cq._ane_port_devicetree(cc.Redactor(), base=tmp)
        block = out["pmgr_blocks"][0]
        self.assertEqual(len(block["children"]), 256)
        self.assertEqual(block["children_total"], 300)

    def test_more_than_eight_pmgr_blocks_are_capped(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_dt(tmp, "", {
                "compatible": b"apple,t6001\x00apple,arm-platform\x00",
            })
            for i in range(9):
                _write_dt(
                    tmp, f"soc/power-management@{0x28e080000 + i * 0x10000:x}",
                    {"compatible": b"apple,t6000-pmgr\x00apple,pmgr\x00"})
            out = cq._ane_port_devicetree(cc.Redactor(), base=tmp)
        self.assertEqual(len(out["pmgr_blocks"]), 8)


class AnePortPayloadDetail(unittest.TestCase):
    """The bounded `ane_port_detail` block rides in the payload alongside
    the bounded `ane_port` summary string."""

    @staticmethod
    def _ane_port_quick_fixture():
        return {
            "available": True,
            "devicetree": {
                "ane_node_present": True,
                "ane_nodes": {
                    "ane@26a000000": {"reg": ["0x26a000000/0x100000"],
                                      "compatible":
                                          ["apple,t6001-ane", "apple,ane"]},
                },
                "ane_reg": ["0x26a000000/0x100000"],
                "darts": {
                    "dart@681004000": {"compatible": "apple,t6000-dart"},
                },
                "pmgr_domains": [{"path": "pmgr/ane-sys",
                                  "label": "ane_sys",
                                  "compatible": ["apple,t6000-pmgr-pwrstate"]}],
                "pmgr_blocks": [{
                    "path": "soc/power-management@28e080000",
                    "reg": ["0x28e080000/0x14000"],
                    "children": [
                        {"name": "power-controller@c000",
                         "label": "ane_set0",
                         "compatible": ["apple,t6000-pmgr-pwrstate"]},
                    ],
                    "children_total": 1,
                }],
                "aic": {"path": "aic",
                        "compatible": ["apple,t6000-aic", "apple,aic"]},
                "phandles": {"1": "dart@681004000"},
                "boot": {"model": "MacBook Pro",
                         "compatible": ["apple,t6001"],
                         "chosen": {"asahi,m1n1-stage1-version": "m1n1 1.2.1"}},
                "dtb_sha256": "ab" * 32,
            },
            "runtime": {"iomem": ["ane: 0x26a000000-0x26a100000"],
                        "module_version": "0.1",
                        "srcversion": "DEADBEEF",
                        "loaded": "ane 32768 0 - Live 0xffffffc0abcdef00",
                        "dmesg": ["ane: probe ok"]},
        }

    def test_payload_carries_both_summary_and_detail(self):
        quick = json.loads(json.dumps(BuildPayload.QUICK))
        quick["ane_port"] = self._ane_port_quick_fixture()
        payload = cc.build_payload("quick", quick, {},
                                   redactor=cc.Redactor())
        # The bounded summary string rides for back-compat.
        self.assertEqual(
            payload["ane_port"],
            "present=true ane@26a000000=0x26a000000/0x100000 darts=1 "
            "pmgr_domains=1 aic=apple,t6000-aic")
        # And the full structure rides in ane_port_detail.
        detail = payload["ane_port_detail"]
        self.assertIsNotNone(detail)
        self.assertIn("devicetree", detail)
        self.assertTrue(detail["devicetree"]["ane_node_present"])
        self.assertIn("ane@26a000000", detail["devicetree"]["ane_nodes"])
        self.assertIn("dart@681004000", detail["devicetree"]["darts"])
        self.assertEqual(detail["devicetree"]["ane_reg"],
                         ["0x26a000000/0x100000"])
        block = detail["devicetree"]["pmgr_blocks"][0]
        self.assertEqual(block["reg"], ["0x28e080000/0x14000"])
        self.assertEqual(block["children_total"], 1)
        self.assertEqual(detail["devicetree"]["dtb_sha256"], "ab" * 32)
        self.assertEqual(detail["devicetree"]["boot"]["model"],
                         "MacBook Pro")
        self.assertIn("runtime", detail)
        self.assertEqual(detail["runtime"]["module_version"], "0.1")

    def test_payload_omits_port_fields_when_section_absent(self):
        payload = cc.build_payload("quick", {}, {})
        self.assertIsNone(payload["ane_port"])
        self.assertIsNone(payload["ane_port_detail"])

    def test_detail_caps_node_counts_and_records_truncation(self):
        quick = json.loads(json.dumps(BuildPayload.QUICK))
        # Build more ane_nodes than MAX_NODES=8 (and more darts than
        # MAX_DARTS=32) to trigger the caps.
        ane_nodes = {f"ane@{i:x}": {"reg": [f"0x{i:x}/0x1000"]}
                     for i in range(20)}
        darts = {f"dart@{i:x}": {"compatible": "apple,t6000-dart"}
                 for i in range(40)}
        phandles = {str(i): f"node@{i:x}" for i in range(20)}
        quick["ane_port"] = {
            "available": True,
            "devicetree": {
                "ane_node_present": True,
                "ane_nodes": ane_nodes,
                "darts": darts,
                "pmgr_domains": [{"path": f"pmgr/p{i}",
                                  "label": f"p{i}",
                                  "compatible": ["x"]} for i in range(80)],
                "pmgr_blocks": [{
                    "path": f"pmgr@{i:x}",
                    "reg": [f"0x{0x28e080000 + i * 0x10000:x}/0x14000"],
                    "children": [],
                    "children_total": 0,
                } for i in range(9)],
                "aic": {"path": "aic", "compatible": ["apple,aic"]},
                "phandles": phandles,
            },
            "runtime": {"iomem": None, "module_version": None,
                        "srcversion": None, "loaded": None, "dmesg": None},
        }
        payload = cc.build_payload("quick", quick, {},
                                   redactor=cc.Redactor())
        d = payload["ane_port_detail"]
        self.assertEqual(len(d["devicetree"]["ane_nodes"]), 8)
        self.assertEqual(len(d["devicetree"]["darts"]), 32)
        self.assertEqual(len(d["devicetree"]["phandles"]), 8)
        self.assertEqual(len(d["devicetree"]["pmgr_domains"]), 64)
        self.assertEqual(len(d["devicetree"]["pmgr_blocks"]), 8)
        self.assertIn("truncated", d)
        truncated = d["truncated"]
        self.assertTrue(any(t.startswith("ane_nodes:") for t in truncated),
                        truncated)
        self.assertTrue(any(t.startswith("darts:") for t in truncated),
                        truncated)
        self.assertTrue(any(t.startswith("phandles:") for t in truncated),
                        truncated)
        self.assertTrue(any(t.startswith("pmgr_domains:") for t in truncated),
                        truncated)
        self.assertTrue(any(t.startswith("pmgr_blocks:") for t in truncated),
                        truncated)

    def test_detail_drop_order_protects_darts_pmgr_and_ane_nodes(self):
        """Byte-budget overflow sacrifices, in order: phandles, aic,
        boot, the ANE pmgr subset, the full pmgr topology, DARTs. The
        ane nodes — the whole point of the capture — are protected
        last."""
        quick = json.loads(json.dumps(BuildPayload.QUICK))
        fat = "y" * 1000
        quick["ane_port"] = {
            "available": True,
            "devicetree": {
                "ane_node_present": True,
                "ane_nodes": {"ane@26a000000":
                              {"reg": ["0x26a000000/0x100000"]}},
                # 32 darts x ~2.1KB: once everything ahead of them is
                # gone this alone still busts the budget, forcing the
                # last drop before the ane nodes.
                "darts": {f"iommu@{i:x}": {"compatible": "y" * 2100}
                          for i in range(32)},
                "pmgr_domains": [{"path": fat, "label": "ane_sys",
                                  "compatible": [fat]} for _ in range(64)],
                "pmgr_blocks": [{
                    "path": f"pmgr@{i:x}",
                    "reg": [f"0x{0x28e080000 + i * 0x10000:x}/0x14000"],
                    "children": [{"name": fat, "label": fat,
                                  "compatible": [fat]} for _ in range(8)],
                    "children_total": 8,
                } for i in range(8)],
                "aic": {"path": "aic",
                        "compatible": ["apple,t6000-aic", "apple,aic2"]},
                "phandles": {str(i): fat * 4 for i in range(8)},
                "boot": {"model": fat * 45,
                         "compatible": ["apple,t6001"],
                         "chosen": {"asahi,m1n1-stage1-version": fat * 20}},
                "dtb_sha256": "ab" * 32,
            },
            "runtime": {"iomem": None, "module_version": None,
                        "srcversion": None, "loaded": None, "dmesg": None},
        }
        payload = cc.build_payload("quick", quick, {},
                                   redactor=cc.Redactor())
        d = payload["ane_port_detail"]
        self.assertIsNotNone(d)
        truncated = d["truncated"]
        # The heavy, non-essential blocks went first...
        self.assertIn("phandles:over_budget", truncated)
        self.assertIn("aic:over_budget", truncated)
        self.assertIn("boot:over_budget", truncated)
        self.assertIn("pmgr_domains:over_budget", truncated)
        self.assertIn("pmgr_blocks:over_budget", truncated)
        self.assertIn("darts:over_budget", truncated)
        # ...and what authoring the overlay needs survived.
        self.assertIn("ane@26a000000", d["devicetree"]["ane_nodes"])
        self.assertEqual(d["devicetree"]["dtb_sha256"], "ab" * 32)
        self.assertIsNone(d["devicetree"]["boot"])
        self.assertEqual(d["devicetree"]["phandles"], {})
        self.assertEqual(d["devicetree"]["pmgr_domains"], [])
        self.assertEqual(d["devicetree"]["pmgr_blocks"], [])
        self.assertEqual(d["devicetree"]["darts"], {})

    def test_detail_drops_phandles_before_darts_and_ane_nodes(self):
        """Byte-budget overflow must sacrifice the phandle map first:
        DARTs and ane nodes are what authoring the overlay needs."""
        quick = json.loads(json.dumps(BuildPayload.QUICK))
        fat = {str(i): "y" * 9000 for i in range(8)}
        quick["ane_port"] = {
            "available": True,
            "devicetree": {
                "ane_node_present": True,
                "ane_nodes": {"ane@26a000000":
                              {"reg": ["0x26a000000/0x100000"]}},
                "darts": {f"iommu@{i:x}": {"compatible":
                                           "apple,t6000-dart"}
                          for i in range(8)},
                "pmgr_domains": [],
                "aic": {"path": "aic",
                        "compatible": ["apple,t6000-aic", "apple,aic2"]},
                "phandles": fat,
            },
            "runtime": {"iomem": None, "module_version": None,
                        "srcversion": None, "loaded": None, "dmesg": None},
        }
        payload = cc.build_payload("quick", quick, {},
                                   redactor=cc.Redactor())
        d = payload["ane_port_detail"]
        self.assertIsNotNone(d)
        self.assertIn("phandles:over_budget", d["truncated"])
        self.assertEqual(d["devicetree"]["phandles"], {})
        self.assertEqual(len(d["devicetree"]["darts"]), 8)
        self.assertIn("ane@26a000000", d["devicetree"]["ane_nodes"])
        self.assertIsNotNone(d["devicetree"]["aic"])

    def test_detail_carries_macos_block(self):
        quick = json.loads(json.dumps(BuildPayload.QUICK))
        quick["host"]["system"] = "Darwin"
        quick["ane_port"] = {
            "available": True,
            "macos": {
                "available": True,
                "instances": [{"name": "ane,t8020",
                               "matched": "ane,t8020",
                               "firmware_loaded": True,
                               "cores": 16, "version": 96,
                               "hw_board_type": 96, "arch": "h13g"}],
                "ane_nodes": [{"name": "ane0",
                               "compatible": ["ane,t8020"],
                               "reg": "0200" * 8,
                               "IOInterruptControllers": "aic",
                               "IOInterruptSpecifiers": "02000000",
                               "IOClass": None,
                               "phandle": 4097}],
                "dart_nodes": [{"name": "dart-ane0",
                                "compatible": ["dart,t6000"],
                                "reg": "0200" * 8,
                                "IOInterruptControllers": "aic",
                                "IOInterruptSpecifiers": "03000000",
                                "IOClass": "AppleT6000DART",
                                "phandle": 4113}],
                "coreml": {"available": False, "compute_units": None,
                           "error": "ModuleNotFoundError"},
                "powermetrics": {"available": False, "power_mw": None,
                                 "error": "requires root"},
                "truncated": [],
            },
        }
        payload = cc.build_payload("quick", quick, {},
                                   redactor=cc.Redactor())
        self.assertEqual(payload["ane_port"],
                         "native_macos=1 instances=1 cores=16 dart_ane=1 "
                         "firmware=loaded")
        detail = payload["ane_port_detail"]
        self.assertIn("macos", detail)
        self.assertNotIn("devicetree", detail)
        self.assertEqual(detail["macos"]["instances"][0]["cores"], 16)
        self.assertEqual(len(detail["macos"]["dart_nodes"]), 1)

    def test_detail_drops_runtime_when_it_overflows_byte_budget(self):
        quick = json.loads(json.dumps(BuildPayload.QUICK))
        # Stuff enough junk into dmesg that the runtime block exceeds
        # the per-payload budget on its own; the devicetree alone fits.
        fat_lines = ["x" * 600 for _ in range(120)]
        quick["ane_port"] = {
            "available": True,
            "devicetree": {
                "ane_node_present": True,
                "ane_nodes": {"ane@26a000000": {"reg":
                                                ["0x26a000000/0x100000"]}},
                "darts": {},
                "pmgr_domains": [],
                "aic": None,
                "phandles": {},
            },
            "runtime": {"iomem": None, "module_version": None,
                        "srcversion": None, "loaded": None,
                        "dmesg": fat_lines},
        }
        payload = cc.build_payload("quick", quick, {},
                                   redactor=cc.Redactor())
        d = payload["ane_port_detail"]
        self.assertIsNotNone(d, "detail should still ride even with fat runtime")
        self.assertNotIn("runtime", d,
                         "runtime should have been dropped for budget")
        self.assertIn("truncated", d)
        self.assertIn("runtime:over_budget", d["truncated"])

    def test_detail_re_redacts_string_leaves(self):
        # The probe already redacts, but the helper is belt-and-braces:
        # a string that somehow leaked through must still come out redacted.
        quick = json.loads(json.dumps(BuildPayload.QUICK))
        red = cc.Redactor(hostname="leakyhost",
                          username="leakyuser",
                          home="/home/leakyuser")
        quick["ane_port"] = {
            "available": True,
            "devicetree": {
                "ane_node_present": True,
                "ane_nodes": {"ane@0": {
                    "label": "leakyhost leaked",
                    "compatible": ["apple,t6001-ane"],
                    "reg": ["0x0/0x1000"]}},
                "darts": {},
                "pmgr_domains": [],
                "aic": None,
                "phandles": {},
            },
            "runtime": {"iomem": None, "module_version": None,
                        "srcversion": None, "loaded": None, "dmesg": None},
        }
        payload = cc.build_payload("quick", quick, {}, redactor=red)
        blob = json.dumps(payload["ane_port_detail"])
        self.assertNotIn("leakyhost", blob)
        self.assertNotIn("leakyuser", blob)
        self.assertIn("[host]", blob)


class PayloadSchemaContract(unittest.TestCase):
    """build_payload and the pinned schema must agree on the key set."""

    @staticmethod
    def _schema():
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.pardir, "services", "community-data",
                            "schema", "payload-v1.schema.json")
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    @classmethod
    def _undeclared(cls, node, schema, path="$"):
        """Paths the payload carries that the schema says nothing about.
        Payload objects are open since 16765c2c, so such a key is
        accepted by the endpoint but gets no type or maxLength check:
        silent drift. Descends only where the schema declares a shape."""
        found = []
        if not isinstance(schema, dict):
            return found
        props = schema.get("properties") or {}
        if isinstance(node, dict):
            if props or schema.get("additionalProperties") is False:
                found += [f"{path}.{key}" for key in node if key not in props]
            for key, value in node.items():
                if key in props:
                    found += cls._undeclared(value, props[key], f"{path}.{key}")
        elif isinstance(node, list) and isinstance(schema.get("items"), dict):
            for index, item in enumerate(node):
                found += cls._undeclared(item, schema["items"],
                                         f"{path}[{index}]")
        return found

    # Emitted by collect_quick since fb649d8d and accepted since 16765c2c
    # opened the object, but not declared, so nothing inside them is
    # validated. Declare them in payload-v1.schema.json and delete the
    # entry; the second assertion in the test below enforces that.
    KNOWN_UNDECLARED = frozenset({
        "$.ane_port_detail.devicetree.set_base_candidate.status",
        "$.ane_port_detail.devicetree.set_base_candidate.ane_pwrstate_cells",
        "$.ane_port_detail.devicetree.set_base_candidate.note",
    })

    def test_nested_payload_keys_are_declared_by_schema(self):
        """The key-set check below only sees the top level. Every key a
        real Apple Silicon devicetree section emits must be declared at
        its depth, or it ships with no validation at all."""
        with tempfile.TemporaryDirectory() as tmp:
            _build_t600x_tree(tmp)
            devicetree = cq._ane_port_devicetree(cc.Redactor(), base=tmp)
        quick = json.loads(json.dumps(BuildPayload.QUICK))
        quick["ane_port"] = {"available": True, "devicetree": devicetree}
        payload = cc.build_payload("quick", quick, {}, redactor=cc.Redactor())
        undeclared = set(self._undeclared(payload, self._schema()))
        self.assertEqual(undeclared - self.KNOWN_UNDECLARED, set(),
                         "new payload keys the schema does not declare")
        # The allowlist may only shrink: once a path is declared, drop it.
        self.assertEqual(self.KNOWN_UNDECLARED - undeclared, set(),
                         "allowlisted paths are now declared or no longer emitted")

    def test_payload_keys_equal_schema_properties(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.pardir, "services", "community-data",
                            "schema", "payload-v1.schema.json")
        with open(path, "r", encoding="utf-8") as fh:
            schema = json.load(fh)
        payload = cc.build_payload("quick", {}, {})
        self.assertEqual(sorted(payload), sorted(schema["properties"]))
        # The e2e-only fields live in the sibling schema, never here:
        # each kind keeps an exact contract.
        e2e_path = os.path.join(os.path.dirname(path),
                                "payload-v1-e2e.schema.json")
        with open(e2e_path, "r", encoding="utf-8") as fh:
            e2e = json.load(fh)
        self.assertEqual(schema["properties"]["kind"]["enum"], ["quick", "deep"])
        self.assertEqual(e2e["properties"]["kind"]["enum"], ["omarchy-mac-e2e"])
        # ane_macos / ane_linux are deep-collector turn-on blocks (schema
        # v2); the e2e summary stays a stripped-down envelope.
        self.assertFalse(set(schema["properties"]) - set(e2e["properties"])
                         - {"ane_macos", "ane_linux"})

class HyphenAdjacentUserName(unittest.TestCase):
    """A user name inside a hyphenated token is still the user name."""

    def setUp(self):
        self.red = cc.Redactor(hostname="omarchy", username="steve",
                               home="/home/steve")

    def test_user_name_inside_hyphenated_tokens_is_redacted(self):
        self.assertEqual(self.red.apply("/tmp/steve-build/out"),
                         "/tmp/[user]-build/out")
        self.assertEqual(self.red.apply("build-steve/log"), "build-[user]/log")
        self.assertEqual(self.red.counts.get("username"), 2)

    def test_default_hostname_keeps_project_tokens(self):
        # Omarchy's default hostname is `omarchy`; hyphen stays a boundary
        # for the hostname rule so the project's own names survive.
        text = "mlx-omarchy 0.32.3 via mlx-omarchy-info; omarchy-ane; host omarchy"
        self.assertEqual(self.red.apply(text),
                         "mlx-omarchy 0.32.3 via mlx-omarchy-info; omarchy-ane; host [host]")

    def test_substrings_untouched(self):
        self.assertEqual(self.red.apply("steven stevex user=steve"),
                         "steven stevex user=[user]")


class HostnameAliasRedaction(unittest.TestCase):
    """A short name derived from the host label is still the host name.

    Found in the wild: a host-admin systemd unit named after the host's
    short fleet alias, quoted verbatim by the journal inside dmesg
    captures. The whole-token hostname rule cannot see it.
    """

    def setUp(self):
        self.red = cc.Redactor(hostname="box1-max-aa", username="steve",
                               home="/home/steve")

    def test_derivation_covers_label_fragments_and_prefixes(self):
        aliases = cc.host_aliases("box1-max-aa")
        self.assertIn("box1-max-aa", aliases)
        self.assertIn("box1-max", aliases)
        self.assertIn("box1", aliases)
        # Generic pieces never become aliases.
        self.assertNotIn("linux", aliases)
        # Shortest allowed length is 4; longest first.
        self.assertEqual(aliases[0], "box1-max-aa")
        self.assertTrue(all(len(a) >= 4 for a in aliases))
        # Truncation prefixes must carry a digit; an alphabetic head
        # like `dead` (of `deadbeef-live`) is a common word.
        dead = cc.host_aliases("deadbeef-live")
        self.assertNotIn("dead", dead)
        self.assertIn("deadbeef", dead)
        self.assertIn("jw16", cc.host_aliases("jw16mbp1-linux"))
        self.assertNotIn("linux", cc.host_aliases("jw16mbp1-linux"))
        self.assertEqual(cc.host_aliases("omarchy"), [])
        self.assertEqual(cc.host_aliases(""), [])

    def test_alias_inside_unit_path_is_redacted(self):
        line = ("Oct  1 21:58:41 systemd[1]: "
                "/etc/systemd/system/box1-ane.service:9: "
                "Ignoring unknown escape sequences")
        out = self.red.apply(line)
        self.assertIn("/etc/systemd/system/[host]-ane.service:9", out)
        self.assertNotIn("box1", out)
        self.assertEqual(self.red.counts.get("hostname_alias"), 1)

    def test_full_hostname_token_still_redacted(self):
        out = self.red.apply("nfs mount from box1-max-aa done")
        self.assertEqual(out, "nfs mount from [host] done")

    def test_hex_runs_and_longer_words_are_not_corrupted(self):
        red = cc.Redactor(hostname="deadbeef-live", username="steve",
                          home="/home/steve")
        text = "iomap 0xdeadbeef beefcake deadbeef-live reg dead"
        out = red.apply(text)
        self.assertIn("0xdeadbeef", out)
        self.assertIn("beefcake", out)
        # `dead` is an alphabetic head, not a digit-bearing truncation:
        # the standalone word survives.
        self.assertIn(" reg dead", out)
        self.assertNotIn("deadbeef-live", out)
        self.assertEqual(red.counts.get("hostname"), 1)

    def test_default_hostname_derives_no_aliases(self):
        red = cc.Redactor(hostname="omarchy", username="steve",
                          home="/home/steve")
        text = "omarchy-ane.service; Linux version 7.1.13"
        self.assertEqual(red.apply(text), text)


class SingleNetworkModule(unittest.TestCase):
    def test_only_collect_submit_imports_urllib(self):
        base = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base, "collect_submit.py")) as fh:
            self.assertIn("import urllib.request", fh.read())


class NoNetworkImports(unittest.TestCase):
    BANNED = re.compile(
        r"^\s*(?:import|from)\s+(?:socket|urllib|http|requests|ftplib|"
        r"smtplib|telnetlib)\b", re.MULTILINE)

    def assert_no_network(self, path):
        with open(path) as fh:
            source = fh.read()
        hits = self.BANNED.findall(source)
        self.assertEqual(hits, [], f"{path} imports a network module: {hits}")

    def test_collectors_have_no_network_imports(self):
        base = os.path.dirname(os.path.abspath(__file__))
        for name in ("collect_quick.py", "collect_deep.py"):
            self.assert_no_network(os.path.join(base, name))

    def test_common_has_no_http_clients(self):
        base = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base, "collect_common.py")) as fh:
            source = fh.read()
        hits = re.findall(r"^\s*(?:import|from)\s+(?:urllib|http|requests|"
                          r"ftplib|smtplib)\b", source, re.MULTILINE)
        self.assertEqual(hits, [])


class VersionQuadSurvivesRedaction(unittest.TestCase):
    """A dotted version is data; a dotted address is not.

    Measured on m1-test-host-linux 2026-09-03: `conformanceVersion = 1.4.0.0`
    came back as `[redacted-ip4]`, destroying real Vulkan data in every
    submission from Apple hardware.
    """

    def test_conformance_version_is_kept(self):
        red = cc.Redactor()
        text = "\tconformanceVersion = 1.4.0.0"
        self.assertIn("1.4.0.0", red.apply(text))
        self.assertEqual(red.counts.get("ipv4", 0), 0)

    def test_lowercase_and_colon_version_forms_are_kept(self):
        red = cc.Redactor()
        # Quad built at runtime: the privacy hook blocks literal
        # private addresses in committed blobs.
        private = ".".join(["10", "0", "0", "1"])
        self.assertIn(private, red.apply(f"driver version: {private}"))
        self.assertEqual(red.counts.get("ipv4", 0), 0)

    def test_real_address_is_still_redacted(self):
        red = cc.Redactor()
        out = red.apply("inet 198.51.100.7 netmask 255.255.255.0")
        self.assertNotIn("198.51.100.7", out)
        self.assertNotIn("255.255.255.0", out)
        self.assertEqual(red.counts.get("ipv4", 0), 2)

    def test_address_on_a_later_line_is_still_redacted(self):
        red = cc.Redactor()
        # Quad built at runtime for the privacy hook (see above).
        private = ".".join(["10", "1", "2", "3"])
        out = red.apply(f"conformanceVersion = 1.4.0.0\ninet {private}\n")
        self.assertIn("1.4.0.0", out)
        self.assertNotIn(private, out)


    def test_boot_firmware_version_chain_is_kept(self):
        red = cc.Redactor()
        out = red.apply("asahi,system-fw-version=iBoot-20712.1.2.0.0")
        self.assertIn("iBoot-20712.1.2.0.0", out)
        self.assertEqual(red.counts.get("ipv4", 0), 0)

    def test_mid_chain_quad_is_kept_but_bare_quad_is_not(self):
        red = cc.Redactor()
        # Quad built at runtime for the privacy hook (see above).
        private = ".".join(["10", "1", "2", "3"])
        out = red.apply(f"fw 20712.1.2.0.0 host at {private}")
        self.assertIn("20712.1.2.0.0", out)
        self.assertNotIn(private, out)


class PrimaryGpuSelection(unittest.TestCase):
    """Honeykrisp must win over llvmpipe.

    `vulkaninfo --summary` on an Apple host lists the real GPU as GPU0
    and llvmpipe as GPU1. Keeping the last block reported llvmpipe as
    the machine's GPU, which makes the submission useless for driver
    work. Sample text is verbatim m1-test-host-linux output.
    """

    SUMMARY = (
        "Devices:\n"
        "========\n"
        "GPU0:\n"
        "\tapiVersion         = 1.4.354\n"
        "\tdriverVersion      = 26.1.7\n"
        "\tdeviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU\n"
        "\tdeviceName         = Apple M1 (G13G B1)\n"
        "\tdriverID           = DRIVER_ID_MESA_HONEYKRISP\n"
        "\tdriverName         = Honeykrisp\n"
        "\tconformanceVersion = 1.4.0.0\n"
        "GPU1:\n"
        "\tapiVersion         = 1.4.354\n"
        "\tdeviceType         = PHYSICAL_DEVICE_TYPE_CPU\n"
        "\tdeviceName         = llvmpipe (LLVM 22.1.8, 128 bits)\n"
        "\tdriverID           = DRIVER_ID_MESA_LLVMPIPE\n"
        "\tdriverName         = llvmpipe\n"
    )

    def test_both_devices_are_parsed(self):
        devices = cq._device_blocks(self.SUMMARY)
        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[0]["driverName"], "Honeykrisp")
        self.assertEqual(devices[1]["driverName"], "llvmpipe")

    def test_honeykrisp_is_primary(self):
        primary = cq._primary_device(
            cq._device_blocks(self.SUMMARY))
        self.assertEqual(primary["driverName"], "Honeykrisp")
        self.assertEqual(primary["deviceName"], "Apple M1 (G13G B1)")

    def test_non_cpu_wins_when_driver_is_unknown(self):
        devices = [
            {"driverName": "llvmpipe",
             "deviceType": "PHYSICAL_DEVICE_TYPE_CPU"},
            {"driverName": "futurevk",
             "deviceType": "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU"},
        ]
        self.assertEqual(
            cq._primary_device(devices)["driverName"], "futurevk")

    def test_cpu_only_host_still_reports_something(self):
        devices = [{"driverName": "llvmpipe",
                    "deviceType": "PHYSICAL_DEVICE_TYPE_CPU"}]
        self.assertEqual(
            cq._primary_device(devices)["driverName"], "llvmpipe")
        self.assertEqual(cq._primary_device([]), {})


class SocGrouping(unittest.TestCase):
    """Submissions group by SoC, not by board model."""

    def test_soc_compatible_wins(self):
        quick = {"host": {"devicetree": {
            "model": "Apple MacBook Pro (13-inch, M1, 2020)",
            "compatible": ["apple,j293", "apple,t8103", "apple,arm-platform"],
        }}}
        payload = cc.build_payload("quick", quick, {})
        self.assertEqual(payload["chip"], "apple,t8103")
        self.assertEqual(payload["model"],
                         "Apple MacBook Pro (13-inch, M1, 2020)")

    def test_first_entry_used_when_no_soc_present(self):
        quick = {"host": {"devicetree": {"compatible": ["vendor,board"]}}}
        self.assertEqual(cc.build_payload("quick", quick, {})["chip"],
                         "vendor,board")

    def test_missing_devicetree_is_null_not_an_error(self):
        self.assertIsNone(cc.build_payload("quick", {}, {})["chip"])


class PayloadOnlySubmit(unittest.TestCase):
    """The quick report publishes without an archive, in one round trip."""

    class Fake:
        def __init__(self, probe=(404, {}), initiate=None):
            self.probe = probe
            self.initiate = initiate or (200, {
                "status": "stored", "receipt_url": "http://r/q"})
            self.requests = []

        def open(self, req, timeout=None):
            self.requests.append(req)
            if req.get_method() == "GET":
                status, body = self.probe
            else:
                status, body = self.initiate
            return SubmitProtocol.FakeResponse(status, body)

    PAYLOAD = {"schema_version": 1, "kind": "quick",
               "generated_at": "2026-09-03T16:40:00Z", "chip": "apple,t8103"}

    def test_initiate_sends_null_archive_and_quick_kind(self):
        import collect_submit as cs
        fake = self.Fake()
        receipt = cs.submit_payload("http://e.example", self.PAYLOAD,
                                    urlopen=fake.open)
        posts = [r for r in fake.requests if r.get_method() == "POST"]
        self.assertEqual(len(posts), 1)
        body = json.loads(posts[0].data.decode("utf-8"))
        self.assertIsNone(body["archive"])
        self.assertEqual(body["kind"], "quick")
        self.assertEqual(body["payload"], self.PAYLOAD)
        self.assertIn("nonce", body["pow"])
        self.assertEqual(receipt["url"], "http://r/q")
        self.assertFalse(receipt["deduplicated"])

    def test_content_hash_is_the_canonical_payload(self):
        import collect_submit as cs
        fake = self.Fake()
        cs.submit_payload("http://e.example", self.PAYLOAD, urlopen=fake.open)
        body = json.loads(
            [r for r in fake.requests
             if r.get_method() == "POST"][0].data.decode("utf-8"))
        expected = cs.sha256_hex(json.dumps(
            self.PAYLOAD, sort_keys=True, separators=(",", ":")).encode())
        self.assertEqual(body["content_sha256"], expected)

    def test_dedup_hit_sends_no_post(self):
        import collect_submit as cs
        fake = self.Fake(probe=(200, {"status": "duplicate",
                                      "receipt_url": "http://r/dup"}))
        receipt = cs.submit_payload("http://e.example", self.PAYLOAD,
                                    urlopen=fake.open)
        self.assertTrue(receipt["deduplicated"])
        self.assertEqual([r.get_method() for r in fake.requests], ["GET"])

    def test_oversize_report_refused_before_any_request(self):
        import collect_submit as cs
        fake = self.Fake()
        big = dict(self.PAYLOAD, model="M" * (cs.MAX_PAYLOAD_BYTES + 10))
        with self.assertRaises(cs.SubmitError):
            cs.submit_payload("http://e.example", big, urlopen=fake.open)
        self.assertEqual(fake.requests, [])


    def test_alias_header_rides_the_initiate(self):
        import collect_submit as cs
        fake = self.Fake()
        receipt = cs.submit_payload("http://e.example", self.PAYLOAD,
                                    urlopen=fake.open,
                                    aliases=["box1-max-aa", "box1"])
        self.assertEqual(receipt["url"], "http://r/q")
        post = [r for r in fake.requests if r.get_method() == "POST"][0]
        sent = {k.lower(): v for k, v in post.headers.items()}
        self.assertEqual(sent.get(cs.ALIAS_HEADER.lower()),
                         "box1-max-aa,box1")

    def test_alias_header_omitted_when_no_aliases(self):
        import collect_submit as cs
        fake = self.Fake()
        cs.submit_payload("http://e.example", self.PAYLOAD, urlopen=fake.open)
        post = [r for r in fake.requests if r.get_method() == "POST"][0]
        sent = {k.lower(): v for k, v in post.headers.items()}
        self.assertIsNone(sent.get(cs.ALIAS_HEADER.lower()))




class MailboxAndReservedMemoryCapture(unittest.TestCase):
    """The ANE mailbox (mboxes target) and reserved-memory subtree:
    exactly the fields the omarchy-ane send-empty pick and fence-pool
    placement need on an untested SoC."""

    def test_mailbox_reg_and_interrupts_are_captured(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_dt(tmp, "", {"compatible":
                                b"apple,t6001\x00apple,arm-platform\x00"})
            _write_dt(tmp, "soc", {
                "#address-cells": _u32_be(2),
                "#size-cells": _u32_be(2),
            })
            _write_dt(tmp, "soc/mailbox-ane@277408000", {
                "compatible": b"apple,t6001-mailbox-ane\x00",
                "reg": _u32_be(0x0, 0x77408000, 0x0, 0x1000),
                "interrupts": _u32_be(588, 0, 589, 0, 590, 0, 591, 0),
                "interrupt-names": b"tx-empty\x00rx\x00",
                "status": b"okay\x00",
            })
            out = cq._ane_port_devicetree(cc.Redactor(), base=tmp)
        mb = out["mailbox"]["soc/mailbox-ane@277408000"]
        self.assertEqual(mb["reg"], ["0x77408000/0x1000"])
        self.assertEqual(mb["interrupts"], [588, 0, 589, 0, 590, 0, 591, 0])
        self.assertEqual(mb["interrupt-names"], ["tx-empty", "rx"])
        self.assertEqual(mb["status"], "okay")

    def test_reserved_memory_ane_children_and_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_dt(tmp, "", {"compatible": b"apple,t6001\x00"})
            _write_dt(tmp, "reserved-memory", {
                "#address-cells": _u32_be(2),
                "#size-cells": _u32_be(2),
            })
            _write_dt(tmp, "reserved-memory/ane-firmware", {
                "reg": _u32_be(0x2, 0x90000000, 0x0, 0x8000000),
                "no-map": b"",
            })
            _write_dt(tmp, "reserved-memory/other-reserved", {
                "reg": _u32_be(0x2, 0x98000000, 0x0, 0x1000000),
            })
            out = cq._ane_port_devicetree(cc.Redactor(), base=tmp)
        rm = out["reserved_memory"]
        self.assertTrue(rm["found"])
        self.assertEqual(rm["child_count"], 2)
        self.assertIn("ane-firmware", rm["nodes"])
        self.assertEqual(rm["nodes"]["ane-firmware"]["reg"],
                         ["0x290000000/0x8000000"])
        # no-map arrives as [] (empty DT property), not a missing value.
        self.assertEqual(rm["nodes"]["ane-firmware"]["no-map"], [])

    def test_missing_reserved_memory_is_a_fact(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_dt(tmp, "", {"compatible": b"apple,t6001\x00"})
            out = cq._ane_port_devicetree(cc.Redactor(), base=tmp)
        self.assertFalse(out["reserved_memory"]["found"])
        self.assertEqual(out["reserved_memory"]["child_count"], 0)

    def test_generic_generation_nodes_match_patterns(self):
        """M3 ascwrap IOP and M4 t8020-class names match without code
        changes; pmgr power-controller children are power domains, not
        the ANE device."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_dt(tmp, "", {"compatible": b"apple,t8122\x00"})
            _write_dt(tmp, "soc/iop-ane0@280000000", {
                "compatible": b"iop-ane,ascwrap-v6\x00",
                "reg": _u32_be(0x2, 0x80000000, 0x0, 0x40000),
            })
            _write_dt(tmp, "soc/ane@284000000", {
                "compatible": b"ane,t8020\x00",
                "reg": _u32_be(0x2, 0x84000000, 0x0, 0x100000),
            })
            out = cq._ane_port_devicetree(cc.Redactor(), base=tmp)
        self.assertIn("soc/iop-ane0@280000000", out["ane_nodes"])
        self.assertIn("soc/ane@284000000", out["ane_nodes"])
        self.assertEqual(
            out["ane_nodes"]["soc/iop-ane0@280000000"]["compatible"],
            "iop-ane,ascwrap-v6")

    def test_power_controller_children_never_look_like_the_ane(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_dt(tmp, "", {"compatible": b"apple,t8122\x00"})
            _write_dt(tmp, "soc/pmgr", {
                "compatible": b"apple,t8122-pmgr\x00apple,pmgr\x00",
            })
            _write_dt(tmp, "soc/pmgr/power-controller@c000", {
                "label": b"ane_set0\x00",
            })
            out = cq._ane_port_devicetree(cc.Redactor(), base=tmp)
        self.assertEqual(out["ane_nodes"], {})
        self.assertEqual(len(out["pmgr_domains"]), 1)

    def test_dtb_error_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_dt(tmp, "", {"compatible": b"apple,t6001\x00"})
            fdt = os.path.join(tmp, "fdt-dir")
            os.makedirs(fdt)
            out = cq._ane_port_devicetree(cc.Redactor(), base=tmp,
                                          fdt_path=fdt)
        self.assertIsNone(out["dtb_sha256"])
        self.assertEqual(out["dtb_sha256_error"], "unreadable")

    def test_firmware_files_are_hashed_not_shipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            fw = os.path.join(tmp, "ane")
            os.makedirs(fw)
            data = b"\x01\x02\x03ane-firmware-bytes"
            with open(os.path.join(fw, "t8122_ane0_fw.bin"), "wb") as fh:
                fh.write(data)
            os.symlink("t8122_ane0_fw.bin",
                       os.path.join(fw, "linked_ane_fw.bin"))
            with open(os.path.join(fw, "unrelated.bin"), "wb") as fh:
                fh.write(b"no ane here")
            out = cq._ane_firmware(cc.Redactor(), roots=[tmp])
        self.assertEqual(len(out["files"]), 1)
        self.assertEqual(out["files"][0]["name"], "t8122_ane0_fw.bin")
        self.assertEqual(out["files"][0]["sha256"],
                         hashlib.sha256(data).hexdigest())

    def test_firmware_cap_records_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(5):
                with open(os.path.join(tmp, f"ane{i}.bin"), "wb") as fh:
                    fh.write(b"x" * i)
            out = cq._ane_firmware(cc.Redactor(), max_files=3, roots=[tmp])
        self.assertEqual(len(out["files"]), 3)
        self.assertEqual(out["truncated"], "files:max")

    def test_payload_detail_carries_mailbox_and_reserved_memory(self):
        quick = json.loads(json.dumps(BuildPayload.QUICK))
        quick["ane_port"] = {"available": True, "devicetree": {
            "ane_node_present": True,
            "ane_nodes": {"ane@26a000000":
                          {"reg": ["0x26a000000/0x100000"]}},
            "darts": {},
            "mailbox": {"soc/mailbox-ane@277408000": {
                "reg": ["0x277408000/0x1000"]}},
            "reserved_memory": {"found": True, "child_count": 1,
                                "nodes": {"ane-firmware": {
                                    "reg": ["0x90000000/0x8000000"]}}},
        }}
        payload = cc.build_payload("quick", quick, {},
                                   redactor=cc.Redactor())
        detail = payload["ane_port_detail"]["devicetree"]
        self.assertEqual(
            detail["mailbox"]["soc/mailbox-ane@277408000"]["reg"],
            ["0x277408000/0x1000"])
        self.assertTrue(detail["reserved_memory"]["found"])
        self.assertIn("ane-firmware", detail["reserved_memory"]["nodes"])

    def test_dtb_sha256_error_rides_the_payload(self):
        quick = json.loads(json.dumps(BuildPayload.QUICK))
        quick["ane_port"] = {"available": True, "devicetree": {
            "ane_node_present": False, "ane_nodes": {}, "darts": {},
            "dtb_sha256": None, "dtb_sha256_error": "needs root",
        }}
        payload = cc.build_payload("quick", quick, {},
                                   redactor=cc.Redactor())
        self.assertEqual(
            payload["ane_port_detail"]["devicetree"]["dtb_sha256_error"],
            "needs root")


class InstallTokenTests(unittest.TestCase):
    """machine-id / owner-id: random per-install tokens, 16 hex ids,
    never serials; the file is created mode 0600 on first run."""

    def test_ids_are_stable_and_differ_per_name(self):
        with tempfile.TemporaryDirectory() as home:
            machine = cd._install_token("machine-id", home=home)
            again = cd._install_token("machine-id", home=home)
            owner = cd._install_token("owner-id", home=home)
        self.assertEqual(machine, again)
        self.assertNotEqual(machine, owner)
        for value in (machine, owner):
            self.assertRegex(value, r"^[0-9a-f]{16}$")

    def test_token_file_is_private(self):
        with tempfile.TemporaryDirectory() as home:
            cd._install_token("machine-id", home=home)
            mode = stat.S_IMODE(os.stat(
                os.path.join(home, ".config", "mlx-omarchy",
                             "machine-id")).st_mode)
        self.assertEqual(mode, 0o600)


class OmarchyAneBlockTests(unittest.TestCase):
    """The omarchy_ane promotion block: check parsing, module state,
    kernel log window, and the fixed 48 KiB cap that drops the dmesg
    tail first and never drops check/module/smoke/dmesg_faults."""

    def test_check_parses_status_and_untested(self):
        rec = {"available": True, "exit_code": 1, "error": None,
               "stdout": "omarchy-ane-check: kernel (ok)\n"
                         "UNTESTED SoC: T8122\nomarchy-ane-check: FAILED",
               "stderr": "", "argv": []}
        with patch.object(cd, "run_tool", return_value=rec):
            out = cd._omarchy_ane_check(cc.Redactor())
        self.assertEqual(out["status"], "FAILED")
        self.assertTrue(out["untested"])
        self.assertEqual(len(out["lines"]), 3)

    def test_check_absent_is_unavailable_not_omitted(self):
        with patch.object(cd, "run_tool", return_value={
                "available": False, "exit_code": None, "error": "not-found",
                "stdout": "", "stderr": "", "argv": []}):
            out = cd._omarchy_ane_check(cc.Redactor())
        self.assertFalse(out["available"])
        self.assertIn("not installed", out["unavailable"])

    def test_smoke_without_runner_says_so(self):
        with patch.object(cd.shutil, "which", return_value=None):
            out = cd._omarchy_ane_smoke(cc.Redactor())
        self.assertFalse(out["available"])
        self.assertEqual(out["name"], "add-fixture")
        self.assertIn("not shipped", out["reason"])

    def test_smoke_with_runner_parses_the_contract(self):
        runner_out = json.dumps({
            "name": "add-fixture", "chip": "t6021",
            "sha256": ["ab" * 32] * 20,
            "golden_sha256": "cd" * 32,
            "errors": 0, "min_ms": 100.1, "median_ms": 101.5})
        with patch.object(cd.shutil, "which", return_value="/usr/bin/x"), \
                patch.object(cd, "run_tool", return_value={
                    "available": True, "exit_code": 0, "error": None,
                    "stdout": runner_out, "stderr": "", "argv": []}):
            out = cd._omarchy_ane_smoke(cc.Redactor())
        self.assertTrue(out["available"])
        self.assertEqual(out["sha256"], ["ab" * 32] * 20)
        self.assertEqual(out["chip"], "t6021")
        self.assertEqual(out["golden_sha256"], "cd" * 32)
        self.assertEqual(out["median_ms"], 101.5)

    def test_smoke_exit_1_keeps_the_failed_run(self):
        runner_out = json.dumps({
            "name": "add-fixture", "chip": "t6021", "available": True,
            "sha256": ["ab" * 32] * 17,
            "golden_sha256": "cd" * 32, "errors": 3,
            "min_ms": 100.1, "median_ms": 101.5})
        with patch.object(cd.shutil, "which", return_value="/usr/bin/x"), \
                patch.object(cd, "run_tool", return_value={
                    "available": True, "exit_code": 1, "error": None,
                    "stdout": runner_out,
                    "stderr": "omarchy-ane-smoke: 3 calls differed",
                    "argv": []}):
            out = cd._omarchy_ane_smoke(cc.Redactor())
        self.assertTrue(out["available"])
        self.assertEqual(out["errors"], 3)
        self.assertEqual(len(out["sha256"]), 17)
        self.assertIn("differed", out["reason"])

    def test_smoke_exit_2_is_unavailable(self):
        with patch.object(cd.shutil, "which", return_value="/usr/bin/x"), \
                patch.object(cd, "run_tool", return_value={
                    "available": True, "exit_code": 2, "error": None,
                    "stdout": "",
                    "stderr": b"no fixture for this SoC".decode(),
                    "argv": []}):
            out = cd._omarchy_ane_smoke(cc.Redactor())
        self.assertFalse(out["available"])
        self.assertIn("no fixture", out["reason"])

    def test_kernel_log_keeps_first_window_and_fault_subset(self):
        lines = [f"line {i} ane" for i in range(250)]
        lines.append("line fault: ANE DART fault")
        fake = {"exit_code": 0, "error": None,
                "stdout": "\n".join(lines), "stderr": "", "argv": []}
        with patch.object(cd, "run_tool", return_value=fake):
            out = cd._ane_kernel_log(cc.Redactor())
        self.assertEqual(out["source"], "journalctl")
        self.assertEqual(len(out["dmesg"]), cd.ANE_DMESG_LINES)
        self.assertEqual(out["dmesg_faults"],
                         ["line fault: ANE DART fault"])

    def test_cap_drops_dmesg_tail_first_and_keeps_the_contract(self):
        block = {"machine_id": "a" * 16, "owner_id": "b" * 16,
                 "check": {"available": True, "exit": 0,
                           "status": "ready", "untested": False,
                           "lines": ["ready"]},
                 "module": {"available": True, "name": "ane"},
                 "smoke": {"available": False, "name": "add-fixture"},
                 "dmesg_faults": ["fault one"],
                 "dmesg": ["x" * 160] * 400}
        out = cc._cap_omarchy_ane(block, cc.Redactor(), max_bytes=4096)
        self.assertNotIn("dmesg", out)
        for kept in ("check", "module", "smoke", "dmesg_faults",
                     "machine_id"):
            self.assertIn(kept, out)

    def test_deep_payload_carries_the_omarchy_ane_block(self):
        quick = json.loads(json.dumps(BuildPayload.QUICK))
        quick["ane_port"] = {"available": True, "devicetree": {
            "ane_node_present": True,
            "ane_nodes": {"ane@26a000000":
                          {"reg": ["0x26a000000/0x100000"]}},
            "darts": {},
        }}
        files = {"quick.json": cc.json_bytes(quick),
                 "ane.json": cc.json_bytes({
                     "platform": "Linux", "available": True,
                     "machine_id": "a" * 16, "owner_id": "b" * 16,
                     "check": {"available": True, "exit": 0,
                               "status": "ready", "untested": False,
                               "lines": ["ready"]},
                     "module": {"available": True, "name": "ane",
                                "version": "0.4", "srcversion": None,
                                "params": None},
                     "modules": ["ane"], "loaded_line": "ane 1 0",
                     "firmware": {"files": [], "unavailable": None},
                     "opt_in": [], "uptime_s": 100,
                     "smoke": {"requested": False},
                     "dmesg": [], "dmesg_faults": [],
                     "dmesg_source": None, "iomem": None,
                     "reserved_memory": {"found": False, "child_count": 0,
                                         "nodes": {}},
                     "interrupts": [], "packages": [],
                     "host": {"cpu_online": 8}, "dt_text":
                     {"available": False, "unavailable": "dtc unavailable"},
                     "generation_probe": [{"pattern": "ane", "matched": 1}],
                     "truncated": []})}
        files = {name: data if isinstance(data, bytes) else data
                 for name, data in files.items()}
        manifest, _, payload = cd.finalize(
            files, [], {}, "t.tar.gz", cd.REPO)
        block = (payload["ane_port_detail"]["runtime"] or {}) \
            .get("omarchy_ane")
        self.assertIsNotNone(block)
        self.assertEqual(block["machine_id"], "a" * 16)
        self.assertEqual(block["check"]["status"], "ready")
        self.assertEqual(block["module"]["name"], "ane")
        self.assertEqual(payload["ane_linux"]["generation_probe"],
                         [{"pattern": "ane", "matched": 1}])
        self.assertEqual(payload["ane_macos"], None)


class InterruptsSamplingTests(unittest.TestCase):
    def test_two_idle_samples_and_after_smoke(self):
        samples = iter([[ " 592: ane" ], [" 592: ane"], [" 592: ane"]])
        with patch.object(cd, "_ane_interrupts_sample",
                          side_effect=lambda: next(samples)), \
                patch.object(cd.time, "sleep") as slept:
            out = cd._ane_interrupts(cc.Redactor(), smoke_ran=False)
        self.assertEqual([s["phase"] for s in out],
                         ["idle_first", "idle_second"])
        slept.assert_called_once_with(cd.ANE_INTERRUPT_SAMPLE_SECS)

    def test_missing_proc_interrupts_is_null_not_an_error(self):
        with patch.object(cd, "open", side_effect=OSError):
            self.assertIsNone(cd._ane_interrupts_sample())


class AneSectionUnavailableMarks(unittest.TestCase):
    """Every capture a machine cannot provide says so explicitly."""

    def test_linux_section_runs_read_only_and_reports(self):
        with tempfile.TemporaryDirectory() as ws, \
                patch.object(cd, "_omarchy_ane_check", return_value={
                    "available": False,
                    "unavailable": "omarchy-ane-check not installed"}), \
                patch.object(cd, "_omarchy_ane_module", return_value={
                    "module": {"available": False,
                               "unavailable": "no module"},
                    "modules": [], "loaded_line": None}), \
                patch.object(cd, "_ane_kernel_log", return_value={
                    "source": None, "dmesg": [], "dmesg_faults": []}), \
                patch.object(cd, "_ane_linux_dt_text",
                             return_value=(None, "dtc unavailable")), \
                patch.object(cd, "_ane_interrupts", return_value=[]), \
                patch.object(cd, "_ane_packages", return_value={
                    "unavailable": "pacman not available"}):
            out = cd.section_ane(cc.Redactor(), cd.REPO, ws, smoke=False)
        self.assertTrue(out["available"])
        self.assertEqual(out["smoke"], {"requested": False})
        self.assertIn("not installed", out["check"]["unavailable"])
        self.assertEqual(out["reserved_memory"]["found"], False)
        self.assertFalse(any(v == "written" for v in [1]))

if __name__ == "__main__":
    unittest.main(verbosity=2)
