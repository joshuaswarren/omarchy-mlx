"""Tests for the community-data query CLI and the mirror workflow.

Stdlib unittest only, no network. Fixtures are built in a temp
directory and shaped like the real collector payloads.
"""

import contextlib
import http.client
import importlib.util
import io
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import query_community_data as query  # noqa: E402
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "community-data.yml"


SHA_DEEP_M1 = "b7e34c02" + "a" * 56
SHA_QUICK_M2 = "1c9d" + "f" * 28 + "e" * 32
SHA_PARTIAL = "9a03" + "7" * 60
SHA_FUTURE = "5d" * 32
SHA_FLAT_STOCK = "aa11" + "0" * 62
SHA_NESTED_SHORT = "bb22" + "1" * 62

MALFORMED_LINE = '{"content_sha256": "oops", no closing brace'


def deep_m1_record():
    return {
        "schema_version": 1,
        "kind": "deep",
        "content_sha256": SHA_DEEP_M1,
        "payload": {
            "report": "mlx-omarchy-deep",
            "host": {"available": True, "arch": "aarch64",
                     "kernel_release": "7.1.6-1-ARCH",
                     "devicetree": {"model": "Apple M1"}},
            "mesa": {"available": True, "gpu": {
                "deviceName": "Apple M1 (G13G B1)",
                "driverName": "Mesa Honeykrisp",
                "apiVersion": "1.4.354"}},
            "mesa_package": {"pacman": {
                "available": True, "exit_code": 0,
                "stdout": "mesa 1:25.2.3-1\n"}},
            "mlx": {"available": True, "mlx_version": "0.29.3",
                    "distributions": {"mlx-omarchy": "0.3.2"}},
            "benchmark": {"available": True, "matmul": [
                {"n": 256, "reps": 8, "median_ms": 0.512, "min_ms": 0.500,
                 "max_ms": 0.600, "tflops": 65.536},
                {"n": 512, "reps": 8, "median_ms": 3.0, "min_ms": 2.9,
                 "max_ms": 3.4, "tflops": 89.4784},
                {"n": 1024, "reps": 8, "median_ms": 22.0, "min_ms": 21.5,
                 "max_ms": 24.0, "tflops": 97.6129}],
            },
        },
    }


def quick_m2_record():
    return {
        "schema_version": 1,
        "kind": "quick",
        "content_sha256": SHA_QUICK_M2,
        "payload": {
            "report": "mlx-omarchy-quick",
            "host": {"available": True, "arch": "aarch64",
                     "kernel_release": "7.2.0-1-ARCH",
                     "devicetree": {"model": "Apple M2"}},
            "mesa": {"available": True, "gpu": {
                "deviceName": "Apple M2 (G14G)",
                "driverName": "Mesa Honeykrisp",
                "apiVersion": "1.4.400"}},
            "mlx": {"available": True,
                    "distributions": {"mlx-omarchy": "0.3.1"}},
        },
    }


def partial_record():
    """Older schema: flat fields, no devicetree, no benchmark."""
    return {
        "schema_version": 0,
        "kind": "quick",
        "content_sha256": SHA_PARTIAL,
        "chip": "Apple M1 Pro",
        "kernel": "6.9.1-1-ARCH",
        "payload": {"mlx_omarchy_version": "0.2.0"},
    }


def future_record():
    """Unknown schema_version with an empty payload must still load."""
    return {
        "schema_version": 99,
        "kind": "deep",
        "content_sha256": SHA_FUTURE,
        "payload": {},
    }


def flat_stock_record():
    """Post-deploy read-API shape: new fields ride flat on the record."""
    return {
        "content_sha256": SHA_FLAT_STOCK,
        "schema_version": 1,
        "kind": "quick",
        "chip": "apple,t8103",
        "cpu_online": 8,
        "cpu_present": 8,
        "hotplug_control": True,
        "ane_dt_node": False,
        "ane_dt_compatible": None,
        "boot_chain": "iboot2=iBoot-8422.141.2 m1n1_stage2=v1.5.4",
        "cmdline": "root=UUID=[redacted-uuid] quiet",
        "core_shortfall": False,
        "payload": {"report": "mlx-omarchy-quick"},
    }


def nested_shortfall_record():
    """Local nested shape: one-core incident facts under payload.host."""
    record = deep_m1_record()
    record["content_sha256"] = SHA_NESTED_SHORT
    record["payload"]["host"]["cpu"] = {
        "present": 8, "possible": 64, "online": 1, "offline": 7,
        "hotplug_control": False}
    record["payload"]["host"]["core_shortfall"] = {
        "present": 8, "online": 1}
    record["payload"]["host"]["cmdline"] = \
        "root=UUID=[redacted-uuid] quiet"
    record["payload"]["host"]["boot"] = {
        "m1n1_stage2": "v1.5.2", "iboot2": "iBoot-8422.141.2"}
    record["payload"]["ane"] = {"devicetree": {
        "node": False, "compatible": None}}
    return record


def snapshot_bytes():
    lines = [json.dumps(deep_m1_record()),
             json.dumps(quick_m2_record()),
             MALFORMED_LINE,
             json.dumps(partial_record()),
             "",
             json.dumps(future_record())]
    return ("\n".join(lines) + "\n").encode("utf-8")


# Balanced, so a decoder with room to spare would return a dict.
DEEP_JSON = '{"a":' * 200_000 + "1" + "}" * 200_000


def run_with_small_stack(fn):
    """Return fn() run in a thread with a pinned 4 MiB stack.

    Python 3.14 bounds JSON nesting by the C stack, which follows
    ulimit -s; the pinned stack makes DEEP_JSON overflow on any host.
    """
    outcome = {}

    def target():
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # re-raised in the caller
            outcome["error"] = exc

    previous = threading.stack_size(4 * 1024 * 1024)
    try:
        thread = threading.Thread(target=target)
        thread.start()
    finally:
        threading.stack_size(previous)
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


class SnapshotTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.snapshot = Path(self._tmp.name)
        (self.snapshot / "latest.jsonl").write_bytes(snapshot_bytes())
        (self.snapshot / "index.json").write_text(json.dumps(
            {"generated_at": "2026-09-02T12:00:00Z",
             "schema_version": 1, "count": 4, "truncated": False}),
            encoding="utf-8")

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(err):
            rc = query.main(list(argv))
        return rc, out.getvalue(), err.getvalue()

    def json_list(self, *extra):
        rc, out, err = self.run_cli("--source", "local",
                                    "--snapshot", str(self.snapshot),
                                    "--json", *extra, "list")
        self.assertEqual(rc, 0, err)
        return json.loads(out)


class QueryCliTests(SnapshotTestCase):
    def test_local_snapshot_parses_and_skips_malformed(self):
        data = self.json_list()
        self.assertEqual(data["count"], 4)
        self.assertEqual(data["skipped_malformed"], 1)
        shas = [r["content_sha256"] for r in data["results"]]
        self.assertEqual(shas, [SHA_DEEP_M1, SHA_QUICK_M2,
                                SHA_PARTIAL, SHA_FUTURE])

    def test_too_deeply_nested_line_is_skipped(self):
        with self.assertRaises(RecursionError):
            run_with_small_stack(lambda: json.loads(DEEP_JSON))
        with open(self.snapshot / "latest.jsonl", "a",
                  encoding="utf-8") as handle:
            handle.write(DEEP_JSON + "\n")
        data = run_with_small_stack(self.json_list)
        self.assertEqual(data["count"], 4)
        self.assertEqual(data["skipped_malformed"], 2)

    def test_human_list_has_one_line_per_record(self):
        rc, out, err = self.run_cli("--source", "local",
                                    "--snapshot", str(self.snapshot),
                                    "list")
        self.assertEqual(rc, 0)
        lines = [x for x in out.splitlines() if x.strip()]
        self.assertEqual(len(lines), 4)
        self.assertIn(SHA_DEEP_M1[:12], lines[0])
        self.assertIn("1 malformed line(s) skipped", err)

    def test_filter_kind(self):
        self.assertEqual(
            {r["content_sha256"] for r in self.json_list(
                "--kind", "quick")["results"]},
            {SHA_QUICK_M2, SHA_PARTIAL})
        self.assertEqual(
            {r["content_sha256"] for r in self.json_list(
                "--kind", "deep")["results"]},
            {SHA_DEEP_M1, SHA_FUTURE})

    def test_filter_chip(self):
        self.assertEqual(
            {r["content_sha256"] for r in self.json_list(
                "--chip", "M1")["results"]},
            {SHA_DEEP_M1, SHA_PARTIAL})
        self.assertEqual(
            {r["content_sha256"] for r in self.json_list(
                "--chip", "M1 Pro")["results"]},
            {SHA_PARTIAL})
        self.assertEqual(
            {r["content_sha256"] for r in self.json_list(
                "--chip", "M2")["results"]},
            {SHA_QUICK_M2})

    def test_filter_kernel(self):
        self.assertEqual(
            {r["content_sha256"] for r in self.json_list(
                "--kernel", "7.1.6")["results"]},
            {SHA_DEEP_M1})
        self.assertEqual(
            {r["content_sha256"] for r in self.json_list(
                "--kernel", "6.9.1")["results"]},
            {SHA_PARTIAL})

    def test_filter_mesa(self):
        self.assertEqual(
            {r["content_sha256"] for r in self.json_list(
                "--mesa", "25.2.3")["results"]},
            {SHA_DEEP_M1})
        self.assertEqual(
            {r["content_sha256"] for r in self.json_list(
                "--mesa", "honeykrisp")["results"]},
            {SHA_DEEP_M1, SHA_QUICK_M2})

    def test_filter_mlx_version(self):
        self.assertEqual(
            {r["content_sha256"] for r in self.json_list(
                "--mlx-version", "0.3.1")["results"]},
            {SHA_QUICK_M2})
        self.assertEqual(
            {r["content_sha256"] for r in self.json_list(
                "--mlx-version", "0.2.0")["results"]},
            {SHA_PARTIAL})

    def test_filters_combine(self):
        self.assertEqual(
            {r["content_sha256"] for r in self.json_list(
                "--kind", "deep", "--chip", "M1")["results"]},
            {SHA_DEEP_M1})

    def test_no_match_is_clean_zero_exit(self):
        rc, out, err = self.run_cli("--source", "local",
                                    "--snapshot", str(self.snapshot),
                                    "--chip", "Ryzen", "list")
        self.assertEqual(rc, 0)
        self.assertIn("no records match", out)

    def test_compare_tflops_aggregates_across_sizes(self):
        rc, out, _ = self.run_cli("--source", "local",
                                  "--snapshot", str(self.snapshot),
                                  "--json", "compare", "--metric", "tflops")
        self.assertEqual(rc, 0)
        data = json.loads(out)
        self.assertEqual(data["metric"], "tflops")
        self.assertEqual(data["groups"], [{
            "device": "Apple M1", "count": 3,
            "min": 65.536, "median": 89.4784, "max": 97.6129}])

    def test_compare_by_size(self):
        rc, out, _ = self.run_cli("--source", "local",
                                  "--snapshot", str(self.snapshot),
                                  "--json", "compare",
                                  "--metric", "tflops", "--size", "512")
        self.assertEqual(rc, 0)
        data = json.loads(out)
        self.assertEqual(data["size"], 512)
        self.assertEqual(data["groups"][0]["count"], 1)
        self.assertEqual(data["groups"][0]["median"], 89.4784)

    def test_compare_median_ms(self):
        rc, out, _ = self.run_cli("--source", "local",
                                  "--snapshot", str(self.snapshot),
                                  "--json", "compare",
                                  "--metric", "median_ms")
        self.assertEqual(rc, 0)
        group = json.loads(out)["groups"][0]
        self.assertEqual(group["min"], 0.512)
        self.assertEqual(group["median"], 3.0)
        self.assertEqual(group["max"], 22.0)

    def test_compare_without_data(self):
        rc, out, _ = self.run_cli("--source", "local",
                                  "--snapshot", str(self.snapshot),
                                  "--chip", "M2",
                                  "compare", "--metric", "tflops")
        self.assertEqual(rc, 0)
        self.assertIn("no tflops benchmark data", out)

    def test_show_by_hash_prefix(self):
        rc, out, _ = self.run_cli("--source", "local",
                                  "--snapshot", str(self.snapshot),
                                  "show", SHA_DEEP_M1[:8])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out)["content_sha256"], SHA_DEEP_M1)

    def test_show_unknown_prefix_errors_cleanly(self):
        rc, out, err = self.run_cli("--source", "local",
                                    "--snapshot", str(self.snapshot),
                                    "show", "ffffffff")
        self.assertEqual(rc, 1)
        self.assertIn("no record with hash prefix", err)
        self.assertNotIn("Traceback", err)

    def test_missing_dataset_message(self):
        """An explicit --snapshot that holds nothing fails there.

        It must NOT fall through to the mirrored community-data branch:
        answering from a different dataset than the one named is worse
        than an error, and it made this test pass or fail depending on
        whether the data branch happened to be fetched.
        """
        missing = self.snapshot / "does-not-exist"
        rc, out, err = self.run_cli("--source", "local",
                                    "--snapshot", str(missing), "list")
        self.assertEqual(rc, 1)
        self.assertIn("no dataset at the requested --snapshot", err)
        self.assertIn(str(missing), err)
        self.assertNotIn("Traceback", err)

    def test_auto_uses_local_snapshot_without_network(self):
        rc, out, _ = self.run_cli("--snapshot", str(self.snapshot),
                                  "--json", "list")
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out)["count"], 4)

    def test_unknown_schema_version_tolerated(self):
        data = self.json_list()
        future = [r for r in data["results"]
                  if r["schema_version"] == 99]
        self.assertEqual(len(future), 1)
        rc, out, err = self.run_cli("--source", "local",
                                    "--snapshot", str(self.snapshot),
                                    "--json", "compare")
        self.assertEqual(rc, 0)  # future record simply has no benchmarks

    def test_partial_record_extracts_flat_fields(self):
        rc, out, _ = self.run_cli("--source", "local",
                                  "--snapshot", str(self.snapshot),
                                  "--json", "list")
        records = {r["content_sha256"]: r
                   for r in json.loads(out)["results"]}
        self.assertEqual(query.record_chip(records[SHA_PARTIAL]),
                         "Apple M1 Pro")
        self.assertEqual(query.record_kernel(records[SHA_PARTIAL]),
                         "6.9.1-1-ARCH")
        self.assertEqual(query.record_mlx_version(records[SHA_PARTIAL]),
                         "0.2.0")


class FleetGapAccessors(unittest.TestCase):
    """The new boot/core/ANE facts are readable in both record shapes."""

    def test_flat_record_fields(self):
        record = flat_stock_record()
        self.assertEqual(query.record_cpu_present(record), 8)
        self.assertEqual(query.record_cpu_online(record), 8)
        self.assertIs(query.record_hotplug(record), True)
        self.assertIs(query.record_core_shortfall(record), False)
        self.assertIs(query.record_ane_dt(record)[0], False)
        self.assertIn("iBoot-8422.141.2",
                      query.record_boot_chain(record) or "")
        self.assertIn("quiet", query.record_cmdline(record) or "")

    def test_nested_record_fields(self):
        record = nested_shortfall_record()
        self.assertEqual(query.record_cpu_present(record), 8)
        self.assertEqual(query.record_cpu_online(record), 1)
        self.assertIs(query.record_hotplug(record), False)
        self.assertIs(query.record_core_shortfall(record), True)
        self.assertIs(query.record_ane_dt(record)[0], False)
        self.assertIn("v1.5.2", query.record_boot_chain(record) or "")

    def test_nested_compatible_list_becomes_blob(self):
        record = nested_shortfall_record()
        record["payload"]["ane"]["devicetree"] = {
            "node": True, "compatible": ["apple,t8103-ane"]}
        node, compat = query.record_ane_dt(record)
        self.assertIs(node, True)
        self.assertEqual(compat, "apple,t8103-ane")

    def test_old_records_yield_none(self):
        for record in (deep_m1_record(), quick_m2_record(),
                       partial_record(), future_record()):
            self.assertIsNone(query.record_cpu_present(record))
            self.assertIsNone(query.record_hotplug(record))
            self.assertIsNone(query.record_core_shortfall(record))
            self.assertIsNone(query.record_ane_dt(record)[0])
            self.assertIsNone(query.record_boot_chain(record))
            self.assertIsNone(query.record_cmdline(record))

    def test_derived_shortfall_respects_maxcpus(self):
        record = deep_m1_record()
        record["payload"]["host"]["cpu"] = {"present": 8, "online": 1}
        self.assertIs(query.record_core_shortfall(record), True)
        record["payload"]["host"]["cmdline"] = "maxcpus=1"
        self.assertIsNone(query.record_core_shortfall(record))


class FleetGapFilterTests(unittest.TestCase):
    """--cpu-present/--hotplug/--ane/--shortfall/--boot/--cmdline."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.snapshot = Path(self._tmp.name)
        lines = [json.dumps(flat_stock_record()),
                 json.dumps(nested_shortfall_record()),
                 json.dumps(deep_m1_record())]
        (self.snapshot / "latest.jsonl").write_bytes(
            ("\n".join(lines) + "\n").encode("utf-8"))

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(err):
            rc = query.main(list(argv))
        return rc, out.getvalue(), err.getvalue()

    def json_list(self, *extra):
        rc, out, err = self.run_cli("--source", "local",
                                    "--snapshot", str(self.snapshot),
                                    "--json", *extra, "list")
        self.assertEqual(rc, 0, err)
        return json.loads(out)

    def shas(self, *extra):
        return {r["content_sha256"]
                for r in self.json_list(*extra)["results"]}

    def test_cpu_present_filter(self):
        self.assertEqual(self.shas("--cpu-present", "8"),
                         {SHA_FLAT_STOCK, SHA_NESTED_SHORT})
        self.assertEqual(self.shas("--cpu-present", "1"), set())

    def test_hotplug_filter(self):
        self.assertEqual(self.shas("--hotplug", "yes"), {SHA_FLAT_STOCK})
        self.assertEqual(self.shas("--hotplug", "no"),
                         {SHA_NESTED_SHORT})

    def test_ane_filter(self):
        self.assertEqual(self.shas("--ane", "yes"), set())
        self.assertEqual(self.shas("--ane", "no"),
                         {SHA_FLAT_STOCK, SHA_NESTED_SHORT})

    def test_shortfall_filter(self):
        self.assertEqual(self.shas("--shortfall", "yes"),
                         {SHA_NESTED_SHORT})
        self.assertEqual(self.shas("--shortfall", "no"), {SHA_FLAT_STOCK})

    def test_boot_filter(self):
        self.assertEqual(self.shas("--boot", "iBoot-8422"),
                         {SHA_FLAT_STOCK, SHA_NESTED_SHORT})
        self.assertEqual(self.shas("--boot", "m1n1_stage1"), set())

    def test_cmdline_filter(self):
        self.assertEqual(self.shas("--cmdline", "maxcpus"), set())
        self.assertEqual(self.shas("--cmdline", "quiet"),
                         {SHA_FLAT_STOCK, SHA_NESTED_SHORT})

    def test_human_list_shows_cores_ane_and_shortfall(self):
        rc, out, _ = self.run_cli("--source", "local",
                                  "--snapshot", str(self.snapshot),
                                  "list")
        self.assertEqual(rc, 0)
        self.assertIn("cores=8/8", out)
        self.assertIn("cores=1/8", out)
        self.assertIn("ane=-", out)
        self.assertIn("core-shortfall", out)

    def test_json_list_includes_new_fields(self):
        results = {r["content_sha256"]: r
                   for r in self.json_list()["results"]}
        self.assertEqual(results[SHA_FLAT_STOCK]["cpu_present"], 8)
        self.assertIs(results[SHA_FLAT_STOCK]["ane_dt_node"], False)
        self.assertEqual(results[SHA_NESTED_SHORT]["payload"]["host"]
                         ["cpu"]["present"], 8)


try:
    import yaml
except ImportError:
    yaml = None


@unittest.skipUnless(yaml is not None, "PyYAML not installed")
class WorkflowYamlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding="utf-8")
        cls.data = yaml.safe_load(cls.text)

    def test_workflow_parses_with_expected_triggers(self):
        triggers = self.data.get("on", self.data.get(True))
        self.assertIsInstance(triggers, dict)
        self.assertIn("17 3 * * *", triggers["schedule"][0]["cron"])
        self.assertIn("workflow_dispatch", triggers)

    def test_minimum_permissions(self):
        self.assertEqual(self.data["permissions"], {"contents": "read"})
        job = self.data["jobs"]["mirror"]
        self.assertEqual(job["permissions"], {"contents": "write"})
        self.assertEqual(job["runs-on"], "ubuntu-latest")

    def test_commits_as_bot_to_data_branch_without_secrets(self):
        self.assertIn("github-actions[bot]", self.text)
        self.assertIn("community-data", self.text)
        self.assertNotIn("${{ secrets.", self.text)

    def test_endpoint_default_matches_cli_default(self):
        self.assertIn(query.DEFAULT_BASE_URL, self.text)
        self.assertIn("vars.COMMUNITY_DATA_BASE_URL", self.text)

    def test_mirror_script_is_referenced(self):
        self.assertIn("scripts/mirror_community_data.py", self.text)


@unittest.skipUnless(yaml is not None, "PyYAML not installed")
class WorkflowStepOrderTests(unittest.TestCase):
    """Run the workflow's own shell steps in a throwaway clone.

    The commit step leaves the checkout on the snapshot-only
    `community-data` branch. The schema-identity step must still find its
    script, which lives on main.
    """

    SCRIPT = "services/community-data/scripts/check_schema_identity.py"

    @classmethod
    def setUpClass(cls):
        steps = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))[
            "jobs"]["mirror"]["steps"]
        cls.commit_run = next(
            s["run"] for s in steps
            if s.get("name") == "Commit snapshot to community-data")
        cls.check_run = next(
            s["run"] for s in steps
            if s.get("name") == "Check deployed schema identity")

    def sh(self, *args, cwd, env=None, check=True):
        env = dict(os.environ if env is None else env,
                   GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="core.hooksPath",
                   GIT_CONFIG_VALUE_0=os.devnull)
        return subprocess.run(
            list(args), cwd=cwd, env=env, check=check, text=True,
            capture_output=True)

    def lab(self, tmp, with_remote_branch):
        origin = tmp / "origin.git"
        seed = tmp / "seed"
        self.sh("git", "init", "-q", "--bare", "-b", "main", str(origin),
                cwd=tmp)
        self.sh("git", "init", "-q", "-b", "main", str(seed), cwd=tmp)
        script = seed / self.SCRIPT
        script.parent.mkdir(parents=True)
        script.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
        ident = ("-c", "user.name=t", "-c", "user.email=t@example.com")
        self.sh("git", "add", ".", cwd=seed)
        self.sh("git", *ident, "commit", "-q", "-m", "main", cwd=seed)
        self.sh("git", "remote", "add", "origin", str(origin), cwd=seed)
        self.sh("git", "push", "-q", "origin", "main", cwd=seed)
        if with_remote_branch:
            self.sh("git", "switch", "-q", "--orphan", "community-data",
                    cwd=seed)
            (seed / "snapshot").mkdir()
            (seed / "snapshot" / "latest.jsonl").write_text(
                '{"old": 1}\n', encoding="utf-8")
            self.sh("git", "add", ".", cwd=seed)
            self.sh("git", *ident, "commit", "-q", "-m", "old", cwd=seed)
            self.sh("git", "push", "-q", "origin", "community-data", cwd=seed)
        work = tmp / "work"
        self.sh("git", "clone", "-q", str(origin), str(work), cwd=tmp)
        sha = self.sh("git", "rev-parse", "HEAD", cwd=work).stdout.strip()
        self.sh("git", "checkout", "-q", "--detach", sha, cwd=work)
        return work, sha

    def run_steps(self, with_remote_branch):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            work, sha = self.lab(tmp, with_remote_branch)
            runner_temp = tmp / "runner_temp"
            (runner_temp / "snapshot").mkdir(parents=True)
            (runner_temp / "snapshot" / "latest.jsonl").write_text(
                '{"new": 1}\n', encoding="utf-8")
            env = dict(os.environ, RUNNER_TEMP=str(runner_temp),
                       BASE_URL="http://example.invalid", GITHUB_SHA=sha)
            self.sh("bash", "-e", "-c", self.commit_run, cwd=work, env=env)
            return self.sh("bash", "-e", "-c", self.check_run, cwd=work,
                           env=env, check=False)

    def test_check_script_is_reachable_after_the_first_snapshot(self):
        result = self.run_steps(with_remote_branch=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_check_script_is_reachable_after_a_later_snapshot(self):
        result = self.run_steps(with_remote_branch=True)
        self.assertEqual(result.returncode, 0, result.stderr)


def load_schema_check():
    path = REPO_ROOT / WorkflowStepOrderTests.SCRIPT
    spec = importlib.util.spec_from_file_location("check_schema_identity",
                                                  path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SchemaIdentityCheckTests(unittest.TestCase):
    """check_schema_identity.py against a faked GET /v1/schema reply.

    Exit 0 means the live identity matches, 2 means a JSON object came
    back with a different identity, and 1 means no usable reply.
    """

    URL = "https://worker.example"

    @classmethod
    def setUpClass(cls):
        cls.check = load_schema_check()
        cls.good = cls.check.expected()

    def run_check(self, reply, url=URL):
        """Run main(); reply is the body bytes or an exception to raise."""
        urlopen = mock.Mock(side_effect=reply) \
            if isinstance(reply, BaseException) \
            else mock.Mock(return_value=io.BytesIO(reply))
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(self.check.urllib.request, "urlopen",
                               urlopen), \
                contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(err):
            code = self.check.main(["--url", url])
        return code, out.getvalue(), err.getvalue(), urlopen

    def reply(self, **changes):
        return json.dumps(dict(self.good, **changes)).encode()

    def test_matching_identity_is_ok(self):
        code, out, err, _ = self.run_check(self.reply())
        self.assertEqual(code, 0, err)
        self.assertIn("[schema-identity] OK", out)

    def test_trailing_slash_url_requests_v1_schema(self):
        for url in (self.URL + "/", self.URL + "//"):
            with self.subTest(url=url):
                code, _, err, urlopen = self.run_check(self.reply(), url)
                self.assertEqual(code, 0, err)
                self.assertEqual(urlopen.call_args.args[0].full_url,
                                 self.URL + "/v1/schema")

    def test_different_identity_exits_2(self):
        for key, value in (
                ("schema_sha256", "0" * 64),
                ("schema_versions", [1]),
                ("schema_versions", 1),
                ("schema_versions", [1, "2"]),
                ("schema_versions", [1, None]),
                ("schema_versions", [True, 2]),
                ("schema_versions", [1.0, 2]),
                ("fields_sha256", [self.good["fields_sha256"]])):
            with self.subTest(key=key, value=value):
                code, out, err, _ = self.run_check(
                    self.reply(**{key: value}))
                self.assertEqual(code, 2, out + err)
                self.assertIn("STALE DEPLOY", err)
                self.assertIn(f"  {key}: expected=", err)

    def test_missing_key_exits_2(self):
        body = {k: v for k, v in self.good.items() if k != "schema_versions"}
        code, _, err, _ = self.run_check(json.dumps(body).encode())
        self.assertEqual(code, 2, err)
        self.assertIn("schema_versions: expected=[1, 2] live=None", err)

    def assert_unreadable(self, reply, reason, small_stack=False):
        code, out, err, _ = (
            run_with_small_stack(lambda: self.run_check(reply))
            if small_stack else self.run_check(reply))
        self.assertEqual(code, 1, out + err)
        self.assertEqual(out, "")
        self.assertTrue(err.startswith(
            f"[schema-identity] cannot read {self.URL}/v1/schema: "), err)
        self.assertIn(reason, err)

    def test_non_object_reply_exits_1(self):
        for body, name in ((b"[]", "list"), (b"null", "NoneType"),
                           (b'"ok"', "str"), (b"123", "int")):
            with self.subTest(body=body):
                self.assert_unreadable(
                    body, f"reply is {name}, not a JSON object")

    def test_undecodable_reply_exits_1(self):
        for body, reason in ((b"{not json", "Expecting property name"),
                             (b"", "Expecting value"),
                             (b"\xff\xfe\xfa", "codec can't decode")):
            with self.subTest(body=body):
                self.assert_unreadable(body, reason)

    def test_network_failure_exits_1(self):
        body = io.BytesIO(b"oops")
        http_500 = urllib.error.HTTPError(
            self.URL, 500, "Internal Server Error", {}, body)
        self.addCleanup(http_500.close)
        for exc in (
                http_500,
                urllib.error.URLError("[Errno -2] Name or service not known"),
                TimeoutError("timed out"),
                ConnectionResetError("reset by peer"),
                http.client.IncompleteRead(b"{", 10),
                http.client.InvalidURL("nonnumeric port: 'x'"),
                # Python 3.11 getaddrinfo on a port above C long.
                OverflowError("Python int too large to convert to C long")):
            with self.subTest(exc=type(exc).__name__):
                self.assert_unreadable(exc, str(exc))
        self.assertTrue(body.closed)

    def test_too_deep_reply_exits_1(self):
        self.assert_unreadable(DEEP_JSON.encode("ascii"),
                               "while decoding a JSON", small_stack=True)

    def test_value_too_deep_to_print_is_reported(self):
        # Python 3.14 decodes this on the pinned stack but cannot repr it;
        # Python 3.11 cannot decode it. Either way, no traceback.
        deep = '{"a":' * 25_000 + "1" + "}" * 25_000
        body = json.dumps(dict(self.good, schema_sha256="DEEP")).replace(
            '"DEEP"', deep).encode("ascii")
        code, out, err, _ = run_with_small_stack(
            lambda: self.run_check(body))
        self.assertIn(code, (1, 2), out + err)
        self.assertEqual(out, "")
        self.assertTrue(err.startswith(
            "[schema-identity] STALE DEPLOY" if code == 2
            else "[schema-identity] cannot read"), err)

    def test_unprintable_live_value_exits_2(self):
        # The same handler as above, independent of the stack size.
        class Unprintable:
            def __str__(self):
                raise RecursionError("too deep")

        with mock.patch.object(self.check, "live", return_value=dict(
                self.good, schema_sha256=Unprintable())):
            code, out, err, _ = self.run_check(b"{}")
        self.assertEqual(code, 2, out + err)
        self.assertIn(
            "schema_sha256: expected="
            f"{self.good['schema_sha256']} "
            "live=<Unprintable nested too deeply to print>", err)

    def test_recursion_error_while_reading_exits_1(self):
        # The same handler as above, independent of the stack size.
        with mock.patch.object(self.check, "live",
                               side_effect=RecursionError("too deep")):
            self.assert_unreadable(b"{}", "too deep")


class ReadApiSummaryShape(unittest.TestCase):
    """The read API serves FLAT fields; extraction must use them.

    Verbatim shape of one record from GET /v1/results on 2026-09-03,
    submitted by m1-test-host-linux. Reading only the nested archive payload
    printed "mesa=unknown mlx-omarchy=unknown" for a record that plainly
    carried Honeykrisp and the wheel version.
    """

    RECORD = {
        "content_sha256": "75ab0797" + "9" * 56,
        "schema_version": 1,
        "kind": "quick",
        "generated_at": "2026-09-03T16:44:00Z",
        "arch": "aarch64",
        "model": "Apple MacBook Pro (13-inch, M1, 2020)",
        "chip": "apple,t8103",
        "kernel": "7.1.6-1-1-ARCH",
        "mesa_driver": "Honeykrisp",
        "mesa_device": "Apple M1 (G13G B1)",
        "mlx_version": "0.32.2.dev202609030512",
        "mlx_device": "Device(gpu, 0)",
    }

    def test_mesa_driver_is_reported(self):
        self.assertEqual(query.record_mesa(self.RECORD), "Honeykrisp")

    def test_mlx_version_is_reported(self):
        self.assertEqual(query.record_mlx_version(self.RECORD),
                         "0.32.2.dev202609030512")

    def test_chip_and_kernel_are_reported(self):
        self.assertEqual(query.record_chip(self.RECORD), "apple,t8103")
        self.assertEqual(query.record_kernel(self.RECORD), "7.1.6-1-1-ARCH")

    def test_mesa_filter_matches_driver_and_device(self):
        blob = query.record_mesa_blob(self.RECORD)
        self.assertIn("honeykrisp", blob)
        self.assertIn("apple m1", blob)

    def _matches_chip(self, record, chip):
        filters = query.build_parser().parse_args(["--chip", chip, "list"])
        return query.record_matches(record, filters)

    def test_chip_filter_matches_flat_mesa_device(self):
        mesa_only = {k: v for k, v in self.RECORD.items() if k != "model"}
        for rec in (self.RECORD, mesa_only):
            self.assertTrue(self._matches_chip(rec, "M1"))
            self.assertFalse(self._matches_chip(rec, "M2"))
            self.assertFalse(self._matches_chip(rec, "M3"))

    def test_chip_filter_matches_flat_model_without_mesa_device(self):
        base = {
            "content_sha256": "86bc1808" + "8" * 56,
            "schema_version": 1,
            "kind": "quick",
            "arch": "aarch64",
            "chip": "apple,t6001",
            "model": "Apple MacBook Pro (16-inch, M1 Max, 2021)",
        }
        for rec in (base, dict(base, mesa_device=None),
                    dict(base, mesa_device="")):
            self.assertTrue(self._matches_chip(rec, "M1"))
            self.assertTrue(self._matches_chip(rec, "M1 Max"))
            self.assertTrue(self._matches_chip(rec, "Apple M1"))
            self.assertFalse(self._matches_chip(rec, "M2"))
            self.assertFalse(self._matches_chip(rec, "M3"))

    def test_chip_filter_ignores_the_machine_name_in_model(self):
        base_m1 = {k: v for k, v in self.RECORD.items()
                   if k != "mesa_device"}
        self.assertTrue(self._matches_chip(base_m1, "M1"))
        self.assertFalse(self._matches_chip(base_m1, "Pro"))
        self.assertFalse(self._matches_chip(base_m1, "MacBook"))

    def test_chip_filter_reads_model_chip_only_on_apple_machines(self):
        board = {"content_sha256": "c3d4" + "2" * 60, "kind": "quick",
                 "chip": "rockchip,rk3399", "model": "FriendlyElec NanoPi M4"}
        self.assertTrue(self._matches_chip(board, "rk3399"))
        self.assertFalse(self._matches_chip(board, "M4"))

    def test_nested_payload_still_works(self):
        nested = {
            "content_sha256": "b7e34c02" + "a" * 56,
            "payload": {"mesa": {"gpu": {"driverName": "Honeykrisp",
                                         "driverVersion": "26.1.7"}},
                        "mlx": {"distributions": {"mlx-omarchy": "0.3.2"}}},
        }
        self.assertEqual(query.record_mesa(nested), "Honeykrisp 26.1.7")
        self.assertEqual(query.record_mlx_version(nested), "0.3.2")


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), \
            contextlib.redirect_stderr(err):
        rc = query.main(list(argv))
    return rc, out.getvalue(), err.getvalue()


class RemoteShowTests(unittest.TestCase):
    """--source remote show resolves hash prefixes via latest.jsonl."""

    BASE_URL = "https://example.invalid"
    DATASET_URL = f"{BASE_URL}/v1/dataset/latest.jsonl"
    RESULTS_URL = f"{BASE_URL}/v1/results/"

    def make_fetcher(self, dataset_records, results_by_sha):
        calls = []
        dataset_bytes = (
            "\n".join(json.dumps(r) for r in dataset_records) + "\n"
        ).encode("utf-8")

        def fake_fetch(url, timeout=60):
            calls.append(url)
            if url == self.DATASET_URL:
                return dataset_bytes
            if url.startswith(self.RESULTS_URL):
                sha = url[len(self.RESULTS_URL):]
                if sha in results_by_sha:
                    payload = results_by_sha[sha]
                    if isinstance(payload, bytes):
                        return payload
                    return json.dumps(payload).encode("utf-8")
            raise query.DatasetError(f"HTTP 404 from {url}")

        return calls, fake_fetch

    def run_remote(self, fake_fetch, *argv):
        with mock.patch.object(query, "fetch_url", side_effect=fake_fetch):
            return run_cli(
                "--source", "remote", "--base-url", self.BASE_URL, *argv)

    def test_remote_show_unique_prefix_resolves_and_fetches_full_sha(self):
        summary = {"content_sha256": SHA_DEEP_M1, "kind": "deep",
                   "chip": "Apple M1"}
        full = dict(deep_m1_record(), archive=None)
        calls, fake_fetch = self.make_fetcher(
            [summary, quick_m2_record()], {SHA_DEEP_M1: full})
        rc, out, err = self.run_remote(fake_fetch, "show", SHA_DEEP_M1[:12])
        self.assertEqual(rc, 0, err)
        self.assertEqual(err, "")
        self.assertEqual(calls, [
            self.DATASET_URL,
            self.RESULTS_URL + SHA_DEEP_M1,
        ])
        self.assertEqual(json.loads(out), full)

    def test_remote_show_full_sha_fetches_directly_and_propagates_404(self):
        fresh_sha = "c" * 64
        missing_sha = "d" * 64
        full = {"content_sha256": fresh_sha, "kind": "quick", "chip": "M3"}
        calls, fake_fetch = self.make_fetcher(
            [deep_m1_record()], {fresh_sha: full})
        rc, out, err = self.run_remote(fake_fetch, "show", fresh_sha)
        self.assertEqual(rc, 0, err)
        self.assertEqual(calls, [self.RESULTS_URL + fresh_sha])
        self.assertEqual(json.loads(out), full)

        calls.clear()
        rc, out, err = self.run_remote(fake_fetch, "show", missing_sha)
        self.assertEqual(rc, 1)
        self.assertEqual(calls, [self.RESULTS_URL + missing_sha])
        self.assertIn("HTTP 404", err)
        self.assertNotIn("Traceback", err)

    def test_remote_show_missing_prefix_errors_without_result_fetch(self):
        calls, fake_fetch = self.make_fetcher(
            [deep_m1_record(), quick_m2_record()], {})
        rc, out, err = self.run_remote(fake_fetch, "show", "ffffffff")
        self.assertEqual(rc, 1)
        self.assertEqual(out, "")
        self.assertIn("no record with hash prefix", err)
        self.assertNotIn("Traceback", err)
        self.assertEqual(calls, [self.DATASET_URL])

    def test_remote_show_ambiguous_prefix_errors_without_result_fetch(self):
        sha_a = "abcd1234" + "0" * 56
        sha_b = "abcd1234" + "1" * 56
        records = [
            {"content_sha256": sha_a, "kind": "quick", "chip": "Apple M1"},
            {"content_sha256": sha_b, "kind": "quick", "chip": "Apple M2"},
        ]
        calls, fake_fetch = self.make_fetcher(records, {})
        rc, out, err = self.run_remote(fake_fetch, "show", "abcd1234")
        self.assertEqual(rc, 1)
        self.assertEqual(out, "")
        self.assertIn("is ambiguous (2 records)", err)
        self.assertNotIn("Traceback", err)
        self.assertEqual(calls, [self.DATASET_URL])

    def test_remote_show_prefix_applies_cli_filters(self):
        sha_m1 = "abcd1234" + "0" * 56
        sha_m2 = "abcd1234" + "1" * 56
        rec_m1 = {"content_sha256": sha_m1, "kind": "quick",
                  "chip": "Apple M1"}
        rec_m2 = {"content_sha256": sha_m2, "kind": "quick",
                  "chip": "Apple M2"}
        calls, fake_fetch = self.make_fetcher(
            [rec_m1, rec_m2], {sha_m1: rec_m1, sha_m2: rec_m2})
        rc, out, err = self.run_remote(
            fake_fetch, "--chip", "M2", "show", "abcd1234")
        self.assertEqual(rc, 0, err)
        self.assertEqual(calls, [
            self.DATASET_URL,
            self.RESULTS_URL + sha_m2,
        ])
        self.assertEqual(json.loads(out), rec_m2)

        calls.clear()
        rc, out, err = self.run_remote(
            fake_fetch, "--chip", "M3", "show", "abcd1234")
        self.assertEqual(rc, 1)
        self.assertIn("no record with hash prefix", err)
        self.assertEqual(calls, [self.DATASET_URL])

    def test_remote_show_malformed_matched_sha_errors_cleanly(self):
        bad_record = {"content_sha256": "bad12345/../evil", "kind": "quick"}
        calls, fake_fetch = self.make_fetcher([bad_record], {})
        rc, out, err = self.run_remote(fake_fetch, "show", "bad12345")
        self.assertEqual(rc, 1)
        self.assertEqual(out, "")
        self.assertIn("invalid sha256", err)
        self.assertNotIn("Traceback", err)
        self.assertEqual(calls, [self.DATASET_URL])

    def test_remote_show_prefix_applies_phandle_correction(self):
        swapped_sha = sorted(query._PHANDLE_BYTESWAP_ROWS)[0]
        summary = {"content_sha256": swapped_sha, "kind": "quick"}
        full = {
            "content_sha256": swapped_sha,
            "kind": "quick",
            "summary": {
                "ane_port_detail": {
                    "macos": {
                        "ane_nodes": [{"phandle": 0x69010000}],
                        "dart_nodes": [{"phandle": 0x2a000000}],
                    },
                },
            },
        }
        calls, fake_fetch = self.make_fetcher([summary], {swapped_sha: full})
        rc, out, err = self.run_remote(fake_fetch, "show", swapped_sha[:12])
        self.assertEqual(rc, 0)
        self.assertEqual(calls, [
            self.DATASET_URL,
            self.RESULTS_URL + swapped_sha,
        ])
        parsed = json.loads(out)
        macos = parsed["summary"]["ane_port_detail"]["macos"]
        self.assertEqual(macos["ane_nodes"][0]["phandle"], 0x169)
        self.assertEqual(macos["dart_nodes"][0]["phandle"], 0x2A)
        self.assertIn("read-side correction", parsed["_read_side_correction"])
        self.assertIn("read-side correction", err)

    def test_remote_show_too_deeply_nested_payload_errors_cleanly(self):
        calls, fake_fetch = self.make_fetcher(
            [], {SHA_DEEP_M1: DEEP_JSON.encode("ascii")})
        rc, out, err = run_with_small_stack(
            lambda: self.run_remote(fake_fetch, "show", SHA_DEEP_M1))
        self.assertEqual(rc, 1)
        self.assertEqual(out, "")
        self.assertIn("invalid JSON", err)
        self.assertEqual(calls, [self.RESULTS_URL + SHA_DEEP_M1])

    def test_remote_show_malformed_result_payload_errors_cleanly(self):
        summary = {"content_sha256": SHA_DEEP_M1, "kind": "deep"}

        calls, fake_fetch = self.make_fetcher(
            [summary], {SHA_DEEP_M1: b"not-json"})
        rc, out, err = self.run_remote(fake_fetch, "show", SHA_DEEP_M1[:12])
        self.assertEqual(rc, 1)
        self.assertEqual(out, "")
        self.assertIn("invalid JSON", err)
        self.assertNotIn("Traceback", err)
        self.assertEqual(calls, [
            self.DATASET_URL,
            self.RESULTS_URL + SHA_DEEP_M1,
        ])

        calls, fake_fetch = self.make_fetcher(
            [summary], {SHA_DEEP_M1: b"[]"})
        rc, out, err = self.run_remote(fake_fetch, "show", SHA_DEEP_M1)
        self.assertEqual(rc, 1)
        self.assertEqual(out, "")
        self.assertIn("expected JSON object", err)
        self.assertNotIn("Traceback", err)
        self.assertEqual(calls, [self.RESULTS_URL + SHA_DEEP_M1])

        calls, fake_fetch = self.make_fetcher(
            [summary], {SHA_DEEP_M1: b"\xff\xfe"})
        rc, out, err = self.run_remote(fake_fetch, "show", SHA_DEEP_M1)
        self.assertEqual(rc, 1)
        self.assertEqual(out, "")
        self.assertIn("invalid JSON", err)
        self.assertNotIn("Traceback", err)
        self.assertEqual(calls, [self.RESULTS_URL + SHA_DEEP_M1])

        embedded_bad_utf8 = (
            b'{"content_sha256":"' + SHA_DEEP_M1.encode("ascii")
            + b'","kind":"quick","note":"\xff"}'
        )
        calls, fake_fetch = self.make_fetcher(
            [summary], {SHA_DEEP_M1: embedded_bad_utf8})
        rc, out, err = self.run_remote(fake_fetch, "show", SHA_DEEP_M1)
        self.assertEqual(rc, 1)
        self.assertEqual(out, "")
        self.assertIn("invalid JSON", err)
        self.assertNotIn("Traceback", err)
        self.assertEqual(calls, [self.RESULTS_URL + SHA_DEEP_M1])


class Hung(BaseException):
    """Raised by the watchdog; a BaseException, so no handler swallows it."""


@contextlib.contextmanager
def watchdog(seconds=5):
    def fire(signum, frame):
        raise Hung(f"blocked for {seconds} s")
    previous = signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def sparse(path, size):
    with open(path, "wb") as stream:
        stream.truncate(size)


class ReadCappedTests(unittest.TestCase):
    def test_short_reads_are_joined_and_never_pass_cap_plus_one(self):
        asked = []

        def chunky(data):
            stream = io.BytesIO(data)

            def read(n):
                asked.append(n)
                return stream.read(min(n, 3))
            return read

        self.assertEqual(query.read_capped(chunky(b"abcdefghij"), 10),
                         b"abcdefghij")
        asked.clear()
        self.assertEqual(query.read_capped(chunky(b"x" * 100), 10),
                         b"x" * 11)
        self.assertEqual(asked, [11, 8, 5, 2])
        # Reads that land exactly on the cap still take one more byte.
        self.assertEqual(query.read_capped(chunky(b"x" * 100), 9), b"x" * 10)
        self.assertEqual(query.read_capped(lambda n: None, 10), b"")


class LocalSnapshotBoundsTests(unittest.TestCase):
    """latest.jsonl is read only as a regular file of at most 32 MiB."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.path = self.dir / "latest.jsonl"

    def list_local(self):
        return run_cli("--source", "local", "--snapshot", str(self.dir),
                       "--json", "list")

    def assert_refused(self, rc, out, err, message):
        self.assertEqual(rc, 1)
        self.assertEqual(out, "")
        self.assertIn(message, err)
        self.assertNotIn("Traceback", err)

    def test_file_of_exactly_the_cap_loads(self):
        line = json.dumps(quick_m2_record()).encode("utf-8") + b"\n"
        with open(self.path, "wb") as stream:
            stream.write(line)
            stream.write(b" " * (query.MAX_JSONL_BYTES - len(line) - 1))
            stream.write(b"\n")
        self.assertEqual(self.path.stat().st_size, query.MAX_JSONL_BYTES)
        rc, out, err = self.list_local()
        self.assertEqual(rc, 0, err)
        self.assertEqual(json.loads(out)["count"], 1)

    def test_one_byte_over_the_cap_is_refused_unread(self):
        sparse(self.path, query.MAX_JSONL_BYTES + 1)
        with mock.patch.object(query, "read_capped",
                               side_effect=AssertionError("read")):
            rc, out, err = self.list_local()
        self.assert_refused(rc, out, err, "is larger than 33554432 bytes")

    def test_the_read_stops_at_cap_plus_one_when_the_size_lies(self):
        self.path.write_bytes(b"x" * 100)
        lengths = []
        real_read_capped, real_fstat = query.read_capped, os.fstat

        def spy(read, max_bytes):
            data = real_read_capped(read, max_bytes)
            lengths.append(len(data))
            return data

        def small_fstat(fd):
            info = real_fstat(fd)
            return mock.Mock(st_mode=info.st_mode, st_size=0)

        with mock.patch.object(query, "MAX_JSONL_BYTES", 10), \
                mock.patch.object(query, "read_capped", spy), \
                mock.patch.object(query.os, "fstat", small_fstat):
            rc, out, err = self.list_local()
        self.assert_refused(rc, out, err, "is larger than 10 bytes")
        self.assertEqual(lengths, [11])

    def test_fifo_swapped_in_after_the_type_check_is_refused(self):
        os.mkfifo(self.path)
        for writer in (False, True):
            with self.subTest(writer=writer):
                fd = os.open(self.path, os.O_RDWR) if writer else None
                try:
                    with watchdog(), mock.patch.object(
                            Path, "is_file", return_value=True):
                        rc, out, err = self.list_local()
                finally:
                    if fd is not None:
                        os.close(fd)
                self.assert_refused(rc, out, err, "is not a regular file")

    def test_invalid_utf8_is_replaced_not_fatal(self):
        self.path.write_bytes(snapshot_bytes()
                              + b'{"kind": "deep", "chip": "M\xff"}\n')
        rc, out, err = self.list_local()
        self.assertEqual(rc, 0, err)
        data = json.loads(out)
        self.assertEqual((data["count"], data["skipped_malformed"]), (5, 1))
        self.assertEqual(data["results"][-1]["chip"], "M\ufffd")

    def test_symlink_to_a_regular_file_loads(self):
        real = self.dir / "real.jsonl"
        real.write_bytes(snapshot_bytes())
        self.path.symlink_to(real)
        rc, out, err = self.list_local()
        self.assertEqual(rc, 0, err)
        self.assertEqual(json.loads(out)["count"], 4)

    @unittest.skipIf(os.geteuid() == 0, "root reads any file")
    def test_unreadable_file_is_refused(self):
        self.path.write_bytes(snapshot_bytes())
        self.path.chmod(0)
        self.assert_refused(*self.list_local(), "Permission denied")

    def test_environment_snapshot_over_the_cap_is_refused(self):
        sparse(self.path, query.MAX_JSONL_BYTES + 1)
        with mock.patch.dict(os.environ,
                             {"MLX_OMARCHY_DATA_DIR": str(self.dir)}):
            rc, out, err = run_cli("--source", "local", "list")
        self.assert_refused(rc, out, err, "is larger than 33554432 bytes")


class GitSnapshotBoundsTests(unittest.TestCase):
    """The community-data branch rung reads at most 32 MiB too."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = Path(tmp.name)
        git = ["git", "-C", str(self.repo), "-c", "user.name=t",
               "-c", "user.email=t@example.invalid"]
        for argv in (["init", "-q", "-b", "community-data"],
                     ["commit", "-q", "--allow-empty", "-m", "empty"]):
            subprocess.run(git + argv, check=True, capture_output=True)
        (self.repo / "snapshot").mkdir()
        (self.repo / "snapshot" / "latest.jsonl").write_bytes(
            snapshot_bytes())
        for argv in (["add", "snapshot"], ["commit", "-q", "-m", "snap"]):
            subprocess.run(git + argv, check=True, capture_output=True)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("MLX_OMARCHY_DATA_DIR", None)

    def load(self):
        return query.load_local(mock.Mock(snapshot=None), self.repo)

    def test_branch_snapshot_loads_the_blob_it_checked(self):
        calls = []
        real_run = subprocess.run

        def run(argv, *args, **kwargs):
            calls.append(argv)
            return real_run(argv, *args, **kwargs)

        with mock.patch.object(query.subprocess, "run", run):
            records, skipped, source = self.load()
        self.assertEqual((len(records), skipped), (4, 1))
        self.assertEqual(source, "git community-data:snapshot/latest.jsonl")
        oid = subprocess.run(
            ["git", "-C", str(self.repo), "rev-parse",
             "community-data:snapshot/latest.jsonl"],
            check=True, capture_output=True, text=True).stdout.strip()
        self.assertEqual(calls[-1], ["git", "cat-file", "blob", oid])

    def test_branch_snapshot_over_the_cap_is_refused_unread(self):
        size = len(snapshot_bytes())
        calls = []
        real_run = subprocess.run

        def run(argv, *args, **kwargs):
            calls.append(argv[:3])
            return real_run(argv, *args, **kwargs)

        with mock.patch.object(query, "MAX_JSONL_BYTES", size - 1), \
                mock.patch.object(query.subprocess, "run", run), \
                self.assertRaises(query.DatasetError) as caught:
            self.load()
        self.assertIn(f"larger than {size - 1} bytes", str(caught.exception))
        self.assertEqual(calls, [["git", "cat-file", "--batch-check"]] * 2)
        with mock.patch.object(query, "MAX_JSONL_BYTES", size):
            self.assertEqual(len(self.load()[0]), 4)

    def test_path_that_is_not_a_blob_is_skipped(self):
        path = self.repo / "snapshot" / "latest.jsonl"
        self.assertIsNone(query.git_blob(self.repo, "community-data:snapshot"))
        self.assertIsNone(query.git_blob(self.repo, "community-data:missing"))
        self.assertEqual(query.git_blob(
            self.repo, "community-data:snapshot/latest.jsonl"),
            path.read_bytes())


class Loopback:
    """HTTP server on 127.0.0.1 that sends finite scripted replies.

    routes maps a path to (head, body): head is the status line and
    headers, with BASE replaced by the server's URL; body is bytes, or
    an int for that many b"x" bytes. sent[path] counts the body bytes
    that left the socket before the client hung up.
    """

    def __init__(self, routes):
        self.routes, self.sent, self.threads = routes, {}, []
        self.stopped = threading.Event()
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.listener.settimeout(0.1)
        self.base = "http://127.0.0.1:%d" % self.listener.getsockname()[1]
        self.acceptor = threading.Thread(target=self._accept, daemon=True)
        self.acceptor.start()

    def _accept(self):
        while not self.stopped.is_set():
            try:
                conn, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            thread = threading.Thread(target=self._serve, args=(conn,),
                                      daemon=True)
            self.threads.append(thread)
            thread.start()

    def _serve(self, conn):
        with conn:
            conn.settimeout(10)
            request = b""
            while b"\r\n\r\n" not in request and len(request) < 65536:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                request += chunk
            path = request.split(b" ", 2)[1].decode("ascii")
            head, body = self.routes[path]
            sent = 0
            try:
                conn.sendall(head.replace("BASE", self.base)
                             .encode("latin-1") + b"\r\n")
                if isinstance(body, bytes):
                    conn.sendall(body)
                    sent = len(body)
                while isinstance(body, int) and sent < body:
                    block = min(1 << 16, body - sent)
                    conn.sendall(b"x" * block)
                    sent += block
            except OSError:
                pass
            self.sent[path] = sent

    def close(self):
        self.stopped.set()
        self.acceptor.join(10)
        self.listener.close()
        for thread in self.threads:
            thread.join(15)


BIG = 64 * 1024 * 1024  # finite, and far more than socket buffers hold


class FetchUrlBoundsTests(unittest.TestCase):
    """fetch_url reads at most max_bytes + 1 bytes of one 200 response."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ,
                                  {"no_proxy": "*", "NO_PROXY": "*"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def serve(self, routes):
        server = Loopback(routes)
        self.addCleanup(server.close)
        return server

    def refused(self, url, message, max_bytes=1024):
        with self.assertRaises(query.DatasetError) as caught:
            query.fetch_url(url, timeout=10, max_bytes=max_bytes)
        self.assertIn(message, str(caught.exception))
        self.assertIn(url, str(caught.exception))

    def test_body_up_to_the_cap_is_returned(self):
        server = self.serve({
            "/full": ("HTTP/1.1 200 OK\r\nContent-Length: 1024\r\n",
                      b"y" * 1024),
            "/zeros": ("HTTP/1.1 200 OK\r\nContent-Length: " + "0" * 30
                       + "5\r\n", b"hello"),
            "/space": ("HTTP/1.1 200 OK\r\nContent-Length: 5 \t\r\n",
                       b"hello"),
        })
        self.assertEqual(query.fetch_url(server.base + "/full", timeout=10,
                                         max_bytes=1024), b"y" * 1024)
        for path in ("/zeros", "/space"):
            with self.subTest(path=path):
                self.assertEqual(query.fetch_url(
                    server.base + path, timeout=10, max_bytes=1024),
                    b"hello")

    def test_redirect_is_followed_and_its_body_left_unread(self):
        server = self.serve({
            "/start": ("HTTP/1.1 302 Found\r\nLocation: BASE/final\r\n"
                       f"Content-Length: {BIG}\r\n", BIG),
            "/final": ("HTTP/1.1 200 OK\r\nContent-Length: 2\r\n", b"ok"),
        })
        self.assertEqual(query.fetch_url(server.base + "/start", timeout=10),
                         b"ok")
        server.close()
        self.assertLess(server.sent["/start"], BIG // 2)

    def test_content_length_over_the_cap_is_refused_unread(self):
        server = self.serve({
            "/big": (f"HTTP/1.1 200 OK\r\nContent-Length: {BIG}\r\n", BIG),
        })
        self.refused(server.base + "/big", "larger than 1024 bytes")
        server.close()
        self.assertLess(server.sent["/big"], BIG // 2)

    def test_body_over_the_cap_without_a_length_stops_at_the_cap(self):
        server = self.serve({
            "/stream": ("HTTP/1.1 200 OK\r\nConnection: close\r\n", BIG),
        })
        self.refused(server.base + "/stream", "larger than 1024 bytes")
        server.close()
        self.assertLess(server.sent["/stream"], BIG // 2)

    def test_malformed_content_length_is_refused(self):
        # http.client cannot parse a value past 4,300 digits, even one
        # that is mostly leading zeros, and would read to the close.
        routes = {f"/{i}": (f"HTTP/1.1 200 OK\r\nContent-Length: {value}\r\n",
                            b"hello")
                  for i, value in enumerate((
                      "+5", "-5", "5 5", "0x5", "5_0", "\u00b2", "5.0", "",
                      "1" * 5000, "0" * 5000 + "100",
                      "5\r\nContent-Length: 5",
                      "5\r\nContent-Length: 999",
                      "5\r\nTransfer-Encoding: chunked"))}
        server = self.serve(routes)
        for path, (head, _) in routes.items():
            with self.subTest(head=head):
                self.refused(server.base + path, "invalid Content-Length")

    def test_body_cut_short_is_refused(self):
        server = self.serve({
            "/cut": ("HTTP/1.1 200 OK\r\nContent-Length: 100\r\n", b"y" * 40),
            "/chunk": ("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n",
                       b"5\r\nhel"),
        })
        self.refused(server.base + "/cut", "ended 60 bytes short")
        self.refused(server.base + "/chunk", "bad HTTP response")

    def test_malformed_status_or_headers_are_refused(self):
        server = self.serve({
            "/garbage": ("GARBAGE\r\n", b""),
            "/headers": ("HTTP/1.1 200 OK\r\n"
                         + "X-Pad: 1\r\n" * 150, b""),
        })
        self.refused(server.base + "/garbage", "BadStatusLine")
        self.refused(server.base + "/headers", "more than 100 headers")

    def test_status_other_than_200_is_refused(self):
        server = self.serve({
            "/partial": ("HTTP/1.1 206 Partial Content\r\n"
                         "Content-Length: 5\r\n", b"hello"),
            "/empty": ("HTTP/1.1 204 No Content\r\n", b""),
        })
        self.refused(server.base + "/partial", "HTTP 206")
        self.refused(server.base + "/empty", "HTTP 204")

    def test_http_error_is_closed(self):
        url = "https://example.invalid/v1/dataset/latest.jsonl"
        body = io.BytesIO(b"not found")
        error = urllib.error.HTTPError(url, 404, "Not Found", {}, body)
        with mock.patch.object(query, "urlopen", side_effect=error):
            self.refused(url, "HTTP 404")
        self.assertTrue(body.closed)

    def test_remote_list_uses_the_32_mib_cap(self):
        server = self.serve({
            "/v1/dataset/latest.jsonl": (
                f"HTTP/1.1 200 OK\r\nContent-Length: {BIG}\r\n", BIG),
        })
        rc, out, err = run_cli("--source", "remote", "--base-url",
                               server.base, "list")
        self.assertEqual(rc, 1)
        self.assertEqual(out, "")
        self.assertIn("larger than 33554432 bytes from "
                      f"{server.base}/v1/dataset/latest.jsonl", err)
        self.assertNotIn("Traceback", err)


if __name__ == "__main__":
    unittest.main()
