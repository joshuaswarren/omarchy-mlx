"""Tests for the community-data query CLI and the mirror workflow.

Stdlib unittest only, no network. Fixtures are built in a temp
directory and shaped like the real collector payloads.
"""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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

    def test_nested_payload_still_works(self):
        nested = {
            "content_sha256": "b7e34c02" + "a" * 56,
            "payload": {"mesa": {"gpu": {"driverName": "Honeykrisp",
                                         "driverVersion": "26.1.7"}},
                        "mlx": {"distributions": {"mlx-omarchy": "0.3.2"}}},
        }
        self.assertEqual(query.record_mesa(nested), "Honeykrisp 26.1.7")
        self.assertEqual(query.record_mlx_version(nested), "0.3.2")




if __name__ == "__main__":
    unittest.main()
