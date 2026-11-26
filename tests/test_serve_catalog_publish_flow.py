"""Flow tests for tools/serve_catalog_publish.sh (local git fixture).

Simulates consecutive CI runs: first run creates the branch + PR, second
run with nothing new is a retry-safe no-op, third run with fresh drift
updates the branch FROM CURRENT MAIN (never a stale remote base) and does
not re-create the PR. gh is a stub on PATH that logs its calls, so the
explicit PR query and loud gh failures stay testable offline.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "tools" / "serve_catalog_publish.sh"

BRANCH = "serve-catalog-refresh"


class PublishFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.remote = self.tmp / "remote.git"
        self.work = self.tmp / "work"
        self.gh_ok = self.tmp / "gh-ok"
        self.gh_fail = self.tmp / "gh-fail-bin"
        self.gh_log = self.tmp / "gh-calls.log"
        for d in (self.gh_ok, self.gh_fail):
            d.mkdir()
        (self.gh_ok / "gh").write_text(
            "#!/bin/sh\necho \"gh $*\" >> %s\n"
            "state=%s/pr-state\n"
            "case \"$*\" in *'pr create'*) echo created > \"$state\";; "
            "*'pr list'*) if [ -f \"$state\" ]; then echo 1; else echo 0; fi;; esac\n"
            % (self.gh_log, self.tmp))
        (self.gh_ok / "gh").chmod(0o755)
        (self.gh_fail / "gh").write_text(
            "#!/bin/sh\necho 'gh: auth expired' >&2\nexit 1\n")
        (self.gh_fail / "gh").chmod(0o755)

        r = lambda *cmd: subprocess.run(list(cmd), check=True, capture_output=True)
        r("git", "init", "--bare", "-q", str(self.remote))
        r("git", "init", "-q", "-b", "main", str(self.work))
        cfg = ["git", "-C", str(self.work)]
        # This test exercises git flow, not the host's identity-guard
        # hooks: point the fixture repo at an empty hooks dir so commits
        # with the CI bot identity are not refused by machine policy.
        hooks = self.tmp / "empty-hooks"
        hooks.mkdir()
        r(*cfg, "config", "core.hooksPath", str(hooks))
        r(*cfg, "config", "user.name", "tester")
        r(*cfg, "config", "user.email", "tester@example.invalid")
        r(*cfg, "remote", "add", "origin", str(self.remote))
        (self.work / "serve").mkdir()
        self.write_catalog("seed")
        r(*cfg, "add", "-A")
        r(*cfg, "commit", "-q", "-m", "base")
        r(*cfg, "push", "-q", "origin", "main")

    def write_catalog(self, body):
        (self.work / "serve" / "catalog.json").write_text(
            '{"seed": true, "v": "%s"}\n' % body, encoding="utf-8")

    def branch_catalog(self):
        out = subprocess.run(
            ["git", "-C", str(self.work), "show",
             "origin/%s:serve/catalog.json" % BRANCH],
            capture_output=True, text=True, check=True)
        return out.stdout

    def work_catalog(self):
        return (self.work / "serve" / "catalog.json").read_text(encoding="utf-8")

    def run_publish(self, gh=None):
        bin_dir = str(self.gh_fail if gh == "fail" else self.gh_ok)
        env = dict(os.environ, GH_TOKEN="t", PATH=bin_dir + os.pathsep + os.environ["PATH"])
        return subprocess.run(
            ["sh", str(SCRIPT), "serve/catalog.json", BRANCH, "main"],
            cwd=str(self.work), env=env, capture_output=True, text=True, timeout=60)

    def test_first_run_creates_branch_commit_and_pr(self):
        self.write_catalog("r1")
        proc = self.run_publish()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.branch_catalog(), self.work_catalog())
        calls = self.gh_log.read_text()
        self.assertIn("pr list", calls)
        self.assertIn("pr create", calls)

    def test_retry_after_failed_pr_still_ensures_pr(self):
        self.write_catalog("r1")
        proc = self.run_publish(gh="fail")
        self.assertNotEqual(proc.returncode, 0)  # loud failure
        self.assertIn("auth expired", proc.stderr)
        # branch was pushed before the PR step failed
        self.assertEqual(self.branch_catalog(), self.work_catalog())
        proc = self.run_publish()  # rerun with the healthy gh stub
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("pr create", self.gh_log.read_text())

    def test_second_run_with_no_new_drift_is_retry_safe_noop(self):
        self.write_catalog("r1")
        self.run_publish()
        proc = self.run_publish()  # sizes unchanged, no local diff
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("pr list", self.gh_log.read_text())  # PR ensured, not duplicated
        self.assertEqual(self.gh_log.read_text().count("pr create"), 1)

    def test_commit_succeeds_without_any_preexisting_git_identity(self):
        # Fresh GitHub Actions checkouts have NO git identity. The script
        # must configure the bot identity BEFORE the first commit; this
        # fixture strips every identity source to prove it.
        self.write_catalog("r1")
        cfg = ["git", "-C", str(self.work)]
        for key in ("user.name", "user.email"):
            subprocess.run(cfg + ["config", "--unset-all", key],
                           capture_output=True)
        env = dict(os.environ,
                   GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null",
                   GH_TOKEN="t",
                   PATH=str(self.gh_ok) + os.pathsep + os.environ["PATH"])
        proc = subprocess.run(
            ["sh", str(SCRIPT), "serve/catalog.json", BRANCH, "main"],
            cwd=str(self.work), env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        author = subprocess.run(
            ["git", "-C", str(self.work), "log", "-1", "--format=%an %ae",
             BRANCH], capture_output=True, text=True, check=True).stdout
        self.assertIn("github-actions[bot]", author)

    def test_fresh_drift_rebases_branch_on_current_main(self):
        self.write_catalog("r1")
        self.run_publish()
        subprocess.run(["git", "-C", str(self.work), "commit", "-q",
                        "--allow-empty", "-m", "main moved"], check=True)
        self.write_catalog("r2")
        proc = self.run_publish()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        merge_base = subprocess.run(
            ["git", "-C", str(self.work), "merge-base", "main", BRANCH],
            capture_output=True, text=True, check=True).stdout.strip()
        main_head = subprocess.run(
            ["git", "-C", str(self.work), "rev-parse", "main"],
            capture_output=True, text=True, check=True).stdout.strip()
        # branch was seeded from CURRENT main, never the stale first-run base
        self.assertEqual(merge_base, main_head)
        self.assertIn('"v": "r2"', self.branch_catalog())


if __name__ == "__main__":
    unittest.main()
