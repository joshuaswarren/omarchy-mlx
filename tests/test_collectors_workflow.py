"""collectors.yml starts for the code trees that its steps run or read.

A PR that touched only serve/ (#753) started no collectors job, so the
serve and assistant tests in tests/ never ran on it.
"""

import shlex
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "collectors.yml"

# Whole trees, so one narrowed entry fails here rather than in a skipped PR.
REQUIRED_PATHS = {
    "requirements-dev.txt",
    "patches/**",
    "scripts/**",
    "serve/**",
    "tools/**",
    "overlay/**",
    "harness/**",
    "tests/**",
    "qualification-kit/**",
    ".github/workflows/community-data.yml",
    ".github/workflows/serve-catalog-refresh.yml",
}


class CollectorsWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        cls.triggers = data.get("on", data.get(True))
        cls.qkit_steps = [
            s["run"] for s in data["jobs"]["test"]["steps"]
            if s.get("run", "").startswith("python -m pytest -q qualification-kit/")]

    def test_code_trees_trigger_the_workflow(self):
        for event in ("pull_request", "push"):
            with self.subTest(event=event):
                missing = REQUIRED_PATHS - set(self.triggers[event]["paths"])
                self.assertFalse(missing, f"{event} misses {sorted(missing)}")

    def test_qualification_kit_step_runs_the_cpu_tests(self):
        self.assertEqual(len(self.qkit_steps), 1)
        targets = {a for a in shlex.split(self.qkit_steps[0])
                   if a.startswith("qualification-kit/")}
        cpu_tests = {f"qualification-kit/{t.name}"
                     for t in (ROOT / "qualification-kit").glob("test_*.py")}
        self.assertTrue(cpu_tests)
        self.assertEqual(targets, cpu_tests)


if __name__ == "__main__":
    unittest.main()
