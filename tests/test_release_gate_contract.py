import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION_ASSIGNMENT = re.compile(
    r"\bMLX_OMARCHY_PAIR_DEV_QUALIFICATION\s*=\s*['\"]?1\b"
)

# These are isolated model-measurement tools, not release build or gate runners.
DEV_MEASUREMENT_SCRIPTS = {
    "card_defect_capture.py",
    "idle_quality27b_perf.sh",
    "pair_memory_turns.py",
    "pair_memory_v2.py",
    "quality27b_perf_api.py",
    "ticket_card_timing.sh",
    "ticket_memory_turns.sh",
    "ticket_memory_v2.sh",
    "ticket_quality27b_perf_api.sh",
}


class ReleaseGateContractTests(unittest.TestCase):
    def test_scripts_and_release_runbooks_do_not_enable_dev_qualification(self):
        scripts = ROOT / "scripts"
        checked = [
            path
            for path in scripts.rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".sh"}
            and path.relative_to(scripts).as_posix() not in DEV_MEASUREMENT_SCRIPTS
        ]
        checked.extend(
            path
            for path in (ROOT / "receipts").glob("*-release.md")
            if path.is_file()
        )

        violations = [
            f"{path.relative_to(ROOT)}"
            for path in checked
            if QUALIFICATION_ASSIGNMENT.search(path.read_text(errors="replace"))
        ]
        self.assertEqual(violations, [])


class ReleaseGateHarnessTests(unittest.TestCase):
    """Contract for scripts/release-gates/ (the draft-first gate battery)."""

    HARNESS = ROOT / "scripts" / "release-gates"
    EXPECTED_FILES = (
        "README.md",
        "env.sh",
        "run-all.sh",
        "gate-probe.py",
        "gate3-card-runner.py",
        "g8-kokoro-driver.py",
        "g1-clean-install.sh",
        "g2-online-9b.sh",
        "g3-online-4b-card.sh",
        "g4-offline.sh",
        "g5-laya.sh",
        "g6-codec.sh",
        "g7a-packaged-icd.sh",
        "g7b-system-install.sh",
        "g7c-ane-worker-verify.sh",
        "g7d-fresh-transcribe.sh",
    )
    # Real fleet identities must never be committed into the harness; every
    # host-specific value arrives via env with placeholder defaults.
    FORBIDDEN = (
        "jw14m2", "jw16mbp1", "16m1mbp", "macstudio",
        "/home/joshuawarren", "joshuawarren@",
        "192.168.", re.compile(r"\b100\.\d+\.\d+\.\d+\b"),
        "omarchy-mplus-private",
    )
    RUN_ORDER = (
        "g1-clean-install", "g2-online-9b", "g3-online-4b-card", "g4-offline",
        "g5-laya", "g6-codec", "g7a-packaged-icd", "g7b-system-install",
        "g8-kokoro", "g7c-ane-worker-verify", "g7d-fresh-transcribe",
    )
    GOLDEN_TRANSCRIPT_SHA = (
        "db501a8c080380ea027ffa50a4b4956c39df77cb692c4fb78e556311a11a0790"
    )

    def test_harness_files_present(self):
        missing = [n for n in self.EXPECTED_FILES
                   if not (self.HARNESS / n).is_file()]
        self.assertEqual(missing, [])

    def test_harness_commits_no_real_hosts_or_paths(self):
        violations = []
        for path in self.HARNESS.iterdir():
            if not path.is_file():
                continue
            text = path.read_text(errors="replace")
            for marker in self.FORBIDDEN:
                found = marker.findall(text) if hasattr(marker, "findall") \
                    else ([marker] if marker in text else [])
                if found:
                    violations.append(f"{path.name}: {marker}")
        self.assertEqual(violations, [])

    def test_env_defaults_are_placeholders(self):
        env = (self.HARNESS / "env.sh").read_text()
        self.assertIn("M2_SSH_ALIAS", env)
        self.assertIn("JW16_SSH_ALIAS", env)
        self.assertNotIn("joshuaswarren@macstudio", env)

    def test_run_all_executes_gates_in_order(self):
        run_all = (self.HARNESS / "run-all.sh").read_text()
        positions = [run_all.find(name) for name in self.RUN_ORDER]
        self.assertEqual([p for p in positions if p == -1], [])
        self.assertEqual(positions, sorted(positions))
        # The jw16 leg must refuse to silently skip: a partial battery
        # must fail the run.
        self.assertIn("RUN_JW16", run_all)
        self.assertIn("SKIPPED", run_all)

    def test_g7d_pins_golden_transcript_and_failure_string(self):
        g7d = (self.HARNESS / "g7d-fresh-transcribe.sh").read_text()
        self.assertIn(self.GOLDEN_TRANSCRIPT_SHA, g7d)
        self.assertIn("missing runtime dependencies for transcribe", g7d)
        # The defect surface is the user-style entry: shebang exec AND the
        # bare `python3 -S` form must both be exercised.
        self.assertIn('"$PY_SYS" -S', g7d)
        self.assertIn("GOLDEN_MATCH", g7d)
        self.assertIn("MLX_OMARCHY_CACHE_DIR", g7d)

    def test_g7d_cpu_mode_stops_after_dependency_probe(self):
        g7d = (self.HARNESS / "g7d-fresh-transcribe.sh").read_text()
        self.assertIn("G7D_MODE", g7d)
        self.assertIn("DEP_BOUNDARY_CROSSED", g7d)

    def test_g7c_guards_the_serving_venv(self):
        g7c = (self.HARNESS / "g7c-ane-worker-verify.sh").read_text()
        self.assertIn("SERVING_VENV", g7c)
        self.assertIn("REFUSE", g7c)


if __name__ == "__main__":
    unittest.main()
