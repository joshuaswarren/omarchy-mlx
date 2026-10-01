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


if __name__ == "__main__":
    unittest.main()
