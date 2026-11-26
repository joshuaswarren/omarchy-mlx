import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "release-gates"))

from voice_status import primer_failure_line  # noqa: E402


def voice(missing=(), detail=None, usable=False, state="missing"):
    return {
        "usable": usable,
        "state": state,
        "dependencies": {"present": [], "missing": list(missing), "detail": detail or {}},
    }


class PrimerFailureLineTests(unittest.TestCase):
    def test_names_a_missing_package(self):
        line = primer_failure_line(voice(missing=["mlx_audio"]))
        self.assertIn("dependencies.missing=['mlx_audio']", line)
        self.assertIn("usable=False", line)
        self.assertIn("state=missing", line)

    def test_names_a_pin_conflict_that_leaves_missing_empty(self):
        detail = {
            "misaki": {"version": "0.8.0", "constraint": "==0.7.4", "satisfies": False},
            "num2words": {"version": "0.5.14", "constraint": "==0.5.14", "satisfies": True},
        }
        line = primer_failure_line(voice(detail=detail))
        self.assertIn("conflicts=['misaki 0.8.0 violates ==0.7.4']", line)
        self.assertNotIn("num2words", line)

    def test_a_package_that_failed_to_import_is_missing_not_a_conflict(self):
        detail = {"spacy": "ImportError: no module named spacy"}
        line = primer_failure_line(voice(missing=["spacy"], detail=detail))
        self.assertIn("dependencies.missing=['spacy']", line)
        self.assertIn("conflicts=[]", line)

    def test_does_not_print_a_reasons_field(self):
        self.assertNotIn("reasons", primer_failure_line(voice(missing=["mlx_audio"])))


if __name__ == "__main__":
    unittest.main()
