"""check-dtbs-override.sh flags a non-empty DTBS= in update-m1n1 defaults and stays quiet otherwise."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check-dtbs-override.sh"

AURORA_SEP_STANZA = """\
# aurora-sep: build m1n1's stage 2 from the device trees the installed
# linux-aurora package owns, which carry the Touch ID sensor node.
#
# Not from a module directory whose pkgbase says linux-aurora: right after an
# upgrade the running kernel's modules are restored unowned but still carry that
# pkgbase, so both kernels' DTBs would be bundled. m1n1 keeps the last matching
# DTB, and the glob sorts 11.10 before 11.9, so the stale one can win.
# (update-m1n1 runs under set -e, so this assignment must always succeed.)
DTBS=$(pacman -Qlq linux-aurora 2>/dev/null | grep '/dtbs/[^/]*\\.dtb$'; true)
"""


def run(config):
    return subprocess.run(["bash", str(SCRIPT), str(config)],
                          capture_output=True, text=True)


class DtbsOverrideDetectionTests(unittest.TestCase):
    def write(self, root, text):
        path = root / "update-m1n1"
        path.write_text(text)
        return path

    def test_aurora_sep_stanza_warns(self):
        with tempfile.TemporaryDirectory() as temp:
            r = run(self.write(Path(temp), AURORA_SEP_STANZA))
        self.assertEqual(r.returncode, 1)
        self.assertIn("WARNING", r.stdout)
        self.assertIn("silently ignored", r.stdout)
        self.assertIn("DTBS=$(pacman -Qlq linux-aurora", r.stdout)

    def test_plain_and_command_substitution_lists_warn(self):
        for text in ('DTBS="/boot/dtbs/a.dtb /boot/dtbs/b.dtb"\n',
                     "DTBS=$(tr '\\n' ' ' < /tmp/list)\n",
                     "M1N1=\nexport DTBS=/boot/dtbs/x.dtb\n",
                     "  DTBS=/boot/dtbs/x.dtb  \n"):
            with tempfile.TemporaryDirectory() as temp:
                r = run(self.write(Path(temp), text))
            self.assertEqual(r.returncode, 1, text)
            self.assertIn("WARNING", r.stdout)

    def test_no_override_stays_silent(self):
        for text in (AURORA_SEP_STANZA.replace(
                         "DTBS=$(pacman -Qlq linux-aurora 2>/dev/null | "
                         "grep '/dtbs/[^/]*\\.dtb$'; true)", "DTBS="),
                     'DTBS=""\nM1N1=\n',
                     "M1N1=m1n1\nU_BOOT=/boot/u-boot.bin\n",
                     "# DTBS=/boot/dtbs/x.dtb\n",
                     ""):
            with tempfile.TemporaryDirectory() as temp:
                r = run(self.write(Path(temp), text))
            self.assertEqual(r.returncode, 0, repr(text))
            self.assertNotIn("WARNING", r.stdout)

    def test_missing_config_is_no_override(self):
        with tempfile.TemporaryDirectory() as temp:
            r = run(Path(temp) / "absent")
        self.assertEqual(r.returncode, 0)
        self.assertIn("no DTBS override", r.stdout)

    def test_unreadable_config_is_an_error(self):
        with tempfile.TemporaryDirectory() as temp:
            r = run(temp)
        self.assertEqual(r.returncode, 2)

    def test_usage_error(self):
        r = subprocess.run(["bash", str(SCRIPT), "a", "b"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)

    def test_docs_reference_the_script(self):
        text = (ROOT / "docs/gpu-base-pstate.md").read_text()
        self.assertIn("scripts/check-dtbs-override.sh", text)
        self.assertIn("If DTBS is set", text)


if __name__ == "__main__":
    unittest.main()
