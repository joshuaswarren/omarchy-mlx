"""scripts/adt_ane_names.py on a synthetic ADT file.

The helper wraps aurora's `aurora-adt-extract.py`, which this repo does not carry. Point
AURORA_ADT_READER at a copy (sha256 3563a3bb...) to run these; they skip without it.
"""

import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "adt_ane_names.py"
READER = os.environ.get("AURORA_ADT_READER", "")


def prop(name, value):
    raw = name.encode().ljust(32, b"\0") + struct.pack("<I", len(value)) + value
    return raw + b"\0" * (-len(value) % 4)


def node(name, props, children=()):
    props = [("name", name.encode() + b"\0")] + list(props)
    body = b"".join(prop(n, v) for n, v in props)
    return struct.pack("<II", len(props), len(children)) + body + b"".join(children)


def adt(with_ane):
    ane = [node("ane", [("compatible", b"apple,ane\0"), ("serial-number", b"C02ABCDEF123\0")])] if with_ane else []
    arm_io = node("arm-io", [("compatible", b"arm-io,t6031\0")], ane + [node("uart0", [])])
    return node("device-tree", [("serial-number", b"C02ABCDEF123\0")], [arm_io])


@unittest.skipUnless(READER and Path(READER).is_file(), "AURORA_ADT_READER not set")
class AdtAneNamesTests(unittest.TestCase):
    def run_script(self, data, *extra):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adt.bin"
            path.write_bytes(data)
            return subprocess.run([sys.executable, str(SCRIPT), "--reader", READER, *extra, str(path)],
                                  capture_output=True, text=True)

    def test_ane_node_is_printed_without_the_serial(self):
        done = self.run_script(adt(with_ane=True))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("/arm-io/ane", done.stdout)
        self.assertIn("apple,ane", done.stdout)
        self.assertIn("ane node in /arm-io: yes", done.stdout)
        self.assertNotIn("C02ABCDEF123", done.stdout)
        self.assertIn("#   uart0", done.stdout)

    def test_absent_ane_node_is_reported_as_absent(self):
        done = self.run_script(adt(with_ane=False))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("ane node in /arm-io: no", done.stdout)
        self.assertNotIn("/arm-io/ane", done.stdout)
        self.assertNotIn("C02ABCDEF123", done.stdout)

    def test_wrong_reader_hash_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "reader.py"
            fake.write_text("print('not the reader')\n")
            done = subprocess.run([sys.executable, str(SCRIPT), "--reader", str(fake), "--check"],
                                  capture_output=True, text=True)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("not the pinned", done.stderr)


if __name__ == "__main__":
    unittest.main()
