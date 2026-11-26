"""The census scripts read kernel names from compute.h by line. An enumerator that carries a trailing comment (a retired id that stays, append-only)
must still be counted, or every later name shifts by one."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HEADER = (
    "enum class ComputeKernel : uint16_t {\n"
    "  Alpha,\n"
    "  Retired, // retired: no shader; the id stays\n"
    "  // a comment line with a brace }\n"
    "  Gamma,\n"
    "  Count,\n"
    "};\n")


def load(script):
    path = REPO_ROOT / "scripts" / script
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KernelEnumParserTests(unittest.TestCase):
    def check(self, script, function):
        with tempfile.TemporaryDirectory() as td:
            header = Path(td) / "compute.h"
            header.write_text(HEADER, encoding="utf-8")
            names = getattr(load(script), function)(str(header))
        self.assertEqual(names, ["Alpha", "Retired", "Gamma"])

    def test_chain_census(self):
        self.check("chain_census.py", "kernel_names")

    def test_sdpa_counts(self):
        self.check("sdpa_counts.py", "kernel_names")

    def test_kv_state_views(self):
        self.check("kv_state_views.py", "kernel_names")


if __name__ == "__main__":
    unittest.main()
