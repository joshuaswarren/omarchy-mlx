"""Dev-box tests for packaging/omlx-linux/ast_import_scan.py (issue 206 item 3).

The scanner is the gate install.sh runs before an oMLX tree is accepted on
Linux (`ast_import_scan.py $OMLX_DIR/omlx --allow patches`): unguarded
module-level imports of WATCHED packages must exit 1. These tests pin the
guard rules exactly as shipped, both sides of each rule, through the
subprocess CLI (exit codes, stderr) and in-process through main().

Runs on the stdlib only, under pytest and unittest.
"""

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "packaging" / "omlx-linux" / "ast_import_scan.py"
sys.path.insert(0, str(SCRIPT.parent))

import ast_import_scan  # noqa: E402


def make_tree(files: dict[str, str]) -> Path:
    tmp = tempfile.mkdtemp(prefix="ast-scan-test-")
    root = Path(tmp)
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return root


def run_scan(root: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(root), *extra],
        capture_output=True, text=True, timeout=60,
    )


def call_main(argv: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    with mock.patch.object(sys, "argv", ["ast_import_scan.py", *argv]), \
            contextlib.redirect_stdout(out):
        rc = ast_import_scan.main()
    return rc, out.getvalue()


UNGUARDED = '"""m"""\nimport torch\n'
# import sits on line 2 in every snippet below (docstring on line 1)
TRY_EXCEPT = '"""m"""\ntry:\n    import torch\nexcept ImportError:\n    torch = None\n'
FUNC_BODY = '"""m"""\ndef load():\n    import torch\n    return torch\n'
PLATFORM_IF = '"""m"""\nimport platform\nif platform.system() == "Darwin":\n    import AppKit\n'
FIND_SPEC_IF = ('"""m"""\nimport importlib.util\n'
                'if importlib.util.find_spec("torch") is not None:\n    import torch\n')
# NOT a guard as shipped: guard_reason only accepts tests containing
# platform/find_spec/importlib/import, and "TYPE_CHECKING" contains none.
TYPE_CHECKING_IF = '"""m"""\nfrom typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import torch\n'


class ScannerCli(unittest.TestCase):
    """Subprocess runs: exit codes are the contract install.sh depends on."""

    def test_unguarded_module_import_exits_1_naming_file_and_line(self):
        root = make_tree({"mod.py": UNGUARDED})
        r = run_scan(root)
        self.assertEqual(r.returncode, 1)
        self.assertIn("UNGUARDED watched imports:", r.stdout)
        self.assertIn("mod.py:2: unguarded import of torch", r.stdout)

    def test_unguarded_import_deep_in_tree_reports_relpath(self):
        root = make_tree({"pkg/sub/mod.py": UNGUARDED})
        r = run_scan(root)
        self.assertEqual(r.returncode, 1)
        self.assertIn("pkg/sub/mod.py:2: unguarded import of torch", r.stdout)

    def test_guarded_imports_exit_0_and_report_only_with_all(self):
        root = make_tree({"m.py": TRY_EXCEPT + FUNC_BODY + PLATFORM_IF + FIND_SPEC_IF})
        r = run_scan(root)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("OK: no unguarded watched imports (4 guarded)", r.stdout)
        self.assertNotIn("guarded  ", r.stdout)  # hidden without --all
        r_all = run_scan(root, "--all")
        self.assertEqual(r_all.returncode, 0, r_all.stdout)
        for fragment in ("(try/except)", "(deferred (function body))",
                         "(guard: platform.system() == 'Darwin')",
                         "(guard: importlib.util.find_spec"):
            self.assertIn(fragment, r_all.stdout)

    def test_type_checking_if_is_not_a_guard_as_shipped(self):
        # Pins current behaviour: an `if TYPE_CHECKING:` import is reported
        # unguarded (exit 1) because guard_reason matches only platform- or
        # availability-style tests. If the rule ever grows to accept
        # TYPE_CHECKING, this test changes with it.
        root = make_tree({"tci.py": TYPE_CHECKING_IF})
        r = run_scan(root)
        self.assertEqual(r.returncode, 1)
        self.assertIn("tci.py:4: unguarded import of torch", r.stdout)

    def test_non_matching_if_is_not_a_guard(self):
        root = make_tree({"cond.py": '"""m"""\nx = 1\nif x:\n    import torch\n'})
        r = run_scan(root)
        self.assertEqual(r.returncode, 1)
        self.assertIn("cond.py:4: unguarded import of torch", r.stdout)

    def test_from_import_of_watched_package(self):
        root = make_tree({"fi.py": '"""m"""\nfrom mlx_embeddings import load\n'})
        r = run_scan(root)
        self.assertEqual(r.returncode, 1)
        self.assertIn("fi.py:2: unguarded import of mlx_embeddings", r.stdout)
        guarded = make_tree({"fi.py": '"""m"""\ndef f():\n    from mlx_embeddings import load\n'})
        self.assertEqual(run_scan(guarded).returncode, 0)

    def test_submodule_import_flags_watched_top_level_name(self):
        root = make_tree({"sub.py": '"""m"""\nimport mlx_audio.tts\n'})
        r = run_scan(root)
        self.assertEqual(r.returncode, 1)
        self.assertIn("sub.py:2: unguarded import of mlx_audio.tts", r.stdout)

    def test_non_watched_module_level_import_is_ignored(self):
        root = make_tree({"fine.py": "import os\nimport requests\n"
                                      "from collections import defaultdict\n"})
        r = run_scan(root)
        self.assertEqual(r.returncode, 0)
        self.assertIn("OK: no unguarded watched imports (0 guarded)", r.stdout)

    def test_allow_prefix_skips_that_prefix_and_only_that_prefix(self):
        root = make_tree({
            "pkg/bad.py": UNGUARDED,
            "lib/bad.py": UNGUARDED,
            "ok.py": "import json\n",
        })
        r = run_scan(root, "--allow", "pkg")
        self.assertEqual(r.returncode, 1)
        self.assertIn("lib/bad.py:2: unguarded import of torch", r.stdout)
        self.assertNotIn("pkg/bad.py", r.stdout)
        self.assertIn("scanned 2 files under", r.stdout)
        self.assertIn("(skipping pkg via --allow)", r.stdout)
        r2 = run_scan(root, "--allow", "pkg", "--allow", "lib")
        self.assertEqual(r2.returncode, 0, r2.stdout)
        self.assertIn("(skipping pkg,lib via --allow)", r2.stdout)

    def test_syntax_error_file_fails_loudly_not_silently(self):
        # As shipped, ast.parse raises: the CLI dies with a traceback and a
        # nonzero status. Pinned so the gate can never silently pass on an
        # unreadable file.
        root = make_tree({"broken.py": "def f(:\n", "ok.py": "import json\n"})
        r = run_scan(root)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("SyntaxError", r.stderr)


class ScannerInProcess(unittest.TestCase):
    """main() directly: same contract, no subprocess."""

    def test_main_returns_1_on_unguarded(self):
        root = make_tree({"mod.py": UNGUARDED})
        rc, out = call_main([str(root)])
        self.assertEqual(rc, 1)
        self.assertIn("mod.py:2: unguarded import of torch", out)

    def test_main_returns_0_on_clean_tree(self):
        root = make_tree({"ok.py": "import json\n", "g.py": FUNC_BODY})
        rc, out = call_main([str(root)])
        self.assertEqual(rc, 0)
        self.assertIn("OK: no unguarded watched imports (1 guarded)", out)

    def test_main_lists_guarded_only_with_all(self):
        root = make_tree({"g.py": TRY_EXCEPT})
        rc, out = call_main([str(root), "--all"])
        self.assertEqual(rc, 0)
        self.assertIn("guarded  g.py:3: torch (try/except)", out)

    def test_main_returns_2_without_exactly_one_root(self):
        self.assertEqual(call_main([])[0], 2)
        self.assertEqual(call_main(["/no/such", "/other"])[0], 2)


class WatchedList(unittest.TestCase):
    def test_watched_names_are_importable_style_identifiers(self):
        self.assertTrue(ast_import_scan.WATCHED)
        for name in ast_import_scan.WATCHED:
            self.assertTrue(name.isidentifier(), f"not an identifier: {name!r}")
            self.assertEqual(name.split(".")[0], name)


if __name__ == "__main__":
    unittest.main()
