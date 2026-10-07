"""End-to-end integration test: CLI module_artifact_problem resolves a
real Bonsai pack via the validate_artifact hook.

This is the positive path of the original integration bug: before the
validate_artifact hook landed on agent/bonsai2-serving, the CLI fell
into the generic 'no validator' branch and refused every valid Bonsai
pack with a misleading 'no validate_artifact hook and no
<pkg>.convert.checkpoint_state classifier' message. With the hook on
the server module (where the CLI imports from), the CLI returns None
for a real pack and never reaches the convert-classifier fallback.

The CLI is part of the SAME unified checkout -- this test asserts the
actual CLI module file exists in REPO/serve (the unified integration
path) and fails when it does not. If the agent/bonsai2-serving branch
does not yet carry the CLI, this test FAILS (not skip) with a clear
assertion pointing at the missing file.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SERVE = REPO / "serve"
TESTS = Path(__file__).resolve().parent

# OUR package source in REPO/serve is mandatory. Assert both module
# files exist in the checkout before importing. A missing file is a
# regression, not a skip case.
_BONSAI_SERVER_PY = SERVE / "mlx_omarchy_bonsai2" / "server.py"
_CLI_MAIN_PY = SERVE / "mlx_omarchy_serve" / "__main__.py"
assert _BONSAI_SERVER_PY.is_file(), (
    "mlx_omarchy_bonsai2.server missing at %s -- checkout is broken; "
    "this test must FAIL, not skip." % _BONSAI_SERVER_PY
)
assert _CLI_MAIN_PY.is_file(), (
    "mlx_omarchy_serve.__main__ missing at %s -- the CLI is part of the "
    "unified checkout; if this branch does not carry the CLI yet, the "
    "test fails until both packages are co-located." % _CLI_MAIN_PY
)

for p in (str(SERVE), str(TESTS)):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Import our own packages -- no skipUnless. A missing/import-broken
# package in REPO/serve is a regression; the test collection fails
# loudly.
from mlx_omarchy_serve.__main__ import module_artifact_problem  # noqa: E402


@unittest.skipUnless(importlib.util.find_spec("mlx"), "needs mlx (the Bonsai pack loader imports it)")
class CliArtifactIntegrationTests(unittest.TestCase):
    """The unified CLI must resolve a real Bonsai pack via the
    validate_artifact hook on the server module."""

    def setUp(self):
        import bonsai2_fixture  # noqa: F401

        self.tmp = Path(tempfile.mkdtemp())
        self.pack_dir, _ = bonsai2_fixture.build_tiny_pack(self.tmp / "pack")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cli_resolves_bonsai_pack_via_validate_artifact(self):
        """The CLI's module_artifact_problem MUST return None for a real
        Bonsai pack. The order in the CLI is validate_artifact FIRST;
        only if it is missing does the CLI fall back to <pkg>.convert.
        Bonsai has no convert module and no conversion step;
        validate_artifact is the contract."""
        result = module_artifact_problem("mlx_omarchy_bonsai2.server", self.pack_dir)
        self.assertIsNone(
            result,
            "CLI module_artifact_problem must return None for a real Bonsai pack, "
            "got %r. The validate_artifact hook is the contract; "
            "MODULE_CONVERT_HINTS and <pkg>.convert are fallbacks that should "
            "never be reached for Bonsai packs." % result,
        )

    def test_cli_rejects_invalid_bonsai_artifact(self):
        """The CLI must surface a clear error for a directory that is
        not a Bonsai pack. validate_artifact returns a string; the CLI
        propagates it verbatim."""
        empty = self.tmp / "empty-pack"
        empty.mkdir()
        result = module_artifact_problem("mlx_omarchy_bonsai2.server", empty)
        self.assertIsNotNone(
            result, "empty dir must produce an error string, got None"
        )
        # The error must reference the missing required file; not be a
        # generic "no validator" message.
        self.assertIn("config.json", result)


if __name__ == "__main__":
    unittest.main()
