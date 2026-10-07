"""Tests for mlx_omarchy_bonsai2.validate_artifact (positive + negative).

The unified serve CLI (mlx_omarchy_serve.__main__.module_artifact_problem)
calls `getattr(importlib.import_module("mlx_omarchy_bonsai2.server"),
"validate_artifact")(model_dir)` before admit-and-launch. Bonsai packs
ARE the upstream artifact -- there is no .convert checkpoint_state
classifier and no conversion step -- so the hook MUST exist on the
server module (where the CLI imports from) and reuse pack_footprint
(config.json + safetensors header, ZERO tensor bytes).

Coverage:
  - positive: real Bonsai pack (tiny fixture) -> None
  - negative: empty dir, missing config.json, missing model.safetensors,
    wrong schema, safetensors with no language_model.* tensors,
    nonexistent path -> informative error string

REPO/serve is mandatory. We do NOT skipWhen our own packages fail to
import -- that is a regression, not a host-safety net. An import
failure here makes the test FAIL with a clear assertion, so CI sees
the breakage immediately.
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

# REPO/serve is mandatory. Assert the module file actually exists in
# the checkout before importing -- a missing file is a regression, not
# a skip case.
_BONSAI_SERVER_PY = SERVE / "mlx_omarchy_bonsai2" / "server.py"
assert _BONSAI_SERVER_PY.is_file(), (
    "mlx_omarchy_bonsai2.server missing at %s -- checkout is broken; "
    "this test must FAIL, not skip." % _BONSAI_SERVER_PY
)

for p in (str(SERVE), str(TESTS)):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Import our own package -- no skipUnless. If this raises, the test
# collection fails loudly (test count = 0 with collection error), which
# is the correct signal: a missing/import-broken mlx_omarchy_bonsai2
# in REPO/serve is a REGRESSION.
from mlx_omarchy_bonsai2.server import validate_artifact  # noqa: E402


@unittest.skipUnless(importlib.util.find_spec("mlx"), "needs mlx (validation loads the pack with it)")
class ValidateArtifactPositiveTests(unittest.TestCase):
    """A real Bonsai pack must validate to None."""

    def test_tiny_fixture_validates(self):
        """The tiny fixture pack from bonsai2_fixture is a fully-built
        prism_hadamard_qwen35 schema-2 pack with a fake vision tensor;
        validate_artifact must accept it."""
        import bonsai2_fixture  # noqa: F401

        with tempfile.TemporaryDirectory() as tmp:
            pack_dir, _ = bonsai2_fixture.build_tiny_pack(Path(tmp))
            result = validate_artifact(pack_dir)
            self.assertIsNone(
                result,
                "expected validate_artifact to return None for a built tiny fixture, got %r"
                % result,
            )


class ValidateArtifactNegativeTests(unittest.TestCase):
    """A non-pack directory or an invalid pack must return an error string
    that the CLI can surface verbatim."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_empty_directory_fails(self):
        result = validate_artifact(self.tmp)
        self.assertIsNotNone(result, "empty directory must fail")
        self.assertIn("missing config.json", result)

    def test_directory_missing_model_safetensors_fails(self):
        (self.tmp / "config.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "model_type": "prism_hadamard_qwen35",
                    "text_config": {"num_hidden_layers": 2},
                }
            )
        )
        result = validate_artifact(self.tmp)
        self.assertIsNotNone(result)
        self.assertIn("missing model.safetensors", result)

    def test_wrong_schema_fails(self):
        """A config.json with the wrong model_type is rejected by
        pack_footprint's _check_config and surfaced verbatim."""
        (self.tmp / "config.json").write_text(
            json.dumps(
                {
                    "schema_version": 99,  # not 2
                    "model_type": "wrong_arch",
                    "text_config": {},
                }
            )
        )
        (self.tmp / "model.safetensors").write_bytes(b"")
        result = validate_artifact(self.tmp)
        self.assertIsNotNone(result)
        self.assertIn("pack validation failed", result)

    def test_safetensors_without_language_model_prefix_fails(self):
        """A safetensors header with no language_model.* tensors means
        the pack has no servable text model -- validator must refuse."""
        (self.tmp / "config.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "model_type": "prism_hadamard_qwen35",
                    "text_config": {"num_hidden_layers": 2},
                }
            )
        )
        (self.tmp / "model.safetensors").write_bytes(b"\x00" * 8)
        result = validate_artifact(self.tmp)
        self.assertIsNotNone(result)
        self.assertTrue(
            "language_model" in result
            or "pack validation failed" in result
            or "header" in result.lower(),
            "expected a validation error mentioning language_model / "
            "pack validation / header, got %r" % result,
        )

    def test_nonexistent_directory_fails(self):
        """A path that doesn't exist must return a clear error, not crash."""
        result = validate_artifact(self.tmp / "does-not-exist")
        self.assertIsNotNone(result)
        self.assertIn("not a directory", result)


if __name__ == "__main__":
    unittest.main()
