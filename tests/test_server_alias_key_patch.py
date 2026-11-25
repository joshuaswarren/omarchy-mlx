"""Regression tests for the mlx-lm server --alias/--api-key patch wiring.

w7Q's review of joshuaswarren/omarchy-mlx#932 caught two failure classes:

- the patch name sat in SERIES_PATCHES but no `apply` call ever passed the
  file to patch(1), so the patch shipped inert;
- the handler saved the alias as `self.alias` but the models route read
  `self.server.alias`, which the handler instance does not carry.

Each test pins one of those against the real files.
"""

import pathlib
import unittest

from tests.test_apply_mlx_lm_patches import make_fakebin, run_script

REPO = pathlib.Path(__file__).resolve().parent.parent

PATCH_FILES = (
    REPO / "patches" / "mlx-lm-server-alias-key.patch",
    REPO / "patches" / "mlx-lm-0.32" / "mlx-lm-server-alias-key.patch",
)


class SeriesAppliesAliasPatch(unittest.TestCase):
    def test_apply_call_passes_the_alias_patch_to_patch_1(self):
        # A working patch(1) must make the script report the alias patch as
        # applied; before the fix the name only sat in SERIES_PATCHES and
        # this line never appeared. python3 is an exit-0 no-op so the
        # mid-series python patchers cannot abort before the alias call.
        root = pathlib.Path(__file__).resolve().parent / "_alias_patch_fixture"
        root.mkdir(exist_ok=True)
        self.addCleanup(_quiet_rmtree, root, True)
        fakebin = make_fakebin(root, "#!/bin/sh\nexit 0\n")
        (fakebin / "python3").unlink()
        (fakebin / "python3").write_text("#!/bin/sh\nexit 0\n")
        (fakebin / "python3").chmod(0o700)
        out = run_script(root / "case", fakebin)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("applied: mlx-lm-server-alias-key.patch", out.stdout)


class AliasReadsHandlerAttribute(unittest.TestCase):
    def test_both_series_read_self_alias_not_self_server_alias(self):
        for pf in PATCH_FILES:
            with self.subTest(patch=str(pf.relative_to(REPO))):
                text = pf.read_text()
                self.assertIn("+        self.alias = alias", text)
                self.assertIn("+        if self.alias:", text)
                self.assertNotIn("self.server.alias", text)


def _quiet_rmtree(root, ignore):
    import shutil

    shutil.rmtree(root, ignore_errors=ignore)


if __name__ == "__main__":
    unittest.main()
