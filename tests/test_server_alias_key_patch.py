"""Regression tests for the mlx-lm server --alias/--api-key patch wiring.

w7Q's review of joshuaswarren/omarchy-mlx#932 caught two failure classes:

- the patch name sat in SERIES_PATCHES but no `apply` call ever passed the
  file to patch(1), so the patch shipped inert;
- the handler saved the alias as `self.alias` but the models route read
  `self.server.alias`, which the handler instance does not carry.

Each test pins one of those against the real files.
"""

import pathlib
import textwrap
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


class BearerComparisonIsConstantTime(unittest.TestCase):
    def test_both_series_compare_with_hmac_compare_digest(self):
        # w7Q review of #932: a plain == on the bearer header leaks the key
        # length/prefix through early-exit timing. The patches must compare
        # the full header bytes with hmac.compare_digest.
        for pf in PATCH_FILES:
            with self.subTest(patch=str(pf.relative_to(REPO))):
                text = pf.read_text()
                self.assertIn("+        return hmac.compare_digest(", text)
                self.assertNotIn('== f"Bearer {self.api_key}"', text)


class ApiKeyDefaultParsesWithoutNameError(unittest.TestCase):
    def test_the_0_31_series_imports_os_at_module_level_for_the_default(self):
        # w7b 9ae64f9e59: the 0.31.3 series now adds a module-level `import os`
        # hunk, so the --api-key default may read bare `os.environ.get(...)` -
        # the name exists at argparse time. The 0.32 series keeps the inline
        # `import os as _os` from the #932 review; both must build the default.
        import argparse
        import os
        from unittest import mock

        for pf in PATCH_FILES:
            with self.subTest(patch=str(pf.relative_to(REPO))):
                text = pf.read_text()
                rel = str(pf.relative_to(REPO))
                if "mlx-lm-0.32" not in rel:
                    # 0.31 series (9ae64f9e59): module-level import os rides
                    # along so a later env-read default cannot regress to the
                    # jw16 NameError; today the key arrives as a parameter.
                    self.assertIn("\n+import os\n", text)
                    self.assertNotIn("default=os.environ.get(", text)
                else:
                    # 0.32 series: inline `import os as _os` from the #932
                    # review carries the env default.
                    self.assertIn("import os as _os", text)
                    self.assertIn('default=_os.environ.get("MLX_API_KEY", "")', text)
                if "MLX_API_KEY" in text:
                    # 0.32 series: the env-read hunk adds --alias and --api-key
                    # together; exec it against a bare namespace that mirrors
                    # unpatched 0.31.3 (no ambient `os`) and build the defaults.
                    key_hunk = next(h for h in text.split("@@") if "MLX_API_KEY" in h)
                    added = "\n".join(
                        line[1:]
                        for line in key_hunk.splitlines()
                        if line.startswith("+") and not line.startswith("+++")
                    )
                    parser = argparse.ArgumentParser()
                    ns = {"parser": parser}
                    with mock.patch.dict(os.environ, {"MLX_API_KEY": "secret-123"}, clear=False):
                        exec(compile(textwrap.dedent(added), str(pf), "exec"), ns)
                    parsed = parser.parse_args(["--alias", "compact"])
                    self.assertEqual(parsed.alias, "compact")
                    self.assertEqual(parsed.api_key, "secret-123")


class PatchedServerModuleImportsAndBuildsDefaultArgs(unittest.TestCase):
    """Main 1433 order: a test that imports the patched module and builds the
    default args - the exact surface that crashed jw16 chat workers with
    NameError: name 'os' is not defined at startup."""

    def test_the_patched_0_31_server_imports_and_survives_help(self):
        import importlib.util
        import os
        import shutil
        import subprocess
        import sys
        import tempfile

        override = os.environ.get("MLX_LM_SITE_PACKAGES")
        if override:
            installed = pathlib.Path(override)
        else:
            try:
                import mlx_lm  # noqa: F401
            except ImportError:
                self.skipTest("mlx_lm is not installed in the running interpreter")
            spec = importlib.util.find_spec("mlx_lm")
            if spec is None or spec.submodule_search_locations is None:
                self.skipTest("cannot locate the installed mlx_lm package")
            installed = pathlib.Path(list(spec.submodule_search_locations)[0])
        if "MLX_API_KEY" not in (installed / "server.py").read_text():
            self.skipTest("this mlx_lm/server.py predates the alias/key patch "
                          "(set MLX_LM_SITE_PACKAGES to the patched tree)")

        with tempfile.TemporaryDirectory() as tmp:
            venv = pathlib.Path(tmp) / "venv"
            site = venv / "lib" / "python3.14" / "site-packages"
            site.mkdir(parents=True)
            shutil.copytree(installed, site / "mlx_lm")
            out = subprocess.run(
                ["bash", str(REPO / "scripts" / "apply-mlx-lm-patches.sh"), str(venv)],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(out.returncode, 0, out.stderr)

            patched = (site / "mlx_lm" / "server.py").read_text()
            self.assertIn("\nimport os\n", patched)

            probe = (
                "import sys; sys.path.insert(0, %r); sys.argv = ['server', '--help'];\n"
                "import mlx_lm.server as s; s.main()" % (str(site),)
            )
            out = subprocess.run([sys.executable, "-c", probe],
                                 capture_output=True, text=True, timeout=120)
            self.assertNotIn("NameError", out.stderr)
            self.assertEqual(out.returncode, 0, out.stderr)


def _quiet_rmtree(root, ignore):
    import shutil

    shutil.rmtree(root, ignore_errors=ignore)


if __name__ == "__main__":
    unittest.main()
