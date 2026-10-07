"""Regression tests for apply-mlx-lm-patches.sh's patch(1) handling.

The M2 gate window of 2026-10-07 caught the failure class: a host without
patch(1) made every probe fail with exit 127 and the script reported a
misleading "mlx-lm version mismatch". These tests pin the contract:

- a missing patch binary exits 6 with a clear install hint;
- a PRESENT but failing patch (rc 1) reports the real mismatch, and never
  the missing-binary message;
- a working patch reports "applied: <name>".

Each case runs the real script in a fixture tree with a constructed PATH, so
no mlx-lm wheel and no network are needed.
"""

import os
import pathlib
import shutil
import stat
import subprocess
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "apply-mlx-lm-patches.sh"

SHADOW_TOOLS = ("dirname", "ls", "grep", "python3")


def make_fakebin(root: pathlib.Path, fake_patch: str | None) -> pathlib.Path:
    """A bin dir whose PATH search sees only what we put there."""
    fakebin = root / "bin"
    fakebin.mkdir(parents=True)
    for tool in SHADOW_TOOLS:
        src = shutil.which(tool)
        if src:
            link = fakebin / tool
            link.symlink_to(src)
    if fake_patch is not None:
        patch = fakebin / "patch"
        patch.write_text(fake_patch)
        patch.chmod(stat.S_IRWXU)
    return fakebin


def make_fixture(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """venv fixture with a mlx_lm tree (0.31.3 marker: no StopSequences) and
    a patches/ dir copied from the repo."""
    site = root / "venv" / "lib" / "python3.14" / "site-packages" / "mlx_lm"
    (site / "models").mkdir(parents=True)
    (site / "generate.py").write_text("# fixture: no StopSequences class\n")
    patches = root / "patches"
    patches.mkdir()
    for p in sorted((REPO / "patches").glob("*.patch")):
        shutil.copy(p, patches / p.name)
    shutil.copy(REPO / "scripts" / "apply-mlx-lm-patches.sh", root / "apply.sh")
    return root / "venv", root / "apply.sh"


def run_script(case_root: pathlib.Path, fakebin: pathlib.Path, path_only: bool = False):
    venv, apply_sh = make_fixture(case_root)
    env = dict(os.environ)
    env["PATH"] = str(fakebin) if path_only else f"{fakebin}:/usr/bin:/bin"
    env.pop("MLX_OMARCHY_CONV_RING", None)
    return subprocess.run(
        ["/usr/bin/bash", str(apply_sh), str(venv)],
        capture_output=True, text=True, env=env, timeout=120,
    )


class PatchBinaryContract(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(__file__).resolve().parent / "_apply_patches_fixture"
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_missing_patch_binary_exits_6_with_install_hint(self):
        # PATH holds an empty bin dir: patch is nowhere to be found, and the
        # up-front check must fire before anything else touches the tree.
        fakebin = make_fakebin(self.root / "case1", fake_patch=None)
        out = run_script(self.root / "case1", fakebin, path_only=True)
        self.assertEqual(out.returncode, 6)
        self.assertIn("patch is not installed", out.stderr)
        self.assertNotIn("version mismatch", out.stderr)

    def test_failing_patch_reports_mismatch_not_missing_binary(self):
        # A patch that exists and rejects the tree (rc 1): the REAL mismatch.
        fake = "#!/bin/sh\nexit 1\n"
        fakebin = make_fakebin(self.root / "case2", fake_patch=fake)
        out = run_script(self.root / "case2", fakebin)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("mlx-lm version mismatch", out.stderr)
        self.assertNotIn("patch is not installed", out.stderr)

    def test_working_patch_reports_applied(self):
        fake = (
            "#!/bin/sh\n"
            'for a in "$@"; do [ "$a" = "--dry-run" ] && exit 0; done\n'
            "exit 0\n"
        )
        fakebin = make_fakebin(self.root / "case3", fake_patch=fake)
        out = run_script(self.root / "case3", fakebin)
        self.assertIn("applied: mlx-lm-tool-call-arguments.patch", out.stdout)


if __name__ == "__main__":
    unittest.main()
