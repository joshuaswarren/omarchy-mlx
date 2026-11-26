# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Issue #217 regression: every documented install path must end with the
vendored mlx-lm serve patches applied.

The metric is the issue's own: the number of .py files of the installed
mlx-lm whose sha256 differs from the wheel's RECORD. The documented paths
and their expected counts:

- a pristine wheel tree (what the release-wheel section produced before the
  fix) counts 0 -- that is the documented bug this issue fixes;
- the checkout's apply script on a fresh tree of each supported line
  (0.31.3, 0.32.0) sets the reference count for that line;
- an installed-prefix layout (what install.sh stages under $PREFIX) must
  reach the same count on both lines. The 0.32.0 leg is the regression the
  issue caught: the installer downloaded only the flat 0.31.3 series, so
  `patches/mlx-lm-0.32` was missing from the prefix (exit 5) and both the
  Muse quick start and any venv moved to the 0.32 line ran stock mlx-lm.

Offline contract cases pin the fix's plumbing: install.sh grew
--patch-venv, the patch fetch covers both series, and both documented
install paths (docs/INSTALL.md release wheel, docs/muse-spark.md quick
start) end with the --patch-venv command.

Needs network for `pip download --no-deps` of the pinned wheels; skips
those cases when pip or the network cannot provide them.
"""

import base64
import hashlib
import pathlib
import re
import shutil
import subprocess
import sys
import zipfile

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "apply-mlx-lm-patches.sh"
INSTALLER = REPO / "install.sh"
INSTALL_DOC = REPO / "docs" / "INSTALL.md"
MUSE_DOC = REPO / "docs" / "muse-spark.md"
VERSIONS = ["0.31.3", "0.32.0"]

PATCH_COMMAND = ("curl -fsSL https://raw.githubusercontent.com/joshuaswarren/"
                 "omarchy-mlx/main/install.sh | bash -s -- --patch-venv")


def _download_wheel(root: pathlib.Path, version: str) -> pathlib.Path:
    dl = root / "dl"
    r = subprocess.run(
        [sys.executable, "-m", "pip", "download", "--no-deps", "--quiet",
         "-d", str(dl), f"mlx-lm=={version}"],
        capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        pytest.skip(f"pip download mlx-lm=={version} failed, "
                    f"no pristine tree: {r.stderr.strip()[-160:]}")
    (wheel,) = dl.glob("mlx_lm-*.whl")
    return dl / wheel.name


def _pristine_site(wheel: pathlib.Path, root: pathlib.Path) -> pathlib.Path:
    """Extract the wheel into a venv-shaped tree; return its site-packages."""
    site = root / "lib" / "python3.11" / "site-packages"
    with zipfile.ZipFile(wheel) as z:
        z.extractall(site)
    return site


def _count_modified(site: pathlib.Path) -> int:
    """The issue's metric: installed mlx_lm .py files whose hash differs
    from the wheel RECORD."""
    record = next(iter(site.glob("mlx_lm-*.dist-info/RECORD")))
    modified = 0
    for line in record.read_text().splitlines():
        fields = line.split(",")
        if len(fields) < 2 or not fields[0].startswith("mlx_lm/") \
                or not fields[0].endswith(".py"):
            continue
        digest = fields[1]
        if not digest.startswith("sha256="):
            continue
        data = (site / fields[0]).read_bytes()
        got = base64.urlsafe_b64encode(hashlib.sha256(data).digest()) \
            .rstrip(b"=").decode()
        modified += got != digest.removeprefix("sha256=")
    return modified


def _run_apply(script: pathlib.Path, tree: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(script), str(tree)],
                          capture_output=True, text=True, timeout=600)


def _prefix_layout(root: pathlib.Path) -> pathlib.Path:
    """The installed-prefix layout install.sh stages under $PREFIX: the apply
    script at the top, patches/ (flat 0.31.3 names) beside it, and the whole
    patches/mlx-lm-0.32/ directory, plus every scripts/*.py helper it names."""
    prefix = root / "prefix"
    prefix.mkdir()
    shutil.copy(SCRIPT, prefix / "apply-mlx-lm-patches.sh")
    shutil.copytree(REPO / "patches", prefix / "patches")
    # patches/ also carries the core-MLX (non mlx-lm) patches the installer
    # never downloads; mirror its fetch exactly.
    for p in (prefix / "patches").glob("*.patch"):
        if not p.name.startswith("mlx-lm-"):
            p.unlink()
    (prefix / "scripts").mkdir()
    for helper in set(re.findall(r"scripts/([a-z0-9_-]+\.py)",
                                 SCRIPT.read_text())):
        shutil.copy(REPO / "scripts" / helper, prefix / "scripts" / helper)
    return prefix


@pytest.fixture(scope="module", params=VERSIONS)
def version(request):
    return request.param


@pytest.fixture(scope="module")
def wheels(tmp_path_factory, version):
    root = tmp_path_factory.mktemp(f"wheel-{version}")
    return _download_wheel(root, version)


@pytest.fixture(scope="module")
def reference_count(wheels, version, tmp_path_factory):
    """Checkout apply script on a fresh tree of this line: the reference."""
    root = tmp_path_factory.mktemp(f"ref-{version}")
    tree = root / "venv"
    _pristine_site(wheels, tree)
    r = _run_apply(SCRIPT, tree)
    detail = r.stdout[-500:] + r.stderr[-300:]
    assert r.returncode == 0, f"checkout apply failed on {version}:\n{detail}"
    count = _count_modified(tree / "lib" / "python3.11" / "site-packages")
    assert count > 0, f"checkout apply modified nothing on {version}"
    return count


def test_pristine_wheel_is_unpatched(wheels, version, tmp_path):
    """The pre-fix release-wheel path: 0 modified files. Documents the bug
    (and proves the counter can see it)."""
    site = _pristine_site(wheels, tmp_path)
    assert _count_modified(site) == 0


def test_checkout_apply_count(reference_count, version):
    assert reference_count > 0


def test_prefix_layout_reaches_the_reference_count(
        wheels, reference_count, version, tmp_path_factory):
    """install.sh's staged prefix applies both series. 0.32.0 used to exit 5
    ('patches/mlx-lm-0.32 is missing from the prefix'), which left the Muse
    quick start and any upgraded venv on stock mlx-lm."""
    root = tmp_path_factory.mktemp(f"prefix-{version}")
    prefix = _prefix_layout(root)
    tree = root / "venv"
    _pristine_site(wheels, tree)
    r = _run_apply(prefix / "apply-mlx-lm-patches.sh", tree)
    detail = r.stdout[-500:] + r.stderr[-300:]
    assert r.returncode == 0, f"prefix apply failed on {version}:\n{detail}"
    assert f"mlx-lm patch series: {'patches/mlx-lm-0.32' if version == '0.32.0' else 'patches'}" in r.stdout
    count = _count_modified(tree / "lib" / "python3.11" / "site-packages")
    assert count == reference_count, (
        f"prefix layout patched {version} to {count} files, checkout "
        f"reference is {reference_count}")
    record = (tree / "lib" / "python3.11" / "site-packages" / "mlx_lm" / ".omarchy-serve-patches").read_text()
    assert f"mlx_lm={version}\n" in record, f"no completion record for {version}: {record!r}"


def test_a_failed_apply_leaves_no_completion_record(wheels, version, tmp_path):
    """The script edits files one patch at a time. When a later patch fails the
    earlier files stay changed, but the completion record must not exist, so the
    serve status says PARTIAL instead of applied."""
    tree = tmp_path / "venv"
    site = _pristine_site(wheels, tree)
    target = site / "mlx_lm" / "models" / "gated_delta.py"
    target.write_text(target.read_text() + "\nDRIFT = True\n")
    target.write_text(target.read_text().replace("import mlx.core as mx", "import mlx.core as mx  # drift", 1))
    stale = site / "mlx_lm" / ".omarchy-serve-patches"
    stale.write_text(f"series=old\nmlx_lm={version}\n")  # a record left by an earlier complete apply
    r = _run_apply(SCRIPT, tree)
    if r.returncode == 0:
        pytest.skip(f"the drifted file did not stop the series on {version}")
    assert not (site / "mlx_lm" / ".omarchy-serve-patches").exists()
    assert _count_modified(site) >= 2, "an earlier patch should have changed a file besides the drifted one"


# ---------------------------------------------------------------------------
# Offline contract cases: the plumbing the documented paths lean on.


def test_installer_accepts_patch_venv():
    text = INSTALLER.read_text()
    assert "--patch-venv)" in text, "install.sh has no --patch-venv case"
    assert ("supported: --ane, --voice, --system, --uninstall, "
            "--patch-venv PATH)") in text


def test_fetch_covers_both_series():
    """The prefix fetch must lay out patches/mlx-lm-0.32/ as well as the flat
    series, or a venv on the 0.32 line cannot be patched (issue #217)."""
    text = INSTALLER.read_text()
    fetch = text.split("fetch_patch_assets()", 1)[1]
    assert '"https://raw.githubusercontent.com/$REPO/$VERSION/patches/mlx-lm-0.32/$p"' in fetch
    assert '-o "$PREFIX/patches/mlx-lm-0.32/$p"' in fetch
    assert 'grep -oE \'mlx-lm-[a-z0-9-]+\\.patch\'' in fetch


def test_patch_venv_creates_a_missing_prefix(tmp_path):
    """The release-wheel path has no earlier installer step, so the install
    prefix does not exist. --patch-venv must create it and lay out the apply
    script, both patch series and the helper scripts under it (curl exit 23
    'Failure writing output' before the fix). The curl shim serves the same
    release-tag paths out of this checkout; no network."""
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    shim = shim_dir / "curl"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        "url=; out=\n"
        'while [ $# -gt 0 ]; do case $1 in -o) out=$2; shift ;; http*) url=$1 ;; esac; shift; done\n'
        'rel=${url#https://raw.githubusercontent.com/joshuaswarren/omarchy-mlx/vtest/}\n'
        f'src={REPO}/$rel\n'
        '[ -f "$src" ] || exit 22\n'
        'cp "$src" "$out"\n')
    shim.chmod(0o755)
    venv = tmp_path / "venv"
    venv.mkdir()
    prefix = tmp_path / "no-such-prefix"
    env = {"PATH": f"{shim_dir}:/usr/bin:/bin", "HOME": str(tmp_path),
           "MLX_OMARCHY_HOME": str(prefix), "MLX_OMARCHY_VERSION": "vtest"}
    r = subprocess.run(["bash", str(INSTALLER), "--patch-venv", str(venv)],
                       capture_output=True, text=True, env=env, timeout=120, check=False)
    # The apply step itself fails on an empty venv; what matters is the fetch.
    assert "Failure writing output" not in r.stderr + r.stdout, r.stderr[-300:]
    assert (prefix / "apply-mlx-lm-patches.sh").is_file(), r.stdout[-300:] + r.stderr[-300:]
    assert list((prefix / "patches").glob("mlx-lm-*.patch")), "flat series missing"
    assert list((prefix / "patches" / "mlx-lm-0.32").glob("mlx-lm-*.patch")), "0.32 series missing"
    assert (prefix / "scripts").is_dir()


def test_release_wheel_doc_ends_with_the_patch_command():
    section = INSTALL_DOC.read_text().split("## From a release wheel", 1)[1]
    section = section.split("\n## ", 1)[0]
    assert PATCH_COMMAND in section
    assert "--patch-venv ~/.venvs/mlx" in section


def test_muse_quick_start_patches_its_venv():
    text = MUSE_DOC.read_text()
    assert PATCH_COMMAND in text
    assert "--patch-venv ~/.local/share/mlx-omarchy/venv" in text
