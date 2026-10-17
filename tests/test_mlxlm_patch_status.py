# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""The serve status line for the vendored mlx-lm patches.

Changed files alone do not prove a complete apply, so "applied" needs the
completion record the apply script writes after its last step. A synthetic
mlx-lm distribution (no network, no real mlx-lm) covers the four states.
"""

import base64
import hashlib
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "serve"))

from mlx_omarchy_serve import __main__ as serve_main  # noqa: E402

VERSION = "9.9.9"
RECORD_NAME = ".omarchy-serve-patches"


def _digest(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


@pytest.fixture
def site(tmp_path, monkeypatch):
    """A site-packages with a fake mlx-lm 9.9.9 whose RECORD matches its files."""
    sp = tmp_path / "site"
    pkg = sp / "mlx_lm"
    pkg.mkdir(parents=True)
    files = {"mlx_lm/a.py": b"A = 1\n", "mlx_lm/b.py": b"B = 2\n"}
    lines = []
    for name, data in files.items():
        (sp / name).write_bytes(data)
        lines.append(f"{name},{_digest(data)},{len(data)}")
    info = sp / f"mlx_lm-{VERSION}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: mlx-lm\nVersion: {VERSION}\n")
    (info / "RECORD").write_text("\n".join(lines + [f"mlx_lm-{VERSION}.dist-info/METADATA,,", f"mlx_lm-{VERSION}.dist-info/RECORD,,"]) + "\n")
    monkeypatch.syspath_prepend(str(sp))
    return sp


def test_stock_tree_says_not_applied(site):
    assert "NOT applied" in serve_main.mlxlm_patch_status()


def test_changed_file_without_a_record_is_partial(site):
    (site / "mlx_lm/a.py").write_text("A = 1  # patched\n")
    line = serve_main.mlxlm_patch_status()
    assert "PARTIAL or unverified" in line
    assert "1 of 2" in line
    assert "applied (" not in line


def test_changed_file_with_a_matching_record_is_applied(site):
    (site / "mlx_lm/a.py").write_text("A = 1  # patched\n")
    (site / "mlx_lm" / RECORD_NAME).write_text(f"series=patches\nmlx_lm={VERSION}\n")
    line = serve_main.mlxlm_patch_status()
    assert line.startswith("mlx-lm serve patches: applied (1 of 2")
    assert "apply completed" in line


def test_a_record_for_another_version_does_not_count(site):
    (site / "mlx_lm/a.py").write_text("A = 1  # patched\n")
    (site / "mlx_lm" / RECORD_NAME).write_text("series=patches\nmlx_lm=0.0.1\n")
    assert "PARTIAL or unverified" in serve_main.mlxlm_patch_status()


def test_a_record_over_a_restored_tree_says_not_applied(site):
    (site / "mlx_lm" / RECORD_NAME).write_text(f"series=patches\nmlx_lm={VERSION}\n")
    assert "NOT applied" in serve_main.mlxlm_patch_status()
