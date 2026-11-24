# SPDX-License-Identifier: MIT
# Copyright 2026 Joshua Warren
"""PT-T1 packaging regression: the host-shared ANE bundle store.

One copy per content hash lives in the shared store; every other tree
points at it with a symlink; different content hashes stay separate;
a dry run changes nothing; an already-linked tree is not re-deduped."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "mlx_omarchy_parakeet_under_test",
    Path(__file__).resolve().parents[1] / "overlay" / "tools"
    / "mlx-omarchy-parakeet" / "mlx_omarchy_parakeet.py")
pk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pk)


def _tree(base: Path, name: str, payload: bytes) -> Path:
    path = base / "venvs" / name / "lib/python3.14/site-packages/mlx/share" \
        / "mlx-omarchy/parakeet-1/bundles/parakeet-encoder-whole" \
        / "program-0.anec"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_duplicates_collapse_into_one_shared_copy(tmp_path, monkeypatch):
    store = tmp_path / "store"
    monkeypatch.setenv("MLX_OMARCHY_SHARED_BUNDLES", str(store))
    a = _tree(tmp_path, "venv-a", b"bundle-bytes-458MB")
    b = _tree(tmp_path, "venv-b", b"bundle-bytes-458MB")
    c = _tree(tmp_path, "venv-c", b"bundle-bytes-458MB")

    report = pk.dedupe_bundles([tmp_path / "venvs"])

    assert report["bytes_freed"] == 2 * len(b"bundle-bytes-458MB")
    assert len(report["entries"]) == 1
    store_files = list(store.glob("*"))
    assert len(store_files) == 1, "one shared copy in the store"
    for path in (a, b, c):
        assert path.is_symlink(), "each venv points at the shared copy"
        assert path.read_bytes() == b"bundle-bytes-458MB", \
            "the symlink resolves to the identical bytes"
    assert store_files[0].resolve() == path.resolve()


def test_different_hashes_stay_separate(tmp_path, monkeypatch):
    store = tmp_path / "store"
    monkeypatch.setenv("MLX_OMARCHY_SHARED_BUNDLES", str(store))
    a = _tree(tmp_path, "venv-a", b"the-old-upgrade")
    b = _tree(tmp_path, "venv-b", b"the-new-upgrade")

    report = pk.dedupe_bundles([tmp_path / "venvs"])

    assert len(list(store.glob("*"))) == 2, "distinct content stays distinct"
    assert report["bytes_freed"] == 0


def test_dry_run_changes_nothing(tmp_path, monkeypatch):
    store = tmp_path / "store"
    monkeypatch.setenv("MLX_OMARCHY_SHARED_BUNDLES", str(store))
    a = _tree(tmp_path, "venv-a", b"payload")

    report = pk.dedupe_bundles([tmp_path / "venvs"], dry_run=True)

    assert not a.is_symlink() and not list(store.glob("*"))
    assert report["dry_run"] is True


def test_an_already_linked_tree_is_not_re_deduped(tmp_path, monkeypatch):
    store = tmp_path / "store"
    monkeypatch.setenv("MLX_OMARCHY_SHARED_BUNDLES", str(store))
    first = _tree(tmp_path, "venv-a", b"payload")
    pk.dedupe_bundles([tmp_path / "venvs"])
    assert first.is_symlink()

    # a second pass over the same tree: nothing left to move or link
    report = pk.dedupe_bundles([tmp_path / "venvs"])
    assert report["bytes_freed"] == 0 and report["linked"] == 0
    assert first.is_symlink() and first.resolve().is_file()
