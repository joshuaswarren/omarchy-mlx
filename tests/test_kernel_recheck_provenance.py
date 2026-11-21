#!/usr/bin/env python3
"""Gap 2 (2026-11-21): --gpu rows must name the wheel that produced them.

Main's gap-2 numbers matched a libmlx built before the Nov-8 chained-cast
and literal-bf16 fixes: kda_glue_pre emitted an unbalanced `(float(` line
(glslc "unexpected SEMICOLON" at the constructor) and kda_glue_post came
back unrounded at exactly the pre-fix 53/12288 count. The harness output
carried nothing that identified the library, so a stale wheel read as a
current result. `run.py --gpu` now prints a PROVENANCE line before the
KERNEL rows: the mlx package dir, the libmlx.so path and its SHA-256, the
translator source SHA the wheel was built with, and the SPIR-V cache dir.
"""

import hashlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.kernel_recheck import run  # noqa: E402


class _FakeSpec:
    def __init__(self, origin, locations=None):
        self.origin = str(origin) if origin else None
        self.submodule_search_locations = locations


def _fake_mlx(tmp_path):
    pkg = tmp_path / "mlx"
    lib = pkg / "lib"
    lib.mkdir(parents=True)
    payload = b"fake libmlx bytes for provenance"
    (lib / "libmlx.so").write_bytes(payload)
    return _FakeSpec(pkg / "__init__.py"), hashlib.sha256(payload).hexdigest()


def test_provenance_names_libmlx_and_sha(tmp_path, monkeypatch, capsys):
    spec, want_sha = _fake_mlx(tmp_path)
    monkeypatch.setattr(run.importlib.util, "find_spec", lambda name: spec)
    monkeypatch.setenv("MLX_OMARCHY_TRANSLATOR_SOURCE_SHA", "abc123")
    monkeypatch.setenv("MLX_OMARCHY_SPIRV_CACHE", "/tmp/krcache-test")

    run._provenance()
    out = capsys.readouterr().out

    assert out.startswith("PROVENANCE ")
    assert f"libmlx_sha256={want_sha}" in out
    assert "translator_source_sha=abc123" in out
    assert "spirv_cache=/tmp/krcache-test" in out


def test_provenance_namespace_package(tmp_path, monkeypatch, capsys):
    # The lab wheel exposes mlx as a namespace package: spec.origin is None
    # and the dir lives in submodule_search_locations. The first PROVENANCE
    # version only read origin and printed libmlx=missing on the real host.
    pkg = tmp_path / "mlx"
    lib = pkg / "lib"
    lib.mkdir(parents=True)
    payload = b"namespace-package libmlx"
    (lib / "libmlx.so").write_bytes(payload)
    spec = _FakeSpec(None, [str(pkg)])
    monkeypatch.setattr(run.importlib.util, "find_spec", lambda name: spec)

    run._provenance()
    out = capsys.readouterr().out

    assert f"libmlx_sha256={hashlib.sha256(payload).hexdigest()}" in out
    assert f"mlx_dir={pkg}" in out


def test_provenance_survives_missing_mlx(monkeypatch, capsys):
    monkeypatch.setattr(run.importlib.util, "find_spec", lambda name: None)
    monkeypatch.delenv("MLX_OMARCHY_TRANSLATOR_SOURCE_SHA", raising=False)
    monkeypatch.delenv("MLX_OMARCHY_SPIRV_CACHE", raising=False)

    run._provenance()
    out = capsys.readouterr().out

    assert out.startswith("PROVENANCE ")
    assert "libmlx=missing" in out
    assert "translator_source_sha=unset" in out
