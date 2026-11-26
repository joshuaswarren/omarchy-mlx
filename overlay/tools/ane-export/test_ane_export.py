#!/usr/bin/env python3
"""The exporter runs the fixed converter, and passes it the weight blob.

Two defects were fixed in the canonical converter
(ane-linux-experiments 52a3211): the weight payload offset now comes out of
the blob record, and nchw geometry out of the task's tile-DMA counts. Both
reappear the moment the exporter calls an older converter copy, or calls the
fixed one without ``--weights`` - the export stays silent and wrong. These
tests stand a stub converter in the tools dir and fail on either regression.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "ane_export", Path(__file__).with_name("ane_export.py")
)
assert SPEC is not None and SPEC.loader is not None
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)

# A converter that accepts --weights, uses the blob record for the payload
# offset and derive_strides for the geometry, and refuses a blob-backed
# object whose weights.bin is absent.
STUB_CONVERTER = '''#!/usr/bin/env python3
import argparse, pathlib, sys
p = argparse.ArgumentParser()
p.add_argument("src"); p.add_argument("dst")
p.add_argument("in_ch", type=int); p.add_argument("out_ch", type=int)
p.add_argument("--weights")
a = p.parse_args()
if a.weights is None or not pathlib.Path(a.weights).exists():
    sys.exit("hwxv2-to-anec: blob-backed object without its weights.bin")
pathlib.Path(a.dst).write_bytes(b"ANEC" + pathlib.Path(a.src).read_bytes())
pathlib.Path("converter-argv.json").write_text(repr(sys.argv[1:]))
print("wrote={} td-count=3 workspace=0x4000 derive_strides".format(a.dst))
'''

STUB_COMPILER = """#!/bin/sh
mkdir -p "$2" && printf 'HWX' > "$2/model.hwx"
"""

COMMIT = "0" * 40


@pytest.fixture
def tools_dir(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    compiler = tools / "ane-compile-hwx"
    compiler.write_text(STUB_COMPILER)
    compiler.chmod(0o755)
    (tools / EXPORT.CONVERTER).write_text(STUB_CONVERTER)
    return tools


def run_export(tmp_path, tools, descriptor=None):
    desc_path = tmp_path / "desc.json"
    desc_path.write_text(json.dumps(
        descriptor or {"op": "add", "input_shape": [1, 512]}
    ))
    out_dir = tmp_path / "out"
    return subprocess.run(
        [sys.executable, str(Path(__file__).with_name("ane_export.py")),
         str(desc_path), "--out-dir", str(out_dir),
         "--tools-dir", str(tools), "--source-commit", COMMIT,
         "--macos-build", "25A000", "--anecompiler", "stub"],
        cwd=tmp_path, text=True, capture_output=True, timeout=120,
    ), out_dir


def test_converter_is_called_with_the_weight_blob(tmp_path, tools_dir):
    # Without --weights the converter reads 64 words of blob header as
    # coefficients and drops the real tail; the export still succeeds.
    result, out_dir = run_export(tmp_path, tools_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    argv = (tools_dir / "converter-argv.json").read_text()
    weights = out_dir / "capture" / "weights.bin"
    assert "--weights" in argv
    assert str(weights) in argv
    assert weights.exists()
    assert (out_dir / "bundle" / "model.anec").exists()


def test_export_is_refused_when_the_weight_blob_is_missing(
    tmp_path, tools_dir
):
    # The converter refuses a blob-backed object without weights.bin; the
    # exporter must surface that instead of shipping a truncated .anec.
    hostile = tools_dir / EXPORT.CONVERTER
    hostile.write_text(STUB_CONVERTER.replace(
        'a = p.parse_args()',
        'a = p.parse_args()\npathlib.Path(a.weights).unlink()',
    ))
    result, out_dir = run_export(tmp_path, tools_dir)
    assert result.returncode != 0
    assert "hwxv2-to-anec" in result.stderr
    assert not (out_dir / "bundle" / "model.anec").exists()


def test_a_converter_predating_the_fix_is_refused(tmp_path, tools_dir):
    stale = tools_dir / EXPORT.CONVERTER
    stale.write_text("#!/usr/bin/env python3\nprint('td-count=3')\n")
    result, out_dir = run_export(tmp_path, tools_dir)
    assert result.returncode != 0
    assert "predates the export fix" in result.stderr
    assert not (out_dir / "bundle" / "model.anec").exists()


def test_a_missing_converter_names_its_source(tmp_path, tools_dir):
    (tools_dir / EXPORT.CONVERTER).unlink()
    result, _ = run_export(tmp_path, tools_dir)
    assert result.returncode != 0
    assert "ane-linux-experiments" in result.stderr


def test_the_canonical_converter_satisfies_the_markers():
    """The guard tracks the real converter, not a spelling of its own."""
    canonical = Path(
        os.environ.get("ANE_LINUX_EXPERIMENTS",
                       Path.home() / "src" / "ane-linux-experiments")
    ) / "tools" / EXPORT.CONVERTER
    if not canonical.exists():
        pytest.skip(f"no canonical converter at {canonical}")
    source = canonical.read_text()
    for marker in EXPORT.CONVERTER_MARKERS:
        assert marker in source, marker
