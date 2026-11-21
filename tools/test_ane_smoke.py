#!/usr/bin/env python3
"""Failing-first tests for tools/ane_smoke.py.

The smoke must answer without hardware and without MLX: a missing ANE node is
a clean exit-0 JSON line; with a node (a regular file stands in) the pinned
H13 add fixture runs through the omarchy-ane runner once and compares
bit-exact against the CPU; a mismatch is a nonzero failure naming the first
bad index; and the tool never imports mlx, because an mx.* result would claim
a Neural Engine pass from a GPU run.
"""
from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ane_smoke", HERE / "ane_smoke.py")
assert _spec is not None and _spec.loader is not None
ane_smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ane_smoke)

FAKE_NODE = "/tmp/ane-smoke-test-node"


def _fake_node() -> None:
    Path(FAKE_NODE).write_bytes(b"")


def _drop_node() -> None:
    Path(FAKE_NODE).unlink(missing_ok=True)


def test_absent_node_is_clean(capsys):
    """No node anywhere: one JSON line, absent, exit 0."""
    code = ane_smoke.main(["ane-smoke"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ane"] == "absent"
    assert "/dev/accel/accel0" in out["nodes"]


def test_present_node_runs_the_pinned_program_and_compares_to_cpu(capsys, monkeypatch):
    """With a node, the runner gets the pinned fixture and the exact surfaces.

    Only the subprocess boundary is patched: ane_add stands in for the real
    omarchy-ane-run call, everything else (detection, fixture lookup, CPU
    comparison, exit code) runs for real.
    """
    _fake_node()
    anec = "/tmp/ane-smoke-test.anec"
    Path(anec).write_bytes(b"")
    monkeypatch.setattr(ane_smoke.shutil, "which", lambda _name: "/tmp/ane-smoke-stub/omarchy-ane-run")
    seen = {}

    def runner(name, anec_path, a, b):
        seen["name"] = name
        seen["anec"] = anec_path
        seen["shapes"] = (len(a), len(b))
        return ane_smoke.cpu_add(a, b)

    monkeypatch.setattr(ane_smoke, "ane_add", runner)
    try:
        code = ane_smoke.main(["ane-smoke", "--ane-node", FAKE_NODE, "--anec", anec])
    finally:
        _drop_node()
        Path(anec).unlink(missing_ok=True)
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ane"] == "match"
    # The runner is the resolved omarchy-ane-run path, never an mx.*/GPU path.
    assert Path(seen["name"]).name == ane_smoke.RUNNER
    assert seen["anec"] == anec
    assert seen["shapes"] == (ane_smoke.SURFACE_BYTES,) * 2


def test_mismatch_is_a_failure_naming_the_index(capsys, monkeypatch):
    _fake_node()
    anec = "/tmp/ane-smoke-test.anec"
    Path(anec).write_bytes(b"")
    monkeypatch.setattr(ane_smoke.shutil, "which", lambda _name: "/tmp/ane-smoke-stub/omarchy-ane-run")

    def runner(_name, _anec, a, _b):
        wrong = bytearray(a)
        wrong[0:2] = struct.pack("<e", 9.0)
        return bytes(wrong)

    monkeypatch.setattr(ane_smoke, "ane_add", runner)
    try:
        code = ane_smoke.main(["ane-smoke", "--ane-node", FAKE_NODE, "--anec", anec])
    finally:
        _drop_node()
        Path(anec).unlink(missing_ok=True)
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert out["ane"] == "mismatch"
    assert out["first_mismatch_index"] == 0
    assert out["ane_sha256"] != out["cpu_sha256"]


def test_runner_missing_is_unavailable(capsys, monkeypatch):
    _fake_node()
    monkeypatch.setattr(ane_smoke.shutil, "which", lambda _name: None)
    try:
        code = ane_smoke.main(["ane-smoke", "--ane-node", FAKE_NODE])
    finally:
        _drop_node()
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert out["ane"] == "unavailable"
    assert "omarchy-ane-run" in out["error"]


def test_never_imports_mlx():
    """The tool must not load MLX at all: no mlx import, no mx.gpu path."""
    _fake_node()

    def runner(_name, _anec, a, b):
        assert "mlx" not in sys.modules, "the smoke loaded MLX; a pass would be a GPU run"
        return ane_smoke.cpu_add(a, b)

    try:
        code = ane_smoke.check(node=FAKE_NODE, ane_runner=runner)
    finally:
        _drop_node()
    assert code["ane"] == "match"
    assert "mlx" not in sys.modules
    assert "mlx.core" not in sys.modules


def test_fixture_inputs_are_exact_in_fp16():
    """Every a+b must be an exact fp16 value, so CPU and hardware cannot drift."""
    import numpy as np

    a, b = ane_smoke.fixture_inputs()
    sums = np.float16(a) + np.float16(b)
    exact = np.array(a, dtype=np.float64) + np.array(b, dtype=np.float64)
    assert np.array_equal(sums.astype(np.float64), exact)
