#!/usr/bin/env python3
"""Failing-first tests for tools/ane_smoke.py.

The smoke tool must answer three questions without hardware: it reports a
missing ANE cleanly (exit 0, no traceback), it runs one tiny known program on
the accelerator and on CPU and compares the results exactly, and a mismatch is
a nonzero failure. Hardware never runs here: the program runner is injected.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ane_smoke", HERE / "ane_smoke.py")
assert _spec is not None and _spec.loader is not None
ane_smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ane_smoke)


def test_absent_node_is_clean(capsys):
    """No node anywhere: one JSON line, absent, exit 0."""
    code = ane_smoke.main(["ane-smoke", "--ane-node", "/nonexistent/ane"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ane"] == "absent"
    assert out["node"] == "/nonexistent/ane"


def test_present_node_runs_the_program_and_compares(capsys, monkeypatch):
    """A node (regular file stands in) triggers one program on both devices."""
    fake = "/tmp/ane-smoke-test-node"
    Path(fake).write_bytes(b"")
    ran = []

    def runner(_program, device):
        ran.append(device)
        return [6, 8, 10, 12]

    monkeypatch.setattr(ane_smoke, "run_program", runner)
    try:
        code = ane_smoke.main(["ane-smoke", "--ane-node", fake])
    finally:
        Path(fake).unlink()
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ane"] == "match"
    assert sorted(ran) == ["accelerator", "cpu"]
    assert out["expected"] == [6, 8, 10, 12]


def test_mismatch_is_a_failure(capsys, monkeypatch):
    fake = "/tmp/ane-smoke-test-node"
    Path(fake).write_bytes(b"")

    def runner(_program, device):
        return [6, 8, 10, 12] if device == "cpu" else [6, 8, 10, 13]

    monkeypatch.setattr(ane_smoke, "run_program", runner)
    try:
        code = ane_smoke.main(["ane-smoke", "--ane-node", fake])
    finally:
        Path(fake).unlink()
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert out["ane"] == "mismatch"
    assert out["accelerator_result"] == [6, 8, 10, 13]


def test_known_program_is_fixed():
    """The smoke program is one fixed tiny add, identical on both devices."""
    program = ane_smoke.PROGRAM
    assert program["op"] == "add"
    assert [a + b for a, b in zip(program["a"], program["b"])] == program["expected"]
