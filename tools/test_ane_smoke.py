#!/usr/bin/env python3
"""Failing-first tests for tools/ane_smoke.py.

The smoke must answer without hardware and without MLX: a missing ANE node is
a clean exit-0 JSON line; with a node the driver's own smoke runs and its
answer maps to match, mismatch, or unavailable with the reason; and the tool
never imports mlx, because an mx.* result would claim a Neural Engine pass
from a GPU run.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ane_smoke", HERE / "ane_smoke.py")
assert _spec is not None and _spec.loader is not None
ane_smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ane_smoke)

FAKE_NODE = "/tmp/ane-smoke-test-node"
GOOD = {
    "name": "add-fixture",
    "chip": "t6000",
    "available": True,
    "sha256": ["a" * 64] * 20,
    "golden_sha256": "b" * 64,
    "errors": 0,
}


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


def test_present_node_runs_the_driver_smoke_and_reports_match():
    """With a node the driver's smoke runs and its exact answer is passed on."""
    _fake_node()
    seen = {}

    def runner(timeout):
        seen["timeout"] = timeout
        return dict(GOOD)

    try:
        answer = ane_smoke.check(node=FAKE_NODE, smoke_runner=runner)
    finally:
        _drop_node()
    assert answer["ane"] == "match"
    assert answer["chip"] == "t6000"
    assert answer["calls"] == 20
    assert answer["sha256"] == "a" * 64
    assert answer["golden_sha256"] == "b" * 64
    assert seen["timeout"] == ane_smoke.RUN_TIMEOUT_S


def test_mismatch_is_a_failure_with_the_driver_reason():
    _fake_node()
    payload = dict(GOOD, errors=1, reason="call 7: output surface or runner status differs (1)")

    try:
        answer = ane_smoke.check(node=FAKE_NODE, smoke_runner=lambda _t: payload)
    finally:
        _drop_node()
    assert answer["ane"] == "mismatch"
    assert "call 7" in answer["error"]


def test_no_program_for_this_soc_is_unavailable():
    """Aurora's SoC has no packaged program: the answer says so, by name."""
    _fake_node()
    payload = {"name": "add-fixture", "chip": "t8122", "available": False,
               "reason": "t8122: no omarchy-ane driver for this chip"}

    try:
        answer = ane_smoke.check(node=FAKE_NODE, smoke_runner=lambda _t: payload)
    finally:
        _drop_node()
    assert answer["ane"] == "unavailable"
    assert answer["chip"] == "t8122"
    assert "t8122" in answer["error"]


def test_missing_smoke_binary_is_unavailable(monkeypatch):
    _fake_node()
    monkeypatch.setattr(ane_smoke.shutil, "which", lambda _name: None)
    try:
        answer = ane_smoke.check(node=FAKE_NODE)
    finally:
        _drop_node()
    assert answer["ane"] == "unavailable"
    assert "omarchy-ane-smoke" in answer["error"]


def test_smoke_failure_is_unavailable_with_the_reason():
    _fake_node()

    def runner(_timeout):
        raise RuntimeError("no such device")

    try:
        answer = ane_smoke.check(node=FAKE_NODE, smoke_runner=runner)
    finally:
        _drop_node()
    assert answer["ane"] == "unavailable"
    assert "RuntimeError" in answer["error"]


def test_never_imports_mlx():
    """The tool must not load MLX at all: no mlx import, no mx.gpu path."""
    _fake_node()

    def runner(_timeout):
        assert "mlx" not in sys.modules, "the smoke loaded MLX; a pass would be a GPU run"
        return dict(GOOD)

    try:
        answer = ane_smoke.check(node=FAKE_NODE, smoke_runner=runner)
    finally:
        _drop_node()
    assert answer["ane"] == "match"
    assert "mlx" not in sys.modules
    assert "mlx.core" not in sys.modules


def test_chip_comes_from_the_device_tree(tmp_path):
    """The chip name is read from the device tree, so a lab row can quote it."""
    dt = tmp_path / "sys/firmware/devicetree/base"
    dt.mkdir(parents=True)
    (dt / "compatible").write_bytes(b"apple,t8122\x00")
    assert ane_smoke.chip(str(tmp_path)) == "t8122"
    assert ane_smoke.chip(str(tmp_path / "nothing")) is None
