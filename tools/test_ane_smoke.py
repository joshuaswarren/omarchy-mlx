#!/usr/bin/env python3
"""Failing-first tests for tools/ane_smoke.py.

The smoke must answer without hardware and without MLX: a missing ANE node is
a clean exit-0 JSON line; with a node the driver's own smoke runs, the probed
node is forwarded to it, and its answer maps to match, mismatch, or
unavailable with the reason; a driver payload that proves nothing is
unavailable, not a pass; and the tool never imports mlx, because an mx.*
result would claim a Neural Engine pass from a GPU run.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ane_smoke", HERE / "ane_smoke.py")
assert _spec is not None and _spec.loader is not None
ane_smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ane_smoke)

GOOD = {
    "name": "add-fixture",
    "chip": "t6000",
    "available": True,
    "sha256": ["a" * 64] * 20,
    "golden_sha256": "b" * 64,
    "errors": 0,
}


@pytest.fixture
def accel_node(tmp_path, monkeypatch):
    """An existing driver-shaped node path that is not the host's /dev."""
    monkeypatch.setattr(ane_smoke, "ACCEL_PREFIX", str(tmp_path) + "/accel")
    node = tmp_path / "accel3"
    node.write_bytes(b"")
    return str(node)


def test_absent_node_is_clean(capsys, monkeypatch, tmp_path):
    """No node anywhere: one JSON line, absent, exit 0. Host /dev is not read."""
    missing = (str(tmp_path / "accel0"), str(tmp_path / "ane"))
    monkeypatch.setattr(ane_smoke, "NODE_CANDIDATES", missing)
    code = ane_smoke.main(["ane-smoke"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ane"] == "absent"
    assert tuple(out["nodes"]) == missing


def test_present_node_runs_the_driver_smoke_and_reports_match(accel_node):
    """With a node the driver's smoke runs and its exact answer is passed on."""
    seen = {}

    def runner(timeout, env):
        seen["timeout"] = timeout
        seen["env"] = env
        return dict(GOOD)

    answer = ane_smoke.check(node=accel_node, smoke_runner=runner)
    assert answer["ane"] == "match"
    assert answer["chip"] == "t6000"
    assert answer["calls"] == 20
    assert answer["sha256"] == "a" * 64
    assert answer["golden_sha256"] == "b" * 64
    assert seen["timeout"] == ane_smoke.RUN_TIMEOUT_S


def test_mismatch_is_a_failure_with_the_driver_reason(accel_node):
    payload = dict(GOOD, errors=1, reason="call 7: output surface or runner status differs (1)")
    answer = ane_smoke.check(node=accel_node, smoke_runner=lambda _t, _e: payload)
    assert answer["ane"] == "mismatch"
    assert "call 7" in answer["error"]


def test_no_program_for_this_soc_is_unavailable(accel_node):
    """Aurora's SoC has no packaged program: the answer says so, by name."""
    payload = {"name": "add-fixture", "chip": "t8122", "available": False,
               "reason": "t8122: no omarchy-ane driver for this chip"}
    answer = ane_smoke.check(node=accel_node, smoke_runner=lambda _t, _e: payload)
    assert answer["ane"] == "unavailable"
    assert answer["chip"] == "t8122"
    assert "t8122" in answer["error"]


def test_payload_without_an_errors_key_is_unavailable(accel_node):
    """A payload that proves nothing is not a passing ANE run."""
    answer = ane_smoke.check(node=accel_node, smoke_runner=lambda _t, _e: {"available": True})
    assert answer["ane"] == "unavailable"
    assert "errors" not in answer
    assert answer["error"]


def test_payload_with_an_empty_hash_list_is_unavailable(accel_node):
    """Zero calls is not a bit-exact answer."""
    payload = {"available": True, "errors": 0, "sha256": []}
    answer = ane_smoke.check(node=accel_node, smoke_runner=lambda _t, _e: payload)
    assert answer["ane"] == "unavailable"
    assert answer["calls"] == 0
    assert answer["error"]


def test_null_errors_is_unavailable_not_mismatch(accel_node):
    """errors: None proves nothing; it is not a zero-error pass either."""
    payload = {"available": True, "errors": None, "sha256": ["a" * 64] * 20}
    answer = ane_smoke.check(node=accel_node, smoke_runner=lambda _t, _e: payload)
    assert answer["ane"] == "unavailable"
    assert answer["error"]


def test_negative_errors_is_unavailable(accel_node):
    payload = {"available": True, "errors": -5, "sha256": ["a" * 64] * 20}
    answer = ane_smoke.check(node=accel_node, smoke_runner=lambda _t, _e: payload)
    assert answer["ane"] == "unavailable"
    assert answer["error"]


def test_non_list_sha256_is_unavailable(accel_node):
    payload = {"available": True, "errors": 0, "sha256": "a" * 64}
    answer = ane_smoke.check(node=accel_node, smoke_runner=lambda _t, _e: payload)
    assert answer["ane"] == "unavailable"
    assert answer["error"]


def test_missing_smoke_binary_is_unavailable(accel_node, monkeypatch):
    monkeypatch.setattr(ane_smoke.shutil, "which", lambda _name: None)
    answer = ane_smoke.check(node=accel_node)
    assert answer["ane"] == "unavailable"
    assert "omarchy-ane-smoke" in answer["error"]


def test_smoke_failure_is_unavailable_with_the_reason(accel_node):
    def runner(_timeout, _env):
        raise RuntimeError("no such device")

    answer = ane_smoke.check(node=accel_node, smoke_runner=runner)
    assert answer["ane"] == "unavailable"
    assert "RuntimeError" in answer["error"]


def test_never_imports_mlx(accel_node):
    """The tool must not load MLX at all: no mlx import, no mx.gpu path."""

    def runner(_timeout, _env):
        assert "mlx" not in sys.modules, "the smoke loaded MLX; a pass would be a GPU run"
        return dict(GOOD)

    answer = ane_smoke.check(node=accel_node, smoke_runner=runner)
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


def test_the_probed_node_is_forwarded_to_the_driver(accel_node):
    """The answer's node must be the node the driver was told to open."""
    seen = {}

    def runner(_timeout, env):
        seen["env"] = env
        return dict(GOOD)

    answer = ane_smoke.check(node=accel_node, smoke_runner=runner)
    assert answer["ane"] == "match"
    assert answer["node"] == accel_node
    assert seen["env"]["ANE_DEVICE"] == "3"


def test_a_node_the_driver_cannot_be_told_to_open_is_unavailable(tmp_path):
    """The driver opens /dev/accel/accelN; any other existing path is refused."""
    ran = []
    other = tmp_path / "ane"
    other.write_bytes(b"")

    def runner(_timeout, _env):
        ran.append(True)
        return dict(GOOD)

    answer = ane_smoke.check(node=str(other), smoke_runner=runner)
    assert answer["ane"] == "unavailable"
    assert not ran
    assert str(other) in answer["error"]


def test_an_explicit_node_that_does_not_exist_is_absent(capsys):
    """--ane-node probes existence too: a missing node is absent, not a run."""
    code = ane_smoke.main(["ane-smoke", "--ane-node", "/nonexistent/ane-smoke-missing-node"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ane"] == "absent"
    assert out["nodes"] == ["/nonexistent/ane-smoke-missing-node"]


def test_a_non_dict_driver_payload_is_unavailable_not_a_crash(accel_node):
    answer = ane_smoke.check(node=accel_node, smoke_runner=lambda _t, _e: 123)
    assert answer["ane"] == "unavailable"
    assert answer["error"]


def test_smoke_json_skips_non_object_json():
    assert ane_smoke.smoke_json(["123", "null"])["available"] is False


def test_the_last_json_object_wins():
    first = json.dumps(dict(GOOD, errors=9))
    parsed = ane_smoke.smoke_json([first, json.dumps(GOOD)])
    assert parsed["errors"] == 0


def test_empty_stdout_keeps_stderr(monkeypatch):
    """A driver that dies with empty stdout still names why, on stderr."""
    proc = subprocess.CompletedProcess(args=[], returncode=2, stdout="",
                                       stderr="omarchy-ane-smoke: boom\n")
    monkeypatch.setattr(ane_smoke.subprocess, "run", lambda *_a, **_k: proc)
    payload = ane_smoke.run_smoke(60, {})
    assert payload["available"] is False
    assert "boom" in payload["reason"]


def test_empty_stdout_and_stderr_names_the_exit_status(monkeypatch):
    proc = subprocess.CompletedProcess(args=[], returncode=2, stdout="", stderr="")
    monkeypatch.setattr(ane_smoke.subprocess, "run", lambda *_a, **_k: proc)
    payload = ane_smoke.run_smoke(60, {})
    assert payload["available"] is False
    assert "2" in payload["reason"]
