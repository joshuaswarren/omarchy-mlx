#!/usr/bin/env python3
"""Failing-first tests for tools/ane_smoke.py.

The smoke must answer without hardware and without MLX: a missing ANE node is
a clean exit-0 JSON line; with a node the driver's own smoke runs and its
answer maps to match, mismatch, or unavailable with the reason; a driver
payload that proves nothing is unavailable, not a pass; the probed node is
forwarded to the driver; the driver's JSON is found even when it prints more
after it; and the tool never imports mlx, because an mx.* result would claim a
Neural Engine pass from a GPU run.
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

# A node the driver can be told to open: the shape libane enumerates.
FAKE_NODE = "/dev/accel/accel7"
GOOD = {
    "name": "add-fixture",
    "chip": "t6000",
    "available": True,
    "sha256": ["a" * 64] * 20,
    "golden_sha256": "b" * 64,
    "errors": 0,
}


def test_absent_node_is_clean(capsys, monkeypatch):
    """No node anywhere: one JSON line, absent, exit 0."""
    monkeypatch.setattr(ane_smoke, "pick_node", lambda *_a: None)
    code = ane_smoke.main(["ane-smoke"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ane"] == "absent"
    assert "/dev/accel/accel0" in out["nodes"]


def test_present_node_runs_the_driver_smoke_and_reports_match():
    """With a node the driver's smoke runs and its exact answer is passed on."""
    seen = {}

    def runner(timeout, env):
        seen["timeout"] = timeout
        seen["env"] = env
        return dict(GOOD)

    answer = ane_smoke.check(node=FAKE_NODE, smoke_runner=runner)
    assert answer["ane"] == "match"
    assert answer["chip"] == "t6000"
    assert answer["calls"] == 20
    assert answer["sha256"] == "a" * 64
    assert answer["golden_sha256"] == "b" * 64
    assert seen["timeout"] == ane_smoke.RUN_TIMEOUT_S


def test_mismatch_is_a_failure_with_the_driver_reason():
    payload = dict(GOOD, errors=1, reason="call 7: output surface or runner status differs (1)")

    answer = ane_smoke.check(node=FAKE_NODE, smoke_runner=lambda _t, _e: payload)
    assert answer["ane"] == "mismatch"
    assert "call 7" in answer["error"]


def test_no_program_for_this_soc_is_unavailable():
    """Aurora's SoC has no packaged program: the answer says so, by name."""
    payload = {"name": "add-fixture", "chip": "t8122", "available": False,
               "reason": "t8122: no omarchy-ane driver for this chip"}

    answer = ane_smoke.check(node=FAKE_NODE, smoke_runner=lambda _t, _e: payload)
    assert answer["ane"] == "unavailable"
    assert answer["chip"] == "t8122"
    assert "t8122" in answer["error"]


def test_payload_without_an_errors_key_is_unavailable():
    """A payload that proves nothing is not a passing ANE run."""
    answer = ane_smoke.check(node=FAKE_NODE, smoke_runner=lambda _t, _e: {"available": True})
    assert answer["ane"] == "unavailable"
    assert "errors" not in answer
    assert "errors" in answer["error"]


def test_payload_with_an_empty_hash_list_is_unavailable():
    """Zero calls is not a bit-exact answer."""
    payload = {"available": True, "errors": 0, "sha256": []}
    answer = ane_smoke.check(node=FAKE_NODE, smoke_runner=lambda _t, _e: payload)
    assert answer["ane"] == "unavailable"
    assert answer["calls"] == 0
    assert answer["error"]


def test_a_run_that_produced_no_calls_carries_the_driver_reason():
    """Every call failed: the driver's own reason is the answer."""
    payload = {"available": True, "errors": 20, "sha256": [],
               "reason": "call 1: failed to find device with dev_id 1"}
    answer = ane_smoke.check(node=FAKE_NODE, smoke_runner=lambda _t, _e: payload)
    assert answer["ane"] == "unavailable"
    assert answer["calls"] == 0
    assert answer["errors"] == 20
    assert "dev_id 1" in answer["error"]


def test_missing_smoke_binary_is_unavailable(monkeypatch):
    monkeypatch.setattr(ane_smoke.shutil, "which", lambda _name: None)
    answer = ane_smoke.check(node=FAKE_NODE)
    assert answer["ane"] == "unavailable"
    assert "omarchy-ane-smoke" in answer["error"]


def test_smoke_failure_is_unavailable_with_the_reason():

    def runner(_timeout, _env):
        raise RuntimeError("no such device")

    answer = ane_smoke.check(node=FAKE_NODE, smoke_runner=runner)
    assert answer["ane"] == "unavailable"
    assert "RuntimeError" in answer["error"]


def test_never_imports_mlx():
    """The tool must not load MLX at all: no mlx import, no mx.gpu path."""

    def runner(_timeout, _env):
        assert "mlx" not in sys.modules, "the smoke loaded MLX; a pass would be a GPU run"
        return dict(GOOD)

    answer = ane_smoke.check(node=FAKE_NODE, smoke_runner=runner)
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


def test_the_probed_node_is_forwarded_to_the_driver():
    """The answer's node must be the node the driver was told to open."""
    seen = {}

    def runner(_timeout, env):
        seen["env"] = env
        return dict(GOOD)

    answer = ane_smoke.check(node="/dev/accel/accel3", smoke_runner=runner)
    assert answer["ane"] == "match"
    assert answer["node"] == "/dev/accel/accel3"
    assert seen["env"]["ANE_DEVICE"] == "3"


def test_a_node_the_driver_cannot_be_told_to_open_is_unavailable():
    """The driver opens /dev/accel/accelN; any other path is refused, not run."""
    ran = []

    def runner(_timeout, _env):
        ran.append(True)
        return dict(GOOD)

    answer = ane_smoke.check(node="/dev/ane", smoke_runner=runner)
    assert answer["ane"] == "unavailable"
    assert not ran
    assert "/dev/ane" in answer["error"]


def test_the_driver_json_is_found_after_any_trailing_output():
    """A driver that prints after its JSON line still parses."""
    payload = json.dumps(GOOD)
    parsed = ane_smoke.smoke_json([payload, "omarchy-ane-smoke: 20/20 calls bit-exact"])
    assert parsed["errors"] == 0
    assert parsed["chip"] == "t6000"


def test_driver_output_without_json_is_unavailable():
    parsed = ane_smoke.smoke_json(["not json", "still not json"])
    assert parsed["available"] is False
    assert parsed["reason"]
