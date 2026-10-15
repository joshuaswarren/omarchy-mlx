# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host contract of the overlapped decoder load (no ANE, Vulkan or mlx).

The transcribe CLI runs the CPU decoder/joint preparation on one worker
thread started right after audio_load and performs every GPU upload on
the main thread after the join, inside the `decoder_load` stage. These
checks hold the CLI to that contract with fake prepare/finish functions,
so a regression (GPU work moved into the thread, a swallowed thread
exception, a renamed or reshaped stage record) fails on any host before
it can reach hardware.
"""

import sys
import threading
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "overlay" / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "mlx-omarchy-parakeet"))

import coreml.vulkan_decoder as vulkan_decoder  # noqa: E402
import coreml.vulkan_decoder_step as vulkan_decoder_step  # noqa: E402
import mlx_omarchy_parakeet as cli  # noqa: E402

MAIN = threading.main_thread()


@pytest.fixture()
def fake_load(monkeypatch):
    """Replace the load phases; record which thread each ran on."""
    seen: list[tuple[str, bool]] = []
    state: dict = {"cpu_error": None}
    parts, packing, decoder = object(), object(), object()

    def prepare_decoder(package_path):
        seen.append(("prepare_decoder", threading.current_thread() is MAIN))
        if state["cpu_error"] is not None:
            raise state["cpu_error"]
        return parts

    def prepare_step_weights(prepared, joint_package):
        seen.append(("prepare_step_weights", threading.current_thread() is MAIN))
        assert prepared is parts
        return packing

    def finish_decoder(prepared):
        seen.append(("finish_decoder", threading.current_thread() is MAIN))
        assert prepared is parts
        return decoder

    def finish_step_weights(packed):
        seen.append(("finish_step_weights", threading.current_thread() is MAIN))
        assert packed is packing
        return object()

    monkeypatch.setattr(vulkan_decoder, "prepare_decoder", prepare_decoder)
    monkeypatch.setattr(
        vulkan_decoder_step, "prepare_step_weights", prepare_step_weights)
    monkeypatch.setattr(vulkan_decoder, "finish_decoder", finish_decoder)
    monkeypatch.setattr(
        vulkan_decoder_step, "finish_step_weights", finish_step_weights)
    state["seen"] = seen
    return state


def test_cpu_preparation_off_main_thread_gpu_after_join(fake_load):
    thread, state = cli._start_decoder_load(Path("/tmp"))
    assert thread is not MAIN
    loaded = cli._join_decoder_load(thread, state)
    assert fake_load["seen"] == [
        ("prepare_decoder", False),
        ("prepare_step_weights", False),
        ("finish_decoder", True),
    ]
    assert loaded is not None
    assert state["cpu_thread_wall_ns"] >= 0
    # The step upload runs on the main thread too (pipeline line after the
    # decoder_load stage).
    vulkan_decoder_step.finish_step_weights(state["packing"])
    assert fake_load["seen"][-1] == ("finish_step_weights", True)


def test_thread_error_propagates_at_join_and_thread_is_joined(fake_load):
    fake_load["cpu_error"] = RuntimeError("bad package")
    thread, state = cli._start_decoder_load(Path("/tmp"))
    with pytest.raises(RuntimeError, match="bad package"):
        cli._join_decoder_load(thread, state)
    assert not thread.is_alive()


def test_stage_records_keep_name_shape_and_extra_field():
    stage, records = cli._stage_recorder()
    ticks = {"n": 0}

    def snapshot():
        ticks["n"] += 1
        return {"vk_submissions": ticks["n"]}

    assert stage("audio_load", snapshot, lambda: "a") == "a"
    assert stage(
        "decoder_load", snapshot, lambda: "d",
        extra=lambda: {"cpu_thread_wall_ms": 1.5},
    ) == "d"
    assert stage("mel_frontend", snapshot, lambda: "m") == "m"
    assert [record["stage"] for record in records] == [
        "audio_load", "decoder_load", "mel_frontend"]
    for record in records:
        assert set(record) >= {
            "stage", "wall_ns", "wall_ms", "gpu_counter_delta"}
        assert record["gpu_counter_delta"] == {"vk_submissions": 1}
    assert records[1]["cpu_thread_wall_ms"] == 1.5
    assert "cpu_thread_wall_ms" not in records[0]
    assert "cpu_thread_wall_ms" not in records[2]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
