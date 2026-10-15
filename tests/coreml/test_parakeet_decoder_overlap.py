# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host contract of the overlapped decoder load (no ANE, Vulkan or mlx).

The transcribe CLI runs the decoder/joint load (CPU preparation AND GPU
uploads) on one worker thread started right before chunk 0's encoder
stage, on the MAIN thread's stream object (default streams are
thread-local); the main thread joins, synchronizes that stream and builds
the TDT control inside the `decoder_load` stage. These checks hold the
CLI to that contract with fake prepare/finish functions and a fake
`mlx.core`, so a regression (uploads leaving the adopted stream, a
swallowed thread exception, a renamed or reshaped stage record, the join
escaping its position) fails on any host before it can reach hardware.
"""

import sys
import threading
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "overlay" / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "mlx-omarchy-parakeet"))

import coreml.vulkan_decoder as vulkan_decoder  # noqa: E402
import coreml.vulkan_decoder_step as vulkan_decoder_step  # noqa: E402
import mlx_omarchy_parakeet as cli  # noqa: E402

MAIN = threading.main_thread()
GPU = object()


class FakeMx(types.ModuleType):
    """Minimal `mlx.core` stand-in that records stream and sync usage."""

    def __init__(self):
        super().__init__("mlx.core")
        self.gpu = GPU
        self.entered = []      # stream objects entered via mx.stream(...)
        self.synchronized = []  # stream objects passed to mx.synchronize(...)
        self.default_requested = 0

    @contextmanager
    def stream(self, s):
        self.entered.append(s)
        yield

    def default_stream(self, device):
        self.default_requested += 1
        return "thread-local-default"

    def synchronize(self, s):
        self.synchronized.append(s)


@pytest.fixture()
def fake_mx(monkeypatch):
    mx = FakeMx()
    package = types.ModuleType("mlx")
    package.core = mx
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    return mx


@pytest.fixture()
def fake_load(monkeypatch):
    """Replace the load phases; record which thread each ran on."""
    seen: list[tuple[str, bool]] = []
    state: dict = {"cpu_error": None, "error_at": None}
    parts, packing, decoder = object(), object(), object()
    state["packing_sentinel"] = packing
    state["decoder_sentinel"] = decoder

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
        if state["error_at"] == "finish_decoder":
            raise RuntimeError("upload failed")
        return decoder

    def finish_step_weights(packed):
        seen.append(("finish_step_weights", threading.current_thread() is MAIN))
        assert packed is packing
        if state["error_at"] == "finish_step_weights":
            raise RuntimeError("upload failed")
        return packed

    monkeypatch.setattr(vulkan_decoder, "prepare_decoder", prepare_decoder)
    monkeypatch.setattr(
        vulkan_decoder_step, "prepare_step_weights", prepare_step_weights)
    monkeypatch.setattr(vulkan_decoder, "finish_decoder", finish_decoder)
    monkeypatch.setattr(
        vulkan_decoder_step, "finish_step_weights", finish_step_weights)
    state["seen"] = seen
    return state


def test_uploads_run_in_thread_on_the_adopted_stream(
        fake_load, fake_mx, monkeypatch):
    stream = object()  # the "main thread's" stream object, as passed in
    controls = []
    monkeypatch.setattr(
        cli, "_TdtControl",
        lambda packed, scratch_root:
        controls.append((packed, scratch_root)) or "control")
    thread, state = cli._start_decoder_load(Path("/tmp"), stream)
    assert thread is not MAIN
    scratch = Path("/tmp/scratch-x")
    loaded, control = cli._join_decoder_load(thread, state, scratch)
    assert fake_load["seen"] == [
        ("prepare_decoder", False),
        ("prepare_step_weights", False),
        ("finish_decoder", False),
        ("finish_step_weights", False),
    ]
    assert loaded is fake_load["decoder_sentinel"] and control == "control"
    # The worker adopted EXACTLY the passed stream: a thread-local default
    # stream (what mx.stream(mx.gpu) would create) must never appear.
    assert fake_mx.entered == [stream]
    assert fake_mx.synchronized == [stream]
    assert state["cpu_thread_wall_ns"] >= 0
    assert controls == [(fake_load["packing_sentinel"], scratch)]


def test_stream_mismatch_is_visible(fake_load, fake_mx):
    """A regression that uploads on any stream other than the one passed
    in fails this check: the fake records what the worker entered."""
    stream_a, stream_b = object(), object()
    thread_a, _ = cli._start_decoder_load(Path("/tmp"), stream_a)
    thread_a.join()
    assert fake_mx.entered == [stream_a]
    assert stream_b not in fake_mx.entered
    thread_b, state_b = cli._start_decoder_load(Path("/tmp"), stream_b)
    thread_b.join()
    assert fake_mx.entered[-1] is stream_b
    assert fake_mx.default_requested == 0  # never a thread-local default
    cli._join_decoder_load(thread_b, state_b, Path("/tmp/scratch-y"))
    assert fake_mx.synchronized == [stream_b]


def test_upload_error_propagates_at_join_and_thread_is_joined(
        fake_load, fake_mx):
    fake_load["error_at"] = "finish_step_weights"
    thread, state = cli._start_decoder_load(Path("/tmp"), object())
    with pytest.raises(RuntimeError, match="upload failed"):
        cli._join_decoder_load(thread, state, Path("/tmp/scratch-x"))
    assert not thread.is_alive()


def test_decoder_load_stage_lands_after_encoder_before_begin_chunk():
    import inspect

    chunks = inspect.getsource(cli._decode_chunks)
    mel = chunks.index('"mel_frontend')
    start = chunks.index("_start_decoder_load")
    enc = chunks.index('"encoder_ane')
    load = chunks.index('"decoder_load"')
    begin = chunks.index("control.begin_chunk")
    assert mel < start < enc < load < begin
    # The pipeline stages no decoder_load and passes cache_dir through.
    pipeline = inspect.getsource(cli._run_pipeline)
    assert '"decoder_load"' not in pipeline
    assert "_start_decoder_load" not in pipeline
    assert "chunk.control" in pipeline


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
