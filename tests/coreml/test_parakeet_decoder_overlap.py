# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host contract of the overlapped decoder load (no ANE, Vulkan or mlx).

The transcribe CLI runs the decoder/joint load (CPU preparation AND GPU
uploads) on one worker thread started right before chunk 0's encoder
stage, on the MAIN thread's stream object (default streams are
thread-local); the main thread joins, synchronizes that stream and builds
the TDT control inside the `decoder_load` stage. The GPU-warm thread
starts BEFORE the audio_load stage and is stopped, on every exit path, in
a finally that spans audio_load and island_open. These checks hold the
CLI to that contract with fake prepare/finish functions and a fake
`mlx.core`, so a regression (uploads leaving the adopted stream, a
swallowed thread exception, a renamed or reshaped stage record, the join
escaping its position, a warm thread leaking past the chunk-loop
boundary) fails on any host before it can reach hardware.

Fake limits (untested here, by design): the fake cannot demonstrate real
scheduler ordering (one worker per stream), the device queue mutex, or
the StreamContext spare-stream creation in a fresh thread — those rest on
the cited MLX source lines in B2-source-proposal.md (mlx/stream.cpp,
scheduler.cpp, omarchy/encoder.cpp), not on this fake. The warm-position
check is an AST order check, not a behavioural run: a full _run_pipeline
run without mlx would need fakes for the mel/encoder/TDT stages and the
installed mlx-omarchy distribution metadata that _golden_report reads.
"""

import ast
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
    """Minimal `mlx.core` stand-in modelling w7K's source reading.

    `set_default_stream(s)` sets THIS thread's current stream (no
    StreamContext, no spare stream). `stream(mx.gpu)` is the device-form
    context: its constructor resolves `default_stream(default_device())`
    to the CURRENT default stream and then sets that same stream — so a
    device-form context inside the worker resolves to the adopted stream.
    Every call is recorded in `events` (ordered) so the test can require
    set_default_stream to be the worker's FIRST GPU-side action.
    """

    def __init__(self):
        super().__init__("mlx.core")
        self.gpu = GPU
        self.current = None            # this (fake) thread's default stream
        self.events = []               # ordered ("set_default"|"enter"|...)
        self.synchronized = []
        self.default_requested = 0

    def set_default_stream(self, s):
        self.events.append(("set_default", s))
        self.current = s

    def default_stream(self, device):
        self.default_requested += 1
        return self.current

    def synchronize(self, s):
        self.events.append(("synchronize", s))
        self.synchronized.append(s)

    @contextmanager
    def stream(self, s):
        if s is GPU:
            resolved = self.default_stream(GPU)  # StreamContext ctor read
            self.events.append(("enter", resolved))
        else:
            self.events.append(("enter", s))
            self.current = s
        yield

    # _run_pipeline's pre-stage setup on the main thread.
    def set_default_device(self, device):
        pass

    def set_cache_limit(self, limit):
        pass


@pytest.fixture()
def fake_mx(monkeypatch):
    mx = FakeMx()
    package = types.ModuleType("mlx")
    package.core = mx
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    # The decoder modules cache their mx handle; clear so every test's
    # fake instance is the one the inner mx.stream(mx.gpu) blocks see.
    vulkan_decoder._mlx.cache_clear()
    vulkan_decoder_step._mx.cache_clear()
    yield mx
    monkeypatch.undo()
    vulkan_decoder._mlx.cache_clear()
    vulkan_decoder_step._mx.cache_clear()


@pytest.fixture()
def fake_load(monkeypatch, fake_mx):
    """Replace the load phases; record which thread each ran on.

    The fake finish bodies enter the device-form `mx.stream(mx.gpu)`
    context exactly like the real ones, so the recorded events show what
    stream those contexts resolved to inside the worker."""
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
        with fake_mx.stream(fake_mx.gpu):
            if state["error_at"] == "finish_decoder":
                raise RuntimeError("upload failed")
        return decoder

    def finish_step_weights(packed):
        seen.append(("finish_step_weights", threading.current_thread() is MAIN))
        assert packed is packing
        with fake_mx.stream(fake_mx.gpu):
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


class FakeWarm:
    """Stand-in for _GpuWarm: a real parked thread and stop bookkeeping."""

    instances: list = []

    def __init__(self, mx):
        self.started = 0
        self.stopped = 0
        self._release = threading.Event()
        self.thread = threading.Thread(target=self._park, daemon=True)
        FakeWarm.instances.append(self)

    def _park(self):
        self._release.wait(5)

    def start(self):
        self.started += 1
        self.thread.start()

    def stop(self):
        self.stopped += 1
        self._release.set()
        self.thread.join()

    @property
    def thread_alive(self):
        return self.thread.is_alive()


@pytest.fixture()
def fake_warm(monkeypatch):
    FakeWarm.instances = []
    monkeypatch.setattr(cli, "_GpuWarm", FakeWarm)
    return FakeWarm


def _fake_lock():
    lock = types.SimpleNamespace()
    lock.audio = types.SimpleNamespace(sample_rate=16000)
    lock.numerical_contract = types.SimpleNamespace(
        nan_count_allowed=0, inf_count_allowed=0)
    lock.files = [types.SimpleNamespace(path="tokenizer.json", sha256="tok")]
    lock.tdt = types.SimpleNamespace(durations=[0, 2])
    return lock


def _fake_pin():
    return {"e2e": {"decode_control": "host"}, "assets": []}


def _run_pipeline_until_audio_fails(monkeypatch, fake_warm,
                                    raise_in_open=False):
    """Drive _run_pipeline into the audio_load/island_open failure path."""
    args = types.SimpleNamespace(deadline_ms=20000, repeat=1,
                                 keep_scratch=False)
    monkeypatch.setattr(cli, "_load_audio", self_raiser())
    # stage() reads the process-wide GPU counters via coreml.vulkan_mel's
    # trace_snapshot, which needs the installed libmlx — fake it.
    monkeypatch.setattr(
        "coreml.vulkan_mel.trace_snapshot",
        lambda: {"vk_submissions": 0, "vk_buffer_copies": 0})
    if raise_in_open:
        monkeypatch.setattr(cli, "_open_encoder", self_raiser())
    with pytest.raises(RuntimeError, match="boom"):
        cli._run_pipeline(
            args, _fake_pin(), _fake_lock(), Path("/tmp/cache"),
            Path("/tmp/fixture.flac"), "0" * 64, Path("/tmp/worker"),
            Path("/tmp/share"), Path("/tmp/scratch"),
            Path("/tmp/out"), False, None)


def self_raiser():
    def _raise(*args, **kwargs):
        raise RuntimeError("boom")
    return _raise


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
    # The worker's first GPU-side action adopts EXACTLY the passed stream
    # via set_default_stream; the two inner device-form contexts (the
    # `mx.stream(mx.gpu)` blocks in finish_decoder / finish_step_weights)
    # resolve to that same adopted stream and enter no other.
    assert fake_mx.events == [
        ("set_default", stream),
        ("enter", stream),   # finish_decoder's mx.stream(mx.gpu)
        ("enter", stream),   # finish_step_weights' mx.stream(mx.gpu)
        ("synchronize", stream),  # the join's explicit barrier
    ]
    assert fake_mx.synchronized == [stream]
    assert state["cpu_thread_wall_ns"] >= 0
    assert controls == [(fake_load["packing_sentinel"], scratch)]


def test_stream_mismatch_is_visible(fake_load, fake_mx):
    """A regression that uploads on any stream other than the one passed
    in fails this check: the fake records what the worker set and what
    its device-form contexts resolved to."""
    stream_a, stream_b = object(), object()
    thread_a, _ = cli._start_decoder_load(Path("/tmp"), stream_a)
    thread_a.join()
    assert [kind for kind, _ in fake_mx.events] == [
        "set_default", "enter", "enter"]
    assert {s for _, s in fake_mx.events} == {stream_a}
    thread_b, state_b = cli._start_decoder_load(Path("/tmp"), stream_b)
    thread_b.join()
    assert {s for _, s in fake_mx.events[-3:]} == {stream_b}
    # Untested note: the fake cannot show real scheduler/queue ordering
    # (single worker per stream, device queue mutex) — those rest on the
    # cited MLX source lines in the B2 proposal, not on this fake.
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


def test_warm_spans_audio_and_island_open_only_ast():
    """AST order check (behavioural full-run infeasible without mlx —
    see module docstring): warm.start precedes the audio_load stage;
    warm.stop sits in a finally whose body spans audio_load and
    island_open and precedes the _decode_chunks call."""
    tree = ast.parse(
        (TOOLS / "mlx-omarchy-parakeet" / "mlx_omarchy_parakeet.py")
        .read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_run_pipeline")
    start_line = audio_line = stop_line = decode_line = open_line = None
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        name = ""
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        elif isinstance(node.func, ast.Name):
            name = node.func.id
        if name == "start" and isinstance(node.func.value, ast.Name) \
                and node.func.value.id == "warm":
            start_line = node.lineno
        elif name == "stop" and isinstance(node.func.value, ast.Name) \
                and node.func.value.id == "warm":
            stop_line = node.lineno
        elif name == "stage" and any(
                isinstance(a, ast.Constant) and a.value == "audio_load"
                for a in node.args):
            audio_line = node.lineno
        elif name == "stage" and any(
                isinstance(a, ast.Constant) and a.value == "island_open"
                for a in node.args):
            open_line = node.lineno
        elif name == "_decode_chunks":
            decode_line = node.lineno
    assert None not in (start_line, audio_line, stop_line, decode_line)
    assert start_line < audio_line, "warm must start before audio_load"
    assert open_line < stop_line < decode_line, (
        "warm must stop after island_open and before the chunk loop")
    # The stop must sit in a finally whose guarded body contains both the
    # audio_load and island_open stage calls.
    for node in ast.walk(fn):
        if isinstance(node, ast.Try) and node.finalbody:
            stops = [n for n in ast.walk(ast.Module(
                body=node.finalbody, type_ignores=[]))
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "stop"]
            lines = [n.lineno for n in ast.walk(ast.Module(
                body=node.body, type_ignores=[]))
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "stage"]
            if stops and lines:
                assert min(lines) <= audio_line <= max(lines)
                assert min(lines) <= open_line <= max(lines)
                assert stops[0].lineno < decode_line
                break
    else:
        pytest.fail("warm.stop not found in a try/finally")


def test_audio_load_exception_stops_and_joins_warm_thread(
        monkeypatch, fake_warm, fake_mx):
    _run_pipeline_until_audio_fails(monkeypatch, fake_warm)
    warm = fake_warm.instances[-1]
    assert warm.started == 1
    assert warm.stopped == 1
    assert not warm.thread_alive


def test_island_open_exception_stops_and_joins_warm_thread(
        monkeypatch, fake_warm, fake_mx):
    _run_pipeline_until_audio_fails(monkeypatch, fake_warm, raise_in_open=True)
    warm = fake_warm.instances[-1]
    assert warm.started == 1
    assert warm.stopped == 1
    assert not warm.thread_alive


def test_prewarm_off_creates_no_thread(monkeypatch, fake_mx):
    monkeypatch.setenv("MLX_OMARCHY_PK_PREWARM", "off")
    warm = cli._GpuWarm(fake_mx)  # env off: no mx.ones/eval, no thread
    assert warm.thread is None
    warm.start()
    assert warm.thread is None
    warm.stop()
    assert warm.thread is None


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
