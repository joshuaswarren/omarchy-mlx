"""Memory-aware prefill chunk for direct mlx-lm use on a small Honeykrisp heap.

On a 15.1 GiB host the Honeykrisp heap is mesa's 50% default (about 7.55 GiB). Qwen3.8-9B 4-bit (4.71 GiB of weights) prefilling an
8k prompt with mlx-lm's default 2048-token chunk fails with VK_ERROR_OUT_OF_DEVICE_MEMORY; with a 512-token chunk the same run
completes (measured on the 13-inch M1, peak 6.18 GB). The patch makes generate_step pick the chunk from the heap headroom when the
caller gave none. An explicit value always wins, and the rule never raises the chunk above the old default, so a host with room
keeps 2048."""
import ast
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.test_apply_mlxlm_series import _pristine_venv

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "serve"))
from mlx_omarchy_serve import budget  # noqa: E402

PATCH = "mlx-lm-prefill-step-memory.patch"
GIB = 1024 ** 3
HELPERS = {"_omarchy_prefill_step", "_omarchy_prefill_headroom_bytes"}


@pytest.fixture(scope="module", params=["0.31.3", "0.32.0"])
def patched(request, tmp_path_factory):
    root = _pristine_venv(tmp_path_factory.mktemp("prefill-step"), request.param)
    series = "patches" if request.param == "0.31.3" else "patches/mlx-lm-0.32"
    site = root / "lib/python3.11/site-packages"
    subprocess.run(["patch", "--directory", str(site), "--strip=1", "--fuzz=0", "--quiet"],
                   input=(REPO / series / PATCH).read_bytes(), check=True)
    return request.param, site / "mlx_lm/generate.py"


def namespace(source, **extra):
    """The patch's module-level constants and helper functions, executed on their own."""
    tree = ast.parse(source.read_text())
    keep = [node for node in tree.body
            if (isinstance(node, ast.FunctionDef) and node.name in HELPERS)
            or (isinstance(node, ast.Assign) and any(getattr(t, "id", "").startswith("_OMARCHY_PREFILL") for t in node.targets))]
    module = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = dict(extra)
    exec(compile(module, str(source), "exec"), ns)
    return ns


def chooser(source):
    return namespace(source)["_omarchy_prefill_step"]


def test_explicit_value_always_wins(patched):
    step = chooser(patched[1])
    assert step(512, 8192, 0) == 512
    assert step(4096, 1000, 1) == 4096


def test_unknown_headroom_keeps_the_old_default(patched):
    assert chooser(patched[1])(None, 8192, None) == 2048


def test_a_host_with_room_keeps_2048(patched):
    step = chooser(patched[1])
    assert step(None, 8192, 16 * GIB) == 2048
    assert step(None, 32768, 40 * GIB) == 2048


def test_qwen38_9b_on_the_13_inch_is_clamped(patched):
    # heap 15843696 kB / 2, weights about 5.06e9 bytes: about 3.05e9 bytes of headroom.
    headroom = 15843696 * 1024 // 2 - 5_060_000_000
    step = chooser(patched[1])
    assert step(None, 8192, headroom) == 1024
    assert step(None, 32768, headroom) == 256
    assert step(None, 8192, headroom) < 2048


def test_chunk_never_grows_when_headroom_shrinks_or_context_grows(patched):
    step = chooser(patched[1])
    for context in (512, 2048, 8192, 32768):
        sizes = [step(None, context, h) for h in (40 * GIB, 8 * GIB, 4 * GIB, 2 * GIB, GIB, GIB // 4, 1)]
        assert sizes == sorted(sizes, reverse=True)
        assert all(64 <= s <= 2048 for s in sizes)
    for headroom in (GIB, 3 * GIB, 8 * GIB):
        sizes = [step(None, c, headroom) for c in (512, 2048, 8192, 32768)]
        assert sizes == sorted(sizes, reverse=True)


def test_estimator_constants_match_the_serve_budget(patched):
    ns = namespace(patched[1])
    assert ns["_OMARCHY_PREFILL_TOKEN_BYTES"] == budget.PREFILL_TOKEN_BYTES
    assert ns["_OMARCHY_PREFILL_SCORE_HEADS"] == budget.PREFILL_SCORE_HEADS
    assert ns["_OMARCHY_PREFILL_SCORE_TENSOR_BYTES"] == budget.PREFILL_SCORE_TENSOR_BYTES
    assert ns["_OMARCHY_PREFILL_MARGIN"] == budget.PREFILL_MARGIN
    # the chosen chunk fits under the same estimate the serve path admits requests with
    step = chooser(patched[1])
    for context, headroom in ((8192, 3_050_000_000), (32768, 3_050_000_000), (4096, 2 * GIB)):
        chosen = step(None, context, headroom)
        assert budget.prefill_scratch_bytes(chosen, context) <= headroom or chosen == 64


class Fs:
    def __init__(self, meminfo):
        self.meminfo = meminfo

    def open(self, path, *args, **kwargs):
        if path != "/proc/meminfo" or self.meminfo is None:
            raise FileNotFoundError(path)
        import io
        return io.StringIO(self.meminfo)


MEMINFO = "MemTotal:       15843696 kB\nMemFree:         100 kB\n"


def headroom_fn(source, monkeypatch, *, meminfo=MEMINFO, active=5_000_000_000, installed=True, hk=None):
    import importlib.metadata as md

    def version(name):
        if not installed:
            raise md.PackageNotFoundError(name)
        return "0.32.4.dev"

    monkeypatch.setattr(md, "version", version)
    if hk is None:
        monkeypatch.delenv("HK_SYSMEM", raising=False)
    else:
        monkeypatch.setenv("HK_SYSMEM", hk)
    mx = SimpleNamespace(get_active_memory=lambda: active)
    return namespace(source, mx=mx, open=Fs(meminfo).open)["_omarchy_prefill_headroom_bytes"]


def test_headroom_defaults_to_half_of_memtotal_minus_arrays(patched, monkeypatch):
    fn = headroom_fn(patched[1], monkeypatch)
    assert fn() == 15843696 * 1024 // 2 - 5_000_000_000


def test_headroom_honours_the_backends_hk_sysmem(patched, monkeypatch):
    fn = headroom_fn(patched[1], monkeypatch, hk=str(11 * GIB))
    assert fn() == 11 * GIB - 5_000_000_000


def test_headroom_ignores_an_unparsable_hk_sysmem(patched, monkeypatch):
    fn = headroom_fn(patched[1], monkeypatch, hk="lots")
    assert fn() == 15843696 * 1024 // 2 - 5_000_000_000


def test_headroom_is_none_off_omarchy_or_without_procfs(patched, monkeypatch):
    assert headroom_fn(patched[1], monkeypatch, installed=False)() is None
    assert headroom_fn(patched[1], monkeypatch, meminfo=None)() is None


def test_headroom_is_none_when_the_memory_api_is_missing(patched, monkeypatch):
    import importlib.metadata as md
    monkeypatch.setattr(md, "version", lambda name: "x")
    monkeypatch.delenv("HK_SYSMEM", raising=False)
    fn = namespace(patched[1], mx=SimpleNamespace(), open=Fs(MEMINFO).open)["_omarchy_prefill_headroom_bytes"]
    assert fn() is None


def _assignments_from_call(fn, target, func):
    """Assignments to `target` whose value is a call of the plain name `func`."""
    def targets_name(node):
        return any(getattr(t, "id", "") == target for t in node.targets)

    def calls_func(node):
        return isinstance(node.value, ast.Call) and getattr(node.value.func, "id", "") == func

    return [n for n in ast.walk(fn) if isinstance(n, ast.Assign) and targets_name(n) and calls_func(n)]


def test_generate_step_resolves_the_chunk_before_prefill(patched):
    version, source = patched
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "generate_step")
    args = {a.arg: d for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults)}
    assert isinstance(args["prefill_step_size"], ast.Constant) and args["prefill_step_size"].value is None
    resolved = _assignments_from_call(fn, "prefill_step_size", "_omarchy_prefill_step")
    assert len(resolved) == 1
    first_use = next(n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "min"
                     and any(getattr(a, "id", "") == "prefill_step_size" for a in n.args))
    assert resolved[0].lineno < first_use.lineno


def test_the_032_cli_no_longer_forces_the_old_default(patched):
    version, source = patched
    if version != "0.32.0":
        pytest.skip("0.31.3 has no --prefill-step-size option")
    tree = ast.parse(source.read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"
                and n.args and getattr(n.args[0], "value", "") == "--prefill-step-size")
    default = next(k.value for k in call.keywords if k.arg == "default")
    assert isinstance(default, ast.Constant) and default.value is None
