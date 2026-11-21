# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""max_kv_size on the prompt-lookup route (mlx-lm 0.31.3 series).

stream_generate gates the lookup branch on ``kwargs.get("max_kv_size") is
None``, which is true both when the key is absent and when the caller passed
``max_kv_size=None`` explicitly. Wrappers that build a kwargs dict with
explicit Nones take the lookup branch carrying the key, and the 0.31.3-series
pop list drops prompt_progress_callback and input_embeddings but not
max_kv_size — so speculative_generate_step is called with an unexpected
keyword argument and the stream raises TypeError. The 0.32-series port pops
max_kv_size with the others; the 0.31.3 patch needs the same.

These tests exec the patched stream_generate with stub collaborators and drive
it through the dispatch: the spec stub takes exactly the kwargs its real
signature accepts, so a leftover key surfaces as TypeError on the holed patch
and passes after the fix. A real max_kv_size value still routes plain decode.
"""
import ast
import contextlib
import subprocess
import time
import types
import typing

import pytest

from tests.test_apply_mlxlm_series import SCRIPT, _pristine_venv


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    root = _pristine_venv(tmp_path_factory.mktemp("maxkv-lookup"), "0.31.3")
    r = subprocess.run(["bash", str(SCRIPT), str(root)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-400:] + r.stderr[-400:]
    return root / "lib" / "python3.11" / "site-packages" / "mlx_lm"


def _harness(site):
    """stream_generate exec'd from the patched tree plus recording stubs."""
    src = (site / "generate.py").read_text()
    tree = ast.parse(src)
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "stream_generate")
    fn.decorator_list = []
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)

    calls = {}

    def speculative_generate_step(prompt, model, draft_model, *,
                                  num_draft_tokens, max_tokens):
        calls["spec"] = {"num_draft_tokens": num_draft_tokens,
                         "max_tokens": max_tokens}
        return iter([(7, None, True)])

    def generate_step(prompt, model, **kw):
        calls["gen"] = kw
        return iter([(7, None)])

    class _Arr:
        size = 4

    class _Detok:
        last_segment = ""

        def add_token(self, t):
            pass

        def finalize(self):
            pass

    class _Tok:
        detokenizer = _Detok()
        eos_token_ids = frozenset()
        bos_token = None

    class _Resp:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    @contextlib.contextmanager
    def wired_limit(model, streams):
        yield

    ns = {
        "os": __import__("os"),
        "time": time,
        "mx": types.SimpleNamespace(array=_Arr, get_peak_memory=lambda: 0),
        "nn": types.SimpleNamespace(Module=object),
        "Union": typing.Union,
        "Optional": typing.Optional,
        "List": typing.List,
        "Tuple": typing.Tuple,
        "Any": typing.Any,
        "Generator": typing.Generator,
        "PreTrainedTokenizer": object,
        "TokenizerWrapper": _Tok,
        "GenerationResponse": _Resp,
        "generate_step": generate_step,
        "speculative_generate_step": speculative_generate_step,
        "wired_limit": wired_limit,
        "generation_stream": object(),
    }
    exec(compile(module, "generate.py", "exec"), ns)
    return ns["stream_generate"], calls, _Tok, _Arr


def test_explicit_none_max_kv_size_takes_the_lookup_route(site, monkeypatch):
    """max_kv_size=None passes the gate; the dispatch must not leak it."""
    stream_generate, calls, tok, arr = _harness(site)
    monkeypatch.setenv("MLX_OMARCHY_PROMPT_LOOKUP", "3")
    next(stream_generate(object(), tok(), arr(), max_tokens=8,
                         max_kv_size=None))
    assert calls["spec"] == {"num_draft_tokens": 3, "max_tokens": 8}


def test_real_max_kv_size_still_routes_plain_decode(site, monkeypatch):
    stream_generate, calls, tok, arr = _harness(site)
    monkeypatch.setenv("MLX_OMARCHY_PROMPT_LOOKUP", "3")
    next(stream_generate(object(), tok(), arr(), max_tokens=8,
                         max_kv_size=512))
    assert calls["gen"]["max_kv_size"] == 512
    assert calls["gen"]["max_tokens"] == 8
    assert "spec" not in calls
