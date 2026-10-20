"""A finished request's extracted KV must not stay reachable while the worker waits for the next request (#245).

mlx-lm's generation loop keeps `r` (the last response of the finished request) alive after the loop ends, and a finished
response carries `prompt_cache=extract_cache(i)`: the request's whole KV. The worker thread then waits for the next request
with that frame alive, so serve held about one extra KV copy after an answer until the next request replaced it."""
from collections import deque
import gc
import logging
from queue import Queue
import weakref
from types import SimpleNamespace

import pytest

import subprocess
from pathlib import Path

import pytest

from tests.test_apply_mlxlm_series import _pristine_venv
from tests.test_mlxlm_generation_failure import extract

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", params=["0.31.3", "0.32.0"])
def patched_source(request, tmp_path_factory):
    root = _pristine_venv(tmp_path_factory.mktemp("release-finished"), request.param)
    series = "patches" if request.param == "0.31.3" else "patches/mlx-lm-0.32"
    site = root / "lib/python3.11/site-packages"
    for name in ("mlx-lm-generation-failure.patch", "mlx-lm-release-finished.patch"):
        if name == "mlx-lm-release-finished.patch" and not (REPO / series / name).exists():
            continue
        subprocess.run(["patch", "--directory", str(site), "--strip=1", "--fuzz=0", "--quiet"],
                       input=(REPO / series / name).read_bytes(), check=True)
    return site / "mlx_lm/server.py"


def test_finished_request_cache_released_while_waiting(patched_source):
    server_source = patched_source
    queue = Queue()
    owned = []
    alive_while_waiting = []
    tokenizer = SimpleNamespace(has_tool_calling=False, has_thinking=False, tool_parser=None,
                                detokenizer=SimpleNamespace(add_token=lambda token: None, finalize=lambda: None, last_segment="ok"))
    provider = SimpleNamespace(load_default=lambda: None, model_key="model", load=lambda *args: (None, tokenizer),
                               cli_args=SimpleNamespace(decode_concurrency=1, prompt_concurrency=1,
                                                        prefill_step_size=512, prompt_cache_bytes=None))

    class Buffer:
        nbytes = 8

        def __init__(self):
            self.tokens = [1]

        def is_trimmable(self):
            return False

    class Batch:
        def __init__(self, *args, **kwargs):
            pass

        def insert_segments(self, **kwargs):
            self.prompt = kwargs["segments"][0][0]
            return [1]

        def close(self):
            pass

        def next(self):
            buffer = Buffer()
            owned.append(weakref.ref(buffer))
            return [], [SimpleNamespace(uid=1, token=7, current_state="normal", match_sequence=None,
                                        logprobs={7: SimpleNamespace(item=lambda: 0.0)}, finish_reason="length",
                                        all_tokens=self.prompt + [7], prompt_cache=[buffer])]

        def extract_cache(self, ids):
            return {}

    namespace = {"mx": SimpleNamespace(default_stream=lambda device: None, default_device=lambda: None, clear_cache=lambda: None),
                 "BatchGenerator": Batch, "GenerationContext": lambda **kwargs: SimpleNamespace(**kwargs, _should_stop=False),
                 "Response": lambda *args: args, "logging": logging,
                 "_make_sampler": lambda *args: None, "_make_logits_processors": lambda *args: None,
                 "_format_top_logprobs": lambda *args: [], "gc": gc}
    cls = extract(server_source, "ResponseGenerator", {"_generate", "cli_args", "_fail_batch"}, namespace)
    worker = cls()
    worker.model_provider = provider
    # a zero-size prompt cache, as `serve` runs (--prompt-cache-size 0): nothing keeps the extracted KV
    worker.prompt_cache = SimpleNamespace(insert_cache=lambda *a, **k: None, fetch_nearest_cache=lambda *a, **k: ([], a[1]),
                                          nbytes=0, trim_to=lambda **k: None, __len__=lambda self: 0)
    worker._is_distributed = False
    worker._stop = False
    worker._time_budget = [None]
    args = SimpleNamespace(model=SimpleNamespace(model="model", adapter=None, draft=None), stop_words=[], max_tokens=1, top_logprobs=0)
    calls = deque([(queue, [1, 2, 3], args), None, "wait", None])

    def next_request(timeout):
        item = calls.popleft()
        if item == "wait":
            # the worker is idle here, waiting for the next request
            gc.collect()
            alive_while_waiting.extend(ref() is not None for ref in owned)
            worker._stop = True
            return None
        return item

    worker._next_request = next_request
    worker._is_batchable = lambda args: True
    worker._tokenize = lambda tokenizer, prompt, args: (prompt, [prompt], ["user"], "normal")
    worker._make_state_machine = lambda *args: (None, None)
    worker._log_cache_stats = lambda: None
    worker._share_object = lambda value: value
    provider.reset = lambda: None
    worker._generate()
    assert owned, "the fake batch never produced a finished response"
    assert alive_while_waiting, "the worker never reached its idle wait"
    assert not any(alive_while_waiting), "a finished request's KV is still reachable while the worker waits"
