"""A finished request's KV must not stay reachable while the worker decodes or waits for the next request (#245).

Two bindings in mlx-lm's generation loop kept KV alive:
- `r` (the last response of the finished request) carries `prompt_cache=extract_cache(i)`, the request's whole KV, and stayed bound while
  the worker waited for the next request;
- `cache` (the loop variable of the end-of-segment extraction `for uid, (cache, cache_key) in caches.items()`) stayed bound after `del caches`:
  with --prompt-cache-size 0 that is the whole prompt's KV, alive through decode and while the worker waits.

A finished response also must not stay reachable through the next decode step when other requests are still decoding: the old responses
are only rebound after `batch_generator.next()` returns."""
from collections import deque
import gc
import logging
from pathlib import Path
from queue import Queue
import subprocess
from types import SimpleNamespace
import weakref

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
        subprocess.run(["patch", "--directory", str(site), "--strip=1", "--fuzz=0", "--quiet"],
                       input=(REPO / series / name).read_bytes(), check=True)
    return site / "mlx_lm/server.py"


class Buffer:
    nbytes = 8

    def __init__(self):
        self.tokens = [1]

    def is_trimmable(self):
        return False


def run_worker(server_source, segmented, steps=1):
    """One request through the patched worker. Returns (finished-response KV alive while waiting,
    segment KV alive during decode, segment KV alive while waiting)."""
    queue = Queue()
    final = []
    segment = []
    seen = {"decode": [], "waiting": [], "final_waiting": [], "next_step": []}
    tokenizer = SimpleNamespace(has_tool_calling=False, has_thinking=False, tool_parser=None,
                                detokenizer=SimpleNamespace(add_token=lambda token: None, finalize=lambda: None, last_segment="ok"))
    provider = SimpleNamespace(load_default=lambda: None, model_key="model", load=lambda *args: (None, tokenizer),
                               cli_args=SimpleNamespace(decode_concurrency=1, prompt_concurrency=1,
                                                        prefill_step_size=512, prompt_cache_bytes=None))

    class Batch:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def insert_segments(self, **kwargs):
            self.prompt = kwargs["segments"][0][0]
            return [1]

        def close(self):
            pass

        def next(self):
            self.calls += 1
            if steps > 1 and self.calls == 2:
                # a second decode step while the first finished request is gone: its response must already be unreachable
                gc.collect()
                seen["next_step"].extend(ref() is not None for ref in final)
                return [], []
            if segmented and self.calls == 1:
                # the user segment ends: the loop extracts the segment cache (the prompt's KV)
                return [SimpleNamespace(uid=1, progress=(0, 1), end_of_segment=True, end_of_prompt=False)], []
            if segmented and self.calls == 2:
                gc.collect()
                seen["decode"].extend(ref() is not None for ref in segment)
            buffer = Buffer()
            final.append(weakref.ref(buffer))
            return [], [SimpleNamespace(uid=1, token=7, current_state="normal", match_sequence=None,
                                        logprobs={7: SimpleNamespace(item=lambda: 0.0)}, finish_reason="length",
                                        all_tokens=self.prompt + [7], prompt_cache=[buffer])]

        def extract_cache(self, ids):
            if not (segmented and ids):
                return {}
            buffer = Buffer()
            segment.append(weakref.ref(buffer))
            return {1: ([buffer], [9, 9])}

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
    worker._time_budget = [None] * steps
    args = SimpleNamespace(model=SimpleNamespace(model="model", adapter=None, draft=None), stop_words=[], max_tokens=1, top_logprobs=0)
    calls = deque([(queue, [1, 2, 3], args), None, None, "wait", None])

    def next_request(timeout):
        item = calls.popleft()
        if item == "wait":
            # the worker is idle here, waiting for the next request
            gc.collect()
            seen["final_waiting"].extend(ref() is not None for ref in final)
            seen["waiting"].extend(ref() is not None for ref in segment)
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
    assert final, "the fake batch never produced a finished response"
    assert seen["final_waiting"], "the worker never reached its idle wait"
    if segmented:
        assert segment and seen["decode"] and seen["waiting"], "the segment extraction never ran"
    if steps > 1:
        assert seen["next_step"], "the second decode step never ran"
    return seen


def test_finished_request_cache_released_while_waiting(patched_source):
    seen = run_worker(patched_source, segmented=False)
    assert not any(seen["final_waiting"]), "a finished request's KV is still reachable while the worker waits"


def test_segment_cache_released_during_decode_and_while_waiting(patched_source):
    seen = run_worker(patched_source, segmented=True)
    assert not any(seen["final_waiting"]), "a finished request's KV is still reachable while the worker waits"
    assert not any(seen["decode"]), "the extracted segment cache (the prompt's KV) is still bound during decode"
    assert not any(seen["waiting"]), "the extracted segment cache (the prompt's KV) is still bound while the worker waits"


def test_finished_response_released_before_the_next_decode_step(patched_source):
    seen = run_worker(patched_source, segmented=False, steps=2)
    assert not any(seen["next_step"]), "a finished request's KV is still reachable during the next decode step"
    assert not any(seen["final_waiting"]), "a finished request's KV is still reachable while the worker waits"
