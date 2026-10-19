"""Exercise the patched upstream worker and HTTP handler without a GPU."""
import ast
from collections import deque
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
from queue import Queue
import subprocess
from threading import Thread
from types import SimpleNamespace
import weakref

import pytest

from tests.test_apply_mlxlm_series import _pristine_venv

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", params=["0.31.3", "0.32.0"])
def server_source(request, tmp_path_factory):
    root = _pristine_venv(tmp_path_factory.mktemp("generation-failure"), request.param)
    series = "patches" if request.param == "0.31.3" else "patches/mlx-lm-0.32"
    patch = REPO / series / "mlx-lm-generation-failure.patch"
    subprocess.run(["patch", "--directory", str(root / "lib/python3.11/site-packages"),
                    "--strip=1", "--fuzz=0", "--quiet"], input=patch.read_bytes(), check=True)
    return root / "lib/python3.11/site-packages/mlx_lm/server.py"


def extract(source, class_name, methods, namespace):
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    cls.bases = []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    module = ast.parse("from __future__ import annotations\n")
    module.body.append(cls)
    ast.fix_missing_locations(module)
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[class_name]


def prompt_cache(source):
    tree = ast.parse(source.with_name("models").joinpath("cache.py").read_text())
    names = {"PromptTrieResult", "PromptTrie", "LRUPromptCache", "can_trim_prompt_cache"}
    module = ast.parse("import copy\nfrom collections import deque\nfrom dataclasses import dataclass\nfrom typing import Any, Dict, List, Optional\n")
    module.body.extend(node for node in tree.body if getattr(node, "name", None) in names)
    ast.fix_missing_locations(module)
    namespace = {"__name__": __name__}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["LRUPromptCache"]()


@pytest.mark.parametrize("close_fails", [False, True])
def test_failed_batch_releases_cache_and_serves_next_request(server_source, close_fails, caplog):
    queues = [Queue(), Queue(), Queue()]
    owned = []
    batches = []
    provider = SimpleNamespace(load_default=lambda: None, model_key="model",
                               load=lambda *args: (None, tokenizer),
                               cli_args=SimpleNamespace(decode_concurrency=1, prompt_concurrency=1,
                                                        prefill_step_size=512, prompt_cache_bytes=None))
    tokenizer = SimpleNamespace(has_tool_calling=False, has_thinking=False, tool_parser=None,
                                detokenizer=SimpleNamespace(add_token=lambda token: None, finalize=lambda: None, last_segment="ok"))
    class Buffer:
        nbytes = 8
        def __init__(self):
            self.tokens = [42]
        def is_trimmable(self):
            return False
    class Batch:
        def __init__(self, *args, **kwargs):
            self.buffer = Buffer()
            owned.append(weakref.ref(self.buffer))
            self.index = len(batches)
            batches.append(weakref.ref(self))
        def insert_segments(self, **kwargs):
            self.count = getattr(self, "count", 0) + 1
            self.prompt = kwargs["segments"][0][0]
            self.caches = kwargs["caches"]
            if self.caches[0] is not None:
                self.caches[0][0].tokens.append(999)
            return [self.count]
        def close(self):
            if close_fails and self.index == 0:
                raise RuntimeError("private close failed")
        def next(self):
            if self.index == 0:
                assert self.count == 2
                raise RuntimeError("private prompt /private/model failed")
            worker._stop = True
            return [], [SimpleNamespace(uid=1, token=7, current_state="normal", match_sequence=None,
                logprobs={7: SimpleNamespace(item=lambda: 0.0)}, finish_reason="length",
                all_tokens=self.prompt + [7], prompt_cache=[])]
        def extract_cache(self, ids):
            return {}
    namespace = {"mx": SimpleNamespace(default_stream=lambda device: None, default_device=lambda: None,
                                        clear_cache=lambda: None),
                 "BatchGenerator": Batch, "GenerationContext": lambda **kwargs: SimpleNamespace(**kwargs, _should_stop=False),
                 "Response": lambda *args: args, "logging": logging,
                 "_make_sampler": lambda *args: None, "_make_logits_processors": lambda *args: None,
                 "_format_top_logprobs": lambda *args: [], "gc": __import__("gc")}
    cls = extract(server_source, "ResponseGenerator", {"_generate", "cli_args", "_fail_batch"}, namespace)
    worker = cls()
    worker.model_provider = provider
    cache = worker.prompt_cache = prompt_cache(server_source)
    cache.insert_cache("model", [1, 2], [Buffer()])
    worker._is_distributed = False
    worker._stop = False
    worker._time_budget = [None]
    args = SimpleNamespace(model=SimpleNamespace(model="model", adapter=None, draft=None),
                           stop_words=[], max_tokens=1, top_logprobs=0)
    requests = deque([(queues[0], [1, 2, 11], args), (queues[1], [1, 2, 22], args), None,
                      (queues[2], [9], args), None])
    worker._next_request = lambda timeout: requests.popleft()
    worker._is_batchable = lambda args: True
    worker._tokenize = lambda tokenizer, prompt, args: (prompt, [prompt], ["user"], "normal")
    worker._make_state_machine = lambda *args: (None, None)
    worker._log_cache_stats = lambda: None
    worker._share_object = lambda value: value
    provider.reset = lambda: None
    worker._generate()
    for queue in queues[:2]:
        failed = list(queue.queue)
        assert len(failed) == 2 and isinstance(failed[1], RuntimeError)
        assert failed[1].__traceback__ is None
        assert str(failed[1]) == "generation_failed"
    assert list(queues[2].queue)[-1] is None
    assert cache.fetch_nearest_cache("model", [9, 7]) == ([], [])
    retained, rest = cache.fetch_nearest_cache("model", [1, 2])
    assert retained[0].tokens == [42] and rest == []
    assert cache.fetch_nearest_cache("model", [1, 2, 11])[1] == [11]
    assert "private" not in caplog.text and "Traceback" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
    assert len(batches) == 2 and all(ref() is None for ref in batches)
    assert all(ref() is None for ref in owned)


@pytest.mark.parametrize("stream", [False, True])
def test_http_failure_ends_response(server_source, stream):
    namespace = {"logging": logging, "json": json,
                 "GenerationArguments": SimpleNamespace, "ModelDescription": SimpleNamespace,
                 "SamplingArguments": SimpleNamespace, "LogitsProcessorArguments": SimpleNamespace,
                 "ToolCallFormatter": lambda *args: None}
    cls = extract(server_source, "APIHandler", {"handle_completion", "_send_generation_error"}, namespace)
    stopped = []
    ctx = SimpleNamespace(tool_parser=None, initial_state="normal",
                          text_sm=SimpleNamespace(make_state=lambda state: None), stop=lambda: stopped.append(True))
    def responses():
        raise RuntimeError("private prompt /private/model failed")
        yield
    class Handler(cls, BaseHTTPRequestHandler):
        def do_POST(self):
            self.stream = stream
            self.response_generator = SimpleNamespace(generate=lambda *args, **kwargs: (ctx, responses()))
            for name in ("requested_model", "requested_draft_model", "adapter", "temperature", "top_p", "top_k",
                         "min_p", "xtc_probability", "xtc_threshold", "logit_bias", "repetition_penalty",
                         "repetition_context_size", "presence_penalty", "presence_context_size", "frequency_penalty",
                         "frequency_context_size", "max_tokens", "num_draft_tokens", "logprobs", "top_logprobs",
                         "seed", "chat_template_kwargs"):
                setattr(self, name, None)
            self.handle_completion(SimpleNamespace(tools=[]), [])
        def _set_completion_headers(self, status):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
        def _set_stream_headers(self, status):
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream")
        def log_message(self, *args):
            pass
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        connection = HTTPConnection(*server.server_address, timeout=2)
        try:
            connection.request("POST", "/v1/chat/completions", body=b"{}")
            response = connection.getresponse()
            body = response.read().decode()
            assert response.status == (200 if stream else 500)
            if stream:
                assert body.count('"error"') == 1 and body.endswith("data: [DONE]\n\n")
                assert "HTTP/1." not in body
            else:
                assert json.loads(body)["error"] == "generation_failed"
            assert "private" not in body
            assert stopped == [True]
        finally:
            connection.close()
            server.shutdown()
            thread.join(timeout=2)
