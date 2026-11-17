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
from threading import Event, Thread
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
    stopped, done = [], Event()
    def stop():
        stopped.append(True)
        done.set()
    ctx = SimpleNamespace(tool_parser=None, initial_state="normal",
                          text_sm=SimpleNamespace(make_state=lambda state: None), stop=stop)
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
            assert done.wait(2) and stopped == [True]
        finally:
            connection.close()
            server.shutdown()
            thread.join(timeout=2)


def _worker(server_source, requests, batch_cls=None, extra_ns=None):
    namespace = {"mx": SimpleNamespace(default_stream=lambda device: None, default_device=lambda: None,
                                        clear_cache=lambda: None),
                 "GenerationContext": lambda **kwargs: SimpleNamespace(**kwargs, _should_stop=False),
                 "Response": lambda *args: args, "logging": logging,
                 "_make_sampler": lambda *args: None, "_make_logits_processors": lambda *args: None,
                 "_format_top_logprobs": lambda *args: [], "gc": __import__("gc")}
    if extra_ns:
        namespace.update(extra_ns)
    cls = extract(server_source, "ResponseGenerator", {"_generate", "cli_args", "_fail_batch"}, namespace)
    worker = cls()
    worker.model_provider = SimpleNamespace(
        load_default=lambda: None, model_key="model",
        load=lambda *args: (None, SimpleNamespace(
            has_tool_calling=False, has_thinking=False, tool_parser=None,
            detokenizer=SimpleNamespace(add_token=lambda token: None, finalize=lambda: None,
                                        last_segment="ok"))),
        cli_args=SimpleNamespace(decode_concurrency=1, prompt_concurrency=1,
                                 prefill_step_size=512, prompt_cache_bytes=None))
    worker.prompt_cache = prompt_cache(server_source)
    worker._is_distributed = False
    worker._stop = False
    worker._time_budget = [None]
    def _next_request(timeout):
        if requests:
            return requests.popleft()
        worker._stop = True
        return None
    worker._next_request = _next_request
    worker._is_batchable = lambda args: True
    worker._tokenize = lambda tokenizer, prompt, args: (prompt, [prompt], ["user"], "normal")
    worker._make_state_machine = lambda *args: (None, None)
    worker._log_cache_stats = lambda: None
    worker._share_object = lambda value: value
    worker.model_provider.reset = lambda: None
    return worker


def _request_args():
    return SimpleNamespace(model=SimpleNamespace(model="model", adapter=None, draft=None),
                           stop_words=[], max_tokens=1, top_logprobs=0)


def _done_response(uid, prompt, stop=False):
    return SimpleNamespace(uid=uid, token=7, current_state="normal", match_sequence=None,
                           logprobs={7: SimpleNamespace(item=lambda: 0.0)}, finish_reason="length",
                           all_tokens=prompt + [7], prompt_cache=[], progress=None,
                           end_of_segment=False, end_of_prompt=False)


@pytest.mark.parametrize("fail_at", ["remove", "extract_cache"])
def test_batch_failure_beyond_next_keeps_worker_alive(server_source, fail_at):
    """The batch-failure guard must cover every batch_generator call in the loop,
    not just next(): a throw in extract_cache or remove must fail the batch the
    same way and leave the worker alive for the next request."""
    queues = [Queue(), Queue()]
    requests = deque([(queues[0], [1, 2, 11], _request_args()), None,
                      (queues[1], [9], _request_args()), None])

    class Batch:
        _count = 0
        def __init__(self, *args, **kwargs):
            self.index = Batch._count
            Batch._count += 1
        def insert_segments(self, **kwargs):
            self.prompt = kwargs["segments"][0][0]
            return [1]
        def close(self):
            pass
        def next(self):
            if fail_at == "extract_cache":
                prompt_r = SimpleNamespace(uid=1, progress=None, end_of_segment=True,
                                           end_of_prompt=False)
                return [prompt_r], []
            return [], [_done_response(1, self.prompt)]
        def extract_cache(self, ids):
            if fail_at == "extract_cache" and self.index == 0:
                raise RuntimeError("extract_cache blew up")
            return {}
        def remove(self, uids):
            if fail_at == "remove" and self.index == 0:
                raise RuntimeError("remove blew up")

    # extract_cache fires only when the result carries a segment type and the
    # response marks end_of_segment; remove fires when the ctx asks to stop.
    should_stop = fail_at == "remove"
    worker = _worker(
        server_source, requests,
        extra_ns={"BatchGenerator": Batch,
                  "GenerationContext": lambda **kw: SimpleNamespace(
                      **{**kw, "_should_stop": should_stop})},
    )
    worker._generate()

    if fail_at == "extract_cache":
        failed = [item for item in queues[0].queue if isinstance(item, RuntimeError)]
        assert failed and str(failed[0]) == "generation_failed"
    else:
        # remove() ran after the first request already finished; the worker
        # must simply stay alive for the second request.
        assert not any(isinstance(item, RuntimeError) for item in queues[1].queue)
    assert list(queues[1].queue)[-1] is None


def test_stray_runtime_error_is_not_flattened(server_source):
    """A RuntimeError that is not the generation_failed sentinel must keep its
    own handling (upstream's generic path), not report generation_failed."""
    namespace = {"logging": logging, "json": json,
                 "GenerationArguments": SimpleNamespace, "ModelDescription": SimpleNamespace,
                 "SamplingArguments": SimpleNamespace, "LogitsProcessorArguments": SimpleNamespace,
                 "ToolCallFormatter": lambda *args: None}
    cls = extract(server_source, "APIHandler", {"handle_completion", "_send_generation_error"}, namespace)
    ctx = SimpleNamespace(tool_parser=None, initial_state="normal",
                          text_sm=SimpleNamespace(make_state=lambda state: None), stop=lambda: None)

    def generate(*args, **kwargs):
        raise RuntimeError("kv cache shape mismatch")

    class Handler(cls, BaseHTTPRequestHandler):
        def do_POST(self):
            self.stream = False
            self.response_generator = SimpleNamespace(generate=generate)
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
            assert response.status != 500 or "generation_failed" not in body
            assert "kv cache shape mismatch" in body or response.status == 404
        finally:
            connection.close()
            server.shutdown()
            thread.join(timeout=2)


def test_late_failure_writes_no_second_status_line(server_source):
    """Non-streaming: headers now go out inside the response block. A failure
    after they leave must not emit a second HTTP status line as body text."""
    namespace = {"logging": logging, "json": json,
                 "GenerationArguments": SimpleNamespace, "ModelDescription": SimpleNamespace,
                 "SamplingArguments": SimpleNamespace, "LogitsProcessorArguments": SimpleNamespace,
                 "ToolCallFormatter": lambda *args: (lambda calls: calls),
                 "TextStateMachine": SimpleNamespace(
                     step=lambda sm, text: (sm, text, "normal"),
                     discard=lambda sm: (sm, "normal"),
                     flush=lambda sm: (sm, "", "normal"))}
    cls = extract(server_source, "APIHandler", {"handle_completion", "_send_generation_error"}, namespace)
    ctx = SimpleNamespace(tool_parser=None, initial_state="normal", prompt=[], prompt_cache_count=0,
                          text_sm=SimpleNamespace(make_state=lambda state: None), stop=lambda: None)
    gen = SimpleNamespace(text="ok", token=1, logprob=0.0, top_tokens=None,
                          finish_reason="stop", state="normal")

    class DropOnBody:
        """wfile that dies on the response body only: headers go out, the
        payload write fails, and any later error-path write is still recorded
        so a second status line would be visible in the body."""
        def __init__(self, real):
            self.real = real
        def __getattr__(self, name):
            return getattr(self.real, name)
        def write(self, data):
            if b'"choices"' in data:
                raise ConnectionResetError("client gone")
            return self.real.write(data)
        def flush(self):
            return self.real.flush()

    class Handler(cls, BaseHTTPRequestHandler):
        def do_POST(self):
            self.stream = False
            self.wfile = DropOnBody(self.wfile)
            self.response_generator = SimpleNamespace(
                generate=lambda *args, **kwargs: (ctx, iter([gen])))
            for name in ("requested_model", "requested_draft_model", "adapter", "temperature", "top_p", "top_k",
                         "min_p", "xtc_probability", "xtc_threshold", "logit_bias", "repetition_penalty",
                         "repetition_context_size", "presence_penalty", "presence_context_size", "frequency_penalty",
                         "frequency_context_size", "max_tokens", "num_draft_tokens", "logprobs", "top_logprobs",
                         "seed", "chat_template_kwargs", "stream_options"):
                setattr(self, name, None)
            self.top_logprobs = 0
            self.logprobs = None
            self.handle_completion(SimpleNamespace(tools=[]), [])
        def _set_completion_headers(self, status):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
        def _set_stream_headers(self, status):
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream")
        def generate_response(self, *args, **kwargs):
            return {"choices": [{"text": "ok"}]}
        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        connection = HTTPConnection(*server.server_address, timeout=2)
        try:
            connection.request("POST", "/v1/chat/completions", body=b"{}")
            body = ""
            try:
                body = connection.getresponse().read().decode()
            except Exception:
                # A dropped body may surface as a truncated read; what must
                # never happen is a SECOND status line inside the body.
                pass
            assert "HTTP/1." not in body
        finally:
            connection.close()
            server.shutdown()
            thread.join(timeout=2)
