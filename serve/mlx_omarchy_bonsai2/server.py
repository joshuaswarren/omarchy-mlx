"""Bonsai-2 text chat endpoint for mlx-omarchy.

Foreground stdlib HTTP server (frozen serve contract):

    serve_main(["--model", <pack dir>, "--host", "127.0.0.1", "--port", "8080"])

Endpoints: GET /health (+/healthz), GET /v1/models,
POST /v1/chat/completions (OpenAI-style; stream and non-stream).

No network egress, no automatic downloads, no silent CPU fallback:
without --allow-cpu the server refuses to start unless the default MLX
device is mx.gpu, and any backend evaluation error surfaces as a 500
with the exact backend message. Responses carry a `timings` block
(prompt_n, cached_n, prompt_ms, prompt_per_second, predicted_n,
predicted_ms, predicted_per_second); cached_n is honestly 0 because no
prompt cache survives across requests. The endpoint enforces a hard
context cap of its own (--max-context, default: the pack's
max_position_embeddings): a prompt plus max_tokens beyond it is a 400.

--managed is the admission-control mode: admit + reserve against
mlx_omarchy_serve.budget BEFORE any weights are touched (weights + KV
at the served context + workspace via budget.estimate_required),
transition the reservation to resident only after the load, release on
exit, and fail closed (exit 3) if any of that cannot be done. Without
--managed the same reservation runs best-effort and never blocks
standalone serving.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_BODY_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_TOKENS = 256


def validate_artifact(model_dir) -> str | None:
    """The CLI integration hook (mlx_omarchy_serve.__main__.

    module_artifact_problem) calls ``getattr(importlib.import_module(
    "mlx_omarchy_bonsai2.server"), "validate_artifact")(model_dir)``.
    The hook must live on the server module (this file), not the
    package __init__, because importlib gives the CLI the server
    module directly. Returns None on a real Bonsai pack, or a short
    error string on failure.

    Implementation reuses pack_footprint (config.json schema 2 +
    safetensors header ONLY -- ZERO tensor bytes). pack_footprint calls
    _check_config which validates model_type, schema_version,
    base_model_type, quantization, text_config, modules list, and
    tensor_namespace -- those PackError diagnostics are surfaced
    verbatim. Tokenizer presence is NOT validated here: the tokenizer
    is required at SERVE time (transformers.AutoTokenizer.from_pretrained
    in load_tokenizer, called from Bonsai2State.__init__ after
    pack_footprint succeeds) and any missing/corrupt tokenizer file
    fails there with the exact transformers error message. No
    <pkg>.convert.checkpoint_state is needed: Bonsai packs ARE the
    upstream artifact and require no conversion step.
    """
    pack = Path(model_dir)
    if not pack.is_dir():
        return f"{pack}: not a directory"
    config_path = pack / "config.json"
    if not config_path.is_file():
        return f"{pack}: missing config.json"
    safetensors = pack / "model.safetensors"
    if not safetensors.is_file():
        return f"{pack}: missing model.safetensors"
    try:
        from .loader import pack_footprint

        facts = pack_footprint(pack)
    except Exception as exc:
        return f"pack validation failed: {exc}"
    if not facts.get("live_weights_bytes"):
        return f"{pack}: no language_model.* tensors in checkpoint"
    return None


def _parse_args(argv):
    p = argparse.ArgumentParser(prog="mlx-omarchy-bonsai2", description="Bonsai-2 text chat server")
    p.add_argument("--model", required=True, help="resolved pack directory")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--model-id", default="bonsai-2-27b-mlx-2bit", help="id reported by /v1/models")
    p.add_argument("--max-context", type=int, default=None,
                   help="hard cap on prompt+generated tokens (default: pack max_position_embeddings)")
    p.add_argument("--source-revision", default=None,
                   help="pack revision to report in /health (not derivable from local files)")
    p.add_argument("--managed", action="store_true",
                   help="fail closed if memory admission/reservation cannot be completed before load")
    p.add_argument("--allow-cpu", action="store_true",
                   help="explicitly permit the CPU device (reference/testing only; never in serving)")
    args = p.parse_args(argv)
    if args.max_context is not None and args.max_context < 1:
        p.error("--max-context must be >= 1")
    return args


def _fail_managed(detail):
    print(
        "bonsai2: --managed: refusing to serve before model load (%s); an unreserved "
        "8.6 GiB-class resident set must not race other tenants" % detail,
        file=sys.stderr,
    )
    sys.exit(3)


def _reservation_name(catalog_id: str) -> str:
    """Per-process reservation key: same catalog id must never collide on
    concurrent servers, or one process's exit would clear the other's
    live reservation."""
    return "%s-%d-%s" % (catalog_id, os.getpid(), uuid.uuid4().hex[:8])


def _footprint_total(budget, total_header_bytes, kv_bytes_per_token, context_tokens):
    est = budget.estimate_required(
        {"weights_bytes": total_header_bytes, "kv_bytes_per_token": kv_bytes_per_token},
        context_tokens=context_tokens,
    )
    return est.total


def _budget_module():
    try:
        from mlx_omarchy_serve import budget

        return budget
    except Exception:
        return None


def _owner_token(budget) -> str:
    try:
        return budget.generate_owner()
    except Exception:
        # Same shape as budget.generate_owner(); only ever used by the
        # non-managed best-effort path (managed requires the real module).
        return "pid%d-%s" % (os.getpid(), uuid.uuid4().hex[:12])


def _preflight_managed(args, reservation_name):
    """Managed mode: atomic admit+reserve BEFORE any weights are touched.

    Runs from config.json + the safetensors header only. Requires the
    shared atomic admit+reserve budget API (refusals raise BudgetError):
    a separate admit() then set_reservation() is NOT acceptable in
    managed mode — the gap is an overadmission race — so its absence
    fails closed.
    """
    from .loader import pack_footprint

    budget = _budget_module()
    if budget is None:
        _fail_managed("budget module unavailable")
    atomic = getattr(budget, "admit_and_reserve", None)
    if atomic is None:
        _fail_managed(
            "installed mlx_omarchy_serve.budget has no atomic admit_and_reserve "
            "(separate admit + set_reservation races other tenants)"
        )
    try:
        facts = pack_footprint(args.model)
        max_context = args.max_context or facts["max_position_embeddings"]
        total = _footprint_total(
            budget, facts["total_header_bytes"], facts["kv_bytes_per_token"], max_context
        )
        admission = atomic(
            reservation_name,
            total,
            note="bonsai2",
            owner=args.owner,
            state="pending",
        )
        if not admission.fits:  # defensive: refusals raise BudgetError
            _fail_managed("admission refused: %s" % "; ".join(admission.lines))
        return total
    except SystemExit:
        raise
    except Exception as exc:
        _fail_managed("atomic admission/reservation failed: %s" % exc)


def _register_reservation(args, state):
    """Post-load reservation lifecycle. Managed failures are fatal."""
    budget = _budget_module()
    if budget is None:
        if args.managed:
            _fail_managed("budget module unavailable")
        return False
    total = _footprint_total(
        budget, state.info["resident_bytes"], state.info["kv_bytes_per_token"], state.max_context
    )
    floor = state.info["resident_bytes"]
    if not args.managed:
        try:
            budget.set_reservation(
                state.reservation_name, total, note="bonsai2",
                owner=state.owner, state="pending",
            )
        except Exception as exc:  # budget absent/broken must not block standalone serving
            print("bonsai2: memory reservation skipped (%s)" % exc, file=sys.stderr)
            return False
    if args.managed:
        # No fallback: an API that cannot hold the resident floor ignores
        # future KV/workspace headroom and overadmits under load.
        try:
            budget.set_reservation_state(
                state.reservation_name, "resident",
                owner=state.owner, resident_floor_bytes=floor,
            )
        except Exception as exc:
            _fail_managed("resident transition failed: %s" % exc)
        state.reservation_bytes = total
        return True
    try:
        try:
            budget.set_reservation_state(
                state.reservation_name, "resident",
                owner=state.owner, resident_floor_bytes=floor,
            )
        except TypeError:
            # Pre-owner budget API: best-effort relabel without ownership.
            budget.set_reservation_state(state.reservation_name, "resident")
        state.reservation_bytes = total
        return True
    except Exception as exc:
        print("bonsai2: reservation resident transition skipped (%s)" % exc, file=sys.stderr)
        return False


def _release_reservation(reservation_name: str, owner: str | None = None):
    """Clear the per-process reservation.

    The owner token is REQUIRED for any owned managed reservation: the
    budget API refuses to clear a reservation held by another holder
    ("owned by another holder"), so a release without the matching token
    raises BudgetError and the entry stays on disk. Managed-mode call
    sites always pass owner=getattr(args, "owner", None); the unowned
    legacy path (no --managed) is preserved for backward compatibility
    but a failed clear is logged, not silently swallowed -- a dead owner
    entry is the bug this helper exists to prevent."""
    budget = _budget_module()
    if budget is None:
        return
    try:
        budget.clear_reservation(reservation_name, owner=owner)
    except Exception as exc:
        print(
            "bonsai2: reservation release for %r raised %s; the entry may need "
            "manual cleanup via budget.clear_reservation(name, owner=%r)"
            % (reservation_name, exc, owner),
            file=sys.stderr,
        )


class Bonsai2State:
    def __init__(self, args):
        import mlx.core as mx

        from .loader import load_text_model, load_tokenizer

        self.mx = mx
        self.pack_dir = Path(args.model).resolve()
        self.model, self.info = load_text_model(self.pack_dir)
        self.tokenizer = load_tokenizer(self.pack_dir)
        self.device = "gpu" if mx.default_device() == mx.gpu else "cpu"
        self.catalog_id = args.model_id
        self.source_revision = args.source_revision
        self.reservation_name = getattr(args, "reservation_name", None) or _reservation_name(
            args.model_id
        )
        self.owner = getattr(args, "owner", None) or _owner_token(None)
        self.max_context = args.max_context or self.info["max_position_embeddings"]
        # One shared model + cache per process: requests queue, they never
        # interleave inside a forward pass. The job queue also pins ALL mx
        # work to ONE thread: the omarchy backend keeps command encoders
        # thread_local (encoder.cpp get_command_encoders), so GPU ops must
        # run on the same thread every time — a fresh per-request HTTP
        # thread wedged on its first submit (observed on t6001-test-host, attempt 3).
        self.generate_lock = threading.Lock()
        self.job_queue = queue.Queue()
        self.reservation_bytes = None

    def submit(self, job):
        """Run job(emit) on the mx worker thread; returns the emit sink."""
        sink = queue.Queue()
        self.job_queue.put((job, sink))
        return sink


def _worker_loop(state):
    """Drain the job queue. Runs on the process main thread under
    serve_main (GPU thread affinity), or on a helper thread in tests."""
    while True:
        job, sink = state.job_queue.get()
        if job is None:
            break
        try:
            job(sink.put)
        except Exception as exc:
            sink.put({"__error__": str(exc)})
        sink.put(None)


def _timings(prompt_n: int, prompt_tps: float, predicted_n: int, predicted_tps: float) -> dict:
    return {
        "prompt_n": prompt_n,
        "cached_n": 0,
        "prompt_ms": round(prompt_n / prompt_tps * 1000, 2) if prompt_tps else 0,
        "prompt_per_second": round(prompt_tps, 2) if prompt_tps else 0,
        "predicted_n": predicted_n,
        "predicted_ms": round(predicted_n / predicted_tps * 1000, 2) if predicted_tps else 0,
        "predicted_per_second": round(predicted_tps, 2) if predicted_tps else 0,
    }


def _sampler(temperature, top_p):
    from mlx_lm.sample_utils import make_sampler

    return make_sampler(temp=temperature, top_p=top_p or 0.0)


def _generate(state, prompt_ids, max_tokens, temperature, top_p):
    """Run generation; returns (text, finish_reason, timings).

    stream_generate is looked up on each call via the mlx_lm package
    attribute, so the ids-probe env-gated wrap (which rebinds
    `mlx_lm.generate` after this module is imported) is picked up here.
    The captured-at-import fallback was a silent miss; per-call lookup
    is the documented fix."""
    # Per-call stream_generate lookup via the mlx_lm package
    # attribute. In mlx_lm 0.31.3, `mlx_lm.generate` is the stream_generate
    # function (re-exported at the package level), so re-reading the
    # In mlx_lm 0.31.3, `mlx_lm.generate` is the non-streaming wrapper
    # that returns str, while the streaming generator is exposed as
    # `mlx_lm.stream_generate`. Reading the package attribute on each
    # call picks up the ids-probe env-gated wrap (which rebinds
    # sys.modules['mlx_lm'].stream_generate after this module is
    # imported). The captured-at-import `from mlx_lm.generate import
    # stream_generate` form was a silent miss because it bound the
    # ORIGINAL function name before the wrap could rebind the package
    # attribute.
    stream_generate = sys.modules["mlx_lm"].stream_generate

    kwargs = {"sampler": _sampler(temperature, top_p)}
    prompt_n = len(prompt_ids)
    predicted_n = 0
    prompt_tps = predicted_tps = 0.0
    segments = []
    finish_reason = "length"
    tic = time.perf_counter()
    for response in stream_generate(
        state.model, state.tokenizer, list(prompt_ids), max_tokens=max_tokens, **kwargs
    ):
        if response.prompt_tps:
            prompt_tps = response.prompt_tps
        predicted_n = response.generation_tokens
        predicted_tps = response.generation_tps
        if response.finish_reason:
            finish_reason = response.finish_reason
            segments.append(response.text)
            break
        segments.append(response.text)
    wall = time.perf_counter() - tic
    if not predicted_tps and predicted_n:
        predicted_tps = predicted_n / max(wall, 1e-9)
    return "".join(segments), finish_reason, _timings(prompt_n, prompt_tps, predicted_n, predicted_tps)


class _RequestError(ValueError):
    """A chat request violates the endpoint contract (HTTP 400)."""


def _finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def validate_chat_request(req):
    """Strict shape/type validation. Returns (messages, max_tokens, temperature, top_p, stream)."""
    if not isinstance(req, dict):
        raise _RequestError("request body must be a JSON object")
    messages = req.get("messages")
    if not isinstance(messages, list) or not messages:
        raise _RequestError("'messages' must be a non-empty array")
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise _RequestError("messages[%d] must be an object" % i)
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise _RequestError("messages[%d].role must be a non-empty string" % i)
        content = message.get("content")
        if not isinstance(content, str) or not content:
            raise _RequestError(
                "messages[%d].content must be a non-empty string (text-only endpoint)" % i
            )
    max_tokens = req.get("max_tokens", DEFAULT_MAX_TOKENS)
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise _RequestError("'max_tokens' must be a positive integer")
    temperature = req.get("temperature", 0.0)
    if not _finite_number(temperature) or temperature < 0:
        raise _RequestError("'temperature' must be a finite non-negative number")
    top_p = req.get("top_p")
    if top_p is not None and (not _finite_number(top_p) or not 0 < top_p <= 1):
        raise _RequestError("'top_p' must be a finite number in (0, 1]")
    stream = req.get("stream", False)
    if not isinstance(stream, bool):
        raise _RequestError("'stream' must be a boolean")
    return messages, max_tokens, temperature, top_p, stream


def _make_handler(state: Bonsai2State):
    class Handler(BaseHTTPRequestHandler):
        server_version = "mlx-omarchy-bonsai2/0.1"

        def log_message(self, fmt, *args):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _send_json(self, code: int, payload: dict):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/health", "/healthz"):
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "model": state.catalog_id,
                        "backend": "mlx-omarchy-bonsai2",
                        "device": state.device,
                        "model_type": state.info["model_type"],
                        "checkpoint": str(state.pack_dir),
                        "source_revision": state.source_revision,
                        "config_sha256": state.info["config_sha256"],
                        "quantization": state.info["quantization"],
                        "resident_bytes": state.info["resident_bytes"],
                        "kv_bytes_per_token": state.info["kv_bytes_per_token"],
                        "excluded_bytes": state.info["excluded_bytes"],
                        "license_files": state.info["license_files"],
                        "max_context": state.max_context,
                        "reservation_name": state.reservation_name,
                        "reservation_bytes": state.reservation_bytes,
                    },
                )
            elif self.path == "/v1/models":
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": state.catalog_id,
                                "object": "model",
                                "owned_by": "prism-ml",
                                "kind": "chat",
                                "max_context": state.max_context,
                            }
                        ],
                    },
                )
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self._send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"error": "invalid Content-Length"})
                return
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send_json(400, {"error": "body must be 1..%d bytes" % MAX_BODY_BYTES})
                return
            raw = self.rfile.read(length)
            try:
                req = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                self._send_json(400, {"error": "invalid JSON body: %s" % exc})
                return
            try:
                messages, max_tokens, temperature, top_p, stream = validate_chat_request(req)
            except _RequestError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            try:
                prompt_ids = state.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True
                )
            except Exception as exc:
                self._send_json(400, {"error": "chat template failed: %s" % exc})
                return
            if len(prompt_ids) + max_tokens > state.max_context:
                self._send_json(
                    400,
                    {
                        "error": "prompt (%d tokens) + max_tokens (%d) exceeds the hard context cap %d"
                        % (len(prompt_ids), max_tokens, state.max_context)
                    },
                )
                return
            created = int(time.time())

            def _nonstream_job(emit):
                with state.generate_lock:
                    text, finish_reason, timings = _generate(
                        state, prompt_ids, max_tokens, temperature, top_p
                    )
                emit({
                    "response": {
                        "id": "chatcmpl-bonsai2-%d" % created,
                        "object": "chat.completion",
                        "created": created,
                        "model": state.catalog_id,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": text},
                                "finish_reason": finish_reason,
                            }
                        ],
                        "usage": {
                            "prompt_tokens": timings["prompt_n"],
                            "completion_tokens": timings["predicted_n"],
                            "total_tokens": timings["prompt_n"] + timings["predicted_n"],
                        },
                        "timings": timings,
                    }
                })

            def _stream_job(emit):
                with state.generate_lock:
                    self._stream_job(state, prompt_ids, max_tokens, temperature, req, created, emit)

            try:
                if stream:
                    sink = state.submit(_stream_job)
                    self._send_stream_response(sink, state, created)
                else:
                    sink = state.submit(_nonstream_job)
                    payload = None
                    while True:
                        item = sink.get()
                        if item is None:
                            break
                        if "__error__" in item:
                            self._send_json(500, {"error": "backend evaluation failed: %s" % item["__error__"]})
                            return
                        payload = item.get("response", payload)
                    self._send_json(200, payload)
            except Exception as exc:  # backend failure: exact error, never a fallback
                self._send_json(500, {"error": "backend evaluation failed: %s" % exc})

        def _stream_job(self, state, prompt_ids, max_tokens, temperature, req, created, emit):
            """Worker-side streaming generation: emits chunk dicts, never
            touches the socket (the handler thread drains and writes).

            Per-call stream_generate lookup (see _generate docstring)."""
            # Per-call stream_generate lookup (see _generate).
            stream_generate = sys.modules["mlx_lm"].stream_generate

            def chunk(delta, finish=None, extra=None):
                payload = {
                    "id": "chatcmpl-bonsai2-%d" % created,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": state.catalog_id,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
                if extra:
                    payload.update(extra)
                emit({"__chunk__": payload})

            kwargs = {"sampler": _sampler(temperature, req.get("top_p"))}
            prompt_n = len(prompt_ids)
            prompt_tps = predicted_tps = 0.0
            predicted_n = 0
            finish_reason = "length"
            for response in stream_generate(
                state.model, state.tokenizer, list(prompt_ids), max_tokens=max_tokens, **kwargs
            ):
                if response.prompt_tps:
                    prompt_tps = response.prompt_tps
                predicted_n = response.generation_tokens
                predicted_tps = response.generation_tps
                finish_reason = response.finish_reason or finish_reason
                chunk({"content": response.text}, finish=response.finish_reason)
            timings = _timings(prompt_n, prompt_tps, predicted_n, predicted_tps)
            emit({"__final__": {"finish_reason": finish_reason, "timings": timings}})

        def _send_stream_response(self, sink, state, created):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def write(payload):
                self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")

            while True:
                item = sink.get()
                if item is None:
                    break
                if "__error__" in item:
                    write({"error": "backend evaluation failed: %s" % item["__error__"]})
                    break
                if "__chunk__" in item:
                    write(item["__chunk__"])
                elif "__final__" in item:
                    final = item["__final__"]
                    write({
                        "id": "chatcmpl-bonsai2-%d" % created,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": state.catalog_id,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": final["finish_reason"]}],
                        "timings": final["timings"],
                    })
            self.wfile.write(b"data: [DONE]\n\n")

    return Handler


def serve_main(argv):
    args = _parse_args(argv)
    import faulthandler
    import signal

    import mlx.core as mx

    def _terminate(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    # All-thread stack dumps to stderr (server.log) every 180 s: a wedged
    # submit self-documents where every thread is stuck.
    faulthandler.enable(all_threads=True)
    faulthandler.dump_traceback_later(180, repeat=True)
    if mx.default_device() != mx.gpu and not args.allow_cpu:
        print(
            "bonsai2: refusing to start: default MLX device is %r; serving requires mx.gpu "
            "(pass --allow-cpu only for reference/testing)" % mx.default_device(),
            file=sys.stderr,
        )
        sys.exit(2)
    args.reservation_name = _reservation_name(args.model_id)
    args.owner = _owner_token(_budget_module())
    # Env-gated ids-probe hook (bench-only, default OFF, byte-identical
    # otherwise). Mirrors the contract in serve.mlx_omarchy_serve._mlxlm_server.
    from . import ids_probe as _bonsai_ids_probe
    _bonsai_ids_probe.install_ids_probe()
    state = None
    httpd = None
    server_thread = None
    try:
        if args.managed:
            # Fail-closed BEFORE any allocation: atomic admit+reserve from the
            # pack header alone; nothing below this line loads weights.
            _preflight_managed(args, args.reservation_name)
        state = Bonsai2State(args)
        if args.max_context is not None and args.max_context > state.info["max_position_embeddings"]:
            print(
                "bonsai2: refusing to start: --max-context %d exceeds the pack's trained "
                "max_position_embeddings %d"
                % (args.max_context, state.info["max_position_embeddings"]),
                file=sys.stderr,
            )
            sys.exit(2)
        registered = _register_reservation(args, state)
        httpd = ThreadingHTTPServer((args.host, args.port), _make_handler(state))
        # GPU thread affinity: the worker loop (ALL mx evaluation) runs on the
        # process MAIN thread — the same topology as every successful CLI
        # forward. The HTTP server only marshals in daemon threads.
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        print(
            "mlx-omarchy-bonsai2: serving %s on http://%s:%d (device=%s, packed_modules=%d, "
            "resident_bytes=%d, excluded_bytes=%s, max_context=%d, reservation_bytes=%s)"
            % (
                state.catalog_id,
                args.host,
                httpd.server_address[1],
                state.device,
                state.info["packed_modules"],
                state.info["resident_bytes"],
                state.info["excluded_bytes"],
                state.max_context,
                state.reservation_bytes,
            ),
            flush=True,
        )
        print(state.info["attribution"], flush=True)
        _worker_loop(state)
    except KeyboardInterrupt:
        pass
    finally:
        # Spans BaseException: a SIGTERM/KeyboardInterrupt during load,
        # relabel, bind, or serve must never leak the reservation. The
        # worker sentinel and httpd shutdown are conditional because the
        # failure may precede either. Single owner of release: covers
        # clean exit, KeyboardInterrupt, any Exception during
        # load/relabel/bind, AND SystemExit from sys.exit(2) in the
        # max-context overflow path (finally derives from BaseException).
        # Cleanup errors are LOGGED, never silently swallowed -- a dead
        # owner entry on disk is the very bug this finally exists to
        # prevent, and a silent fail here would defeat the fix.
        if state is not None:
            state.job_queue.put((None, None))
        if httpd is not None:
            httpd.shutdown()
        if httpd is not None:
            httpd.server_close()
        try:
            _release_reservation(args.reservation_name, owner=getattr(args, "owner", None))
        except Exception as release_exc:
            print(
                "bonsai2: reservation release raised %s; entry may need manual cleanup via "
                "budget.clear_reservation(name, owner=%r)" % (release_exc, getattr(args, "owner", None)),
                file=sys.stderr,
            )


if __name__ == "__main__":
    serve_main(sys.argv[1:])
