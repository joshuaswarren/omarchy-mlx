"""Typed-decisions HTTP endpoint for Laya on mlx-omarchy.

A foreground stdlib HTTP server exposing the upstream system_one envelope
over POST /v1/decisions. This is NOT chat completions: Laya is a
non-autoregressive decision model, every response has output_tokens == 0.

serve_main(argv) is the integration entry point called by
mlx_omarchy_serve (catalog serve.backend = "module"):

    serve_main(["--model", <ckpt dir>, "--host", "127.0.0.1", "--port", "8081"])

No network egress, no automatic downloads, no silent CPU fallback: without
--allow-cpu the server refuses to start unless the default MLX device is
mx.gpu, and any backend evaluation error surfaces as a 500 with the exact
backend message.

Memory admission (Main review round 3):
  - the reserve is DTYPE-AWARE and happens BEFORE any weight loads:
    total = parameter_bytes(run dtype) + workspace bound
    (fp32 run = 2x fp16 parameter bytes; workspace derives from
    max_questions x seq x hidden x dtype including the fp32 softmax upcast)
  - the post-load relabel NEVER grows the admitted total: it relabels
    pending -> resident with resident_floor_bytes = the run dtype's exact
    parameter nbytes; a floor exceeding the admitted total is a mismatch
    and is refused
  - under --managed anything less than the shared atomic
    budget.admit_and_reserve API is fail-closed (exit 3); standalone runs
    degrade to warn-and-continue
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_BODY_BYTES = 8 * 1024 * 1024

# Shared atomic budget API (serve-cli feat/serve-cli-catalog): capability is
# detected so older checkouts degrade safely; --managed accepts nothing less.
_ATOMIC_API_NAMES = ("admit_and_reserve",)

_DTYPE_SIZES = {"float16": 2, "bfloat16": 2, "float32": 4}

_MAX_OPTIONS = 64  # per-question option bound used in the workspace estimate

# Measured on the dev box (CPU fp16 reference run, 16 questions x 512 tokens):
# allocator peak growth 1,287,019,628 B vs the raw term sum 673,193,984 B =
# 1.91x. The graph scheduler keeps more transient buffers alive than the
# minimal lifetime count, so the admitted workspace applies a 2x safety
# factor over the term sum. The on-host workspace audit re-verifies the
# factor against real GPU hardware inside the window.
_WORKSPACE_SAFETY_FACTOR = 2.0


def _parse_args(argv):
    p = argparse.ArgumentParser(prog="mlx-omarchy-laya", description="Laya typed-decisions server")
    p.add_argument("--model", required=True, help="converted checkpoint directory")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument(
        "--allow-cpu",
        action="store_true",
        help="explicitly permit the CPU device (reference/testing only; never in serving)",
    )
    p.add_argument(
        "--managed",
        action="store_true",
        help="managed co-serving: admission via the shared atomic budget API is "
        "REQUIRED or startup fails (exit 3). Without this flag a failed "
        "reservation only warns, for standalone runs where no admission "
        "control relies on it.",
    )
    p.add_argument(
        "--max-questions",
        type=int,
        default=64,
        help="per-request question cap (batch size bound for the memory budget)",
    )
    return p.parse_args(argv)


def _read_manifest(model_dir: Path) -> dict:
    path = model_dir / "manifest.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# --------------------------------------------------------------------------- admission math


def parameter_bytes(manifest: dict, dtype: str) -> int:
    """Exact parameter nbytes of the checkpoint at the run dtype.

    The stored weights are fp16 (2 B/element); an fp32 run materializes 2x.
    This is the resident floor: the parameters themselves, never a global
    allocator counter (which mixes in transient activations). Prefers the
    manifest's counted param_elements so a future non-fp16 pack cannot
    silently halve the floor."""
    header = manifest.get("weights_header", {})
    elements = header.get("param_elements")
    if elements is None:
        elements = int(header.get("weight_bytes") or 0) // 2
    return int(elements) * _DTYPE_SIZES[dtype]


def workspace_bytes(manifest: dict, enc_cfg, dtype: str, max_questions: int,
                    context_override: int | None = None) -> int:
    """Conservative transient-workspace bound for one worst-case request.

    Derives from the batch bound (max_questions), the context length, hidden
    size and run dtype — including the fp32 softmax upcast — instead of a
    flat constant. Transients are layer-reused, so the bound counts one
    attention-score buffer plus the live residual/FFN buffers plus fp32
    output logits:
      scores:  B*H*T*T*(sz + 4)            (run-dtype scores + fp32 softmax)
      residual/attention/FFN buffers: 4*B*T*D*sz + B*T*(3D + 2I + 4D)*sz
      logits:  2*B*max_options*4           (fp32 logits + probabilities)
    """
    sz = _DTYPE_SIZES[dtype]
    B = max(1, int(max_questions))
    T = int(context_override or manifest.get("context_max_tokens") or 512)
    D = enc_cfg.hidden_size
    H = enc_cfg.num_attention_heads
    I = enc_cfg.intermediate_size
    scores = B * H * T * T * (sz + 4)
    residual = 4 * B * T * D * sz
    ffn_qkv = B * T * (3 * D + 2 * I + 4 * D) * sz
    logits = 2 * B * _MAX_OPTIONS * 4
    return int(_WORKSPACE_SAFETY_FACTOR * (scores + residual + ffn_qkv + logits))


def total_estimate_bytes(manifest: dict, enc_cfg, dtype: str, max_questions: int) -> int:
    return parameter_bytes(manifest, dtype) + workspace_bytes(manifest, enc_cfg, dtype, max_questions)


def resident_floor_bytes(manifest: dict, dtype: str) -> int:
    return parameter_bytes(manifest, dtype)


def floor_matches_admission(admitted_total: int, floor: int) -> bool:
    """The admitted total is a commitment: relabel verifies the floor against
    it and never grows it. A floor above the admitted total is a mismatch."""
    return floor <= admitted_total


def relabel_admissible(admitted_total: int, floor: int, workspace: int) -> bool:
    """Main round-3 drift guard: at relabel the SAME run-args workspace bound
    is recomputed and the full committed requirement (parameters + workspace)
    must still fit inside the admitted total. Holds by construction when the
    run args are unchanged; catches any drift between reserve-time and
    relabel-time estimates."""
    return floor + workspace <= admitted_total


# --------------------------------------------------------------------------- budget registry


def _memory_api_tier(budget):
    """Capability tiers (Main-directed adoption order).

    - "atomic": budget.admit_and_reserve (shared atomic admit+reserve with
      unique per-process ownership)
    - "legacy": admit() + set_reservation/set_reservation_state two-call path
    - "none": no usable registry surface

    Under --managed only "atomic" is acceptable (exit 3); the legacy two-call
    path is not atomic and is allowed only for standalone runs.
    """
    for name in _ATOMIC_API_NAMES:
        if hasattr(budget, name):
            return "atomic", name
    if hasattr(budget, "set_reservation") and hasattr(budget, "admit"):
        return "legacy", None
    return "none", None


class ManagedBudgetUnavailable(RuntimeError):
    """managed co-serving cannot run against this budget module (exit 3)."""


def _budget_or_fail(managed):
    try:
        from mlx_omarchy_serve import budget
    except ImportError as exc:
        if managed:
            raise ManagedBudgetUnavailable(
                "managed co-serving requires mlx_omarchy_serve.budget (the serve-cli "
                "package) on PYTHONPATH: %s" % exc
            ) from exc
        print("laya: mlx_omarchy_serve.budget not importable; memory reservation skipped (%s)" % exc,
              file=sys.stderr)
        return None
    tier, _ = _memory_api_tier(budget)
    if tier != "atomic":
        if managed:
            raise ManagedBudgetUnavailable(
                "managed co-serving requires the shared atomic admit+reserve budget API "
                "(one of %s); this budget module only exposes the legacy two-call path "
                "(admit/set_reservation are not atomic together)" % (_ATOMIC_API_NAMES,)
            )
        print("laya: budget module lacks the atomic admit API; reservation skipped", file=sys.stderr)
        return None
    return budget


def _register_pending(model_dir: Path, name: str, managed: bool, dtype: str, max_questions: int):
    """Dtype-aware atomic admit + pending reservation BEFORE weights load.

    total = parameter_bytes(run dtype) + workspace bound. Under --managed the
    fit check and the registry write happen in one locked step; any refusal
    surfaces the budget table and stops startup before a single weight is
    resident. Standalone runs fall back to warn-and-continue.
    """
    manifest = _read_manifest(model_dir)
    if not manifest:
        if managed:
            raise RuntimeError(
                "managed co-serving requires a converted manifest (estimated_bytes, "
                "weights_header, context_max_tokens); %s has none. Run "
                "mlx_omarchy_laya.convert first or pass an unmanaged standalone run."
                % model_dir
            )
        print("laya: no manifest; memory reservation skipped", file=sys.stderr)
        return None, None, 0
    from .model import load_encoder_config

    enc_cfg = load_encoder_config(model_dir / "encoder")
    total = total_estimate_bytes(manifest, enc_cfg, dtype, max_questions)
    budget = _budget_or_fail(managed)
    if budget is None:
        return None, None, total
    owner = budget.generate_owner()
    admission = budget.admit_and_reserve(
        name, int(total), note=manifest.get("source_revision", "laya"), owner=owner
    )
    for line in getattr(admission, "lines", []):
        print("laya: admit | %s" % line, file=sys.stderr)
    return admission, owner, int(total)


def _relabel_resident(model_dir: Path, name: str, managed: bool, registered, owner,
                      dtype: str, admitted_total: int, max_questions: int):
    """Flip pending -> resident once weights are materialized.

    The floor is the run dtype's exact parameter nbytes. The admitted total is
    NEVER grown here: the same run-args workspace bound is recomputed and the
    full committed requirement (parameters + workspace) must still fit inside
    the admitted total — a mismatch refuses under --managed and degrades to
    full-bytes conservative counting standalone."""
    if registered is None:
        return None
    manifest = _read_manifest(model_dir)
    floor = resident_floor_bytes(manifest, dtype)
    from .model import load_encoder_config

    workspace = workspace_bytes(manifest, load_encoder_config(model_dir / "encoder"),
                                dtype, max_questions)
    if not relabel_admissible(admitted_total, floor, workspace):
        if managed:
            raise RuntimeError(
                "resident requirement %d (params %d + recomputed workspace %d) exceeds "
                "the admitted total %d for %s; refusing to relabel — the admission "
                "commitment is never grown post hoc"
                % (floor + workspace, floor, workspace, admitted_total, name)
            )
        print("laya: resident requirement %d exceeds admitted total %d; relabeling "
              "without floor (full bytes stay counted)" % (floor + workspace, admitted_total),
              file=sys.stderr)
        from mlx_omarchy_serve import budget

        budget.set_reservation_state(name, "resident", owner=owner)
        return {"bytes": admitted_total, "state": "resident", "floor_verified": False}
    from mlx_omarchy_serve import budget

    budget.set_reservation_state(name, "resident", resident_floor_bytes=floor, owner=owner)
    return {"bytes": admitted_total, "state": "resident", "floor_verified": True,
            "floor_bytes": floor}


def _clear_memory(name: str, managed: bool, why: str, owner=None):
    """Remove the reservation so a dead process never blocks future admissions.

    Ownership-scoped: clear_reservation refuses names held by another owner,
    so shutdown can never remove someone else's live reservation."""
    if owner is None:
        return
    try:
        from mlx_omarchy_serve import budget

        budget.clear_reservation(name, owner=owner)
    except Exception as exc:
        msg = "laya: could not clear %s memory reservation for %s: %s" % (why, name, exc)
        if managed:
            print(msg + " — co-serving registry may be stale; launcher must clear it", file=sys.stderr)
        else:
            print(msg, file=sys.stderr)


def _budget_or_none():
    try:
        from mlx_omarchy_serve import budget
    except ImportError:
        return None
    return budget


def _read_claim_payload(fd: int) -> dict:
    budget = _budget_or_none()
    if budget is None:
        raise RuntimeError(
            "claim mode requires mlx_omarchy_serve.budget on PYTHONPATH")
    return budget.read_claim_fd(fd)


def _claim_reserved(model_dir: Path, catalog_id: str, payload: dict, managed: bool,
                    dtype: str, max_questions: int):
    """Bind this child to the parent's batch reservation instead of running
    its own admission. The claimed allocation is validated against THIS
    child's recomputed requirement (manifest + run args) — a mismatch is a
    fatal refusal, never a silent growth — and against the record under the
    budget lock (token, bytes, parent identity)."""
    from .model import load_encoder_config

    name = str(payload["name"])
    if name != catalog_id:
        raise RuntimeError(
            "claim refused: batch record %r does not match this checkpoint's "
            "catalog_id %r" % (name, catalog_id))
    enc_cfg = load_encoder_config(model_dir / "encoder")
    total = total_estimate_bytes(_read_manifest(model_dir), enc_cfg, dtype, max_questions)
    if int(payload["bytes"]) != int(total):
        raise RuntimeError(
            "claim refused: batch record %r reserves %d bytes but this run "
            "(dtype=%s, max_questions=%d) requires %d; claims never change an "
            "allocation" % (name, int(payload["bytes"]), dtype, max_questions, total))
    try:
        from mlx_omarchy_serve import budget
    except ImportError as exc:
        raise RuntimeError(
            "claim mode requires mlx_omarchy_serve.budget on PYTHONPATH: %s" % exc) from exc
    record = budget.claim_reservation(
        name, str(payload["claim_token"]), expected_bytes=int(payload["bytes"]),
        home=budget.default_home(), expected_pair=payload.get("pair_id"))
    print("laya: claimed %s (%d bytes) from pair %s" % (
        name, int(payload["bytes"]), payload.get("pair_id")), file=sys.stderr)
    return {"state": "pending", "claimed": record["claimed"]}, record["owner"], int(total)


class LayaState:
    def __init__(self, args):
        import mlx.core as mx

        from .api import LayaEngine

        dtype = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}[args.dtype]
        self.mx = mx
        self.managed = args.managed
        self.max_questions = args.max_questions
        self.model_dir = Path(args.model).resolve()
        self.manifest = _read_manifest(self.model_dir)
        self.catalog_id = str(self.manifest.get("catalog_id") or self.engine_id_fallback(args))
        claim_payload = None
        budget = _budget_or_none()
        raw_fd = os.environ.get(budget.CLAIM_FD_ENV) if budget is not None else None
        if raw_fd:
            if not self.managed:
                raise RuntimeError(
                    "claim mode (managed pair child) requires --managed; refusing "
                    "to bind a batch reservation without fail-closed admission")
            claim_payload = _read_claim_payload(int(raw_fd))
        if claim_payload is not None:
            # Claimed child: the parent manager fit-checked and reserved this
            # worker's allocation in the atomic batch transaction. The child
            # re-derives its own requirement with the SAME run-args math and
            # refuses any mismatch before a single weight loads.
            self.reservation, self.owner, self.admitted_total = _claim_reserved(
                self.model_dir, self.catalog_id, claim_payload,
                self.managed, args.dtype, args.max_questions)
        else:
            # pending phase: dtype-aware atomic admit + reserve BEFORE any weight
            # is resident; concurrent admissions subtract it from MemAvailable
            self.reservation, self.owner, self.admitted_total = _register_pending(
                self.model_dir, self.catalog_id, self.managed, args.dtype, args.max_questions
            )
        # Phase-scoped BaseException guards: whichever phase fails clears the
        # held reservation exactly once with a DISTINCT registry label
        # (forensics: failed-startup vs failed-relabel), then re-raises the
        # original error — including SystemExit/KeyboardInterrupt from the
        # SIGTERM handler arriving mid-load.
        try:
            self.engine = LayaEngine(args.model, dtype=dtype, require_gpu=not args.allow_cpu)
        except BaseException:
            _clear_memory(self.catalog_id, self.managed, why="failed-startup", owner=self.owner)
            raise
        try:
            # weights are now resident (LayaEngine evals them); relabel with the
            # parameter-nbytes floor — the admitted total is never grown
            self.reservation = _relabel_resident(self.model_dir, self.catalog_id, self.managed,
                                                 self.reservation, self.owner,
                                                 args.dtype, self.admitted_total, args.max_questions)
        except BaseException:
            _clear_memory(self.catalog_id, self.managed, why="failed-relabel", owner=self.owner)
            raise

    @staticmethod
    def engine_id_fallback(args) -> str:
        return Path(args.model).resolve().name


def _make_handler(state: LayaState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "mlx-omarchy-laya/0.1"

        def log_message(self, fmt, *args):  # one line, stderr, keep foreground logs readable
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
                        "backend": "mlx-omarchy",
                        "device": state.engine.device(),
                        "dtype": str(state.engine.dtype).rsplit(".", 1)[-1],
                        "checkpoint": str(state.model_dir),
                        "source_revision": state.manifest.get("source_revision"),
                        "reservation": (state.reservation or {}).get("state")
                        if isinstance(state.reservation, dict) else None,
                        "managed": state.managed,
                        "max_questions": state.max_questions,
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
                                "owned_by": "convaiinnovations",
                                "kind": "decisions",
                                "max_len": state.engine.max_len,
                            }
                        ],
                    },
                )
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/decisions":
                self._send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"error": "invalid Content-Length"})
                return
            if length <= 0 or length > MAX_BODY_BYTES:
                # drain what the client is still sending so the 400 lands cleanly
                # (rejecting without reading causes a client-side broken pipe)
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(1 << 16, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                self._send_json(400, {"error": "body must be 1..%d bytes" % MAX_BODY_BYTES})
                return
            raw = self.rfile.read(length)
            try:
                req = json.loads(raw)
            except json.JSONDecodeError as exc:
                self._send_json(400, {"error": "invalid JSON: %s" % exc})
                return
            if not isinstance(req.get("state"), (str, dict, list)):
                self._send_json(400, {"error": "'state' must be a string or JSON object"})
                return
            questions = req.get("questions")
            if not isinstance(questions, dict) or not questions:
                self._send_json(400, {"error": "'questions' must be a non-empty object"})
                return
            if len(questions) > state.max_questions:
                self._send_json(400, {"error": "at most %d questions per request (got %d)"
                                      % (state.max_questions, len(questions))})
                return
            for qid, qdef in questions.items():
                if not isinstance(qdef, dict) or qdef.get("type") not in ("choice", "score", "noul"):
                    self._send_json(400, {"error": "question %r: 'type' must be choice|score|noul" % qid})
                    return
                if "instructions" not in qdef:
                    self._send_json(400, {"error": "question %r: missing 'instructions'" % qid})
                    return
            try:
                result = state.engine.system_one(req["state"], questions)
            except ValueError as exc:  # upstream: options do not fit in head_max_len
                self._send_json(400, {"error": str(exc)})
                return
            except Exception as exc:  # backend failure: exact error, never a fallback
                self._send_json(500, {"error": "backend evaluation failed: %s" % exc})
                return
            self._send_json(200, result)

    return Handler


def serve_main(argv):
    args = _parse_args(argv)
    # install BEFORE LayaState: a SIGTERM arriving mid-load must raise
    # SystemExit so the constructor's BaseException guard clears the
    # reservation instead of the default handler killing the process mid-load
    def _on_sigterm(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)
    # Managed pair child: stop when the parent manager's lifetime pipe closes
    # (manager exited or died), so workers never outlive their owner.
    budget = _budget_or_none()
    raw_lifetime = os.environ.get(budget.LIFETIME_FD_ENV, "") if budget is not None else ""
    if raw_lifetime.isdigit():
        def _parent_gone():
            os.kill(os.getpid(), signal.SIGTERM)
        threading.Thread(
            target=budget.watch_parent_fd,
            args=(int(raw_lifetime), _parent_gone),
            daemon=True,
        ).start()
        print("laya: watching parent lifetime fd %s" % raw_lifetime, file=sys.stderr)
    state = None
    try:
        # the admit -> construct -> bind -> serve span is wrapped so any
        # failure (including EADDRINUSE at bind, after the reservation is
        # held) clears the reservation before the process exits
        state = LayaState(args)
    except ManagedBudgetUnavailable as exc:
        print("laya: %s" % exc, file=sys.stderr)
        sys.exit(3)
    try:
        httpd = ThreadingHTTPServer((args.host, args.port), _make_handler(state))
        print(
            "mlx-omarchy-laya: serving %s on http://%s:%d (device=%s, dtype=%s, max_len=%d)"
            % (
                state.catalog_id,
                args.host,
                args.port,
                state.engine.device(),
                args.dtype,
                state.engine.max_len,
            ),
            flush=True,
        )
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()
    finally:
        if state.reservation is not None:
            # the registry must not outlive the process: a resident entry keeps
            # contributing its unmaterialized headroom (bytes - floor) to every
            # future admission for as long as it exists, so a stale entry
            # permanently shrinks what later models may admit. Ownership-scoped:
            # another owner's live reservation is refused.
            _clear_memory(state.catalog_id, state.managed, why="shutdown", owner=state.owner)


if __name__ == "__main__":
    serve_main(sys.argv[1:])
