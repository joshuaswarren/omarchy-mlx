"""Total-context-capped launcher for mlx_lm.server (pinned 0.31.3).

The upstream server treats --max-tokens as a per-request DEFAULT: a client
may send any prompt plus any max_tokens of its own, and nothing bounds
prompt + output. This shim adds the missing cap at the tokenization trust
boundary — the one place where the full prompt token list exists — and
rejects any request whose prompt + max_tokens exceeds the admitted context
budget before a single token is generated.

Aggregate bound: the serve CLI launches this server with decode and prompt
concurrency 1 and the prompt cache disabled, so exactly ONE request's
tokens are ever resident and the aggregate equals the admitted context.
A larger --prompt-cache-size re-enables prefix caching at a documented
cost: worst-case resident KV becomes (cache + 1) x the admitted context.

Build-stamp match: the venv carries this package as unpackaged copies
(install.sh SERVE_PACKAGES), so a wheel-only reinstall can leave an old
shim against a new wheel. main() refuses to launch (exit 4) unless the
.build-commit stamp written by install.sh matches the installed
mlx-omarchy wheel version; MLX_OMARCHY_ALLOW_SERVE_DRIFT=1 downgrades the
refusal to a loud warning. A missing stamp does not refuse when every
copy hashes byte-identical to the installed wheel's dist-info RECORD:
content-verified copies are the same build by definition.

Fail-closed pinning: the shim supports exactly the pinned mlx-lm version.
A different mlx_lm.__version__ refuses to launch unless
MLX_OMARCHY_ALLOW_UNPINNED_MLXLM=1 is set explicitly (loud warning); if
the hooked internals moved anyway, the launch still fails loudly instead
of silently dropping the cap.
"""

from __future__ import annotations

import functools
import inspect
import json
import math
import os
import stat
import sys
import threading
import time
from pathlib import Path

# PAR-32: prevent OpenBLAS worker threads from spinning CPU (~9.5% on Apple
# Silicon) during token decode; callers may still override explicitly.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# PairGates: install /v1/internal/memory probe if the patcher is on
# PYTHONPATH. Side-effect import — no-op if absent.
try:
    import _mlxlm_server_with_memory  # noqa: F401
except ImportError:
    pass

LIMIT_ENV = "MLX_OMARCHY_SERVE_CONTEXT_LIMIT"
TRAINED_ENV = "MLX_OMARCHY_SERVE_TRAINED_CONTEXT"  # optional: the model's trained horizon, warn-only
UNPINNED_ENV = "MLX_OMARCHY_ALLOW_UNPINNED_MLXLM"
STAMP_ENV = "MLX_OMARCHY_ALLOW_SERVE_DRIFT"
STAMP_FILE = ".build-commit"  # written by install.sh next to this module
SUPPORTED_MLXLM_VERSIONS = ("0.31.3",)


def _wheel_commit(version: str | None) -> str | None:
    """The build token a wheel advertises: the commit in the local
    version segment, else the release version itself."""
    if not version:
        return None
    version = str(version).strip()
    if not version:
        return None
    if "+" in version:
        return version.rsplit("+", 1)[1][:10] or None
    return version.lstrip("v") or None


def resolve_stamp_mismatch(stamp, wheel_version, drift_allowed=False,
                           installed=True, record_verified=False,
                           wheel_present=False):
    """Compare the serve copies' build stamp with the installed wheel.

    Returns (refusal, warning): refusal is a loud error string when the
    launch must not proceed, warning when the check cannot bite or drift
    is allowed but named. `installed` marks the copies as living in a
    site-packages venv; a checkout only ever warns, so dev runs and the
    subprocess tests stay unaffected. `record_verified` marks the copies
    as hash-identical to the wheel's dist-info RECORD, which replaces a
    missing stamp instead of refusing byte-identical content.
    `wheel_present` distinguishes an installed wheel whose metadata was
    unreadable from an absent wheel.
    """
    build = _wheel_commit(wheel_version)
    demote = not installed
    if not build:
        if drift_allowed or demote:
            return (None, f"no installed mlx-omarchy wheel version to "
                    f"compare the serve copies against"
                    + ("; continuing on " + STAMP_ENV + "=1"
                       if drift_allowed else "; the serve-drift check "
                       "cannot run here"))
        if not stamp and not wheel_present:
            return (None, "no build stamp and no installed mlx-omarchy "
                    "wheel; the serve-drift check cannot run")
        return ("cannot read the installed mlx-omarchy wheel version; "
                "the serve copies may not match it (reinstall with "
                f"install.sh, or set {STAMP_ENV}=1 to continue)", None)
    if not stamp:
        if drift_allowed or demote:
            return (None, f"this venv's serve copies carry no build stamp"
                    + ("; continuing on " + STAMP_ENV + "=1"
                       if drift_allowed else "; refusing only applies to "
                       "installed copies"))
        if record_verified:
            return (None, "this venv's serve copies carry no build "
                    "stamp; content verified byte-identical to the "
                    "installed mlx-omarchy wheel's RECORD")
        return ("this venv's serve copies carry no build stamp; "
                "reinstall with install.sh so the wheel and the serve "
                f"copies come from one build, or set {STAMP_ENV}=1 to "
                "continue", None)
    stamp = str(stamp).strip().lstrip("v")
    if stamp and (build.startswith(stamp) or stamp.startswith(build)):
        return (None, None)
    if drift_allowed or demote:
        return (None, f"serve copies ({stamp}) do not match the installed "
                f"mlx-omarchy wheel ({build})"
                + ("; continuing on " + STAMP_ENV + "=1" if drift_allowed
                   else "; refusing only applies to installed copies"))
    return (f"serve copies ({stamp}) do not match the installed "
            f"mlx-omarchy wheel ({build}); reinstall with install.sh, or "
            f"set {STAMP_ENV}=1 to continue", None)


BUILD_STAMP_MAX_BYTES = 4096
METADATA_MAX_BYTES = 1 << 20
RECORD_MAX_BYTES = 4 << 20
SERVE_COPY_MAX_BYTES = 4 << 20
_MAX_WALK_DIRS = 1024
_READ_CHUNK_BYTES = 64 << 10
_REGULAR_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NONBLOCK", 0)
    | getattr(os, "O_NOCTTY", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


def _read_regular_bounded(path: Path, max_bytes: int, sink) -> bool:
    """Stream up to max_bytes from a regular file into sink(chunk).

    Follows symlinks (venv or editable installs may symlink regular files),
    opens with O_NONBLOCK so a FIFO without a writer cannot hang, requires
    S_ISREG and st_size <= max_bytes on the opened fd, and loops in unbuffered
    chunks up to max_bytes + 1 so files that grow after fstat (or report
    st_size == 0) stay bounded. Returns True only on EOF within max_bytes.
    """
    fd = None
    try:
        fd = os.open(path, _REGULAR_OPEN_FLAGS)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return False
        if info.st_size > max_bytes:
            return False
        total = 0
        while True:
            chunk = os.read(fd, min(_READ_CHUNK_BYTES, max_bytes + 1 - total))
            if not chunk:
                return True
            total += len(chunk)
            if total > max_bytes:
                return False
            sink(chunk)
    except (OSError, ValueError):
        return False
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def _is_stdlib_md_callable(fn, name: str) -> bool:
    return (
        inspect.isfunction(fn)
        and getattr(fn, "__module__", None) == "importlib.metadata"
        and getattr(fn, "__name__", None) == name
    )


def _read_wheel_version() -> tuple[str | None, bool]:
    """Return (version, wheel_present) without blocking or unbounded reads.

    `wheel_present` is False only when no mlx-omarchy distribution is found
    (`PackageNotFoundError`). When a distribution exists but its METADATA or
    PKG-INFO is unreadable, non-regular, oversized, or invalid UTF-8, returns
    `(None, True)` so an installed server fails closed instead of treating
    corrupt metadata as an absent wheel.
    """
    import email.parser
    import importlib.metadata as md

    if not _is_stdlib_md_callable(md.version, "version"):
        try:
            return (md.version("mlx-omarchy"), True)
        except md.PackageNotFoundError:
            return (None, False)
        except Exception:
            return (None, True)
    try:
        dist = md.distribution("mlx-omarchy")
    except md.PackageNotFoundError:
        return (None, False)
    except Exception:
        return (None, True)
    try:
        dist_path = getattr(dist, "_path", None)
        if isinstance(dist, md.PathDistribution) and isinstance(dist_path, os.PathLike):
            root = Path(dist_path)
            if stat.S_ISDIR(os.stat(root).st_mode):
                target = None
                for name in ("METADATA", "PKG-INFO"):
                    candidate = root / name
                    try:
                        os.lstat(candidate)
                    except FileNotFoundError:
                        continue
                    target = candidate
                    break
                if target is None:
                    return (None, True)
            else:
                target = root
            raw = bytearray()
            if not _read_regular_bounded(target, METADATA_MAX_BYTES, raw.extend):
                return (None, True)
            msg = email.parser.Parser().parsestr(raw.decode("utf-8"))
            return (msg.get("Version") or None, True)
        return (dist.version, True)
    except Exception:
        return (None, True)


def _read_record_hashes() -> dict[str, str] | None:
    """Return {posix_path: sha256_b64} from the installed mlx-omarchy RECORD.

    Reads RECORD through _read_regular_bounded for filesystem PathDistributions
    so a FIFO or oversized RECORD cannot hang or exhaust memory. Preserves
    importlib.metadata.files() when patched or backed by a non-filesystem finder.
    Conflicting duplicate rows for the same normalized path fail closed.
    """
    import csv
    import importlib.metadata as md
    import io
    from pathlib import PurePosixPath

    pairs: list[tuple[str, str | None]] = []
    try:
        if _is_stdlib_md_callable(md.files, "files"):
            dist = md.distribution("mlx-omarchy")
            dist_path = getattr(dist, "_path", None)
            if isinstance(dist, md.PathDistribution) and isinstance(dist_path, os.PathLike):
                raw = bytearray()
                if not _read_regular_bounded(
                        Path(dist_path) / "RECORD", RECORD_MAX_BYTES, raw.extend):
                    return None
                text = raw.decode("utf-8")
                for row in csv.reader(io.StringIO(text, newline="")):
                    if not row or not row[0]:
                        continue
                    spec = row[1] if len(row) >= 2 and row[1] else None
                    pairs.append((str(PurePosixPath(row[0])), spec))
            else:
                for entry in dist.files or []:
                    spec = (f"{entry.hash.mode}={entry.hash.value}"
                            if entry.hash else None)
                    pairs.append((str(entry), spec))
        else:
            for entry in md.files("mlx-omarchy") or []:
                spec = (f"{entry.hash.mode}={entry.hash.value}"
                        if entry.hash else None)
                pairs.append((str(entry), spec))
    except Exception:
        return None

    seen_specs: dict[str, str | None] = {}
    recorded: dict[str, str] = {}
    for key, spec in pairs:
        if key in seen_specs and seen_specs[key] != spec:
            return None
        seen_specs[key] = spec
        if spec is None:
            continue
        mode, sep, value = spec.partition("=")
        if sep and mode == "sha256":
            recorded[key] = value
    return recorded or None


def _collect_py_copies(packages: Path) -> list[Path] | None:
    """Return sorted *.py entries under packages, failing closed on any
    traversal error, unreadable subdirectory, or directory symlink cycle."""
    try:
        root_st = os.stat(packages)
        if not stat.S_ISDIR(root_st.st_mode):
            return None
        root_key = (root_st.st_dev, root_st.st_ino)
        copies: list[Path] = []
        visited_dirs = 0
        stack: list[tuple[Path, frozenset[tuple[int, int]]]] = [
            (packages, frozenset((root_key,)))
        ]
        while stack:
            current, ancestors = stack.pop()
            visited_dirs += 1
            if visited_dirs > _MAX_WALK_DIRS:
                return None
            with os.scandir(current) as entries:
                for entry in entries:
                    entry_path = Path(entry.path)
                    if entry.name.endswith(".py"):
                        copies.append(entry_path)
                    if entry.is_dir(follow_symlinks=False):
                        st = entry.stat(follow_symlinks=False)
                        key = (st.st_dev, st.st_ino)
                        if key in ancestors:
                            return None
                        stack.append((entry_path, ancestors | {key}))
                    elif entry.is_symlink():
                        try:
                            st = entry.stat(follow_symlinks=True)
                        except FileNotFoundError:
                            continue
                        if stat.S_ISDIR(st.st_mode):
                            key = (st.st_dev, st.st_ino)
                            if key in ancestors:
                                return None
                            stack.append((entry_path, ancestors | {key}))
    except (OSError, ValueError):
        return None
    copies.sort()
    return copies


def _decode_record_sha256(encoded) -> bytes | None:
    """Decode a canonical PEP 376 urlsafe-base64-nopad SHA-256 digest."""
    import base64

    if (not isinstance(encoded, str) or len(encoded) != 43
            or not encoded.isascii() or "+" in encoded
            or "/" in encoded or "=" in encoded):
        return None
    try:
        want = base64.b64decode(encoded + "=", altchars=b"-_", validate=True)
    except Exception:
        return None
    if len(want) != 32:
        return None
    if base64.urlsafe_b64encode(want).rstrip(b"=").decode("ascii") != encoded:
        return None
    return want


def _is_installed_copy(module_path: Path | None = None) -> bool:
    """True when this module lives under an installation site-packages root
    (including when site-packages is a symlink whose target has another name)."""
    raw_path = Path(module_path) if module_path is not None else Path(__file__)
    if "site-packages" in raw_path.parts:
        return True
    try:
        resolved = raw_path.resolve()
    except OSError:
        return False
    if "site-packages" in resolved.parts:
        return True
    roots: list[str] = []
    try:
        import site
    except Exception:
        site = None
    if site is not None:
        try:
            roots.extend(site.getsitepackages())
        except Exception:
            pass
        try:
            user_site = site.getusersitepackages()
            if user_site:
                roots.append(user_site)
        except Exception:
            pass
    try:
        import sysconfig
        for key in ("purelib", "platlib"):
            val = sysconfig.get_path(key)
            if val:
                roots.append(val)
    except Exception:
        pass
    for root in roots:
        try:
            if resolved.is_relative_to(Path(root).resolve()):
                return True
        except OSError:
            continue
    return False


def read_build_stamp(directory=None):
    """Read the install.sh stamp next to this module; missing or unreadable
    means the copies cannot prove a stamp."""
    path = (Path(directory) if directory else Path(__file__).parent) / STAMP_FILE
    raw = bytearray()
    if not _read_regular_bounded(path, BUILD_STAMP_MAX_BYTES, raw.extend):
        return None
    try:
        return raw.decode("utf-8").strip() or None
    except UnicodeDecodeError:
        return None


def verify_serve_copies_against_record(directory=None):
    """Content-verify every serve copy against the installed wheel RECORD.

    True when every .py copy hashes to the value the installed
    mlx-omarchy dist-info RECORD lists for the same package path. Only
    canonical PEP 376 urlsafe-base64-nopad sha256 values are accepted;
    anything else fails closed. A copy missing from the RECORD, a hash
    mismatch, an undecodable or non-canonical value, an unreadable,
    non-regular, or oversized copy or RECORD, an unreadable or cyclic
    subdirectory, or no RECORD hashes at all is False: unverifiable
    content must not pass a missing-stamp check.
    """
    import hashlib

    packages = Path(directory) if directory else Path(__file__).parent
    copies = _collect_py_copies(packages)
    if not copies:
        return False
    recorded = _read_record_hashes()
    if not recorded:
        return False
    prefix = packages.name
    for copy in copies:
        rel = copy.relative_to(packages)
        want = _decode_record_sha256(recorded.get(f"{prefix}/{rel.as_posix()}"))
        if want is None:
            return False
        hasher = hashlib.sha256()
        if not _read_regular_bounded(copy, SERVE_COPY_MAX_BYTES, hasher.update):
            return False
        if want != hasher.digest():
            return False
    return True


def check_build_stamp(wheel_version=None):
    """Refuse to launch when serve copies and wheel disagree.

    Refusal applies only to copies living in site-packages; a checkout
    import of this module only warns, so dev runs are never blocked.
    """
    if wheel_version is not None:
        version, wheel_present = wheel_version, bool(wheel_version)
    else:
        version, wheel_present = _read_wheel_version()
    installed = _is_installed_copy()
    stamp = read_build_stamp()
    record_verified = not stamp and verify_serve_copies_against_record()
    refusal, warning = resolve_stamp_mismatch(
        stamp, version,
        drift_allowed=os.environ.get(STAMP_ENV) == "1",
        installed=installed,
        record_verified=record_verified,
        wheel_present=wheel_present)
    if warning:
        print(f"warning: {warning}", file=sys.stderr, flush=True)
    if refusal:
        print(f"error: {refusal}", file=sys.stderr, flush=True)
        raise SystemExit(4)


def _mlxlm_version(server_module) -> str:
    for holder in (server_module, getattr(server_module, "_version", None)):
        version = getattr(holder, "__version__", None)
        if version:
            return str(version)
    return "unknown"


def check_pinned(server_module) -> str:
    version = _mlxlm_version(server_module)
    if version in SUPPORTED_MLXLM_VERSIONS:
        return version
    if os.environ.get(UNPINNED_ENV) == "1":
        print(
            f"warning: {UNPINNED_ENV}=1 — mlx-lm {version} is not the pinned "
            f"{SUPPORTED_MLXLM_VERSIONS}; the cap hooks may not hold",
            file=sys.stderr,
        )
        return version
    raise RuntimeError(
        f"mlx-lm {version} is not the pinned version "
        f"{SUPPORTED_MLXLM_VERSIONS}; refusing to launch without the "
        f"verified total-context cap (export {UNPINNED_ENV}=1 to override "
        "explicitly)"
    )


EXPECTED_TOKENIZE_SIGNATURE = ("self", "tokenizer", "request", "args")

TTFT_TRACE_ENV = "MLX_OMARCHY_TTFT_TRACE"
_ttft_armed = {"first_chunk": False}


def _ttft_trace(event: str, **fields) -> None:
    """One JSON line per request boundary when MLX_OMARCHY_TTFT_TRACE=1.

    Timestamps are time.monotonic() seconds, comparable across processes on
    one host; the assistant's ttft-phases.jsonl uses the same clock."""
    if os.environ.get(TTFT_TRACE_ENV) != "1":
        return
    fields["event"] = event
    fields["t"] = time.monotonic()
    print("[ttft] " + json.dumps(fields), flush=True)


def loaded_models_payload(model_ref: str) -> dict:
    """The OpenAI-compatible /v1/models payload: the LOADED model only.

    Upstream handle_models_request scans the whole HF cache and appends the
    --model entry last, so data[0] is whichever repo downloaded most
    recently - a server serving gemma reported Qwen because that download
    was newer. Clients read data[0].id, so the served model must be the
    only entry. Upstream's /v1/models/<repo_id> filter path still falls
    through to the cache scan; only the bare route is overridden.
    """
    ref = model_ref or "unknown"
    p = Path(ref)
    if p.exists():
        ref = str(p.resolve())
    return {"object": "list",
            "data": [{"id": ref, "object": "model",
                      "created": int(time.time())}]}


def install_models_route(server_module) -> None:
    """Answer GET /v1/models with the loaded model; wrap, not replace, the
    incoming handler class so this composes with the yield gate in any
    install order. A stub server without the HTTP surface skips the route
    (same tolerance as the yield gate)."""
    original_run = getattr(server_module, "_run_http_server", None)
    if not callable(original_run):
        print("warning: no _run_http_server surface; loaded-model /v1/models "
              "route disabled", file=sys.stderr)
        return

    @functools.wraps(original_run)
    def run_with_models(host, port, response_generator, server_class=None,
                        handler_class=None, *args, **kwargs):
        base = handler_class or server_module.APIHandler

        class _ModelsHandler(base):
            disable_nagle_algorithm = True

            def do_GET(self):  # noqa: N802 (stdlib API)
                path = self.path.split("?", 1)[0]
                if path == "/v1/models":
                    auth = getattr(self, "_authorized", None)
                    if callable(auth) and not auth():
                        return base.do_GET(self)
                    model_name = (getattr(self, "alias", "")
                                  or self.response_generator.cli_args.model)
                    raw = json.dumps(loaded_models_payload(
                        model_name)).encode("utf-8")
                    # Upstream answers through _set_completion_headers, which
                    # adds the CORS headers; bypassing it dropped them and
                    # broke cross-origin browser clients (w86 review).
                    self._set_completion_headers(200)
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                return base.do_GET(self)

        return original_run(host, port, response_generator,
                            server_class=server_class
                            or server_module.ThreadingHTTPServer,
                            handler_class=_ModelsHandler, *args, **kwargs)

    server_module._run_http_server = run_with_models


def install(server_module, limit: int, trained: int | None = None) -> None:
    generator = server_module.ResponseGenerator
    original = getattr(generator, "_tokenize", None)
    if not callable(original):
        raise RuntimeError(
            "mlx_lm.server.ResponseGenerator._tokenize not found; this shim "
            "does not know this mlx-lm version and refuses to launch without "
            "the total-context cap"
        )
    # Enforced for the pinned version AND any pin bypass: if the internals
    # moved, the cap cannot be proven to hold and the launch refuses.
    try:
        params = tuple(inspect.signature(original).parameters)
    except (TypeError, ValueError):
        params = ()
    if params != EXPECTED_TOKENIZE_SIGNATURE:
        raise RuntimeError(
            "unexpected _tokenize signature "
            f"{params} != {EXPECTED_TOKENIZE_SIGNATURE}; this shim does not "
            "know this mlx-lm build and refuses to launch without the "
            "verified total-context cap"
        )

    def capped(self, tokenizer, request, args):
        _ttft_t0 = time.monotonic()
        result = original(self, tokenizer, request, args)
        prompt = result[0]
        # Upstream semantics (verified in the pinned source): the HTTP
        # handler resolves max_tokens from the body or the CLI default
        # (an int, default 512) and validates int >= 0 BEFORE queuing, so
        # None or negatives never legitimately reach this boundary.
        # 0 generates nothing (bounded); -1 (infinite) is handler-blocked.
        max_tokens = getattr(args, "max_tokens", None)
        if max_tokens is None or isinstance(max_tokens, bool) \
                or not isinstance(max_tokens, int) or max_tokens < 0:
            raise ValueError(
                "request rejected: max_tokens must be a non-negative int at "
                f"the cap boundary, got {max_tokens!r}"
            )
        total = len(prompt) + max_tokens
        if trained is not None and total > trained:
            # Minefield 55/61: past the trained horizon quality degrades with no error;
            # the admitted window is a memory bound, not a quality bound, so this warns.
            print(f"warning: this request runs {total} tokens past the model's trained "
                  f"window of {trained}; quality beyond the horizon is not guaranteed",
                  file=sys.stderr, flush=True)
        if total > limit:
            raise ValueError(
                f"request rejected: prompt ({len(prompt)} tokens) + max_tokens "
                f"({max_tokens}) exceeds the admitted context budget of {limit}; "
                "lower the prompt or max_tokens"
            )
        _ttft_armed["first_chunk"] = True
        _ttft_trace("tokenized", prompt=len(prompt), max_tokens=max_tokens,
                    tokenize_s=round(time.monotonic() - _ttft_t0, 3))
        return result

    generator._tokenize = capped


PROBE_ENV = "MLX_OMARCHY_SERVE_IDS_PROBE"


# ----------------------------------------------------------- speech yield gate
#
# Cooperative TTS scheduling between generation chunks (offline assistant
# design, "Recognition and synthesis"): the assistant coordinator must be able
# to run bounded speech synthesis on the GPU BETWEEN generation chunks instead
# of only after a whole response. Generation happens in this (separate mlx-lm)
# process, so a coordinator-side lock alone cannot stop it: without a real
# pause the server would keep decoding while the voice worker synthesizes —
# overlapping GPU generation and synthesis — and "not reading the SSE stream"
# is only a fake pause (TCP backpressure freezes the loop mid-flight, not at a
# token boundary).
#
# The gate therefore parks the pinned batched decode loop at a genuine chunk
# boundary: BatchGenerator.next() runs one full decode step synchronously and
# returns with every array materialized and all KV caches resident, so a block
# between steps is a GPU-idle boundary with state preserved. The coordinator
# only releases its GPU lock for synthesis AFTER a hold was acknowledged from
# inside this process ("held" means: no step running now and no step can start
# until release), so the no-overlap invariant is provable, not assumed. When
# the pause cannot be proven (unpatched single-path generation, parked-ack
# timeout, wrong secret), hold fails honestly and the assistant keeps today's
# busy rejection. Nothing is ever SIGSTOPed.

YIELD_CONTROL_PATH = "/v1/internal/yield"
YIELD_SECRET_HEADER = "X-MLX-Yield-Secret"
_ACTIVE_GATE: "YieldGate | None" = None
YIELD_HOLD_MAX_SECONDS = 600.0
YIELD_ACK_MAX_SECONDS = 30.0
YIELD_BODY_MAX_BYTES = 1 << 20
EXPECTED_NEXT_SIGNATURE = ("self",)
EXPECTED_RUN_SIGNATURE = ("host", "port", "response_generator", "server_class",
                          "handler_class")
# patches/mlx-lm-server-alias-key.patch appends (alias, api_key) to
# mlx_lm.server._run_http_server; accept both unpatched and patched forms
# while still refusing any unknown signature.
EXPECTED_RUN_SIGNATURES = (
    EXPECTED_RUN_SIGNATURE,
    EXPECTED_RUN_SIGNATURE + ("alias", "api_key"),
)


class YieldGate:
    """Hold/park state shared by the decode thread and the control route.

    State changes interlock under one condition so a hold can never be
    acknowledged while a decode step is starting: the step side sets
    ``in_step`` and checks ``desire`` in one atomic region, and the control
    side sets ``desire`` and checks ``in_step`` in the same region.
    """

    def __init__(self, hold_max_seconds: float = YIELD_HOLD_MAX_SECONDS):
        self._cond = threading.Condition()
        self._secret: str | None = None
        self._desire = False
        self._parked = False
        self._in_step = False
        self._closed = False
        self._hold_max_seconds = hold_max_seconds
        self.holds_granted = 0
        self.hold_timeouts = 0

    def bind_secret(self, secret) -> None:
        """Bind the secret presented by the current chat stream. Loopback
        trust: generation requests were never authenticated on this server;
        the secret only keeps browsers (no custom headers on simple
        cross-origin requests) and unrelated local tools from flipping the
        gate."""
        if not isinstance(secret, str) or not secret:
            return
        with self._cond:
            self._secret = secret

    def close(self) -> None:
        """Release a parked decode thread at shutdown."""
        with self._cond:
            self._closed = True
            self._desire = False
            self._cond.notify_all()

    def control(self, action, secret, ack_timeout) -> tuple[int, dict]:
        """Handle one control request; returns (http status, payload)."""
        with self._cond:
            if action == "probe":
                if self._secret is None and isinstance(secret, str) and secret:
                    self._secret = secret
                return 200, {"gate": True, "bound": self._secret is not None,
                             "secret_ok": isinstance(secret, str)
                             and secret == self._secret}
            if self._closed:
                return 503, {"error": "worker is shutting down"}
            if not isinstance(secret, str) or secret != self._secret:
                return 403, {"error": "yield secret mismatch"}
            if action == "hold":
                # Validate before raising desire: a raise here would leave the
                # next step parked, and NaN would make the ack wait spin.
                ack_seconds = math.nan
                if (isinstance(ack_timeout, (int, float))
                        and not isinstance(ack_timeout, bool)):
                    try:
                        ack_seconds = float(ack_timeout)
                    except OverflowError:  # int beyond float range
                        pass
                if not math.isfinite(ack_seconds):
                    return 400, {"error": "ack_timeout must be a finite number"}
                self._desire = True
                if self._in_step or self._parked:
                    deadline = time.monotonic() + min(
                        max(ack_seconds, 0.1), YIELD_ACK_MAX_SECONDS)
                    while not self._parked and not self._closed:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            # Honest failure: no proven pause, no fake one.
                            self._desire = False
                            self.hold_timeouts += 1
                            self._cond.notify_all()
                            return 409, {"held": False,
                                         "reason": "park-ack-timeout"}
                        self._cond.wait(remaining)
                if self._closed:
                    self._desire = False
                    return 503, {"error": "worker is shutting down"}
                self.holds_granted += 1
                return 200, {"held": True}
            if action == "release":
                self._desire = False
                self._cond.notify_all()
                return 200, {"released": True}
            return 400, {"error": "action must be probe, hold or release"}

    # Decode-thread side (called from the wrapped BatchGenerator).
    def _before_step(self) -> None:
        with self._cond:
            self._in_step = True
            if not self._desire:
                return
            self._parked = True
            self._cond.notify_all()
            deadline = time.monotonic() + self._hold_max_seconds
            while self._desire and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # Crash net: an assistant that died mid-hold must not
                    # wedge generation forever. It only fires long after any
                    # bounded synthesis could legitimately take.
                    self._desire = False
                    self._cond.notify_all()
                    break
                self._cond.wait(remaining)
            self._parked = False
            self._cond.notify_all()

    def _after_step(self) -> None:
        with self._cond:
            self._in_step = False
            self._cond.notify_all()


def install_thinking_switch(server_module) -> None:
    """Serve default: chat answers in ``content`` unless the request
    opts in.

    Thinking-capable chat templates (gemma-4, the Qwen3 family) default
    the thought channel on; with a small max_tokens the stream never
    closes it and every delta lands in ``delta.reasoning`` with
    ``content`` empty. The clients that hit this - the serve profiler,
    tools/serve-bench - send plain chat bodies, so the shim sets
    ``enable_thinking: False`` for every request that does not carry an
    explicit ``chat_template_kwargs.enable_thinking``. Opt-in is
    ``chat_template_kwargs: {"enable_thinking": true}`` (upstream's own
    per-request knob, which upstream applies on top of args).
    Templates that ignore the kwarg are unchanged.
    """
    generator = getattr(server_module, "ResponseGenerator", None)
    if generator is None:
        # stub servers (tests) have no generator surface; nothing to wrap
        return
    original = getattr(generator, "_tokenize", None)
    if not callable(original):
        raise RuntimeError(
            "no ResponseGenerator._tokenize to wrap; this shim does not "
            "know this mlx-lm build and refuses the thinking switch")
    if getattr(original, "_omarchy_thinking_switch", False):
        return

    @functools.wraps(original)
    def _tokenize(self, tokenizer, request, args):
        kwargs = dict(getattr(args, "chat_template_kwargs", None) or {})
        if "enable_thinking" not in kwargs:
            kwargs["enable_thinking"] = False
            args.chat_template_kwargs = kwargs
        return original(self, tokenizer, request, args)

    _tokenize._omarchy_thinking_switch = True  # noqa: B010 - guard flag
    generator._tokenize = _tokenize
def wrap_generator_with_gate(real_generator, gate):
    """Return a generator whose next() pre-evaluates response logprobs.

    The upstream batch loop reads ``r.logprobs[r.token].item()`` per
    response - a GPU readback that serializes the async decode pipeline
    (16-inch M1 Max measurement: 26.5 ms inside the step plus a 27 ms
    readback = 63 ms per streamed chunk). That read runs on the same
    thread as the loop, so delaying delivery cannot overlap it;
    evaluating the full logprobs array inside the wrapped step can,
    because this step's forward pass was already scheduled at the
    step's start and finishes while ``next()`` waits for the previous
    one. mlx import is deferred: only a real decode step needs it.
    """

    class _EvalBatchGenerator:
        _omarchy_yield_wrapped = True

        def __init__(self):
            self._real = real_generator

        def __getattr__(self, name):
            return getattr(self._real, name)

        def next(self, *args, **kwargs):
            gate._before_step()
            try:
                result = self._real.next(*args, **kwargs)
                prompts, gens = result
                arrays = [r.logprobs for r in list(prompts) + list(gens)
                          if getattr(r, "logprobs", None) is not None]
                if arrays:
                    import mlx.core as mx
                    mx.eval(*arrays)
                if _ttft_armed["first_chunk"] and (prompts or gens):
                    _ttft_armed["first_chunk"] = False
                    _ttft_trace("first_chunk")
                return result
            finally:
                gate._after_step()

    return _EvalBatchGenerator()


def install_yield_gate(server_module, gate: YieldGate) -> None:
    """Wrap the pinned decode loop and HTTP handler with the yield gate.

    Refuses to launch when the pinned internals moved: an unverified hook
    could silently stop parking or park mid-step, so the shim refuses the
    same way it does for the context cap.
    """
    real_generator = server_module.BatchGenerator
    if getattr(real_generator, "_omarchy_yield_wrapped", False):
        return
    next_params = tuple(inspect.signature(real_generator.next).parameters)
    if next_params != EXPECTED_NEXT_SIGNATURE:
        raise RuntimeError(
            "unexpected BatchGenerator.next signature "
            f"{next_params} != {EXPECTED_NEXT_SIGNATURE}; this shim does not "
            "know this mlx-lm build and refuses to launch with the speech "
            "yield gate")
    run_fn = server_module._run_http_server
    run_params = tuple(inspect.signature(run_fn).parameters)
    if run_params not in EXPECTED_RUN_SIGNATURES:
        raise RuntimeError(
            "unexpected _run_http_server signature "
            f"{run_params} not in {EXPECTED_RUN_SIGNATURES}; this shim does not "
            "know this mlx-lm build and refuses to launch with the speech "
            "yield gate")

    class _GatedBatchGenerator:
        """Delegates everything; parks at genuine chunk boundaries."""
        _omarchy_yield_wrapped = True

        def __init__(self, *args, **kwargs):
            self._eval = wrap_generator_with_gate(
                real_generator(*args, **kwargs), gate)

        def __getattr__(self, name):
            return getattr(self._eval, name)

        def next(self, *args, **kwargs):
            return self._eval.next(*args, **kwargs)

    server_module.BatchGenerator = _GatedBatchGenerator

    @functools.wraps(run_fn)
    def _run_with_gate(host, port, response_generator, server_class=None,
                       handler_class=None, *args, **kwargs):
        # Build the gated class HERE, on the incoming handler, so a route
        # installed before the gate stays the base (its do_GET survives)
        # and the gate's do_POST routing is always present.
        base = handler_class or server_module.APIHandler

        class _GatedHandler(base):
            disable_nagle_algorithm = True

            def do_POST(self):  # noqa: N802 (stdlib API)
                path = self.path.split("?", 1)[0]
                if path == YIELD_CONTROL_PATH:
                    return self._handle_yield_control()
                secret = self.headers.get(YIELD_SECRET_HEADER)
                if secret:
                    gate.bind_secret(secret)
                return base.do_POST(self)

            def _handle_yield_control(self):
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    # read(-1) blocks until EOF; a huge length raises or
                    # allocates it.
                    if not 0 <= length <= YIELD_BODY_MAX_BYTES:
                        raise ValueError("Content-Length out of range")
                    body = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(body, dict):
                        raise ValueError("body must be a JSON object")
                except (ValueError, OSError, RecursionError):
                    self._yield_json(400, {"error": "invalid control body"})
                    return
                status, payload = gate.control(
                    body.get("action"), body.get("secret"),
                    body.get("ack_timeout", YIELD_ACK_MAX_SECONDS))
                self._yield_json(status, payload)

            def _yield_json(self, status, payload):
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        return run_fn(host, port, response_generator,
                      server_class=server_class or server_module.ThreadingHTTPServer,
                      handler_class=_GatedHandler, *args, **kwargs)

    server_module._run_http_server = _run_with_gate


def install_ids_probe(server_module, emit=None) -> None:
    """Bench-only ids capture for the managed route (default OFF; enabled
    only when MLX_OMARCHY_SERVE_IDS_PROBE=1). Wraps the streaming
    detokenizer classes so every raw sampled token id is recorded at the
    add_token boundary and the completed per-request id list is emitted as
    one `PROBE: {...}` JSON line on stderr at reset(). Nothing on disk is
    touched; with the env unset this function is never called."""
    if emit is None:
        def emit(event):
            print("PROBE:", json.dumps(event), file=sys.stderr, flush=True)
    state = {"ids": []}
    from mlx_lm.tokenizer_utils import (
        NaiveStreamingDetokenizer,
        BPEStreamingDetokenizer,
        SPMStreamingDetokenizer,
        StreamingDetokenizer,
    )

    def wrap(cls):
        if cls is None or getattr(cls, "_ids_probe_installed", False):
            return
        orig_add = getattr(cls, "add_token", None)
        if orig_add is None:
            return

        def add_token_probe(self, token, _orig=orig_add):
            state["ids"].append(int(token))
            _orig(self, token)

        cls.add_token = add_token_probe
        cls._ids_probe_installed = True

        orig_reset = getattr(cls, "reset", None)
        if orig_reset is not None:

            def reset_probe(self, _orig=orig_reset):
                if state["ids"]:
                    import hashlib
                    emit({"event": "generation", "n": len(state["ids"]),
                          "ids_sha16": hashlib.sha256(
                              json.dumps(state["ids"]).encode()).hexdigest()[:16],
                          "ids": state["ids"]})
                    state["ids"] = []
                return _orig(self)

            cls.reset = reset_probe

        # mlx_lm's stream_generate finalizes (never resets) at request end —
        # the server path's actual flush boundary (generate.py:741).
        orig_finalize = getattr(cls, "finalize", None)
        if orig_finalize is not None:

            def finalize_probe(self, _orig=orig_finalize):
                if state["ids"]:
                    import hashlib
                    emit({"event": "generation", "n": len(state["ids"]),
                          "ids_sha16": hashlib.sha256(
                              json.dumps(state["ids"]).encode()).hexdigest()[:16],
                          "ids": state["ids"]})
                    state["ids"] = []
                return _orig(self)

            cls.finalize = finalize_probe

    for cls in (StreamingDetokenizer, NaiveStreamingDetokenizer,
                BPEStreamingDetokenizer, SPMStreamingDetokenizer):
        wrap(cls)

    # Third capture site: the BATCHED branch (mlx_lm/server.py:884) feeds
    # generated tokens to the detokenizer via add_token but NEVER calls
    # reset/finalize — the batch flush boundary is BatchGenerator.remove
    # (finished uids) and close. With the managed CLI's decode_concurrency
    # of 1, the accumulated state at remove time is exactly the finished
    # request's token array.
    batch_cls = getattr(server_module, "BatchGenerator", None)

    def _flush_uids(uids=None):
        import hashlib
        store = state.get("by_uid", {})
        done = list(store) if not uids else [u for u in uids if u in store]
        for uid in done:
            ids = store.pop(uid)
            emit({"event": "generation", "n": len(ids),
                  "ids_sha16": hashlib.sha256(
                      json.dumps(ids).encode()).hexdigest()[:16],
                  "ids": ids})

    if batch_cls is not None and not getattr(batch_cls, "_ids_probe_flush",
                                             False):
        orig_next = getattr(batch_cls, "next", None)
        if orig_next is not None:
            # Primary batched capture: gen_responses carry (.uid, .token) —
            # accumulate per uid and emit each uid's array when the server
            # removes that uid (request completion).
            def next_probe(self, *_a, _orig=orig_next, **_k):
                prompt_responses, gen_responses = _orig(self, *_a, **_k)
                for r in gen_responses or []:
                    token = getattr(r, "token", None)
                    uid = getattr(r, "uid", None)
                    if token is None or uid is None:
                        continue
                    state.setdefault("by_uid", {}).setdefault(
                        uid, []).append(int(token))
                    # The server never resets the detokenizer on this route;
                    # flush a uid's array the moment its generation finishes.
                    if getattr(r, "finish_reason", None) is not None:
                        _flush_uids([uid])
                return prompt_responses, gen_responses
            batch_cls.next = next_probe

        orig_remove = getattr(batch_cls, "remove", None)
        if orig_remove is not None:
            def remove_probe(self, uids, _orig=orig_remove):
                _flush_uids(list(uids) if uids else None)
                return _orig(self, uids)
            batch_cls.remove = remove_probe
        else:
            orig_remove = None
        orig_close = getattr(batch_cls, "close", None)
        if orig_close is not None:
            def close_probe(self, _orig=orig_close):
                _flush_uids(None)
                return _orig(self)
            batch_cls.close = close_probe
        batch_cls._ids_probe_flush = True

    # Second, independent capture site: mlx_lm.server calls stream_generate
    # on the single-stream path (the managed CLI pins decode_concurrency=1,
    # so this is THE generation route). Collect each yielded response's
    # token and flush the per-request array when the generator exhausts —
    # independent of any detokenizer reset/finalize behavior.
    orig_stream = getattr(server_module, "stream_generate", None)
    if orig_stream is not None:
        def stream_probe(*args, **kwargs):
            gen = orig_stream(*args, **kwargs)
            print("shim: stream_probe ENTERED", file=sys.stderr, flush=True)
            count = 0
            while True:
                try:
                    item = next(gen)
                except StopIteration:
                    break
                token = getattr(item, "token", None)
                if token is not None:
                    state["ids"].append(int(token))
                    count += 1
                try:
                    yield item
                except GeneratorExit:
                    # Client disconnected mid-generation: flush what the
                    # request produced so partial arrays are recorded.
                    if count:
                        import hashlib
                        emit({"event": "generation_client_disconnected",
                              "n": len(state["ids"]),
                              "ids": state["ids"]})
                        state["ids"] = []
                    raise
            if count:
                import hashlib
                emit({"event": "generation", "n": len(state["ids"]),
                      "ids_sha16": hashlib.sha256(
                          json.dumps(state["ids"]).encode()).hexdigest()[:16],
                      "ids": state["ids"]})
                state["ids"] = []

        server_module.stream_generate = stream_probe


def start_managed() -> None:
    """Pair worker under the assistant manager: exit when its lifetime pipe
    closes (manager died), and claim the reservation before any weight
    loads. A standalone launch passes neither fd and this does nothing."""
    try:
        from . import budget
    except ImportError:
        import budget
    try:
        budget.watch_parent_from_env()
        budget.claim_from_env()
    except budget.BudgetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(3)


def main() -> None:
    check_build_stamp()
    raw = os.environ.get(LIMIT_ENV, "")
    try:
        limit = int(raw)
    except ValueError:
        print(f"error: {LIMIT_ENV} must be an integer, got {raw!r}", file=sys.stderr)
        raise SystemExit(3)
    if limit <= 0:
        print(f"error: {LIMIT_ENV} must be positive", file=sys.stderr)
        raise SystemExit(3)
    start_managed()
    # Serving placement, step 1: clamp THIS thread before mlx and mlx_lm start
    # their stream threads, so every thread they create inherits uclamp_min.
    # Same call bench_decode.py makes before any mlx import. Silent fallback:
    # EPERM / old kernel / off switch -> serving is unchanged.
    try:
        from .perf_placement import apply_from_env
    except ImportError:
        from perf_placement import apply_from_env
    early_applied, early_detail = apply_from_env()
    try:
        import mlx_lm.server as server_module
    except ImportError as exc:
        print(f"error: mlx_lm is not installed ({exc})", file=sys.stderr)
        raise SystemExit(3)
    check_pinned(server_module)
    raw_trained = os.environ.get(TRAINED_ENV, "").strip()
    trained = None
    if raw_trained:
        try:
            trained = int(raw_trained)
        except ValueError:
            print(f"error: {TRAINED_ENV} must be an integer, got {raw_trained!r}", file=sys.stderr)
            raise SystemExit(3)
        if trained <= 0:
            print(f"error: {TRAINED_ENV} must be positive", file=sys.stderr)
            raise SystemExit(3)
    # chat answers in content by default; opt-in via the request's
    # chat_template_kwargs. Applies to every template that honors
    # enable_thinking, not only gemma.
    install_thinking_switch(server_module)
    install(server_module, limit, trained=trained)
    gate = None
    if getattr(server_module, "BatchGenerator", None) is not None and getattr(server_module, "_run_http_server", None) is not None:
        gate = YieldGate()
        globals()["_ACTIVE_GATE"] = gate
        install_yield_gate(server_module, gate)
    else:
        # No batched decode surface to park (stub servers, non-batched
        # builds): launch WITHOUT the gate. Assistant speak requests then
        # keep the honest busy refusal instead of a fake pause.
        print("warning: no BatchGenerator surface; speech yield gate disabled",
              file=sys.stderr)
    # The route installs LAST so the gate's _run_http_server signature
    # guard above still inspects the original upstream function.
    install_models_route(server_module)
    probe_wanted = os.environ.get(PROBE_ENV) == "1"
    print(f"shim: PROBE_ENV={os.environ.get(PROBE_ENV)!r} "
          f"installing={probe_wanted}", file=sys.stderr, flush=True)
    if probe_wanted:
        install_ids_probe(server_module)
        print("shim: ids probe installed", file=sys.stderr, flush=True)
    pct = getattr(server_module, "_process_control_tokens", None)
    if pct is not None and getattr(pct, "_omarchy_ttft_wrapped", False) is False:
        def traced_pct(ctx, token_stream, _orig=pct):
            stream = _orig(ctx, token_stream)
            first = True

            def traced():
                nonlocal first
                for gen in stream:
                    if first:
                        first = False
                        _ttft_trace("first_control_yield",
                                    buffer=max((len(s) for s in (ctx.sequences or ())),
                                               default=0))
                    yield gen
            return traced()

        traced_pct._omarchy_ttft_wrapped = True
        server_module._process_control_tokens = traced_pct
    # Step 1 already ran above, before mlx_lm started its threads. This second
    # call only covers threads that predated it; nothing new is imported here.
    applied, detail = apply_from_env()
    print(f"shim: uclamp placement {'applied' if applied else 'skipped'} "
          f"(start: {early_detail}; now: {detail})", file=sys.stderr, flush=True)
    server_module.main()


if __name__ == "__main__":
    main()
