"""Assistant pair runtime: curated pairs, adaptive context, managed workers.

PairManager turns a catalog pair record into two managed worker processes
(chat + typed decisions) under one atomic budget batch admission:

- setup()      resolve + qualify the pair, size the context continuously
               (integer bytes, no RAM tiers), plan or perform pinned
               downloads and the Laya conversion;
- start()      admit the WHOLE pair in one locked transaction, spawn both
               workers, each claiming its reservation through a private
               inherited pipe, watched by the parent-lifetime pipe;
- ensure_context() atomically grow the chat reservation (fresh fit check)
               and restart the chat worker at the new cap; refusal retains
               the prior cap;
- stop()       bounded termination, then record cleanup ONLY after a
               verified worker exit.

Readiness is honest: ready_offline requires the pair's own catalog
qualification receipt plus a verified offline cold start. Running workers
never promote an unqualified pair.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

from mlx_omarchy_serve import _mlxlm_server as cap_shim
from mlx_omarchy_serve import budget, catalog
from mlx_omarchy_serve import __main__ as serve_cli

from . import managed

PREFERENCES = ("fast", "balanced", "long-context")
# Explicit local qualification runs: unqualified candidates may START, with
# the shortfall visible in status(); they are never marked ready.
PAIR_DEV_QUALIFICATION_ENV = "MLX_OMARCHY_PAIR_DEV_QUALIFICATION"

DESKTOP_RESERVE_FRACTION = 0.10
# ponytail: static fraction per the approved design; per-app measured pressure
# would replace the fraction, not add a second reserve.
HEALTH_TIMEOUT_SECONDS = 120.0
CONVERT_TIMEOUT_SECONDS = 900.0
CHAT_PORT_ROLE = "chat"
DECISION_PORT_ROLE = "decision"


class PairError(RuntimeError):
    """A named pair refusal (qualification, memory, artifacts, state)."""


# ------------------------------------------------------------------ sizing

def _opt_int(value, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PairError(f"catalog extension.{field} must be a positive int or absent")
    return value


def desktop_reserve(available_bytes: int) -> int:
    return max(budget.SAFETY_RESERVE_BYTES,
               int(DESKTOP_RESERVE_FRACTION * available_bytes))


def admissible_context(chat_entry: dict, *, other_bytes: int, preference: str,
                       available_bytes: int | None = None,
                       reserved_bytes: int = 0,
                       backend_override: bool = False) -> dict:
    """Largest context satisfying every limit, by integer binary search over
    the monotone memory bound. Limits: the model's positional maximum, the
    backend's separately qualified limit (absent for every current entry —
    reported, never inferred), the selected latency policy (catalog-measured
    only), and atomic memory admission including context-dependent scratch,
    other workers, external reservations, and the desktop reserve. The
    positional limit is a MODEL fact here, never a hardware qualification."""
    if preference not in PREFERENCES:
        raise PairError(f"preference must be one of {PREFERENCES}, got {preference!r}")
    memory = chat_entry["memory"]
    model_max = chat_entry["context"]["max_tokens"] or budget.DEFAULT_CONTEXT_TOKENS
    extension = chat_entry.get("extension") or {}
    backend_limit = _opt_int(extension.get("backend_context_qualified_tokens"),
                             "backend_context_qualified_tokens")
    latency_map = extension.get("latency_context_tokens") or {}
    latency_limit = _opt_int(latency_map.get(preference),
                             f"latency_context_tokens.{preference}")
    effective_backend = backend_limit
    if effective_backend is None and not backend_override:
        effective_backend = budget.DEFAULT_CONTEXT_TOKENS
    named = [n for n in (model_max, effective_backend, latency_limit) if n is not None]
    hi_named = min(named)
    hi = hi_named
    if memory.get("kv_bytes_per_token") is None:
        hi = min(hi, budget.DEFAULT_CONTEXT_TOKENS)
    available = (available_bytes if available_bytes is not None
                 else budget.mem_available())
    reserve = desktop_reserve(available)

    def fits(tokens: int) -> bool:
        total = budget.estimate_required(memory, tokens).total
        return total + other_bytes + reserved_bytes + reserve <= available

    def largest_fitting(upper: int) -> int | None:
        lo, hi_s = catalog.MIN_CONTEXT_TOKENS, upper
        if not fits(lo):
            return None
        while lo < hi_s:
            mid = (lo + hi_s + 1) // 2
            if fits(mid):
                lo = mid
            else:
                hi_s = mid - 1
        return lo

    memory_bound = largest_fitting(catalog.MAX_CONTEXT_TOKENS)
    if memory_bound is None:
        need = budget.estimate_required(memory, catalog.MIN_CONTEXT_TOKENS).total
        raise PairError(
            "memory: even a minimum context does not fit "
            f"(need {need} bytes for chat + {other_bytes} for the rest of the "
            f"pair + {reserve} desktop reserve; {available} available)")
    context = largest_fitting(hi)
    named_bounds = [("model", model_max), ("backend", effective_backend),
                    ("latency", latency_limit)]
    binding = "memory" if memory_bound < hi_named else next(
        name for name, value in named_bounds if value == hi_named)
    return {
        "context_tokens": context,
        "limits": {"model": model_max, "backend": backend_limit,
                   "memory": memory_bound, "latency": latency_limit},
        "binding": binding,
        "backend_context_qualified": backend_limit is not None,
        "reserve_bytes": reserve,
        "available_bytes": available,
    }


def resolve_requested_context(chat_entry: dict, *, requested: int, other_bytes: int,
                              preference: str, available_bytes: int | None = None,
                              reserved_bytes: int = 0,
                              backend_override: bool = False) -> int:
    """An explicit request is honored inside every limit and refused outside
    them — never silently clamped."""
    if isinstance(requested, bool) or not isinstance(requested, int) or \
            requested < catalog.MIN_CONTEXT_TOKENS:
        raise PairError(f"requested context must be an int >= {catalog.MIN_CONTEXT_TOKENS}")
    sizing = admissible_context(chat_entry, other_bytes=other_bytes,
                                preference=preference,
                                available_bytes=available_bytes,
                                reserved_bytes=reserved_bytes,
                                backend_override=backend_override)
    if requested > sizing["context_tokens"]:
        suffix = ""
        if (sizing["binding"] == "backend"
                and not sizing["backend_context_qualified"]):
            suffix = ("; no backend-qualified context evidence for this model — "
                      f"set {PAIR_DEV_QUALIFICATION_ENV}=1 for an explicit "
                      "development smoke run")
        raise PairError(
            f"requested context {requested} exceeds the admissible "
            f"{sizing['context_tokens']} (binding limit: {sizing['binding']}; "
            f"limits: {sizing['limits']}){suffix}")
    return requested


# ------------------------------------------------------------------ selection

def _serveable(entry: dict) -> bool:
    qual = entry["qualification"]
    return (qual["generation"]["status"] == "qualified"
            and qual["http"]["status"] == "qualified")


def _catalog_decision_bytes(decision_entry: dict) -> int:
    """Rough pre-conversion requirement for SELECTION only; the admission
    itself uses the converted manifest's exact math."""
    return budget.estimate_required(decision_entry["memory"],
                                    budget.DEFAULT_CONTEXT_TOKENS).total


def local_chip_arch() -> str | None:
    """This machine's Apple chip id (e.g. 't6021') from the devicetree."""
    return serve_cli.machine_soc()


def _chat_backend(chat: dict) -> str:
    """The runtime that actually serves the chat model for this pair."""
    return (chat.get("serve") or {}).get("backend") or "mlx-lm"


def _pair_evidence(pair: dict) -> dict | None:
    return (pair.get("extension") or {}).get("selection_evidence") or None


# Design-fixed interactive bound: p95 first visible answer within 2 seconds
# (128-token prompt, 256-token response). A bound the measurement must meet,
# never a measurement itself.
FIRST_VISIBLE_P95_MS = 2000.0


def _local_runtime_identity() -> dict | None:
    """The running mlx backend's provenance identity — the same RECORD-hash
    core the voice qualification uses (mlx.core extension + the actually
    loaded libmlx.so). None when it cannot be established and verified."""
    try:
        from . import synthesis
        provenance = synthesis._backend_provenance()
    except Exception:
        return None
    if provenance.get("verified") != "match":
        return None
    identity = provenance.get("identity") or {}
    if not identity.get("extension_sha256") or not identity.get("libmlx_sha256"):
        return None
    return {"mlx_version": provenance.get("mx_version"),
            "extension_sha256": identity["extension_sha256"],
            "libmlx_sha256": identity["libmlx_sha256"]}


def _runtime_mismatch(pid: str, evidence_runtime: dict,
                      local_runtime: dict) -> str | None:
    """Named refusal when the evidence's runtime identity differs from the
    running backend in any bound component."""
    bound = (evidence_runtime["mlx_version"], evidence_runtime["extension_sha256"],
             evidence_runtime["libmlx_sha256"])
    local = (local_runtime.get("mlx_version"), local_runtime.get("extension_sha256"),
             local_runtime.get("libmlx_sha256"))
    if bound == local:
        return None
    return (f"{pid}: evidence is bound to mlx {bound[0]} (extension "
            f"{bound[1][:12]}, libmlx {bound[2][:12]}); the local runtime "
            f"is mlx {local[0]} (extension {(local[1] or '')[:12]}, libmlx "
            f"{(local[2] or '')[:12]}) — different runtime")


def _evidence_refusal(pair: dict, chat: dict, decision: dict,
                      chip_arch: str | None,
                      local_runtime: dict) -> str | None:
    """Named reason a pair may not be recommended automatically, or None when
    its release gate and evidence are valid and scoped to this machine."""
    pid = pair["id"]
    status = pair["qualification"]["status"]
    if status != "qualified":
        return (f"{pid}: pair-level qualification is {status!r} — the pair "
                "release gate has not passed")
    evidence = _pair_evidence(pair)
    if evidence is None:
        return f"{pid}: qualified but carries no measured selection evidence"
    if chip_arch is not None and chip_arch not in evidence["arch"]:
        return (f"{pid}: evidence covers chips {evidence['arch']}, "
                f"local chip is {chip_arch}")
    backend = _chat_backend(chat)
    if evidence["backend"] != backend:
        return (f"{pid}: evidence is for the {evidence['backend']!r} runtime, "
                f"but this pair serves chat via {backend!r}")
    for role, entry, field in (("chat", chat, "chat_revision"),
                               ("decision", decision, "decision_revision")):
        bound = evidence[field]
        if bound != entry["revision"]:
            return (f"{pid}: {role} evidence is bound to revision {bound}, "
                    f"catalog pins {entry['revision']} for "
                    f"{entry['id']} — stale")
    for role, entry in (("chat", chat), ("decision", decision)):
        supported = (entry.get("capability") or {}).get("arch")
        if supported and chip_arch is not None and chip_arch not in supported:
            return (f"{pid}: {role} model {entry['id']} supports chips "
                    f"{supported}, local chip is {chip_arch}")
    runtime = _runtime_mismatch(pid, evidence["runtime"], local_runtime)
    if runtime is not None:
        return runtime
    latency = evidence["latency"]
    if latency["first_visible_p95_ms"] > FIRST_VISIBLE_P95_MS:
        return (f"{pid}: measured first-visible p95 "
                f"{latency['first_visible_p95_ms']} ms exceeds the "
                f"{FIRST_VISIBLE_P95_MS:.0f} ms interactive bound")
    if latency["decode_tokens_per_sec"] < latency["target_tokens_per_sec"]:
        return (f"{pid}: measured decode {latency['decode_tokens_per_sec']} "
                f"tok/s misses the qualified interactive target "
                f"{latency['target_tokens_per_sec']} tok/s")
    return None


def select_pair(pair_records, model_entries, *, preference: str,
                home: Path | None = None,
                available_bytes: int | None = None,
                chip_arch: str | None = None,
                runtime_identity: dict | None = None) -> dict:
    """Evidence-based pair selection: rank by measured task quality and
    measured decode latency from the pair's selection evidence, only when
    that evidence is scoped to this chip, this exact mlx runtime (extension +
    loaded libmlx.so RECORD hashes), and the pinned revisions, and only past
    the pair release gate, the 2000 ms first-visible bound, and a byte-based
    memory admission. Curated priority is the final tie-break, never the
    ranking criterion; weight size and parameter count are never quality.
    Without valid scoped evidence the refusal is named — development
    qualification runs go through explicit setup, not here."""
    if preference not in PREFERENCES:
        raise PairError(f"preference must be one of {PREFERENCES}, got {preference!r}")
    if chip_arch is None:
        chip_arch = local_chip_arch()
    if chip_arch is None:
        raise PairError("cannot determine this machine's chip architecture; "
                        "automatic pair selection is scoped to the detected chip")
    if runtime_identity is None:
        runtime_identity = _local_runtime_identity()
    if runtime_identity is None:
        raise PairError("cannot verify the local mlx runtime provenance; "
                        "selection evidence is bound to an exact runtime "
                        "(mlx.core extension + libmlx.so RECORD hashes)")
    models = {m["id"]: m for m in model_entries}
    refusals: list[str] = []
    candidates = []
    for pair in sorted(pair_records, key=lambda p: p["priority"]):
        chat = models.get(pair["chat_model"])
        decision = models.get(pair["decision_model"])
        if chat is None or decision is None:
            refusals.append(f"{pair['id']}: references a model missing from "
                            "the catalog")
            continue
        gate = True
        for role, entry in (("chat", chat), ("decision", decision)):
            if not _serveable(entry):
                gate = False
                refusals.append(
                    f"{pair['id']}: {role} model {entry['id']} fails the model "
                    f"qualification gate (generation="
                    f"{entry['qualification']['generation']['status']}, "
                    f"http={entry['qualification']['http']['status']})")
        if not gate:
            continue
        reason = _evidence_refusal(pair, chat, decision, chip_arch,
                                   runtime_identity)
        if reason is not None:
            refusals.append(reason)
            continue
        decision_bytes = _catalog_decision_bytes(decision)
        try:
            sizing = admissible_context(chat, other_bytes=decision_bytes,
                                        preference=preference,
                                        available_bytes=available_bytes)
        except PairError as exc:
            refusals.append(f"{pair['id']}: memory: {exc}")
            continue
        candidates.append((pair, _pair_evidence(pair), sizing))
    if not candidates:
        detail = "; ".join(refusals) if refusals else "the catalog lists no pairs"
        raise PairError(f"no evidence-qualified pair for the {preference!r} "
                        f"preference ({detail})")
    if preference == "fast":
        def rank(candidate):
            _pair, evidence, _sizing = candidate
            return (-evidence["latency"]["decode_tokens_per_sec"],
                    -evidence["quality"]["score"], _pair["priority"])
    elif preference == "long-context":
        def rank(candidate):
            _pair, evidence, sizing = candidate
            return (-sizing["context_tokens"],
                    -evidence["quality"]["score"], _pair["priority"])
    else:
        def rank(candidate):
            _pair, evidence, _sizing = candidate
            return (-evidence["quality"]["score"],
                    -evidence["latency"]["decode_tokens_per_sec"],
                    _pair["priority"])
    return min(candidates, key=rank)[0]


# ------------------------------------------------------------------ locks

def pair_lock_path(home: Path, pair_id: str) -> Path:
    return Path(home) / "assistant" / "pair-locks" / f"{pair_id}.json"


def write_pair_lock(home: Path, pair_id: str, lock: dict) -> Path:
    path = pair_lock_path(home, pair_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(lock, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def load_pair_lock(home: Path, pair_id: str) -> dict | None:
    path = pair_lock_path(home, pair_id)
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise PairError(f"pair lock {path} is unreadable: {exc}") from exc
    if not isinstance(lock, dict) or lock.get("version") != 1 \
            or lock.get("pair_id") != pair_id \
            or not isinstance(lock.get("models"), dict):
        raise PairError(f"pair lock {path} is not a valid v1 lock for {pair_id!r}")
    return lock


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_files(path: Path) -> list[dict]:
    """Every regular file under an artifact dir: relative posix name, size,
    sha256. The offline transfer's hashing helper."""
    path = Path(path)
    if not path.is_dir():
        raise PairError(f"artifact path {path} does not exist")
    out = []
    for item in sorted(path.rglob("*")):
        if item.is_file() and not item.is_symlink():
            out.append({"name": item.relative_to(path).as_posix(),
                        "bytes": item.stat().st_size,
                        "sha256": _sha256_file(item)})
    return out


def artifact_manifests(lock: dict) -> list[dict]:
    """Per-model export manifests from a pair lock: identity, license, path,
    and the full hashed file list."""
    out = []
    licenses = lock.get("licenses") or {}
    for role in ("chat", "decision"):
        model = (lock.get("models") or {}).get(role)
        if not model:
            raise PairError(f"pair lock is missing the {role} model")
        path = Path(model["path"])
        out.append({
            "role": role,
            "model_id": model["id"],
            "repo": model["repo"],
            "revision": model["revision"],
            "path": str(path),
            "license": licenses.get(model["id"]),
            "files": artifact_files(path),
        })
    return out


def _boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        return None


# ------------------------------------------------------------------ workers

def _chat_worker_argv(spec: managed.WorkerSpec) -> list[str]:
    return serve_cli.server_argv("mlx-lm", None, spec.model_dir,
                                 "127.0.0.1", spec.port, spec.context_tokens)


def _decision_worker_argv(spec: managed.WorkerSpec) -> list[str]:
    return serve_cli.server_argv("module", "mlx_omarchy_laya.server",
                                 spec.model_dir, "127.0.0.1", spec.port,
                                 spec.context_tokens,
                                 max_questions=spec.max_questions)


def _encoder_cfg(encoder_dir: Path):
    """The three fields the Laya workspace math needs, read with stdlib only
    (mlx_omarchy_laya.model pulls mlx.core)."""
    cfg = json.loads((Path(encoder_dir) / "config.json").read_text(encoding="utf-8"))
    return type("EncoderCfg", (), {k: int(cfg[k]) for k in
                                   ("hidden_size", "num_attention_heads",
                                    "intermediate_size")})()


def _read_manifest(model_dir: Path) -> dict:
    path = Path(model_dir) / "manifest.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def decision_requirement_bytes(model_dir: Path, max_questions: int) -> int:
    """Exact dtype-aware requirement of the CONVERTED decision artifact at
    the pair's question batch bound — the same math the worker relabels
    against, so admission and runtime never drift."""
    from mlx_omarchy_laya.server import total_estimate_bytes

    manifest = _read_manifest(model_dir)
    if not manifest:
        raise PairError(f"{model_dir}: converted decision artifact has no manifest")
    return int(total_estimate_bytes(manifest, _encoder_cfg(Path(model_dir) / "encoder"),
                                    "float16", max_questions))


DOWNLOAD_TIMEOUT_SECONDS = 3600.0


def _download_child_code(repo: str, revision: str, cache_dir: str,
                         patterns) -> str:
    patterns_expr = "" if patterns is None \
        else ", allow_patterns=%r" % (list(patterns),)
    return (
        "import os, threading\n"
        "from huggingface_hub import snapshot_download\n"
        "def _watch():\n"
        "    from mlx_omarchy_serve import budget\n"
        "    budget.watch_parent_fd(int(os.environ[budget.LIFETIME_FD_ENV]),\n"
        "                          lambda: os._exit(3))\n"
        "threading.Thread(target=_watch, daemon=True).start()\n"
        "print(snapshot_download(repo_id=%r, revision=%r, cache_dir=%r%s))\n"
    ) % (repo, revision, cache_dir, patterns_expr)


# ------------------------------------------------------------------ manager

class PairManager:
    """Owns one prepared pair and its managed workers for a home directory."""

    def __init__(self, home: Path):
        self.home = Path(home)
        self.catalog = catalog.load_catalog(self.home)
        self.catalog_models = {m["id"]: m for m in self.catalog["models"]}
        self._op_lock = threading.RLock()
        self._state = "idle"
        self._error: str | None = None
        self._prepared: dict | None = None
        self._last_plan: dict | None = None
        self._pair_id: str | None = None
        self._children: dict[str, managed.ManagedChild] = {}
        self._records: dict[str, dict] = {}
        self._lifetime_write_fd: int | None = None
        self._lifetime_read_fd: int | None = None
        self._context_tokens: int | None = None
        self._available_bytes_override: int | None = None  # test seam
        self._cancel = threading.Event()


    def _available_bytes(self) -> int:
        if self._available_bytes_override is not None:
            return self._available_bytes_override
        return budget.mem_available()

    def _dev_smoke(self) -> bool:
        return os.environ.get(PAIR_DEV_QUALIFICATION_ENV) == "1"

    def _admissible(self, chat, *, other_bytes, preference, available_bytes=None,
                    reserved_bytes=0):
        return admissible_context(chat, other_bytes=other_bytes,
                                 preference=preference,
                                 available_bytes=available_bytes,
                                 reserved_bytes=reserved_bytes,
                                 backend_override=self._dev_smoke())

    def _resolve_requested(self, chat, *, requested, other_bytes, preference,
                           available_bytes=None, reserved_bytes=0):
        return resolve_requested_context(chat, requested=requested,
                                         other_bytes=other_bytes,
                                         preference=preference,
                                 available_bytes=available_bytes,
                                 reserved_bytes=reserved_bytes,
                                 backend_override=self._dev_smoke())

    def _pairs_summary(self) -> list[dict]:
        return [{"id": p["id"], "label": p["label"],
                 "chat_model": p["chat_model"],
                 "decision_model": p["decision_model"],
                 "priority": p["priority"],
                 "qualification": p["qualification"]}
                for p in (self.catalog.get("pairs") or [])]

    def _pair(self, pair_id: str) -> dict:
        for pair in (self.catalog.get("pairs") or []):
            if pair["id"] == pair_id:
                return pair
        raise PairError(f"unknown pair {pair_id!r}; available: "
                        f"{[p['id'] for p in (self.catalog.get('pairs') or [])]}")

    def _child_env(self) -> dict:
        env = dict(os.environ)
        serve_root = str(Path(catalog.__file__).resolve().parent.parent)
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = serve_root if not existing else serve_root + os.pathsep + existing
        env["MLX_OMARCHY_HOME"] = str(self.home)
        return env

    def _ensure_lifetime_pipe(self) -> int:
        if self._lifetime_read_fd is None or self._lifetime_write_fd is None:
            self._close_lifetime()
            read_fd, write_fd = os.pipe()
            os.set_inheritable(read_fd, True)
            self._lifetime_read_fd = read_fd
            self._lifetime_write_fd = write_fd
        return self._lifetime_read_fd

    def _download_snapshot_supervised(self, resolved, patterns, progress,
                                      model_id: str) -> Path:
        """Pinned snapshot fetch in a supervised subprocess: cancellation and
        manager death kill it (bounded, no orphan download); hub resumes
        partial downloads. Returns the verified-complete path."""
        progress("download", role="chat" if patterns is not None else "download",
                 model_id=model_id)
        lifetime_r = self._ensure_lifetime_pipe()
        log_path = self.home / "assistant" / "logs" / f"download-{model_id}.log"
        argv = [sys.executable, "-c", _download_child_code(
            resolved.repo, resolved.revision, str(serve_cli.hf_cache_dir()),
            patterns)]
        proc = managed.spawn_supervised(argv, lifetime_read_fd=lifetime_r,
                                        env=self._child_env(),
                                        log_path=log_path)
        deadline = time.monotonic() + DOWNLOAD_TIMEOUT_SECONDS
        while proc.poll() is None:
            if self._cancel.is_set():
                managed.terminate(proc, timeout=15)
                raise PairError(f"cancelled during {model_id} download")
            if time.monotonic() > deadline:
                managed.terminate(proc, timeout=15)
                raise PairError(
                    f"{model_id}: download exceeded {DOWNLOAD_TIMEOUT_SECONDS:.0f}s "
                    "and was stopped")
            time.sleep(0.2)
        if proc.returncode != 0:
            raise PairError(f"{model_id}: download failed (exit "
                            f"{proc.returncode}); log: {log_path}")
        try:
            lines = [ln for ln in log_path.read_text(encoding="utf-8")
                     .strip().splitlines() if ln.strip()]
            path = Path(lines[-1])
        except (OSError, IndexError) as exc:
            raise PairError(
                f"{model_id}: download finished but reported no snapshot path "
                f"({exc}); log: {log_path}") from exc
        return path

    def _voice_requirement_bytes(self) -> int:
        """Conservative admission estimate for voice mode: the synthesis pack's
        published runtime_estimate_bytes plus the recognition assets' when
        voice is enabled for both directions. Estimates, never qualification:
        readiness is established later by the parent's prepare calls."""
        try:
            from .synthesis import Synthesis
        except ImportError:
            raise PairError(
                "voice synthesis is not available (assistant synthesis module "
                "missing); retry with voice=false") from None
        synthesis_status = Synthesis(self.home).status() or {}
        synthesis_bytes = ((synthesis_status.get("memory") or {})
                           .get("runtime_estimate_bytes"))
        if isinstance(synthesis_bytes, bool) or not isinstance(synthesis_bytes, int) \
                or synthesis_bytes <= 0:
            raise PairError(
                "voice synthesis status exposes no positive "
                "memory.runtime_estimate_bytes; refusing to admit voice with "
                "unaccounted (or zero) memory")
        total = synthesis_bytes
        try:
            from .recognition import Recognition
        except ImportError:
            raise PairError(
                "voice recognition is not available; refusing to admit voice "
                "mode without accounting recognition assets") from None
        recognition_status = Recognition(self.home).status() or {}
        recognition_bytes = ((recognition_status.get("memory") or {})
                             .get("runtime_estimate_bytes"))
        if isinstance(recognition_bytes, bool) or not isinstance(recognition_bytes, int) \
                or recognition_bytes <= 0:
            raise PairError(
                "voice recognition status exposes no positive "
                "memory.runtime_estimate_bytes; refusing to admit voice with "
                "unaccounted (or zero) memory")
        return total + recognition_bytes

    def _ready_offline(self, pair: dict) -> bool:
        """True only when the pair is qualified, names a receipt, and a
        cold offline start was verified on this boot. Running well is never
        promotion."""
        qual = pair["qualification"]
        receipt = qual.get("receipt")
        if qual["status"] != "qualified" or not isinstance(receipt, str) or not receipt.strip():
            return False
        lock = load_pair_lock(self.home, pair["id"])
        if lock is None or not lock.get("started_offline"):
            return False
        if lock.get("boot_id") != _boot_id():
            return False
        chat = lock["models"].get("chat") or {}
        decision = lock["models"].get("decision") or {}
        chat_entry = self.catalog_models.get(pair["chat_model"]) or {}
        decision_entry = self.catalog_models.get(pair["decision_model"]) or {}
        return (chat.get("revision") == chat_entry.get("revision")
                and decision.get("revision") == decision_entry.get("revision"))


    def status(self) -> dict:
        if self._children:
            managed.reap_pair_records(self._pair_id, self.home,
                                      children=tuple(self._children.values()))
        dead = sorted(c.role for c in tuple(self._children.values())
                      if c.popen.poll() is not None)
        if dead and self._state == "ready":
            self._state = "error"
            self._error = f"worker(s) exited: {', '.join(dead)}"
        pair = (self._pair(self._pair_id) if self._pair_id else None)
        children = [{"role": c.role, "pid": c.popen.pid,
                     "alive": c.popen.poll() is None}
                    for c in tuple(self._children.values())]
        records = [{"name": name, "bytes": r["bytes"], "state": r["state"],
                    "claimed": r["claimed"]}
                   for name, r in budget.pair_records(self._pair_id or "",
                                                      self.home).items()] \
            if self._pair_id else []
        prepared = self._prepared or {}
        active = bool(pair) and self._state in ("prepared", "starting", "ready")
        return {
            "state": self._state,
            "error": self._error,
            "pair_id": self._pair_id if active else None,
            "active_pair": self._pair_id if active else None,
            "pairs": self._pairs_summary(),
            "recommendation": self._recommendation(),
            "chat_url": prepared.get("chat_url"),
            "decision_url": prepared.get("decision_url"),
            "chat_model": prepared.get("chat_id"),
            "decision_model": prepared.get("decision_id"),
            "context_tokens": self._context_tokens,
            "context": prepared.get("sizing")
                       or (self._last_plan or {}).get("context"),
            "downloads": (self._last_plan or {}).get("downloads") or [],
            "voice": {"requested": bool((self._last_plan or {}).get("voice")),
                      "bytes": (self._last_plan or {}).get("voice_bytes") or 0},
            "model_paths": {"chat": str(prepared["chat_path"]),
                            "decision": str(prepared["decision_path"])
                            if prepared.get("decision_path") else None}
            if prepared.get("chat_path") else {},
            "ready_offline": bool(pair) and self._ready_offline(pair),
            "qualification": {
                "pair": pair["qualification"] if pair else None,
                "chat": prepared.get("chat_qualification"),
                "decision": prepared.get("decision_qualification"),
            },
            "reservations": records,
            "children": children,
        }

    def _recommendation(self) -> dict | None:
        """The pair this manager would pick right now, from live memory —
        continuous in bytes, never a fixed table."""
        try:
            pick = select_pair(self.catalog.get("pairs") or [],
                               self.catalog["models"],
                               preference="balanced", home=self.home,
                               available_bytes=self._available_bytes())
        except PairError as exc:
            return {"error": str(exc)}
        return {"pair_id": pick["id"], "label": pick["label"],
                "chat_model": pick["chat_model"],
                "decision_model": pick["decision_model"],
                "priority": pick["priority"],
                "qualification": pick["qualification"]}

    def setup(self, pair_id: str, approve_download: bool = False,
              preference: str = "balanced", context_tokens: int | None = None,
              voice: bool = False, progress=None,
              available_bytes: int | None = None) -> dict:
        progress = progress or (lambda stage, **detail: None)
        if preference not in PREFERENCES:
            raise PairError(f"preference must be one of {PREFERENCES}")
        with self._op_lock:
            if self._state in ("starting", "ready"):
                raise PairError("a pair is already active; stop it first")
            self._cancel.clear()
            self._error = None
            self.catalog = catalog.load_catalog(self.home)
            self.catalog_models = {m["id"]: m for m in self.catalog["models"]}
            self._state = "preparing"
            try:
                return self._setup_locked(pair_id, approve_download, preference,
                                          context_tokens, voice, progress,
                                          available_bytes)
            except PairError:
                if self._state == "preparing":
                    self._state = "prepared" if self._prepared else "idle"
                raise

    def _setup_locked(self, pair_id: str, approve_download: bool,
                      preference: str, context_tokens: int | None,
                      voice: bool, progress, available_bytes: int | None) -> dict:
            pair = self._pair(pair_id)
            chat = self.catalog_models[pair["chat_model"]]
            decision = self.catalog_models[pair["decision_model"]]
            for role, entry in (("chat", chat), ("decision", decision)):
                if not _serveable(entry):
                    message = (f"{pair_id}: qualification gate failed: {role} model "
                               f"{entry['id']} is generation="
                               f"{entry['qualification']['generation']['status']}/"
                               f"http={entry['qualification']['http']['status']}; "
                               f"set {PAIR_DEV_QUALIFICATION_ENV}=1 for an explicit "
                               "developer qualification run")
                    if os.environ.get(PAIR_DEV_QUALIFICATION_ENV) == "1":
                        progress("qualification", role=role, warning=message)
                        break
                    raise PairError(message)
            voice_bytes = self._voice_requirement_bytes() if voice else 0
            decision_module = (decision.get("serve") or {}).get("module")
            if decision_module != "mlx_omarchy_laya.server":
                raise PairError(
                    f"{pair_id}: decision model {decision['id']} has no supported "
                    f"module route ({decision_module!r})")

            decision_conv = self.home / "models" / decision["id"]
            decision_bytes = (decision_requirement_bytes(decision_conv, pair["max_questions"])
                              if self._converted_usable(decision_conv, decision)
                              else _catalog_decision_bytes(decision))
            if context_tokens is not None:
                chosen = self._resolve_requested(
                    chat, requested=context_tokens, other_bytes=decision_bytes,
                    preference=preference, available_bytes=available_bytes)
                sizing = self._admissible(chat, other_bytes=decision_bytes,
                                          preference=preference,
                                          available_bytes=available_bytes)
                sizing = dict(sizing, context_tokens=chosen)
            else:
                sizing = self._admissible(chat, other_bytes=decision_bytes,
                                          preference=preference,
                                          available_bytes=available_bytes)
            progress("plan", pair_id=pair_id, context_tokens=sizing["context_tokens"],
                     binding=sizing["binding"])
            result = {
                "pair_id": pair_id,
                "state": "planned",
                "context_tokens": sizing["context_tokens"],
                "context": sizing,
                "preference": preference,
                "max_questions": pair["max_questions"],
                "voice": bool(voice),
                "voice_bytes": voice_bytes,
                "downloads": [],
                "model_paths": {},
            }
            chat_patterns = serve_cli.download_patterns_for(chat)
            decision_patterns = serve_cli.download_patterns_for(decision)
            chat_resolved = serve_cli.Resolved(chat["id"], chat, None,
                                               chat["repo"], chat["revision"])
            decision_resolved = serve_cli.Resolved(decision["id"], decision, None,
                                                   decision["repo"], decision["revision"])
            converted = self._converted_usable(decision_conv, decision)
            chat_path = serve_cli.probe_snapshot(chat_resolved, chat_patterns)
            raw_decision = None if converted else \
                serve_cli.probe_snapshot(decision_resolved, decision_patterns, hf_layout=False)
            conversion_needed = not converted
            chat_fetch = chat_path is None
            decision_fetch = raw_decision is None and not converted
            result["downloads"] = [
                {"role": "chat", "model_id": chat["id"], "needed": chat_fetch,
                 "bytes": chat["availability"]["size_bytes"]},
                {"role": "decision", "model_id": decision["id"],
                 "needed": decision_fetch,
                 "conversion_needed": conversion_needed,
                 "bytes": decision["availability"]["size_bytes"]},
            ]
            fetch_needed = chat_fetch or decision_fetch
            if fetch_needed and not approve_download:
                self._last_plan = result
                self._state = "planned"
                result["state"] = "planned"
                return result
            self._last_plan = result
            self._check_cancel("setup")
            chat_path = self._prepare_chat(chat, chat_resolved, chat_patterns,
                                           progress, allow_fetch=approve_download)
            self._check_cancel("setup")
            decision_path = self._prepare_decision(decision, decision_resolved,
                                                   decision_patterns, decision_conv,
                                                   progress,
                                                   allow_fetch=approve_download)
            decision_bytes = decision_requirement_bytes(decision_path,
                                                        pair["max_questions"])
            if context_tokens is not None:
                chosen = self._resolve_requested(
                    chat, requested=context_tokens, other_bytes=decision_bytes,
                    preference=preference, available_bytes=available_bytes)
                sizing = dict(sizing, context_tokens=chosen)
            else:
                sizing = self._admissible(chat, other_bytes=decision_bytes,
                                          preference=preference,
                                          available_bytes=available_bytes)
            self._prepared = {
                "pair_id": pair_id,
                "chat_id": chat["id"],
                "decision_id": decision["id"],
                "chat_path": chat_path,
                "decision_path": decision_path,
                "context_tokens": sizing["context_tokens"],
                "sizing": sizing,
                "preference": preference,
                "max_questions": pair["max_questions"],
                "voice_bytes": voice_bytes,
                "chat_qualification": chat["qualification"],
                "decision_qualification": decision["qualification"],
            }
            self._state = "prepared"
            result["state"] = "prepared"
            result["context_tokens"] = sizing["context_tokens"]
            result["context"] = sizing
            result["model_paths"] = {"chat": chat_path, "decision": decision_path}
            return result

    def _check_cancel(self, stage: str) -> None:
        if self._cancel.is_set():
            raise PairError(f"cancelled during {stage}")

    def _abort_predicate(self):
        def abort():
            if self._cancel.is_set():
                return True
            return any(c.popen.poll() is not None
                       for c in tuple(self._children.values()))
        return abort

    def _converted_usable(self, conv_dir: Path, decision_entry: dict) -> bool:
        from mlx_omarchy_laya.convert import checkpoint_state

        state, _details = checkpoint_state(conv_dir)
        if state != "converted":
            return False
        manifest = _read_manifest(conv_dir)
        # identity and provenance both bind: a converted artifact stamped
        # with another catalog id (e.g. a pre-rename conversion) is a stale
        # identity, reconverted rather than adopted
        return (manifest.get("catalog_id") == decision_entry["id"]
                and manifest.get("source_revision") == decision_entry["revision"])

    def _prepare_chat(self, chat: dict, resolved, patterns, progress,
                      *, allow_fetch: bool) -> Path:
        path = serve_cli.probe_snapshot(resolved, patterns)
        if path is None and not allow_fetch:
            raise PairError(
                f"{chat['id']}: no complete local snapshot and downloads were "
                "not approved")
        if path is None:
            if catalog.env_offline():
                raise PairError(
                    f"{chat['id']}: no complete local snapshot and downloads are "
                    "disabled (offline mode); run setup once online or pre-seed "
                    "the Hugging Face cache"
                    + serve_cli.offline_snapshot_detail(resolved, patterns))
            need = serve_cli.disk_need_bytes(resolved)
            if need is not None:
                ok, free, where = budget.disk_check(need, serve_cli.hf_cache_dir())
                if not ok:
                    raise PairError(
                        f"{chat['id']}: disk need {need} bytes exceeds {free} free "
                        f"at {where}")
            progress("download", role="chat", model_id=chat["id"])
            path = self._download_snapshot_supervised(resolved, patterns,
                                                      progress, chat["id"])
            if not serve_cli.snapshot_complete(path, patterns):
                raise PairError(
                    f"{chat['id']}: downloaded snapshot is incomplete; refusing")
        serve_cli.check_custom_code(path)
        return path

    def _prepare_decision(self, decision: dict, resolved, patterns, conv_dir: Path,
                          progress, *, allow_fetch: bool) -> Path:
        if self._converted_usable(conv_dir, decision):
            progress("verify", role="decision", reused=True)
            return conv_dir
        snap = serve_cli.probe_snapshot(resolved, patterns, hf_layout=False)
        if snap is None:
            if not allow_fetch:
                raise PairError(
                    f"{decision['id']}: no usable converted artifact, no local "
                    "snapshot to convert, and downloads were not approved")
            if catalog.env_offline():
                raise PairError(
                    f"{decision['id']}: no usable converted artifact and no raw "
                    "snapshot to convert (offline mode); conversion of the pinned "
                    "source snapshot is required before serving"
                    + serve_cli.offline_snapshot_detail(resolved, patterns, hf_layout=False))
            need = serve_cli.disk_need_bytes(resolved)
            if need is not None:
                ok, free, where = budget.disk_check(need, serve_cli.hf_cache_dir())
                if not ok:
                    raise PairError(
                        f"{decision['id']}: disk need {need} bytes exceeds {free} "
                        f"free at {where}")
            progress("download", role="decision", model_id=decision["id"])
            snap = self._download_snapshot_supervised(resolved, patterns,
                                                      progress, decision["id"])
        progress("convert", role="decision", source=str(snap))
        env = self._child_env()
        argv = [sys.executable, "-m", "mlx_omarchy_laya.convert",
                "--from-local", str(snap), "--out", str(conv_dir),
                "--variant", "root", "--revision", decision["revision"], "--yes"]
        # supervised like downloads: manager death or cancel kills the
        # converter (bounded, no orphan), output lands in a readable log
        lifetime_r = self._ensure_lifetime_pipe()
        log_path = (self.home / "assistant" / "logs"
                    / f"convert-{decision['id']}.log")
        proc = managed.spawn_supervised(argv, lifetime_read_fd=lifetime_r,
                                        env=env, log_path=log_path)
        deadline = time.monotonic() + CONVERT_TIMEOUT_SECONDS
        while proc.poll() is None:
            if self._cancel.is_set():
                managed.terminate(proc)
                raise PairError(f"cancelled during {decision['id']} conversion")
            if time.monotonic() > deadline:
                managed.terminate(proc)
                raise PairError(
                    f"{decision['id']}: conversion exceeded "
                    f"{CONVERT_TIMEOUT_SECONDS:.0f}s and was stopped")
            time.sleep(0.2)
        if proc.returncode != 0:
            try:
                tail = " | ".join(log_path.read_text(encoding="utf-8",
                                                     errors="replace")
                                  .strip().splitlines()[-5:])
            except OSError:
                tail = f"log: {log_path}"
            raise PairError(
                f"{decision['id']}: conversion failed (exit {proc.returncode}): "
                f"{tail}")
        if not self._converted_usable(conv_dir, decision):
            raise PairError(
                f"{decision['id']}: conversion output is not a usable converted "
                "artifact")
        return conv_dir


    def adopt_saved(self, pair_id: str | None = None) -> dict:
        """Adopt a saved pair lock after an app restart. Verifies the locked
        artifacts exist and their revisions still match the catalog pins, then
        prepares from them. Cached, verified artifacts need no approval."""
        with self._op_lock:
            if self._state in ("starting", "ready"):
                raise PairError("a pair is already active; stop it first")
            wanted = pair_id or self._newest_lock_pair_id()
            if wanted is None:
                raise PairError("no saved pair lock to adopt")
            lock = load_pair_lock(self.home, wanted)
            if lock is None:
                raise PairError(f"no saved pair lock for {wanted!r}")
            pair = self._pair(wanted)
            chat = self.catalog_models[pair["chat_model"]]
            decision = self.catalog_models[pair["decision_model"]]
            chat_model = lock["models"].get("chat") or {}
            decision_model = lock["models"].get("decision") or {}
            for role, locked, entry in (("chat", chat_model, chat),
                                        ("decision", decision_model, decision)):
                if locked.get("revision") != entry["revision"]:
                    raise PairError(
                        f"{wanted}: saved lock {role} revision "
                        f"{locked.get('revision')!r} no longer matches the "
                        f"catalog pin {entry['revision']!r}; re-run setup")
            chat_path = Path(chat_model["path"])
            decision_path = Path(decision_model["path"])
            patterns = serve_cli.download_patterns_for(chat)
            resolved = serve_cli.Resolved(chat["id"], chat, None,
                                          chat["repo"], chat["revision"])
            if not serve_cli.snapshot_complete(chat_path, patterns):
                raise PairError(f"{wanted}: locked chat artifact {chat_path} is "
                                "not a complete servable snapshot")
            if not self._converted_usable(decision_path, decision):
                raise PairError(f"{wanted}: locked decision artifact "
                                f"{decision_path} is not a usable converted "
                                "artifact")
            decision_bytes = decision_requirement_bytes(decision_path,
                                                        pair["max_questions"])
            sizing = self._admissible(chat, other_bytes=decision_bytes,
                                      preference="balanced",
                                        available_bytes=self._available_bytes())
            context = min(int(lock.get("context_tokens")
                              or sizing["context_tokens"]),
                          sizing["context_tokens"])
            self._prepared = {
                "pair_id": wanted,
                "chat_id": chat["id"],
                "decision_id": decision["id"],
                "chat_path": chat_path,
                "decision_path": decision_path,
                "context_tokens": context,
                "sizing": dict(sizing, context_tokens=context),
                "preference": "balanced",
                "max_questions": pair["max_questions"],
                "voice_bytes": 0,
                "chat_qualification": chat["qualification"],
                "decision_qualification": decision["qualification"],
            }
            self._pair_id = wanted
            self._state = "prepared"
            return self.status()

    def _newest_lock_pair_id(self) -> str | None:
        directory = pair_lock_path(self.home, "unused").parent
        if not directory.is_dir():
            return None
        candidates = sorted(directory.glob("*.json"), key=lambda path: path.stat().st_mtime)
        return candidates[-1].stem if candidates else None

    def start(self, pair_id: str | None = None) -> dict:
        with self._op_lock:
            if self._state == "ready" and self._prepared is not None \
                    and (pair_id is None or pair_id == self._prepared["pair_id"]):
                dead = sorted(c.role for c in self._children.values()
                              if c.popen.poll() is not None)
                if not dead and set(self._children) == {"chat", "decision"}:
                    return self.status()
                self.stop()
            prepared = self._prepared
            if prepared is None:
                try:
                    self.adopt_saved(pair_id)
                except PairError:
                    pass
                prepared = self._prepared
            if prepared is None or (pair_id is not None
                                    and pair_id != prepared["pair_id"]):
                raise PairError(
                    "no prepared pair; run setup(pair_id, approve_download=True) "
                    "or ensure a saved pair lock exists before start")
            self._cancel.clear()
            pair = self._pair(prepared["pair_id"])
            # settle anything retained from a previously failed start: every
            # child still owned here is terminated first so the reaper below
            # can clear its record and the new admission starts clean;
            # a child whose termination cannot be verified stays owned
            survivors = {}
            for child in self._children.values():
                if not managed.terminate(child.popen):
                    survivors[child.role] = child
            managed.reap_pair_records(pair["id"], self.home,
                                      children=self._children.values())
            self._children = survivors
            chat = self.catalog_models[prepared["chat_id"]]
            decision = self.catalog_models[prepared["decision_id"]]
            decision_bytes = decision_requirement_bytes(prepared["decision_path"],
                                                        prepared["max_questions"])
            try:
                sizing = self._admissible(chat, other_bytes=decision_bytes,
                                            preference=prepared["preference"],
                                            available_bytes=self._available_bytes())
            except PairError as exc:
                self._state = "error"
                self._error = str(exc)
                raise
            context = prepared["context_tokens"]
            if context > sizing["context_tokens"]:
                raise PairError(
                    f"memory changed since setup: the prepared context {context} "
                    f"no longer fits (admissible {sizing['context_tokens']}, "
                    f"binding {sizing['binding']}); re-run setup")
            chat_bytes = budget.estimate_required(chat["memory"], context).total
            items = [(chat["id"], chat_bytes, f"pair {pair['id']} chat"),
                     (decision["id"], decision_bytes, f"pair {pair['id']} decisions")]
            if prepared["voice_bytes"]:
                items.append((f"voice-{pair['id']}", prepared["voice_bytes"],
                              f"pair {pair['id']} voice admission"))
            self._state = "starting"
            try:
                records = budget.admit_and_reserve_batch(
                    items, pair_id=pair["id"], home=self.home,
                    available_bytes=self._available_bytes())
            except budget.BudgetError as exc:
                self._state = "error"
                self._error = str(exc)
                raise PairError(f"pair admission refused: {exc}") from exc
            self._records = {r["name"]: r for r in records}
            self._pair_id = pair["id"]
            self._context_tokens = context
            self._error = None
            lifetime_r = self._ensure_lifetime_pipe()
            env = self._child_env()
            log_dir = self.home / "assistant" / "logs"
            chat_port = decision_port = None
            try:
                chat_port = managed.free_port()
                decision_port = managed.free_port()
                specs = {
                    "chat": managed.WorkerSpec("chat", prepared["chat_path"],
                                               chat_port, context, 0),
                    "decision": managed.WorkerSpec("decision", prepared["decision_path"],
                                                   decision_port, context,
                                                   prepared["max_questions"]),
                }
                builders = {"chat": _chat_worker_argv, "decision": _decision_worker_argv}
                child_env = dict(env)
                child_env[cap_shim.LIMIT_ENV] = str(context)
                for role in ("chat", "decision"):
                    self._check_cancel("worker launch")
                    record_name = chat["id"] if role == "chat" else decision["id"]
                    record = self._records[record_name]
                    payload = {"name": record["name"],
                               "claim_token": record["claim_token"],
                               "bytes": record["bytes"],
                               "pair_id": pair["id"]}
                    log_path = log_dir / f"{pair['id']}-{role}.log"
                    popen = managed.spawn_claiming_worker(
                        builders[role](specs[role]), payload,
                        lifetime_read_fd=lifetime_r, env=child_env,
                        log_path=log_path)
                    # register IMMEDIATELY: a later role's failure must be
                    # able to terminate and reap this child
                    self._children[role] = managed.ManagedChild(role, popen,
                                                                record_name)
                self._check_cancel("health wait")

                def _abort():
                    if self._cancel.is_set():
                        return True
                    return any(c.popen.poll() is not None
                               for c in self._children.values())

                chat_ok = managed.wait_health(
                    f"http://127.0.0.1:{chat_port}/v1/models",
                    deadline=HEALTH_TIMEOUT_SECONDS, abort=_abort)
                decision_ok = managed.wait_health(
                    f"http://127.0.0.1:{decision_port}/health",
                    deadline=HEALTH_TIMEOUT_SECONDS, abort=_abort)
                if not (chat_ok and decision_ok):
                    failed = []
                    for role, ok in (("chat", chat_ok), ("decision", decision_ok)):
                        if ok:
                            continue
                        rc = self._children[role].popen.poll()
                        if rc is not None:
                            failed.append(f"{role} exited early (rc={rc}); "
                                          f"log: {log_dir}/{pair['id']}-{role}.log")
                        elif self._cancel.is_set():
                            failed.append(f"{role} cancelled")
                        else:
                            failed.append(role)
                    raise managed.WorkerError(
                        f"workers failed to become healthy: {'; '.join(failed)}")
            except Exception as exc:
                survivors = {}
                for role, child in self._children.items():
                    if not managed.terminate(child.popen):
                        # termination unconfirmed: keep the child owned so
                        # status()/stop() keep watching it and its record
                        survivors[role] = child
                managed.reap_pair_records(pair["id"], self.home,
                                          children=self._children.values())
                self._release_unclaimed_records(pair["id"])
                self._children = survivors
                self._close_lifetime()
                self._state = "error"
                self._error = str(exc)
                raise PairError(f"pair start failed: {exc}") from exc
            prepared["chat_url"] = f"http://127.0.0.1:{chat_port}/v1"
            prepared["decision_url"] = f"http://127.0.0.1:{decision_port}/v1/decisions"
            self._state = "ready"
            self._write_lock(pair, chat, decision, context)
            return self.status()

    def _write_lock(self, pair: dict, chat: dict, decision: dict,
                    context: int) -> None:
        decision_manifest = _read_manifest(self._prepared["decision_path"])
        lock = {
            "version": 1,
            "pair_id": pair["id"],
            "routing_policy": pair["routing_policy"],
            "context_tokens": context,
            "max_questions": pair["max_questions"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "boot_id": _boot_id(),
            "started_offline": bool(catalog.env_offline()),
            "chip": {"arch": (chat.get("capability") or {}).get("arch")},
            "licenses": {m["id"]: m.get("license")
                         for m in (chat, decision) if m.get("license")},
            "models": {
                "chat": {"id": chat["id"], "repo": chat["repo"],
                         "revision": chat["revision"],
                         "path": str(self._prepared["chat_path"]),
                         "weights_bytes": chat["memory"]["weights_bytes"],
                         "kv_bytes_per_token": chat["memory"]["kv_bytes_per_token"]},
                "decision": {"id": decision["id"], "repo": decision["repo"],
                             "revision": decision["revision"],
                             "path": str(self._prepared["decision_path"]),
                             "weights_sha256": decision_manifest.get("weights_sha256"),
                             "context_max_tokens": decision_manifest.get("context_max_tokens")},
            },
        }
        write_pair_lock(self.home, pair["id"], lock)

    def _release_unclaimed_records(self, pair_id: str) -> None:
        """Parent-owned (never claimed) records — e.g. voice admission — are
        released only by the manager that owns their tokens."""
        for name, record in budget.pair_records(pair_id, self.home).items():
            if record.get("claimed") is None:
                try:
                    budget.clear_reservation(name, home=self.home,
                                             owner=record["owner"])
                except budget.BudgetError:
                    pass

    def _close_lifetime(self) -> None:
        for attr in ("_lifetime_write_fd", "_lifetime_read_fd"):
            fd = getattr(self, attr, None)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
                setattr(self, attr, None)

    def cancel(self) -> bool:
        """Abort an in-flight setup or start at the next stage boundary
        (downloads finish or fail on their own; conversion and spawn/health
        waits abort promptly). Returns True when an operation was in flight."""
        in_flight = self._state in ("preparing", "starting")
        self._cancel.set()
        return in_flight

    def stop(self) -> dict:
        """Stop the pair's workers and settle every reservation record.

        Closure contract (no silent success): every child gets bounded
        SIGTERM then SIGKILL; a child is closed only when its exit is
        VERIFIED. stopped is True iff every child is confirmed dead AND no
        pair record was retained. An unkillable (or unverifiable) worker
        RETAINS its reservation — reported in retained — and a later start()
        refuses with that record named until the worker verifiably exits
        (status() reaps it) or it is manually unreserved. stop() is safe to
        call repeatedly and after failures; prepared artifacts survive, so
        start() may be retried without re-setup."""
        self._cancel.set()
        with self._op_lock:
            report = {"stopped": False, "children": {}, "cleared": [],
                      "retained": []}
            for role, child in self._children.items():
                confirmed = managed.terminate(child.popen)
                report["children"][role] = {
                    "pid": child.popen.pid,
                    "confirmed_dead": confirmed,
                    "returncode": child.popen.poll(),
                }
            self._close_lifetime()
            if self._pair_id:
                report["cleared"] = managed.reap_pair_records(
                    self._pair_id, self.home, children=self._children.values())
                self._release_unclaimed_records(self._pair_id)
                report["retained"] = sorted(
                    set(budget.pair_records(self._pair_id, self.home))
                    - set(report["cleared"]))
            self._children = {role: child for role, child in
                              self._children.items()
                              if not report["children"][role]["confirmed_dead"]}
            self._context_tokens = None
            if self._prepared:
                for key in ("chat_url", "decision_url"):
                    self._prepared.pop(key, None)
            self._state = "prepared" if self._prepared else "idle"
            all_dead = all(c["confirmed_dead"]
                           for c in report["children"].values())
            report["stopped"] = all_dead and not report["retained"]
            self._cancel.clear()
            return report

    def ensure_context(self, required_tokens: int) -> dict:
        """Grow the chat worker's admitted context to at least required_tokens:
        one atomic fit-checked reservation resize, then a chat-worker restart
        at the new cap (the mlx-lm cap is fixed at launch). The decision worker
        is untouched. Refusal retains the prior cap."""
        with self._op_lock:
            if self._state != "ready" or self._prepared is None:
                raise PairError("ensure_context requires a running pair")
            current = self._context_tokens
            if not isinstance(current, int):
                raise PairError("ensure_context requires a running pair")
            if isinstance(required_tokens, bool) or not isinstance(required_tokens, int) \
                    or required_tokens < 1:
                raise PairError("required_tokens must be a positive int")
            if required_tokens <= current:
                return {"ok": True, "changed": False,
                        "context_tokens": current,
                        "requested": required_tokens}
            if required_tokens < catalog.MIN_CONTEXT_TOKENS:
                raise PairError(f"required_tokens must be an int >= "
                                f"{catalog.MIN_CONTEXT_TOKENS}")
            prepared = self._prepared
            chat = self.catalog_models[prepared["chat_id"]]
            decision = self.catalog_models[prepared["decision_id"]]
            decision_bytes = decision_requirement_bytes(prepared["decision_path"],
                                                        prepared["max_questions"])
            sizing = self._admissible(chat, other_bytes=decision_bytes,
                                      preference=prepared["preference"],
                                        available_bytes=self._available_bytes())
            if required_tokens > sizing["context_tokens"]:
                return {"ok": False, "changed": False,
                        "reason": sizing["binding"], "limits": sizing["limits"],
                        "context_tokens": current, "requested": required_tokens}
            new_bytes = budget.estimate_required(chat["memory"], required_tokens).total
            current_bytes = budget.estimate_required(chat["memory"], current).total
            record = self._records[chat["id"]]
            log_path = (self.home / "assistant" / "logs"
                        / f"{prepared['pair_id']}-chat.log")
            try:
                budget.resize_reservation(chat["id"], new_bytes,
                                          owner=record["owner"], home=self.home,
                                          available_bytes=self._available_bytes())
            except budget.BudgetError as exc:
                return {"ok": False, "changed": False, "reason": "memory",
                        "error": str(exc), "context_tokens": current,
                        "requested": required_tokens}

            def spawn_chat(tokens: int, port: int):
                spec = managed.WorkerSpec("chat", prepared["chat_path"], port,
                                          tokens, 0)
                payload = {"name": chat["id"],
                           "claim_token": record["claim_token"],
                           "bytes": budget.estimate_required(
                               chat["memory"], tokens).total,
                           "pair_id": prepared["pair_id"]}
                return managed.spawn_claiming_worker(
                    _chat_worker_argv(spec), payload,
                    lifetime_read_fd=self._lifetime_read_fd,
                    env=dict(self._child_env(),
                             **{cap_shim.LIMIT_ENV: str(tokens)}),
                    log_path=log_path)

            def restore_reservation():
                budget.resize_reservation(chat["id"], current_bytes,
                                          owner=record["owner"], home=self.home,
                                          available_bytes=self._available_bytes())

            def stop_note(base: str) -> str:
                # stop() result is part of the claim: never report a verified
                # stop that did not happen
                if self.stop()["stopped"]:
                    return f"{base}; the pair was stopped"
                return (f"{base}; stop did not verify and the worker remains "
                        "retained with its reservation")

            old_child = self._children["chat"]
            if not managed.terminate(old_child.popen):
                # the old worker is still live at `current`: put the
                # reservation back so admitted bytes match the running worker
                restore_reservation()
                return {"ok": False, "changed": False, "reason": "worker",
                        "error": "chat worker did not confirm termination; "
                                 "the running worker keeps the prior "
                                 "reservation",
                        "context_tokens": current,
                        "requested": required_tokens}
            new_port = managed.free_port()
            try:
                popen = spawn_chat(required_tokens, new_port)
            except Exception as exc:
                restore_reservation()
                raise PairError(f"chat worker restart failed to spawn: {exc}; "
                                "reservation restored to the prior context"
                                ) from exc
            self._children = {**self._children,
                              "chat": managed.ManagedChild("chat", popen, chat["id"])}
            abort = self._abort_predicate()
            if managed.wait_health(f"http://127.0.0.1:{new_port}/v1/models",
                                   deadline=HEALTH_TIMEOUT_SECONDS, abort=abort):
                prepared["chat_url"] = f"http://127.0.0.1:{new_port}/v1"
                self._context_tokens = required_tokens
                self._records[chat["id"]]["bytes"] = new_bytes
                return {"ok": True, "changed": True,
                        "context_tokens": required_tokens,
                        "limits": sizing["limits"], "requested": required_tokens}
            # the growth failed: the failed candidate is terminated and
            # verified dead BEFORE the reservation is touched or any
            # fallback spawns, and it stays tracked until then
            candidate = self._children["chat"]
            if not managed.terminate(candidate.popen):
                raise PairError(stop_note(
                    "grown chat worker would not confirm termination"))
            restore_reservation()
            fallback_port = managed.free_port()
            try:
                popen = spawn_chat(current, fallback_port)
            except Exception as exc:
                raise PairError(stop_note(
                    f"chat worker failed health at {required_tokens} tokens "
                    f"and the fallback did not spawn: {exc}")) from exc
            self._children = {**self._children,
                              "chat": managed.ManagedChild("chat", popen, chat["id"])}
            if not managed.wait_health(
                    f"http://127.0.0.1:{fallback_port}/v1/models",
                    deadline=HEALTH_TIMEOUT_SECONDS, abort=abort):
                raise PairError(stop_note(
                    "chat worker failed to restart at either context"))
            prepared["chat_url"] = f"http://127.0.0.1:{fallback_port}/v1"
            self._context_tokens = current
            return {"ok": False, "changed": False, "reason": "worker",
                    "error": f"chat worker failed health at {required_tokens} "
                             "tokens; the prior context is still serving",
                    "context_tokens": current, "requested": required_tokens}
