"""mlx-omarchy serve CLI: recommend, plan, approve, download, launch.

Exit codes: 0 ok, 1 refused (does not fit / fetch failed / user said no),
2 bad input, 3 missing environment (e.g. backend not installed).

Design invariants (do not weaken):
- Approval before any download; noninteractive runs fail closed unless
  --yes is given AND the target was named explicitly.
- The memory budget counts the full checkpoint (MoE total parameters),
  the KV at the explicit context limit, a labeled workspace margin, the
  aggregate active reservations, and a safety reserve. No fit -> refuse.
- Catalog refresh touches ONLY the public GitHub raw URL; no telemetry,
  no background timers, no automatic code or model downloads.
- trust_remote_code is never enabled anywhere.
- The mlx_lm.server path runs behind our total-context cap shim: upstream
  --max-tokens is only a per-request default, so prompt+output is enforced
  <= the admitted context at tokenization time, and the launch fails
  loudly if the mlx-lm pin bump moves the internals.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

from . import budget, catalog
from . import _mlxlm_server as cap_shim

GiB = 1024**3
DISK_SLACK_FRACTION = 0.05  # headroom over the download size for partial files

# allow_patterns declared per entry (extension.download_patterns) are the
# ONLY catalog-controlled download filter; they are validated here so a
# vetted entry can narrow a multi-variant repo to its own variant without
# any pattern able to escape the snapshot directory.
PATTERN_RE = re.compile(r"^[A-Za-z0-9 ._/*\[\]-]{1,128}$")


def download_patterns_for(entry: dict | None) -> list[str] | None:
    if entry is None:
        return None
    extension = entry.get("extension") or {}
    patterns = extension.get("download_patterns")
    if patterns is None:
        return None
    if not isinstance(patterns, list) or not patterns or len(patterns) > 64:
        raise budget.BudgetError(
            f"{entry.get('id', 'entry')}: extension.download_patterns must be "
            "a non-empty list (max 64) of file patterns"
        )
    for pattern in patterns:
        if (not isinstance(pattern, str) or not PATTERN_RE.match(pattern)
                or ".." in pattern):
            raise budget.BudgetError(
                f"{entry.get('id', 'entry')}: unsafe download pattern {pattern!r}"
            )
    return [str(pat) for pat in patterns]

REPO_RE = re.compile(r"^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)+$")

# Catalog data must never choose executable code. A module backend is only
# routed if its module name is on this audited allowlist (ships in-repo);
# extending it is a code review, not a catalog edit.
MODULE_ALLOWLIST = frozenset({
    "mlx_omarchy_laya.server",
    "mlx_omarchy_bonsai2.server",
})

# Modules that accept --managed (admission-controlled launch: the server
# must fail closed if it cannot register its reservation). Modules not
# listed here launch without the flag and are best-effort by contract.
MODULE_MANAGED = frozenset({
    "mlx_omarchy_laya.server",
    "mlx_omarchy_bonsai2.server",
})

# Catalog entries may point at an UPSTREAM RAW repo while a module backend
# needs its own CONVERTED artifact. Until a source->converted route exists
# and is tested, a raw snapshot is a hard preflight refusal naming the
# manual conversion command — never an implicit conversion download.
# Modules that accept a server-side context flag the CLI forwards
# (module_context_flag(module) -> flag name). Modules without an entry
# enforce their own internal cap.
MODULE_CONTEXT_FLAG = {
    "mlx_omarchy_bonsai2.server": "--max-context",
}

# Modules whose per-request question batch bound the CLI forwards. The
# assistant pair route passes the pair's cap (8); the standalone CLI omits
# the flag and keeps the module default.
MODULE_QUESTIONS_FLAG = {
    "mlx_omarchy_laya.server": "--max-questions",
}

MODULE_CONVERT_HINTS = {
    # mlx_omarchy_laya.server: upstream raw snapshots need conversion
    # (mlx_omarchy_laya.convert). mlx_omarchy_bonsai2.server is
    # deliberately ABSENT: Bonsai packs ARE the upstream artifact — there
    # is no conversion step, and the module's validate_artifact hook
    # (landed 61184a21) classifies packs directly.
    "mlx_omarchy_laya.server":
        "python -m mlx_omarchy_laya.convert --out <converted-dir> "
        "--from-local <snapshot-dir> [--variant typed-decisions]",
}


def module_artifact_problem(module: str, model_dir: Path) -> str | None:
    """Ask the module ecosystem whether model_dir is a usable converted
    artifact. Order: the server module's own validate_artifact hook, then
    the sibling <pkg>.convert checkpoint_state classifier (Laya contract:
    converted/raw/invalid). No hook at all -> a conservative generic
    reason, since raw bytes cannot be proven servable."""
    try:
        imported = importlib.import_module(module)
    except Exception as exc:
        return f"server module {module!r} is not importable ({exc})"
    validator = getattr(imported, "validate_artifact", None)
    if callable(validator):
        try:
            return validator(model_dir)
        except Exception as exc:
            return f"artifact validation failed: {exc}"
    convert_module_name = module.rsplit(".", 1)[0] + ".convert"
    try:
        convert_module = importlib.import_module(convert_module_name)
    except Exception:
        return (
            f"module {module!r} exposes no validate_artifact hook and no "
            f"{convert_module_name}.checkpoint_state classifier; cannot "
            "prove this directory is a converted artifact"
        )
    classifier = getattr(convert_module, "checkpoint_state", None)
    if not callable(classifier):
        return (
            f"module {module!r} exposes no artifact validator; cannot prove "
            "this directory is a converted artifact"
        )
    try:
        state, details = classifier(model_dir)
    except Exception as exc:
        return f"artifact classification failed: {exc}"
    if state == "converted":
        return None
    if state == "raw":
        command = details.get("convert_command") if isinstance(details, dict) else None
        hint = command or MODULE_CONVERT_HINTS.get(module, "convert the snapshot first")
        return f"RAW UPSTREAM SNAPSHOT (not converted): {details}. Convert first: {hint}"
    return f"invalid artifact: {details}"

MAX_WEIGHTS_GIB = 4096


def offline_requested(args) -> bool:
    return bool(getattr(args, "offline", False)) or catalog.env_offline()


def parse_weights_gib(value: str) -> int:
    try:
        gib = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--weights-gib {value!r} is not a number")
    if not math.isfinite(gib) or gib <= 0:
        raise argparse.ArgumentTypeError("--weights-gib must be positive and finite")
    if gib > MAX_WEIGHTS_GIB:
        raise argparse.ArgumentTypeError(f"--weights-gib exceeds {MAX_WEIGHTS_GIB} GiB")
    return int(gib * GiB)


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def fail(msg: str, code: int) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return code


def machine_soc() -> str | None:
    try:
        raw = Path("/proc/device-tree/compatible").read_bytes()
    except OSError:
        return None
    for token in raw.split(b"\0"):
        match = re.fullmatch(rb"apple,t(\d+)", token)
        if match:
            return "t" + match.group(1).decode()
    return None


# ---------------------------------------------------------------- resolution


class Resolved:
    def __init__(self, target: str, entry: dict | None, local_path: Path | None,
                 repo: str | None, revision: str | None):
        self.target = target
        self.entry = entry
        self.local_path = local_path
        self.repo = repo
        self.revision = revision

    @property
    def named_explicitly(self) -> bool:
        return self.target is not None


def backend_available(entry: dict) -> bool:
    serve = entry["serve"]
    if serve is None:
        return False
    if serve["backend"] == "omlx":
        return omlx_available()
    if serve["backend"] == "module":
        return serve["module"] in MODULE_ALLOWLIST
    return True


def auto_serve_reason(entry: dict) -> str | None:
    """Why this entry may not be AUTO-picked (manual naming is still allowed)."""
    qual = entry["qualification"]
    if not entry["recommended"]:
        return "not recommended"
    if qual["generation"]["status"] != "qualified":
        return "generation unqualified"
    if qual["http"]["status"] != "qualified":
        return "http serving unqualified"
    if qual["managed"]["status"] != "qualified":
        return "managed-route unqualified"
    if entry["serve"] is None:
        return "no serve route"
    if entry["serve"]["backend"] == "module" and entry["serve"]["module"] not in MODULE_ALLOWLIST:
        return "module not in audited allowlist"
    return None


def entry_context_tokens(entry: dict, requested: int | None) -> int | None:
    """Per-entry context for display/admission; None = requested exceeds limit."""
    limit = entry["context"]["max_tokens"]
    if requested is not None:
        if limit is not None and requested > limit:
            return None
        return requested
    return limit if limit is not None else budget.DEFAULT_CONTEXT_TOKENS


def pick_recommended(cat: dict, kind: str, context_tokens: int | None, home: Path | None):
    """Auto-pick = curated order among entries that are recommended:true
    AND pass every other gate (generation/http/managed qualified, working
    backend, device compatible, fit memory). The first fitting entry wins.

    `recommended: false` is a quarantine: an entry with it NEVER auto-picks
    and NEVER appears in `fits`, no matter how small or well it fits. If no
    recommended entry fits, `fits` is empty and the caller prints
    "pick: none — name a target explicitly": the user's manual override is
    the only path to anything unqualified, and downloads always require
    the standing approval contract. Curated priority reflects quality and
    is preserved as the ordering within the recommended tier; the same
    ordering is the `ranked` display order the CLI table uses verbatim."""
    soc = machine_soc()
    ranked = sorted(
        (e for e in cat["models"] if e["kind"] == kind),
        key=lambda e: (not e["recommended"], e["priority"]),
    )
    fits = []
    for entry in ranked:
        if auto_serve_reason(entry) is not None:
            continue
        if entry["capability"]["arch"] is not None and soc is not None \
                and soc not in entry["capability"]["arch"]:
            continue
        ctx = entry_context_tokens(entry, context_tokens)
        if ctx is None:
            continue
        if entry["capability"]["min_mem_gib"] is not None:
            try:
                if budget.mem_available() < entry["capability"]["min_mem_gib"] * GiB:
                    continue
            except budget.BudgetError:
                pass
        est = budget.estimate_required(entry["memory"], ctx)
        if budget.admit(est.total, home).fits:
            fits.append(entry)
    return fits, ranked, soc


def resolve_target(target: str | None, cat: dict) -> Resolved:
    if target is None:
        return Resolved(None, None, None, None, None)
    for entry in cat["models"]:
        if entry["id"] == target:
            return Resolved(target, entry, None, entry["repo"], entry["revision"])
    path = Path(target).expanduser()
    if path.is_dir():
        return Resolved(target, None, path, None, None)
    if REPO_RE.match(target) and "/" in target and not target.startswith("."):
        return Resolved(target, None, None, target, None)
    raise budget.BudgetError(
        f"{target!r} is not a catalog id, an existing local directory, or an HF repo id"
    )


def quant_label(entry: dict) -> str:
    if entry["quant"] is None:
        return "unquantized"
    mode = entry["quant"]["mode"]
    if mode == "affine":
        return f"{entry['quant']['bits']}bit/g{entry['quant']['group_size']}"
    return mode


def backend_for(resolved: Resolved, server_flag: str | None) -> tuple[str, str | None]:
    entry = resolved.entry
    if server_flag is not None:
        return server_flag, (entry["serve"]["module"] if entry and entry["serve"] else None)
    if entry is not None and entry["serve"] is not None:
        backend = entry["serve"]["backend"]
        if backend == "module":
            return "module", entry["serve"]["module"]
        return backend, None
    return "mlx-lm", None


# ------------------------------------------------------------------- planning


def hf_cache_dir() -> Path:
    # Mirror huggingface_hub's own precedence when it is importable
    # (HF_HUB_CACHE > HF_HOME/hub > default); fall back to the same
    # order manually so the disk check cannot look at the wrong volume.
    env_hub = os.environ.get("HF_HUB_CACHE")
    if env_hub:
        hub = Path(env_hub)
    else:
        try:
            from huggingface_hub import constants as hub_constants

            hub = Path(hub_constants.HF_HUB_CACHE)
        except Exception:
            home = os.environ.get("HF_HOME")
            base = Path(home) if home else Path.home() / ".cache/huggingface"
            hub = base / "hub"
    hub.mkdir(parents=True, exist_ok=True)  # the download would create it; stat it now
    return hub


def download_need_bytes(resolved: Resolved) -> int | None:
    return disk_need_bytes(resolved)


def disk_need_bytes(resolved: Resolved) -> int | None:
    if resolved.local_path is not None:
        return None
    if resolved.entry is not None:
        size = resolved.entry["availability"]["size_bytes"]
        weights = resolved.entry["memory"]["weights_bytes"]
        base = size if size is not None else weights
    else:
        base = None
    if base is None:
        return None
    return int(base * (1 + DISK_SLACK_FRACTION))


def plan_lines(resolved: Resolved, context_tokens: int, backend: str,
               module: str | None, host: str, port: int,
               weights_override: int | None, home: Path | None,
               prompt_cache_size: int = 0,
               patterns: list[str] | None = None) -> tuple[list[str], budget.Admission | None, bool]:
    lines: list[str] = []
    est = None
    admission = None
    if resolved.local_path is not None:
        if weights_override is None:
            raise budget.BudgetError(
                f"cannot budget local model {resolved.local_path} without --weights-gib "
                "(the full checkpoint size on disk; MoE counts total parameters)"
            )
        memory = {"weights_bytes": weights_override, "kv_bytes_per_token": None, "peak_estimate_bytes": None}
    elif resolved.entry is not None:
        memory = resolved.entry["memory"]
    else:
        if weights_override is None:
            raise budget.BudgetError(
                f"{resolved.repo} is not in the catalog; pass --weights-gib so the "
                "memory budget stays honest (or pick a catalog id)"
            )
        memory = {"weights_bytes": weights_override, "kv_bytes_per_token": None, "peak_estimate_bytes": None}

    est = budget.estimate_required(memory, context_tokens)
    if not est.kv_known and context_tokens > budget.DEFAULT_CONTEXT_TOKENS:
        raise budget.BudgetError(
            f"KV-per-token unknown for this model, so the budget cannot honestly "
            f"cover {context_tokens} tokens; use --context <= "
            f"{budget.DEFAULT_CONTEXT_TOKENS} or a catalog entry with KV facts"
        )
    cache_extra = 0
    if backend == "mlx-lm" and prompt_cache_size > 0:
        # Retained prompt-cache entries hold their own KV (each <= the
        # capped request total). The admission must reserve them BEFORE
        # launch; without KV facts they cannot be budgeted at all.
        if not est.kv_known:
            raise budget.BudgetError(
                f"--prompt-cache-size {prompt_cache_size} needs KV facts to "
                "budget the retained caches; this entry has unknown "
                "KV-per-token"
            )
        cache_extra = est.kv * prompt_cache_size
    lines.append(f"model:        {resolved.repo or resolved.local_path}")
    if resolved.revision:
        lines.append(f"revision:     {resolved.revision} (pinned)")
    lines.append(f"backend:      {backend}" + (f" module={module}" if module else ""))
    lines.append(f"bind:         {host}:{port}")
    lines.append(f"context:      {context_tokens} tokens")
    lines.append(f"weights:      {est.peak_override and '(peak override) ' or ''}{est.weights / GiB:.2f} GiB")
    if est.kv_known:
        lines.append(f"kv cache:     {est.kv / GiB:.2f} GiB at {context_tokens} tokens")
    else:
        lines.append(f"kv cache:     UNKNOWN -> flat {budget.UNKNOWN_KV_MARGIN / GiB:.2f} GiB margin included")
    if cache_extra:
        # hold the retained caches inside the estimate: raise the total and
        # the visible margin by exactly the cache KV
        est = budget.Estimate(est.weights, est.kv, est.kv_known,
                              est.workspace + cache_extra, est.peak_override,
                              est.total + cache_extra)
        lines.append(f"prompt cache: +{cache_extra / GiB:.2f} GiB "
                     f"({prompt_cache_size} retained entries x KV at "
                     f"{context_tokens} tokens)")
    if patterns:
        lines.append(f"download filter: {len(patterns)} pattern(s) from the "
                     "catalog entry (whole-repo download prevented)")
    label = "measured peak" if est.peak_override else "workspace margin, unmeasured estimate"
    lines.append(f"margin:       {est.workspace / GiB:.2f} GiB ({label})")
    disk_ok = True
    need = disk_need_bytes(resolved) if resolved.local_path is None else None
    if need is not None:
        ok, free, where = budget.disk_check(need, hf_cache_dir())
        disk_ok = ok
        lines.append(f"disk:         need {need / GiB:.2f} GiB, {free / GiB:.2f} GiB free at {where}"
                     + ("" if ok else "  -> INSUFFICIENT"))
    admission = budget.admit(est.total, home)
    lines.extend("mem:          " + stripped for stripped in (ln.strip() for ln in admission.lines))
    lines.append("verdict:      " + ("FITS" if admission.fits and disk_ok else
                                  "DOES NOT FIT — refusing (no swap/OOM gambles)"))
    return lines, admission, disk_ok


def snapshot_gaps(path: Path, patterns: list[str] | None = None,
                  hf_layout: bool = True) -> list[str]:
    """Names of the files that make this snapshot unusable: weights (index
    shards or single file) must exist nonempty, and every declared download
    pattern must be satisfied (exact patterns: the file exists; wildcard
    patterns: at least one match). With hf_layout (the default, for
    transformers-style checkpoints) config.json and some tokenizer file must
    also be present; a pack with its own layout (Laya keeps encoder/config.json
    and tokenizer/) declares its files in patterns and passes hf_layout=False.
    Empty list means complete."""
    gaps: list[str] = []
    index = path / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        except (OSError, json.JSONDecodeError):
            weight_map = {}
        shards = sorted(set(weight_map.values()))
        if not shards:
            gaps.append("model.safetensors.index.json (no shards mapped)")
        gaps.extend(shard for shard in shards
                    if not (path / shard).is_file() or (path / shard).stat().st_size == 0)
    else:
        single = path / "model.safetensors"
        if not (single.is_file() and single.stat().st_size > 0):
            gaps.append("model.safetensors")
    if hf_layout and not (path / "config.json").is_file():
        gaps.append("config.json")
    if hf_layout and not any((path / name).is_file() for name in
                             ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")):
        gaps.append("a tokenizer file (tokenizer.json et al.)")
    if patterns:
        for pattern in patterns:
            if any(ch in pattern for ch in "*?["):
                if not any(path.glob(pattern)):
                    gaps.append(pattern)
            elif not (path / pattern).is_file():
                gaps.append(pattern)
    return gaps


def snapshot_complete(path: Path, patterns: list[str] | None = None,
                      hf_layout: bool = True) -> bool:
    return not snapshot_gaps(path, patterns, hf_layout)


def offline_snapshot_detail(resolved: Resolved, patterns: list[str] | None = None,
                            hf_layout: bool = True) -> str:
    """Human-readable cache state for an offline refusal: names the exact
    missing files when a snapshot directory exists, or says none does."""
    if resolved.local_path is not None or not resolved.revision:
        return ""
    directory = (hf_cache_dir() / ("models--" + resolved.repo.replace("/", "--"))
                 / "snapshots" / resolved.revision)
    if not directory.is_dir():
        return (f"; the Hugging Face cache has no snapshot for revision "
                f"{resolved.revision[:12]} of {resolved.repo}")
    gaps = snapshot_gaps(directory, patterns, hf_layout)
    if not gaps:
        return ""
    return f"; the cached snapshot for {resolved.repo} is missing: " + ", ".join(gaps[:8])


def probe_snapshot(resolved: Resolved, patterns: list[str] | None = None,
                   hf_layout: bool = True) -> Path | None:
    """Local-only snapshot resolution. Returns a verified-complete path or
    None (missing OR partial download). Never touches the network."""
    if resolved.local_path is not None:
        return resolved.local_path
    hub = _import_huggingface_hub()
    if hub is None:
        return None  # cannot prove presence; the approval flow will say so
    try:
        path = Path(hub.snapshot_download(
            repo_id=resolved.repo,
            revision=resolved.revision,
            local_files_only=True,
            cache_dir=hf_cache_dir(),
            **({"allow_patterns": patterns} if patterns else {}),
        ))
    except Exception:  # huggingface_hub raises several types for a missing snapshot
        return None
    return path if snapshot_complete(path, patterns, hf_layout) else None


def _import_huggingface_hub():
    spec = importlib.util.find_spec("huggingface_hub")
    if spec is None:
        return None
    import huggingface_hub

    return huggingface_hub


def download_snapshot(resolved: Resolved, patterns: list[str] | None = None) -> Path:
    hub = _import_huggingface_hub()
    if hub is None:
        raise budget.BudgetError("huggingface_hub is not installed in this environment")
    return Path(
        hub.snapshot_download(repo_id=resolved.repo, revision=resolved.revision,
                              cache_dir=hf_cache_dir(),
                              **({"allow_patterns": patterns} if patterns else {}))
    )


def check_custom_code(model_dir: Path) -> None:
    config = model_dir / "config.json"
    try:
        obj = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if isinstance(obj, dict) and "auto_map" in obj:
        warn(
            f"{config} declares auto_map (custom modeling code). mlx-omarchy never "
            "enables trust_remote_code; loading this checkpoint will likely fail."
        )


def server_argv(backend: str, module: str | None, model_dir: Path, host: str,
                port: int, context_tokens: int,
                prompt_cache_size: int = 0,
                max_questions: int | None = None) -> list[str]:
    if backend == "mlx-lm":
        # Upstream --max-tokens is only a per-request default (client
        # overridable, verified in the 0.31.3 wheel), so the REAL total
        # cap lives in our _mlxlm_server shim (prompt+output <= budget,
        # enforced at tokenization). The flag remains as the default for
        # clients that omit max_tokens. --trust-remote-code exists on
        # this server; it is never passed.
        # Aggregate bound: concurrency 1 means exactly one request's
        # tokens are resident; prompt cache 0 means no retained KV caches
        # -- so the aggregate equals the admitted context. A larger
        # --prompt-cache-size re-enables caching at a documented
        # (cache + 1) x context worst case.
        # The --max-tokens DEFAULT is bounded to min(512, context): a
        # client that omits max_tokens inherits it, and inheriting the
        # full context would make the shim reject every nonempty prompt
        # (prompt + default > context). The strict prompt+output cap in
        # the shim still applies to every request.
        argv = [sys.executable, "-m", "mlx_omarchy_serve._mlxlm_server",
                "--model", str(model_dir),
                "--host", host, "--port", str(port),
                "--max-tokens", str(min(512, context_tokens)),
                "--decode-concurrency", "1",
                "--prompt-concurrency", "1",
                "--prompt-cache-size", str(prompt_cache_size)]
        return argv
    if backend == "omlx":
        return [sys.executable, "-m", "omlx.server", "--model-dir", str(model_dir),
                "--host", host, "--port", str(port)]
    if backend == "module":
        if not module:
            raise budget.BudgetError("module backend needs a module name (catalog serve.module)")
        code = f"import sys; from {module} import serve_main; serve_main(sys.argv[1:])"
        argv = [sys.executable, "-c", code, "--model", str(model_dir),
                "--host", host, "--port", str(port)]
        if module in MODULE_MANAGED:
            # --managed: admission-controlled launch; the module must fail
            # closed if it cannot register its reservation.
            argv.append("--managed")
        flag = MODULE_CONTEXT_FLAG.get(module)
        if flag:
            argv += [flag, str(context_tokens)]
        if max_questions is not None:
            qflag = MODULE_QUESTIONS_FLAG.get(module)
            if not qflag:
                raise budget.BudgetError(
                    f"module {module!r} has no question-batch flag; "
                    "max_questions cannot be forwarded")
            argv += [qflag, str(int(max_questions))]
        return argv
    raise budget.BudgetError(f"unknown backend {backend!r}")


def omlx_available() -> bool:
    return importlib.util.find_spec("omlx") is not None


# --------------------------------------------------------------------- flows


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mlx-omarchy-serve",
        description="Serve a vetted local model on the Apple GPU (admission-checked, approve-first).",
    )
    sub = parser.add_subparsers(dest="command")

    def common(p, with_target=True):
        p.add_argument("--offline", action="store_true",
                       help="never touch the network (catalog stays cached/bundled)")
        if with_target:
            p.add_argument("target", nargs="?",
                           help="catalog id, HF repo id (org/name), or local model directory; "
                                "default: curated recommendation")
        p.add_argument("--context", type=int, default=None,
                       help="context tokens (KV is budgeted at this); default = model limit or 4096")
        p.add_argument("--server", choices=("mlx-lm", "omlx", "module"), default=None,
                       help="serve backend; default from catalog, else mlx-lm")
        p.add_argument("--weights-gib", type=parse_weights_gib, default=None, dest="weights_gib",
                       help="explicit checkpoint size in GiB for off-catalog/local models "
                            "(positive, finite; the FULL artifact)")
        p.add_argument("--prompt-cache-size", type=int, default=0,
                       help="mlx-lm prompt cache entries; 0 (default) keeps the "
                            "aggregate KV bound at the admitted context, N raises "
                            "the worst case to (N+1) x context and is budgeted")

    p_recommend = sub.add_parser("recommend", help="show curated fits and the pick")
    common(p_recommend, with_target=False)
    p_recommend.add_argument("--kind", default="chat", choices=catalog.KINDS)

    p_plan = sub.add_parser("plan", help="admission + disk check only; downloads nothing")
    common(p_plan)
    p_plan.add_argument("--host", default="127.0.0.1")
    p_plan.add_argument("--port", type=int, default=8080)

    p_serve = sub.add_parser("serve", help="plan, require approval, download, launch foreground")
    common(p_serve)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--yes", action="store_true",
                         help="noninteractive explicit opt-in; requires an explicit target")

    p_cat = sub.add_parser("catalog", help="catalog list/status/refresh")
    csub = p_cat.add_subparsers(dest="catalog_command", required=True)
    for name, help_text in (
        ("list", "print the catalog table with memory verdicts"),
        ("status", "which catalog is live (cache age, source)"),
        ("refresh", "force a GitHub refresh now"),
    ):
        p = csub.add_parser(name, help=help_text)
        p.add_argument("--offline", action="store_true",
                       help="never touch the network")

    p_res = sub.add_parser("reserve", help="declare resident memory another local service owns")
    p_res.add_argument("name")
    p_res.add_argument("gib", type=parse_weights_gib)
    p_res.add_argument("--note", default="")
    p_unres = sub.add_parser("unreserve", help="remove a reservation")
    p_unres.add_argument("name")

    return parser.parse_args(argv)


def cmd_recommend(args, cat, home) -> int:
    requested = args.context
    fits, ranked, soc = pick_recommended(cat, args.kind, requested, home)
    print(f"machine SoC: {soc or 'unknown'}; kind: {args.kind}; curated order "
          f"(auto-serve needs recommended + generation and http qualified + backend):")
    for entry in ranked:
        ctx = entry_context_tokens(entry, requested)
        reason = auto_serve_reason(entry)
        if ctx is None:
            est_total = budget.estimate_required(entry["memory"],
                                                 entry["context"]["max_tokens"]).total
            note = f"context {requested} over limit {entry['context']['max_tokens']}"
        else:
            est_total = budget.estimate_required(entry["memory"], ctx).total
            note = reason or ""
        fits_now = entry in fits
        print(f"  {'*' if fits_now else ' '} {entry['id']:<28} {quant_label(entry):<10}"
              f" p{entry['priority']}"
              f" {'recommended' if entry['recommended'] else '           '}"
              f" {est_total / GiB:6.2f} GiB"
              f" gen:{entry['qualification']['generation']['status']}"
              f" http:{entry['qualification']['http']['status']}"
              + (f"  [{note}]" if note else ""))
    if fits:
        pick = fits[0]
        print(f"pick: {pick['id']} (qualified, compatible, memory fits; other fitting "
              "variants above are servable by id)")
        return 0
    print("pick: none — no entry passes the auto-serve gate; name a target explicitly "
          "to serve something unqualified")
    return 0


def build_resolved(args, cat) -> Resolved:
    resolved = resolve_target(args.target, cat)
    if resolved.target is None and args.command in ("serve", "plan"):
        fits, ranked, _soc = pick_recommended(
            cat, "chat", args.context, None
        )
        if not fits:
            raise budget.BudgetError(
                "no auto-servable chat model (needs recommended + generation and "
                "http qualified + working backend + memory fit); name a target explicitly"
            )
        entry = fits[0]
        if sys.stdin.isatty() and len(fits) > 1:
            # Manual variant selection: every fitting quant variant is offered,
            # the curated pick is just the default. Never all of them.
            for i, candidate in enumerate(fits, 1):
                est = budget.estimate_required(
                    candidate["memory"], args.context or budget.DEFAULT_CONTEXT_TOKENS)
                print(f"  {i}) {candidate['id']:<28} {quant_label(candidate):<10}"
                      f" {est.total / GiB:.2f} GiB")
            answer = input(f"Select variant [1-{len(fits)}] (default 1): ").strip()
            choice = int(answer) if answer.isdigit() and 1 <= int(answer) <= len(fits) else 1
            entry = fits[choice - 1]
        print(f"serving variant: {entry['id']}")
        resolved = Resolved(None, entry, None, entry["repo"], entry["revision"])
    return resolved


def do_plan(args, cat, home) -> tuple[Resolved, list[str], budget.Admission | None, str, str | None]:
    resolved = build_resolved(args, cat)
    context = budget.resolve_context(
        resolved.entry["context"] if resolved.entry else {"max_tokens": None},
        args.context,
    )
    backend, module = backend_for(resolved, args.server)
    if backend == "module":
        if not module:
            raise budget.BudgetError("module backend selected but no serve.module is set")
        if resolved.entry is not None and module not in MODULE_ALLOWLIST:
            raise budget.BudgetError(
                f"module {module!r} is not in the audited allowlist {sorted(MODULE_ALLOWLIST)}; "
                "catalog data must never select executable code"
            )
    if resolved.entry is not None and args.command in ("serve", "plan"):
        reason = auto_serve_reason(resolved.entry)
        if reason is not None and resolved.named_explicitly:
            warn(f"{resolved.entry['id']}: {reason}; you named it explicitly, proceeding "
                 "with an unqualified target")
    cache_size = max(int(getattr(args, "prompt_cache_size", 0) or 0), 0)
    if cache_size > 0 and backend != "mlx-lm":
        raise budget.BudgetError(
            "--prompt-cache-size applies only to the mlx-lm backend"
        )
    patterns = download_patterns_for(resolved.entry)
    lines, admission, disk_ok = plan_lines(
        resolved, context, backend, module, args.host, args.port,
        args.weights_gib if args.weights_gib else None, home,
        prompt_cache_size=cache_size, patterns=patterns,
    )
    return resolved, lines, admission, disk_ok, backend, module, context, patterns


def cmd_plan(args, cat, home) -> int:
    _resolved, lines, _adm, _disk, _backend, _module, _ctx, _pat = do_plan(args, cat, home)
    print("\n".join(lines))
    return 0


def approve_download(lines: list[str], *, interactive: bool, assume_yes: bool,
                     explicit_target: bool, input_fn=None) -> bool:
    print("\n".join(lines))
    print()
    # The invariant holds in BOTH modes: --yes means "pre-approved", and
    # pre-approval requires a named target — interactive or not.
    if assume_yes and not explicit_target:
        print("refusing: --yes requires an explicitly named target; nothing was downloaded",
              file=sys.stderr)
        return False
    if not interactive:
        if not assume_yes:
            print("refusing: noninteractive without --yes; nothing was downloaded", file=sys.stderr)
            return False
        return True
    answer = (input_fn or input)("Download this model now? Type 'yes' to proceed: ")
    return answer.strip().lower() == "yes"


def cmd_serve(args, cat, home) -> int:
    (resolved, lines, admission, disk_ok, backend, module, context,
     patterns) = do_plan(args, cat, home)
    if (admission is not None and not admission.fits) or not disk_ok:
        print("\n".join(lines), file=sys.stderr)
        print("\nrefusing: does not fit (memory or disk); nothing was downloaded or started",
              file=sys.stderr)
        return 1

    interactive = sys.stdin.isatty()
    if resolved.entry is not None and patterns is None and download_need_bytes(resolved):
        warn(f"{resolved.entry['id']} declares no extension.download_patterns; "
             "the WHOLE repo will download — ensure the disk budget covers it")
    model_dir = probe_snapshot(resolved, patterns)
    if model_dir is not None and resolved.local_path is not None \
            and backend == "mlx-lm" and not snapshot_complete(model_dir, patterns):
        return fail(
            f"local model directory {model_dir} is not a complete servable "
            "artifact (weights shards, config.json, and a tokenizer file are "
            "all required)", 2)
    download = model_dir is None
    if download and offline_requested(args):
        return fail("--offline set and a complete model is not on disk; refusing to download", 1)

    if download:
        # F9: a non-catalog repo has no pinned revision, so resolve the
        # actual current commit SHA BEFORE approval and hold it fixed for
        # the download — what the user approves is exactly what fetches.
        if resolved.revision is None and resolved.repo is not None:
            hub = _import_huggingface_hub()
            sha = None
            if hub is not None:
                try:
                    sha = getattr(hub.model_info(resolved.repo), "sha", None)
                except Exception as exc:
                    return fail(f"cannot resolve the current commit of "
                                f"{resolved.repo} ({exc}); refusing an "
                                "un-pinned download", 1)
            if not sha:
                return fail(f"cannot resolve the current commit of "
                            f"{resolved.repo}; refusing an un-pinned download", 1)
            resolved.revision = str(sha)
            lines.append(f"revision:     {sha} (resolved pre-approval)")
        if not approve_download(lines, interactive=interactive, assume_yes=args.yes,
                                explicit_target=resolved.named_explicitly):
            return 1
        model_dir = download_snapshot(resolved, patterns)
        if not snapshot_complete(model_dir, patterns):
            return fail("downloaded snapshot is incomplete (weights, configs, or "
                        "declared files missing); refusing to launch", 3)
        print(f"downloaded to {model_dir}")
    else:
        print("\n".join(lines))
    check_custom_code(model_dir)

    if backend == "module":
        problem = module_artifact_problem(module, model_dir)
        if problem:
            hint = MODULE_CONVERT_HINTS.get(module)
            print(f"error: {problem}", file=sys.stderr)
            if hint:
                print(f"convert the source snapshot first: {hint}", file=sys.stderr)
            return fail("module backend requires its own converted artifact; "
                        "the raw catalog/source snapshot is not servable", 2)

    if backend == "omlx":
        if not omlx_available():
            return fail("omlx is not installed (source install only; see "
                        "docs/serve.md); use --server mlx-lm", 3)
        # omlx has no verified server-side context cap: the admitted
        # context cannot be enforced, so the launch is memory-unsafe.
        return fail("omlx backend has no verified server-side context cap; "
                    "refusing to launch (memory safety cannot be enforced). "
                    "Use --server mlx-lm", 3)
    if host_nonlocal(args.host):
        warn(f"{args.host} is not loopback: this development server has no authentication")

    # Shared launch boundary: backends that do NOT self-manage their
    # reservation (mlx-lm) get an atomic admit+reserve for the FULL
    # requirement BEFORE the child spawns, held for the child's whole
    # lifetime (conservative: the CLI cannot prove residency, so it never
    # relabels). Module backends own their atomic reservation instead.
    #
    # Ordering (Main review): argv/env/signal handler are built FIRST;
    # admit -> spawn -> wait run inside ONE try/finally so no failure
    # between reserve and spawn can leak the reservation. Cleanup is
    # SIGTERM-safe: the handler converts the signal into SystemExit, the
    # finally terminates the child with a bounded wait (escalating to
    # kill), and the reservation is released ONLY once the child is
    # confirmed dead — otherwise it is retained (fail closed) so a future
    # admission cannot double-count the child's RAM.
    argv = server_argv(backend, module, model_dir, args.host, args.port, context,
                       prompt_cache_size=max(int(getattr(args, "prompt_cache_size", 0) or 0), 0))
    child_env = dict(os.environ)
    if backend == "mlx-lm":
        child_env[cap_shim.LIMIT_ENV] = str(context)
    import signal
    import threading

    def _on_sigterm(signum, _frame):
        raise SystemExit(128 + signum)

    in_main_thread = threading.current_thread() is threading.main_thread()
    previous_term = signal.signal(signal.SIGTERM, _on_sigterm) if \
        in_main_thread else None
    launch_owner = None
    launch_name = None
    child = None
    try:
        if backend != "module":
            launch_owner = budget.generate_owner()
            launch_name = f"{backend}-launch-{launch_owner}"
            budget.admit_and_reserve(
                launch_name, admission.required,
                note=f"CLI launch: {resolved.repo or resolved.local_path}",
                owner=launch_owner, home=home, state="pending")
        child = subprocess.Popen(argv, env=child_env)
        returncode = child.wait()
    except budget.BudgetError as exc:
        print("\n".join(lines), file=sys.stderr)
        return fail(f"concurrent launch refused: {exc}", 1)
    except FileNotFoundError as exc:
        return fail(f"server launch failed: {exc}", 3)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        confirmed_dead = child is None or child.poll() is not None
        if launch_name is not None:
            if confirmed_dead:
                budget.clear_reservation(launch_name, home, owner=launch_owner)
            else:
                print(
                    f"error: launched child did not confirm termination; "
                    f"RETAINING its reservation {launch_name!r} (fail closed "
                    "until manually cleared via unreserve)", file=sys.stderr)
        if previous_term is not None:
            signal.signal(signal.SIGTERM, previous_term)
    return returncode


def host_nonlocal(host: str) -> bool:
    return host not in ("127.0.0.1", "localhost", "::1")


def cmd_catalog(args, _cat, home) -> int:
    if args.catalog_command == "list":
        cat = catalog.load_catalog(home)
        print(f"{'id':<28} {'kind':<9} {'p':<3} {'rec':<3} {'gen':<9} {'http':<9} {'GiB':>7}")
        for e in cat["models"]:
            print(f"{e['id']:<28} {e['kind']:<9} {e['priority']:<3}"
                  f" {'yes' if e['recommended'] else '-':<3}"
                  f" {e['qualification']['generation']['status']:<9}"
                  f" {e['qualification']['http']['status']:<9}"
                  f" {e['memory']['weights_bytes'] / GiB:>7.2f}")
        return 0
    if args.catalog_command == "status":
        path = catalog.cache_path(home)
        if path.is_file():
            age_h = 0.0
            import time

            age_h = (time.time() - path.stat().st_mtime) / 3600.0
            print(f"cache:  {path} ({age_h:.1f}h old)")
        else:
            print(f"cache:  none at {path}")
        print(f"bundled: {catalog.bundled_path()}")
        print(f"url:    {os.environ.get('MLX_OMARCHY_CATALOG_URL') or catalog.DEFAULT_CATALOG_URL}")
        print(f"live:   {'cache' if path.is_file() else 'bundled fallback'}")
        return 0
    result = catalog.refresh(home=home, offline=offline_requested(args))
    print(f"catalog refresh: {result['status']} ({result['detail']})")
    return 0 if result["status"] in ("fresh", "updated", "not-modified", "skipped") else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    home = budget.default_home()
    try:
        if args.command in ("reserve", "unreserve"):
            cat = None  # reservation bookkeeping never needs the catalog
        else:
            catalog.refresh(home=home, offline=offline_requested(args))
            cat = catalog.load_catalog(home)
        if args.command == "recommend":
            return cmd_recommend(args, cat, home)
        if args.command == "plan":
            return cmd_plan(args, cat, home)
        if args.command == "serve":
            return cmd_serve(args, cat, home)
        if args.command == "catalog":
            return cmd_catalog(args, cat, home)
        if args.command == "reserve":
            budget.set_reservation(args.name, args.gib, args.note, home)
            print(f"reserved {args.gib / GiB:g} GiB for {args.name}")
            return 0
        if args.command == "unreserve":
            if not budget.clear_reservation(args.name, home):
                return fail(f"no reservation named {args.name!r}", 2)
            print(f"removed reservation {args.name}")
            return 0
    except budget.BudgetError as exc:
        return fail(str(exc), 2)
    except catalog.CatalogError as exc:
        return fail(f"catalog invalid: {exc}", 3)
    return fail("unreachable", 2)


if __name__ == "__main__":
    sys.exit(main())
