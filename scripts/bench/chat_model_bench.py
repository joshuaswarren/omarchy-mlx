#!/usr/bin/env python3
"""Chat-model benchmark for the offline-assistant pair defaults.

Per the assignment:
  - one model per process (clean cold and warm load times)
  - same machine, same prompts, greedy, thinking disabled
  - 2 warmups + 5 measured turns
  - prefill @512, @2048, decode @128, TTFT for ~170-token prompt + assistant
    SCHEMA_PROMPT_COMPACT system prompt
  - 16 frozen card-adherence prompts, 20 frozen GSM8K, 20 frozen IFE
  - peak memory via mx.get_peak_memory, process RSS, /proc/meminfo MemAvailable
  - provenance line printed beside every measurement
  - does NOT mark anything recommended

Output: JSON written to --out. The driver is invoked by run_one.sh for each model.

Usage:
  chat_model_bench.py --model-id <hfrepo> --revision <sha> \
      --prompts-dir <dir> --out <path.json> [--model-label <label>] \
      [--max-tokens 700] [--repetition-penalty 1.1]

The model path passed to mlx_lm.load is the local snapshot dir
(hub/models--<repo>/snapshots/<sha>/) so the load is reproducible and
avoids any network I/O.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Force Honeykrisp / deterministic numerics where possible
os.environ.setdefault("MLX_DISABLE_COMPILE", "1")

REPO_ROOT = Path(os.environ.get(
    "BENCH_REPO_ROOT",
    str(Path(__file__).resolve().parents[2]),
))
PROMPTS_DEFAULT = REPO_ROOT / "scripts/bench/prompts"
# HF cache root. Set HF_HUB_CACHE to override; default matches the host's
# standard location when the env var is unset.
HF_CACHE_ROOT = Path(os.environ.get(
    "HF_HUB_CACHE",
    str(Path.home() / ".cache" / "huggingface" / "hub"),
))

# Match the coordinator's defaults: see coordinator.py
REPETITION_PENALTY = 1.1

# Reproduce coordinator prompt build for chat turns (SCHEMA_PROMPT_COMPACT)
# Import from the project; we are a child process the parent runs under PYTHONPATH=serve
try:
    from mlx_omarchy_assistant.components import (
        SCHEMA_PROMPT_COMPACT,
        SCHEMA_PROMPT,
        validate_components,
    )
except Exception as exc:  # noqa: BLE001
    print(f"WARN cannot import assistant components: {exc}", file=sys.stderr)
    SCHEMA_PROMPT_COMPACT = ""
    SCHEMA_PROMPT = ""
    validate_components = None


def provenance_line() -> str:
    """One line for the side of every measurement."""
    import mlx.core as mx

    mlx_v = getattr(mx, "__version__", "?")
    import mlx_lm  # noqa: PLC0415
    mlxlm_v = getattr(mlx_lm, "__version__", "?")
    try:
        wheel = subprocess.check_output(
            [sys.executable, "-c",
             "import importlib.metadata as m; print(m.version('mlx-omarchy'))"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        wheel = "unknown"
    # Commit lookup: try REPO_ROOT (worktree), then cwd, then env override
    commit = None
    for path in (REPO_ROOT, Path.cwd()):
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "--short=12", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            break
        except Exception:  # noqa: BLE001
            continue
    commit = commit or os.environ.get("BENCH_COMMIT_OVERRIDE", "unknown")
    return (
        f"prov: mlx={mlx_v} mlx_lm={mlxlm_v} wheel={wheel} "
        f"commit={commit} python={sys.version.split()[0]}"
    )


def _save_partial(out_path: str, scope: dict) -> None:
    """Persist a partial result snapshot to <out_path>.tmp then rename.

    `scope` is the locals() dict from the calling function; we copy the
    result lists/metrics we need.
    """
    import os as _os  # noqa: PLC0415
    if not out_path:
        return
    try:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        partial = {
            "checkpoint": True,
            "model_id": scope.get("args").model_id if scope.get("args") else None,
            "model_label": scope.get("args").model_label if scope.get("args") else None,
            "revision": scope.get("args").revision if scope.get("args") else None,
            "cold_load_s": scope.get("cold_load_s"),
            "warm_load_s": scope.get("warm_load_s"),
            "peak_mem_after_load_gb": scope.get("peak_after_load_gb"),
            "ttft_170sys_256tok": scope.get("ttft_stats"),
            "ttft_measured": scope.get("measured"),
            "prefill_512": scope.get("prefill_512"),
            "prefill_2048": scope.get("prefill_2048"),
            "decode_512": scope.get("decode_512_tps"),
            "decode_2048": scope.get("decode_2048_tps"),
            "card_pass": sum(1 for c in scope.get("card_results", []) if c.get("hit")),
            "card_total": len([c for c in scope.get("card_results", []) if "expect" in c]),
            "card_results": _coerce(scope.get("card_results", [])),
            "gsm_pass": sum(1 for g in scope.get("gsm_results", []) if g.get("pass")),
            "gsm_total": len(scope.get("gsm_results", [])),
            "gsm_results": _coerce(scope.get("gsm_results", [])),
            "ife_pass": sum(1 for i in scope.get("ife_results", []) if i.get("pass")),
            "ife_total": len(scope.get("ife_results", [])),
            "ife_results": _coerce(scope.get("ife_results", [])),
            "provenance": provenance_line(),
        }
        tmp = out.with_suffix(out.suffix + ".tmp")
        with open(tmp, "w") as fh:
            json.dump(partial, fh, indent=2)
        _os.replace(tmp, out)
    except Exception as exc:  # noqa: BLE001
        print(f"WARN partial save failed: {exc}", file=sys.stderr)


def _coerce(value):
    """Recursively coerce unserializable objects to strings.

    JSON-serializable primitives pass through. Containers are recursed.
    Anything else (regex Match, MLX arrays, custom classes) becomes str().
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _coerce(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce(v) for v in value]
    return str(value)


def meminfo_available_mib() -> int:
    txt = Path("/proc/meminfo").read_text()
    m = re.search(r"MemAvailable:\s+(\d+)\s+kB", txt)
    return int(m.group(1)) // 1024 if m else -1


def rss_mib() -> int:
    txt = Path(f"/proc/{os.getpid()}/status").read_text()
    m = re.search(r"VmRSS:\s+(\d+)\s+kB", txt)
    return int(m.group(1)) // 1024 if m else -1


def build_prompt(system: str, user: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def load_model(model_path: str, tokenizer_config=None):
    """mlx_lm.load; returns (model, tokenizer)."""
    from mlx_lm import load  # noqa: PLC0415

    t0 = time.monotonic()
    model, tok = load(model_path, tokenizer_config=tokenizer_config)
    dt = time.monotonic() - t0
    return model, tok, dt


def warm_load_model(model_path: str):
    """Second load to measure warm restart (kernel cache, page cache warm)."""
    from mlx_lm import load  # noqa: PLC0415

    t0 = time.monotonic()
    model, tok = load(model_path)
    dt = time.monotonic() - t0
    return model, tok, dt


def time_first_token(model, tok, system: str, user: str, max_tokens: int):
    """Return (ttft_s, generated_text, prompt_tokens, n_tokens)."""
    from mlx_lm.sample_utils import make_sampler, make_logits_processors  # noqa: PLC0415
    from mlx_lm.generate import generate_step  # noqa: PLC0415
    import mlx.core as mx  # noqa: PLC0415

    sampler = make_sampler(temp=0.0, top_p=1.0)
    logits_processors = make_logits_processors(repetition_penalty=REPETITION_PENALTY)
    prompt_text = build_prompt(system, user)
    from mlx_lm.tokenizer_utils import TokenizerWrapper  # noqa: PLC0415

    if isinstance(tok, TokenizerWrapper):
        prompt_str = tok.apply_chat_template(
            prompt_text, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = tok.encode(prompt_str)
    else:
        prompt_str = system + "\n\n" + user
        prompt_tokens = tok.encode(prompt_str)

    mx.reset_peak_memory()
    t0 = time.monotonic()
    gen = generate_step(
        mx.array(prompt_tokens),
        model,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
    )
    first_token_at = None
    all_tokens: list[int] = []
    for token, _prob in gen:
        if first_token_at is None:
            first_token_at = time.monotonic()
        all_tokens.append(int(token))
        if len(all_tokens) >= max_tokens:
            break
    ttft_s = (first_token_at - t0) if first_token_at is not None else float("nan")
    text = tok.decode(all_tokens) if all_tokens else ""
    return ttft_s, text, prompt_tokens, len(all_tokens)


def time_decode_n_tokens(
    model, tok, prompt_tokens: list[int], n_decode: int = 128
) -> dict:
    """Time prefill then decode for n_decode tokens. Return tok/s + prefill s."""
    from mlx_lm.generate import generate_step  # noqa: PLC0415
    from mlx_lm.sample_utils import make_sampler, make_logits_processors  # noqa: PLC0415
    import mlx.core as mx  # noqa: PLC0415

    sampler = make_sampler(temp=0.0, top_p=1.0)
    logits_processors = make_logits_processors(repetition_penalty=REPETITION_PENALTY)
    mx.reset_peak_memory()

    t_pre = time.monotonic()
    first_token = None
    gen = generate_step(
        mx.array(prompt_tokens),
        model,
        max_tokens=n_decode,
        sampler=sampler,
        logits_processors=logits_processors,
    )
    decode_times: list[float] = []
    for i, (token, _prob) in enumerate(gen):
        now = time.monotonic()
        if first_token is None:
            first_token = now
            t_first = now - t_pre
        else:
            decode_times.append(now - t_pre)
        if i + 1 >= n_decode:
            break
    t_end = time.monotonic()

    if not decode_times:
        decode_tps = float("nan")
        prefill_s = float("nan")
    else:
        per_token = []
        prev = t_first
        for t in decode_times:
            per_token.append(t - prev)
            prev = t
        decode_tps = 1.0 / statistics.mean(per_token) if per_token else float("nan")
        prefill_s = t_first
    return {
        "prefill_s": prefill_s,
        "decode_tps": decode_tps,
        "decode_count": len(decode_times),
        "peak_mem_gb": mx.get_peak_memory() / 1e9,
        "t_total_s": t_end - t_pre,
    }


def synthesize_prompt_tokens(tok, n_target: int) -> list[int]:
    """Synthesize a real (coherent-ish) prompt of n_target tokens.

    Strategy: build a long factual paragraph and truncate. This avoids the
    "all-padding tokens" pathology where synthetic IDs don't reflect realistic
    attention patterns.
    """
    seed = (
        "The history of computing begins with early mechanical calculators, "
        "telescopes and clocks, but the modern computer age began with "
        "electronic digital computers in the mid twentieth century. "
        "These machines relied on vacuum tubes, transistors, and later "
        "integrated circuits. Programming languages evolved from machine "
        "code to assembly to high level languages such as Fortran, "
        "Lisp, C, and many successors. Networking protocols like TCP/IP "
        "made distributed computing practical, and the World Wide Web "
        "turned a research network into a global communications substrate. "
        "Today the field continues to evolve with machine learning, "
        "specialized accelerators, and a growing emphasis on privacy, "
        "open weights, and local execution. Open source communities "
        "publish reproducible code and reproducible measurements, "
        "while benchmarks quantify quality, latency, and energy use."
    )
    # Tokenize then repeat with slight separators until we hit the target
    toks = tok.encode(seed)
    out: list[int] = []
    while len(out) < n_target:
        out.extend(toks)
    return out[:n_target]


def evaluate_card(
    text: str, components_module, expected_kind: str
) -> dict:
    """Use assistant components.validate_components to score card adherence."""
    if components_module is None:
        return {"score": "skipped", "reason": "no validator"}
    # Extract fenced block (```assistant-ui ...```)
    fences = re.findall(r"```assistant-ui\s*\n(.*?)```", text, re.DOTALL)
    prose_only = not fences
    invalid_json = False
    leaked_fence = False
    valid_components: list[dict] = []
    for raw in fences:
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                comps = components_module.validate_components(payload)
                valid_components.extend(comps)
        except Exception:  # noqa: BLE001
            invalid_json = True
    # Plain text could still contain markdown structure
    has_checklist = bool(re.search(r"^\s*[-*]\s+\[[ x]\]", text, re.MULTILINE))
    has_pipe_table = bool(re.search(r"\n\|[\s\-:|]+\|\n", text))
    has_timeline = bool(re.search(r"\b(19|20)\d{2}\b", text))
    return {
        "fences_found": len(fences),
        "invalid_json": invalid_json,
        "prose_only": prose_only,
        "valid_components": [c.get("type") for c in valid_components],
        "has_checklist_md": has_checklist,
        "has_pipe_table_md": has_pipe_table,
        "has_timeline_md": has_timeline,
        "expected_kind": expected_kind,
        "hit": (
            (expected_kind == "prose" and prose_only and not invalid_json)
            or (expected_kind == "card" and (len(valid_components) > 0 or has_checklist or has_pipe_table))
        ),
    }


def evaluate_gsm8k(text: str, gold: int) -> bool:
    """Parse the final integer from the response and compare."""
    # Look for the last number in the text
    nums = re.findall(r"-?\d[\d,]*", text)
    if not nums:
        return False
    last = nums[-1].replace(",", "")
    try:
        return int(last) == gold
    except ValueError:
        return False


def evaluate_ife(text: str, spec: dict) -> bool:
    """Apply the deterministic pass/fail check."""
    t = text.strip()
    check = spec["pass_check"]
    if check == "equals":
        expected = spec["expected"]
        if spec.get("case_insensitive"):
            return t.upper() == expected.upper()
        return t == expected
    if check == "line_equals":
        first_line = (t.splitlines() or [""])[0].strip()
        return first_line == spec["expected"]
    if check == "lines":
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        expected = [str(x).lower() if spec.get("case_insensitive") else str(x) for x in spec["expected"]]
        actual = [ln.lower() if spec.get("case_insensitive") else ln for ln in lines]
        return actual == expected
    if check == "sentence_count":
        # Split on . ? ! ignoring trailing whitespace
        sents = [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
        return len(sents) == spec["expected"]
    if check == "comma_count":
        # count commas in first line
        first = (t.splitlines() or [""])[0]
        return first.count(",") == spec["expected"]
    if check == "bullet_count":
        bullets = [ln for ln in t.splitlines() if re.match(r"^\s*[-*]\s", ln)]
        return len(bullets) == spec["expected"]
    if check == "json_keys":
        try:
            obj = json.loads(t)
        except Exception:  # noqa: BLE001
            return False
        if set(obj.keys()) != set(spec["expected_keys"]):
            return False
        for k, v in spec["expected_values"].items():
            if obj.get(k) != v:
                return False
        return True
    if check == "json_array":
        try:
            obj = json.loads(t)
        except Exception:  # noqa: BLE001
            return False
        return obj == spec["expected_values"]
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--model-label", default="")
    ap.add_argument("--prompts-dir", default=str(PROMPTS_DEFAULT))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--n-decode", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-quality", action="store_true",
                    help="skip GSM8K/IFE (perf-only run)")
    ap.add_argument("--skip-cards", action="store_true",
                    help="skip card adherence run")
    args = ap.parse_args()

    snapshot = (
        HF_CACHE_ROOT
        / f"models--{args.model_id.replace('/', '--')}"
        / "snapshots"
        / args.revision
    )
    model_path = str(snapshot)
    if not snapshot.exists():
        print(f"ERROR snapshot missing: {snapshot}", file=sys.stderr)
        return 2

    # Reset everything; record idle mem
    mem_before = meminfo_available_mib()
    rss_before = rss_mib()

    # Cold load
    t0 = time.monotonic()
    model, tok, cold_load_s = load_model(model_path)
    print(f"LOAD_COLD {cold_load_s:.2f}s peak={model.__class__.__name__}")
    import mlx.core as mx  # noqa: PLC0415
    peak_after_load_gb = mx.get_peak_memory() / 1e9
    rss_after_load = rss_mib()

    # Warm restart = second load in the same process (page cache warm)
    # Skip the model; just reload
    _, _, warm_load_s = warm_load_model(model_path)

    # 2 warmups, 5 measured
    sample_user = "Describe what is on the desk in front of you."
    chat_system = SCHEMA_PROMPT_COMPACT
    # Warmups (results discarded)
    for _ in range(2):
        ttft, _, _, n = time_first_token(model, tok, chat_system, sample_user, max_tokens=64)
    measured: list[dict] = []
    for i in range(5):
        mx.reset_peak_memory()
        mem_pre = meminfo_available_mib()
        rss_pre = rss_mib()
        ttft, text, ptoks, n_tok = time_first_token(model, tok, chat_system, sample_user, max_tokens=256)
        mem_post = meminfo_available_mib()
        rss_post = rss_mib()
        peak = mx.get_peak_memory() / 1e9
        measured.append({
            "iteration": i + 1,
            "ttft_s": ttft,
            "n_tokens_generated": n_tok,
            "prompt_tokens": len(ptoks),
            "peak_mem_gb": peak,
            "rss_mib_pre": rss_pre,
            "rss_mib_post": rss_post,
            "mem_avail_mib_pre": mem_pre,
            "mem_avail_mib_post": mem_post,
            "mem_avail_delta_mib": mem_post - mem_pre,
        })

    # Prefill @512 and @2048
    prompt_512 = synthesize_prompt_tokens(tok, 512)
    prompt_2048 = synthesize_prompt_tokens(tok, 2048)
    # 2 warmup decode runs to settle
    time_decode_n_tokens(model, tok, prompt_2048, n_decode=16)
    time_decode_n_tokens(model, tok, prompt_2048, n_decode=16)
    decode_512 = [time_decode_n_tokens(model, tok, prompt_512, n_decode=args.n_decode) for _ in range(5)]
    decode_2048 = [time_decode_n_tokens(model, tok, prompt_2048, n_decode=args.n_decode) for _ in range(5)]

    def stats(values):
        clean = [v for v in values if v == v]  # drop NaN
        if not clean:
            return {"mean": None, "median": None, "min": None, "max": None, "stdev": None, "n": 0}
        return {
            "mean": statistics.mean(clean),
            "median": statistics.median(clean),
            "min": min(clean),
            "max": max(clean),
            "stdev": statistics.stdev(clean) if len(clean) > 1 else 0.0,
            "n": len(clean),
        }

    prefill_512 = stats([d["prefill_s"] for d in decode_512])
    prefill_2048 = stats([d["prefill_s"] for d in decode_2048])
    decode_512_tps = stats([d["decode_tps"] for d in decode_512])
    decode_2048_tps = stats([d["decode_tps"] for d in decode_2048])
    ttft_stats = stats([m["ttft_s"] for m in measured])

    # Resume from existing checkpoint if present (so a reboot/restart can
    # pick up partial card/gsm/ife results without rerunning them).
    card_results = []
    gsm_results = []
    ife_results = []
    existing = Path(args.out)
    if existing.exists() and existing.stat().st_size > 100:
        try:
            old = json.loads(existing.read_text())
            if old.get("model_id") == args.model_id and old.get("revision") == args.revision:
                card_results = old.get("card_results", [])
                gsm_results = old.get("gsm_results", [])
                ife_results = old.get("ife_results", [])
                print(f"RESUMED checkpoint: cards={len(card_results)} gsm={len(gsm_results)} ife={len(ife_results)}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"WARN resume failed: {exc}", file=sys.stderr)

    # Save partial after perf work is done (before cards/quality)
    _save_partial(args.out, locals())

    # Card adherence: 16 frozen prompts
    if not args.skip_cards:
        prompts_path = Path(args.prompts_dir) / "cards_16.jsonl"
        with prompts_path.open() as fh:
            prompts = [json.loads(ln) for ln in fh if ln.strip()]
        from mlx_lm.sample_utils import make_sampler, make_logits_processors  # noqa: PLC0415
        from mlx_lm.generate import generate_step  # noqa: PLC0415
        from mlx_lm.tokenizer_utils import TokenizerWrapper  # noqa: PLC0415

        sampler = make_sampler(temp=0.0, top_p=1.0)
        logits_processors = make_logits_processors(repetition_penalty=REPETITION_PENALTY)

        def chat_generate(system: str, user: str, max_tokens: int) -> str:
            messages = build_prompt(system, user)
            if isinstance(tok, TokenizerWrapper):
                prompt_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_tokens = tok.encode(prompt_str)
            else:
                prompt_tokens = tok.encode(system + "\n\n" + user)
            out_tokens: list[int] = []
            for token, _ in generate_step(
                mx.array(prompt_tokens), model,
                max_tokens=max_tokens, sampler=sampler,
                logits_processors=logits_processors,
            ):
                out_tokens.append(int(token))
                if len(out_tokens) >= max_tokens:
                    break
            return tok.decode(out_tokens)

        # 2 warmups on a sample
        for _ in range(2):
            chat_generate(chat_system, "Hi", 8)

        done_card_indices = {c.get("index") for c in card_results if "index" in c}
        for p in prompts:
            if p["index"] in done_card_indices:
                continue  # already done in a previous run
            uses_full = bool(re.search(
                r"\b(charts?|graphs?|forms?|decisions?|options?|facts?|sources?)\b",
                p["prompt"], re.IGNORECASE))
            sys_prompt = SCHEMA_PROMPT if uses_full else SCHEMA_PROMPT_COMPACT
            try:
                text = chat_generate(sys_prompt, p["prompt"], args.max_tokens)
            except Exception as exc:  # noqa: BLE001
                card_results.append({
                    "index": p["index"], "error": str(exc)[:200],
                    "expect": p["expect"],
                })
                continue
            verdict = evaluate_card(text, sys.modules.get("mlx_omarchy_assistant.components"),
                                    p["expect"])
            verdict.update({
                "index": p["index"], "expect": p["expect"],
                "category": p["category"], "uses_full_schema": uses_full,
                "text_excerpt": text[:300],
            })
            card_results.append(verdict)
            _save_partial(args.out, locals())

    # GSM8K: 20 frozen items
    gsm_results = []
    if not args.skip_quality:
        gsm_path = Path(args.prompts_dir) / "gsm8k_20.jsonl"
        with gsm_path.open() as fh:
            gsm_prompts = [json.loads(ln) for ln in fh if ln.strip()]
        from mlx_lm.sample_utils import make_sampler, make_logits_processors  # noqa: PLC0415
        from mlx_lm.generate import generate_step  # noqa: PLC0415
        from mlx_lm.tokenizer_utils import TokenizerWrapper  # noqa: PLC0415

        sampler = make_sampler(temp=0.0, top_p=1.0)
        logits_processors = make_logits_processors(repetition_penalty=REPETITION_PENALTY)

        def chat_generate_local(system: str, user: str, max_tokens: int) -> str:
            messages = build_prompt(system, user)
            if isinstance(tok, TokenizerWrapper):
                prompt_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_tokens = tok.encode(prompt_str)
            else:
                prompt_tokens = tok.encode(system + "\n\n" + user)
            out_tokens: list[int] = []
            for token, _ in generate_step(
                mx.array(prompt_tokens), model,
                max_tokens=max_tokens, sampler=sampler,
                logits_processors=logits_processors,
            ):
                out_tokens.append(int(token))
                if len(out_tokens) >= max_tokens:
                    break
            return tok.decode(out_tokens)

        gsm_system = (
            "You are a careful math tutor. Solve the problem step by step, "
            "then end your reply with the final answer as a single line:\n"
            "Final answer: <number>"
        )
        done_gsm_indices = {g.get("index") for g in gsm_results if "index" in g}
        for g in gsm_prompts:
            if g["index"] in done_gsm_indices:
                continue
            try:
                text = chat_generate_local(gsm_system, g["q"], 512)
            except Exception as exc:  # noqa: BLE001
                gsm_results.append({"index": g["index"], "error": str(exc)[:200]})
                continue
            ok = evaluate_gsm8k(text, g["gold"])
            gsm_results.append({"index": g["index"], "gold": g["gold"],
                                "pass": ok, "excerpt": text[-200:]})
            _save_partial(args.out, locals())

    # IFE: 20 frozen items
    if not args.skip_quality:
        ife_path = Path(args.prompts_dir) / "ife_20.jsonl"
        with ife_path.open() as fh:
            ife_prompts = [json.loads(ln) for ln in fh if ln.strip()]
        ife_system = (
            "You follow instructions exactly. Reply ONLY with what the "
            "instruction asks; do not add commentary."
        )
        done_ife_indices = {i.get("index") for i in ife_results if "index" in i}
        for it in ife_prompts:
            if it["index"] in done_ife_indices:
                continue
            try:
                text = chat_generate_local(ife_system, it["instruction"], 256)
            except Exception as exc:  # noqa: BLE001
                ife_results.append({"index": it["index"], "error": str(exc)[:200]})
                continue
            ok = evaluate_ife(text, it)
            ife_results.append({"index": it["index"], "pass": ok,
                                "check": it["pass_check"],
                                "excerpt": text[:160]})
            _save_partial(args.out, locals())

    # Build summary
    card_pass = sum(1 for c in card_results if c.get("hit"))
    card_total = len([c for c in card_results if "expect" in c])
    gsm_pass = sum(1 for g in gsm_results if g.get("pass"))
    ife_pass = sum(1 for i in ife_results if i.get("pass"))

    summary = {
        "model_id": args.model_id,
        "model_label": args.model_label or args.model_id,
        "revision": args.revision,
        "provenance": provenance_line(),
        "cold_load_s": cold_load_s,
        "warm_load_s": warm_load_s,
        "peak_mem_after_load_gb": peak_after_load_gb,
        "rss_after_load_mib": rss_after_load,
        "mem_before_run_mib": mem_before,
        "rss_before_run_mib": rss_before,
        "ttft_170sys_256tok": ttft_stats,
        "ttft_measured": measured,
        "prefill_512": prefill_512,
        "prefill_2048": prefill_2048,
        "decode_512": decode_512_tps,
        "decode_2048": decode_2048_tps,
        "card_pass": card_pass,
        "card_total": card_total,
        "card_results": card_results,
        "gsm_pass": gsm_pass,
        "gsm_total": len(gsm_results),
        "gsm_results": gsm_results,
        "ife_pass": ife_pass,
        "ife_total": len(ife_results),
        "ife_results": ife_results,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps({
        "model": args.model_id,
        "label": args.model_label,
        "revision": args.revision,
        "cold_load_s": round(cold_load_s, 2),
        "warm_load_s": round(warm_load_s, 2),
        "peak_mem_gb": round(peak_after_load_gb, 2),
        "ttft_p50_s": ttft_stats["median"],
        "prefill_512_p50_s": prefill_512["median"],
        "prefill_2048_p50_s": prefill_2048["median"],
        "decode_512_tps": decode_512_tps["median"],
        "decode_2048_tps": decode_2048_tps["median"],
        "card_valid": f"{card_pass}/{card_total}",
        "gsm8k": f"{gsm_pass}/{len(gsm_results)}",
        "ife": f"{ife_pass}/{len(ife_results)}",
        "provenance": provenance_line(),
    }, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())