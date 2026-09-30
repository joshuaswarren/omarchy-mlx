#!/usr/bin/env python3
"""Chat-model benchmark for the offline-assistant pair defaults.

One model per process. Greedy, thinking disabled (``enable_thinking=False``
in the chat template, as the coordinator sends), repetition penalty 1.1.
Generation stops at the tokenizer's EOS tokens except for the fixed-length
decode measurement, which ignores EOS on purpose.

Phases, each checkpointed to --out after every item so a killed turn
resumes where it stopped:
  perf    cold/warm load, TTFT for the coordinator card prompt, prefill at
          512 and 2048 tokens, decode over --n-decode tokens, peak memory
  cards   16 frozen prompts, scored with components.validate_components and
          card_promotion.extract_text
  gsm     20 GSM8K test items (proxy, not a benchmark)
  ife     20 deterministic instruction-following checks (proxy)

Paths come from BENCH_REPO_ROOT (default: this checkout) and HF_HUB_CACHE.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("MLX_DISABLE_COMPILE", "1")

HARNESS_VERSION = 2
REPO_ROOT = Path(os.environ.get("BENCH_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
HF_CACHE_ROOT = Path(os.environ.get("HF_HUB_CACHE", str(Path.home() / ".cache" / "huggingface" / "hub")))
sys.path.insert(0, str(REPO_ROOT / "serve"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from mlx_omarchy_assistant.components import (  # noqa: E402
    SCHEMA_PROMPT, ComponentError, validate_components)
from mlx_omarchy_assistant.card_promotion import extract_text  # noqa: E402
from mlx_omarchy_assistant.coordinator import FULL_CARD_CUES, REPETITION_PENALTY  # noqa: E402

SYSTEM_PREFIX = "Answer the user using their supplied facts. "
TTFT_USER = "Describe what is on the desk in front of you."
GSM_SYSTEM = ("You are a careful math tutor. Solve the problem step by step, then end your "
              "reply with the final answer as a single line:\nFinal answer: <number>")
IFE_SYSTEM = "You follow instructions exactly. Reply ONLY with what the instruction asks."
FENCE = "```assistant-ui\n"


def provenance() -> str:
    try:
        import mlx_provenance
        return mlx_provenance.provenance_line(mlx_provenance.installed_provenance())
    except Exception as exc:  # noqa: BLE001
        return f"provenance unavailable: {type(exc).__name__}: {exc}"


def mem_available_mib() -> int:
    m = re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text())
    return int(m.group(1)) // 1024


def rss_mib() -> int:
    m = re.search(r"VmRSS:\s+(\d+)", Path("/proc/self/status").read_text())
    return int(m.group(1)) // 1024


def other_gpu_processes() -> int:
    """Count other processes holding a DRM render node (the shared GPU)."""
    count = 0
    for fd_dir in Path("/proc").glob("[0-9]*/fd"):
        if fd_dir.parent.name == str(os.getpid()):
            continue
        try:
            if any(os.readlink(fd).startswith("/dev/dri/renderD") for fd in fd_dir.iterdir()):
                count += 1
        except OSError:
            continue
    return count


def stats(values):
    vals = [v for v in values if v == v]
    if not vals:
        return None
    return {"median": statistics.median(vals), "max": max(vals), "min": min(vals), "n": len(vals)}


class Bench:
    def __init__(self, model, tok):
        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_logits_processors, make_sampler
        self.mx, self.model, self.tok = mx, model, tok
        self.generate_step = generate_step
        self.sampler = make_sampler(temp=0.0)
        self.processors = make_logits_processors(repetition_penalty=REPETITION_PENALTY)
        self.eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])
        self.min_mem_avail = mem_available_mib()

    def encode_chat(self, system: str, user: str) -> list[int]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
        bos = getattr(self.tok, "bos_token", None)
        return self.tok.encode(text, add_special_tokens=bos is None or not text.startswith(bos))

    def run(self, tokens: list[int], max_tokens: int, stop_on_eos: bool = True):
        """Return (generated ids, per-token wall times from start, stopped_on_eos)."""
        start = time.monotonic()
        ids, times, eos_hit = [], [], False
        gen = self.generate_step(self.mx.array(tokens), self.model, max_tokens=max_tokens,
                                 sampler=self.sampler, logits_processors=self.processors)
        for token, _ in gen:
            token = int(token)
            times.append(time.monotonic() - start)
            if stop_on_eos and token in self.eos:
                eos_hit = True
                break
            ids.append(token)
            if len(ids) >= max_tokens:
                break
        self.min_mem_avail = min(self.min_mem_avail, mem_available_mib())
        return ids, times, eos_hit

    def chat(self, system: str, user: str, max_tokens: int) -> dict:
        ids, times, eos_hit = self.run(self.encode_chat(system, user), max_tokens)
        return {"text": self.tok.decode(ids), "tokens": len(ids),
                "truncated": not eos_hit, "seconds": times[-1] if times else 0.0}


def synthetic_prompt(tok, n: int) -> list[int]:
    seed = tok.encode(
        "The history of computing begins with mechanical calculators and clocks. Electronic digital "
        "computers used vacuum tubes, then transistors and integrated circuits. Programming moved "
        "from machine code to high-level languages, and networks connected machines worldwide. ")
    out = []
    while len(out) < n:
        out.extend(seed)
    return out[:n]


def perf_phase(bench: Bench, args) -> dict:
    mx = bench.mx
    card_system = SYSTEM_PREFIX
    ttft_prompt = bench.encode_chat(card_system, TTFT_USER)
    for _ in range(2):
        bench.run(ttft_prompt, 8)
    ttft = [bench.run(ttft_prompt, 8)[1][0] for _ in range(args.perf_n)]

    def prefill_decode(n_prompt: int, reps: int):
        prompt = synthetic_prompt(bench.tok, n_prompt)
        prefill, decode = [], []
        for _ in range(reps):
            _, times, _ = bench.run(prompt, args.n_decode, stop_on_eos=False)
            prefill.append(times[0])
            decode.append((len(times) - 1) / (times[-1] - times[0]))
        return prefill, decode

    bench.run(synthetic_prompt(bench.tok, 512), 8, stop_on_eos=False)
    p512, d512 = prefill_decode(512, args.perf_n)
    p2048, d2048 = prefill_decode(2048, args.prefill2048_n)
    return {
        "ttft_card_prompt_s": stats(ttft), "ttft_prompt_tokens": len(ttft_prompt),
        "prefill_512_s": stats(p512), "prefill_512_tok_s": 512 / statistics.median(p512),
        "prefill_2048_s": stats(p2048),
        "prefill_2048_tok_s": 2048 / statistics.median(p2048) if p2048 else "skipped (--prefill2048-n 0)",
        "decode_after_512_tok_s": stats(d512), "decode_after_2048_tok_s": stats(d2048),
        "decode_tokens": args.n_decode,
        "peak_mem_gb_mx": mx.get_peak_memory() / 1e9,
        "rss_mib_after_perf": rss_mib(),
    }


def score_card(text: str, user_text: str) -> dict:
    lines = text.splitlines()
    out = {
        "md_checklist": any(re.match(r"\s*[-*]\s+\[[ xX]\]", ln) for ln in lines),
        "md_pipe_table": any(re.match(r"\s*\|?\s*:?-{3,}", ln) for ln in lines),
        "md_numbered_or_bullets": sum(bool(re.match(r"\s*([-*]|\d+[.)])\s+", ln)) for ln in lines) >= 3,
    }
    start = text.find(FENCE)
    if start < 0:
        out["outcome"] = "leaked-fence" if "```assistant-ui" in text else "prose-only"
    else:
        body = text[start + len(FENCE):]
        end = body.find("```")
        if end < 0:
            out["outcome"] = "leaked-fence"
        else:
            try:
                comps = validate_components(json.loads(body[:end]))
                out["outcome"], out["types"] = "valid-card", [c["type"] for c in comps]
            except (ValueError, TypeError, KeyError, ComponentError) as exc:
                out["outcome"], out["error"] = "invalid-json", str(exc)[:160]
    promoted = extract_text(text, user_text)
    if promoted is not None:
        try:
            validate_components({"version": 1, "components": [promoted]})
            out["promotable"] = promoted.get("type")
        except (ValueError, ComponentError):
            pass
    return out


def grade_gsm(text: str, gold: int) -> bool:
    m = re.findall(r"Final answer:\s*\**\s*\$?\s*(-?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    nums = m or re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not nums:
        return False
    try:
        return float(nums[-1].replace(",", "")) == gold
    except ValueError:
        return False


def grade_ife(text: str, spec: dict) -> bool:
    t = text.strip()
    check, want = spec["check"], spec["expected"]
    if check == "equals":
        norm = spec.get("normalize", "")
        if "remove_spaces" in norm:
            t = t.replace(" ", "")
        if "strip_period" in norm:
            t = t.rstrip(".").strip()
        if "casefold" in norm:
            t = t.casefold()
        return t == want
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if check == "line_count_plain_words":
        return len(lines) == want and all(re.fullmatch(r"[A-Za-z][A-Za-z ]*", ln) for ln in lines)
    if check == "lines_equal":
        return lines == want
    if check == "sentence_count":
        return len(lines) >= 1 and len(re.findall(r"[^.!?]+[.]", t)) == want and t.endswith(".")
    if check == "comma_items":
        return len(lines) == 1 and len([p for p in lines[0].split(",") if p.strip()]) == want
    if check == "hyphen_bullets":
        return len(lines) == want and all(ln.startswith("- ") for ln in lines)
    if check == "max_words":
        return 0 < len(t.split()) <= want
    if check == "json_equals":
        try:
            return json.loads(t) == want
        except ValueError:
            return False
    raise ValueError(f"unknown IFE check {check}")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--model-label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts-dir", default=str(REPO_ROOT / "scripts/bench/prompts"))
    ap.add_argument("--max-tokens", type=int, default=700, help="card reply cap")
    ap.add_argument("--n-decode", type=int, default=128)
    ap.add_argument("--perf-n", type=int, default=5)
    ap.add_argument("--prefill2048-n", type=int, default=5)
    ap.add_argument("--phases", default="perf,cards,gsm,ife")
    args = ap.parse_args()
    phases = args.phases.split(",")

    out = Path(args.out)
    state = json.loads(out.read_text()) if out.exists() else {}
    if (state.get("model_id"), state.get("revision"), state.get("harness_version")) != (
            args.model_id, args.revision, HARNESS_VERSION):
        state = {"model_id": args.model_id, "revision": args.revision, "label": args.model_label,
                 "harness_version": HARNESS_VERSION, "turns": []}

    def save():
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        os.replace(tmp, out)
        os.sync()

    snapshot = HF_CACHE_ROOT / f"models--{args.model_id.replace('/', '--')}" / "snapshots" / args.revision
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    turn = {"boot_id": boot_id,
            "loadavg_start": Path("/proc/loadavg").read_text().strip(),
            "other_gpu_processes": other_gpu_processes(),
            "mem_available_mib_before_load": mem_available_mib(),
            "page_cache": ("warm: loaded earlier in this boot"
                           if any(t.get("boot_id") == boot_id and "load_s" in t for t in state["turns"])
                           else "cold or unknown: first load recorded in this boot"),
            "provenance": provenance(), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print(turn["provenance"], flush=True)
    state["turns"].append(turn)
    if "perf" in phases and "perf" not in state and turn["other_gpu_processes"]:
        turn["perf_refused"] = f"{turn['other_gpu_processes']} other process(es) hold the GPU"
        save()
        print("PERF_REFUSED", turn["perf_refused"], flush=True)
        return 4

    import mlx.core as mx
    from mlx_lm import load
    t0 = time.monotonic()
    try:
        model, tok = load(str(snapshot))
    except Exception as exc:  # noqa: BLE001  a model that cannot load is a finding
        turn["load_error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
        save()
        print("LOAD_ERROR", turn["load_error"], flush=True)
        return 3
    turn["load_s"] = time.monotonic() - t0
    turn["peak_mem_gb_after_load"] = mx.get_peak_memory() / 1e9
    save()
    bench = Bench(model, tok)

    if "perf" in phases and "perf" not in state:
        perf = perf_phase(bench, args)
        perf.update({"mem_available_mib_before_load": turn["mem_available_mib_before_load"],
                     "mem_available_min_mib": bench.min_mem_avail})
        perf["mem_available_delta_mib"] = perf["mem_available_mib_before_load"] - perf["mem_available_min_mib"]
        state["perf"] = perf
        save()
        print("PERF", json.dumps(perf), flush=True)

    prompts_dir = Path(args.prompts_dir)
    if "cards" in phases:
        done = state.setdefault("cards", {})
        for p in load_jsonl(prompts_dir / "cards_16.jsonl"):
            if str(p["index"]) in done:
                continue
            full = bool(FULL_CARD_CUES.search(p["prompt"]))
            reply = bench.chat(SYSTEM_PREFIX + (SCHEMA_PROMPT if full else ""),
                               p["prompt"], args.max_tokens)
            done[str(p["index"])] = {"expect": p["expect"], "category": p["category"],
                                     "full_schema": full, **score_card(reply["text"], p["prompt"]),
                                     **{k: reply[k] for k in ("tokens", "truncated", "seconds")},
                                     "text": reply["text"]}
            save()
    if "gsm" in phases:
        done = state.setdefault("gsm", {})
        for g in load_jsonl(prompts_dir / "gsm8k_20.jsonl"):
            if str(g["index"]) in done:
                continue
            reply = bench.chat(GSM_SYSTEM, g["q"], 768)
            done[str(g["index"])] = {"gold": g["gold"], "pass": grade_gsm(reply["text"], g["gold"]),
                                     "truncated": reply["truncated"], "text": reply["text"]}
            save()
    if "ife" in phases:
        done = state.setdefault("ife", {})
        for spec in load_jsonl(prompts_dir / "ife_20.jsonl"):
            if str(spec["index"]) in done:
                continue
            reply = bench.chat(IFE_SYSTEM, spec["instruction"], 256)
            done[str(spec["index"])] = {"pass": grade_ife(reply["text"], spec),
                                        "truncated": reply["truncated"], "text": reply["text"]}
            save()
    turn["peak_mem_gb_turn"] = mx.get_peak_memory() / 1e9
    turn["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save()
    print("DONE", args.model_label, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
