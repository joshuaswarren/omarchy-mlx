#!/usr/bin/env python3
"""wheel_card_row.py — one-command packaged-wheel qualification that prints a
Parity-card row.

Runs the card's exact cell on a packaged mlx-omarchy wheel: one discarded
warm-up plus five timed runs of decode-128 and single-forward prefill at one
context, greedy output hashed per run, then prints a ready-to-paste README
"Parity card" row and writes the full JSON receipt.

    # qualify a packaged wheel in a fresh venv (the intended contributor path):
    python3 scripts/wheel_card_row.py \
        --wheel dist/mlx_omarchy-0.32.4.dev*-py3-none-any.whl \
        --machine "M4 Pro MacBook Pro 14-inch (20-core GPU)" --model 4b

    # qualify whatever mlx is already importable in this environment:
    python3 scripts/wheel_card_row.py --installed --model 4b \
        --machine "16-inch M1 Max MacBook Pro (32-core GPU)"

    # fill the macOS columns when you have the stock-mlx numbers too:
    ... --macos-decode 104.53 --macos-prefill 660.26 --mesa 53ad9af196f5

The card's fixed rules are enforced, not options: MLX_OMARCHY_DEFER_TOKENS=0,
greedy sampling, the same prompt, 5 timed runs + 1 discarded warm-up, and an
asserted served-model revision for the pinned snapshots.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
from itertools import islice
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from statistics import median
from time import perf_counter

MODELS = {
    # label -> (HF repo, pinned revision, README display name)
    "2b": ("SiddhJagani/Qwen3.8-2B-mlx-4Bit",
           "0867d98bfb174b042d88461c0e7c97b86b34b381"),
    "4b": ("mlx-community/Qwen3-4B-Instruct-2507-4bit",
           "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"),
    "9b": ("nvythong/Qwen3.8-9B-Distill-mlx-4Bit",
           "e827c31fbd588828f43180a87ab34415a6d8a4bf"),
}
PROMPT = "The history of France begins in antiquity."
DECODE_TOKENS = 128          # card cell: "decode 128"
SAMPLES = 5                  # 5 timed runs after 1 discarded warm-up
CTX_CHOICES = (512, 2048)


def digest(ids) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def live_boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return ""


def check_guard() -> None:
    """Card rule: deferral off. An explicit non-zero setting is a hard stop;
    an unset variable is pinned to 0 in-process before the cell runs."""
    defer = os.environ.get("MLX_OMARCHY_DEFER_TOKENS")
    if defer is None:
        os.environ["MLX_OMARCHY_DEFER_TOKENS"] = "0"
    elif defer != "0":
        raise SystemExit(
            f"REFUSED: MLX_OMARCHY_DEFER_TOKENS={defer!r}; card cells run with =0. "
            "Re-run inside `env MLX_OMARCHY_DEFER_TOKENS=0 ...`.")


def run_cell(model_label: str, ctx: int, out_dir: Path, local_only: bool) -> dict:
    check_guard()
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache

    mx.set_default_device(mx.gpu)
    repo, revision = MODELS[model_label]
    snap = Path(snapshot_download(repo, revision=revision,
                                  local_files_only=local_only))
    model, tokenizer = load(str(snap))
    base = tokenizer.encode(PROMPT)
    if not base:
        raise SystemExit("tokenizer produced no ids for the card prompt")

    core_so = Path(os.path.realpath(mx.__file__)).parent / "lib" / "libmlx.so"
    meta = {
        "runner": "scripts/wheel_card_row.py",
        "runner_sha256": sha256_file(Path(__file__)),
        "repo": repo, "revision": revision, "prompt": PROMPT,
        "device": mx.device_info(), "os": platform.platform(),
        "boot_id": live_boot_id(),
        "defer_tokens": os.environ.get("MLX_OMARCHY_DEFER_TOKENS"),
        "decode_tokens": DECODE_TOKENS, "warmups_discarded": 1, "samples": SAMPLES,
        "core_so_sha256": sha256_file(core_so) if core_so.is_file() else None,
        "mlx_lm": metadata.version("mlx-lm"),
        "mlx": next((metadata.version(d) for d in ("mlx-omarchy", "mlx")
                     if _has(d)), "?"),
    }

    ids = (base * (ctx // len(base) + 1))[:ctx]
    tokens = mx.array(ids)
    result = {**meta, "ctx": ctx, "input_sha256": digest(ids)}
    for mode in ("decode", "prefill"):
        samples, outputs = [], []
        for rep in range(SAMPLES + 1):
            cache = make_prompt_cache(model)
            if mode == "decode":
                with_stream = generate_step(tokens, model,
                                            max_tokens=DECODE_TOKENS,
                                            prompt_cache=cache)
                first, _ = next(with_stream)
                seq = [int(first)]
                t0 = perf_counter()
                seq.extend(int(t) for t, _ in islice(with_stream,
                                                     DECODE_TOKENS - 1))
                assert len(seq) == DECODE_TOKENS
                mx.synchronize()
                rate = (DECODE_TOKENS - 1) / (perf_counter() - t0)
            else:
                t0 = perf_counter()
                y = model(tokens[None], cache=cache)
                mx.eval(y)
                rate = ctx / (perf_counter() - t0)
                seq = [int(mx.argmax(y[0, -1]))]
            if rep:
                samples.append(rate)
                outputs.append(seq)
        mid = median(samples)
        result[mode] = {
            "timed_tok_s": samples, "median_tok_s": mid,
            "first_vs_median_pct": 100 * (samples[0] / mid - 1),
            "last_vs_median_pct": 100 * (samples[-1] / mid - 1),
            "output_sha256": [digest(s) for s in outputs],
            "output_ids": outputs,
            "all_outputs_identical": len({digest(s) for s in outputs}) == 1,
        }
        print(f"{mode} {model_label} ctx{ctx}: {mid:.2f} tok/s "
              f"(runs {['%.2f' % s for s in samples]})", flush=True)
    return result


def _has(dist: str) -> bool:
    try:
        metadata.version(dist)
        return True
    except metadata.PackageNotFoundError:
        return False


def mesa_pin() -> str:
    """Best-effort Mesa build pin from the live Vulkan driver, if available."""
    try:
        import mlx.core as mx
        info = str(mx.device_info())
        m = re.search(r"(?:borealis|honeykrisp|mesa)[^\"']*?([0-9a-f]{10,12})\b",
                      info, re.IGNORECASE)
        if m:
            return m.group(1)
    except Exception:
        pass
    return ""


def print_row(machine: str, model_label: str, ctx: int, result: dict,
              macos_decode: float | None, macos_prefill: float | None,
              mesa: str, pr: str) -> None:
    repo, revision = MODELS[model_label]
    dec = result["decode"]["median_tok_s"]
    pre = result["prefill"]["median_tok_s"]
    ids12 = result["decode"]["output_sha256"][0][:12]
    if macos_decode:
        share = dec / macos_decode
        sign = "+" if share >= 1 else "−"
        share_s = f"{share*100:.1f}% ({sign}{abs(share-1)*100:.1f}%)"
    else:
        share_s = "pending"
    mac_dec = f"{macos_decode:.2f} tok/s" if macos_decode else "pending"
    mac_pre = f"{macos_prefill:.2f} tok/s" if macos_prefill else "pending"
    runs = f"{SAMPLES} timed / {SAMPLES}+1 runs"
    print("\nCard row (paste under the Parity card table):\n")
    print(f"| {machine} | `{repo}` (`{revision[:8]}`) | {ctx} | "
          f"{dec:.2f} tok/s | {pre:.2f} tok/s | {mac_dec} | {mac_pre} | "
          f"{share_s} | {runs} | `{ids12}` | {pr or 'pending'} | "
          f"`{mesa or 'pending'}` |")


def make_venv_and_reexec(args) -> None:
    venv = Path(args.venv).resolve()
    py = venv / "bin" / "python3"
    if not py.exists():
        import shutil
        uv = shutil.which("uv")
        print(f"[1/3] venv {venv} ({'uv' if uv else 'venv'})", flush=True)
        if uv:
            subprocess.run([uv, "venv", str(venv)], check=True)
        else:
            subprocess.run([sys.executable, "-m", "venv", str(venv)],
                           check=True)
        pkgs = [str(Path(args.wheel).resolve()), "mlx-lm", "huggingface_hub"]
        print("[2/3] install wheel + mlx-lm", flush=True)
        if uv:
            subprocess.run([uv, "pip", "install", "--python", str(py),
                            "--quiet", *pkgs], check=True)
        else:
            subprocess.run([str(py), "-m", "ensurepip", "--upgrade"],
                           check=True)
            subprocess.run([str(py), "-m", "pip", "install", "--quiet",
                            *pkgs], check=True)
    env = dict(os.environ, MLX_OMARCHY_DEFER_TOKENS="0")
    rest = ["--installed", "--machine", args.machine, "--model", args.model,
            "--ctx", str(args.ctx), "--out", str(args.out)]
    for flag, val in (("--macos-decode", args.macos_decode),
                      ("--macos-prefill", args.macos_prefill),
                      ("--mesa", args.mesa), ("--pr", args.pr)):
        if val:
            rest += [flag, str(val)]
    if args.local_files_only:
        rest.append("--local-files-only")
    print("[3/3] cell run inside the wheel venv", flush=True)
    os.execve(str(py), [str(py), str(Path(__file__).resolve()), *rest], env)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--wheel", help="path to the mlx-omarchy .whl to qualify "
                     "in a fresh venv")
    src.add_argument("--installed", action="store_true",
                     help="qualify the mlx already importable in this env")
    ap.add_argument("--machine", required=True,
                    help='card "Machine" column, e.g. "M4 Pro MacBook Pro '
                         '14-inch (20-core GPU)"')
    ap.add_argument("--model", choices=sorted(MODELS), default="4b")
    ap.add_argument("--ctx", type=int, choices=CTX_CHOICES, default=512)
    ap.add_argument("--out", type=Path, default=Path("wheel-card-out"))
    ap.add_argument("--venv", default="wheel-card-venv")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--macos-decode", type=float, default=None,
                    help="stock-mlx macOS decode tok/s for the same cell")
    ap.add_argument("--macos-prefill", type=float, default=None,
                    help="stock-mlx macOS prefill tok/s for the same cell")
    ap.add_argument("--mesa", default="", help="Mesa build pin (auto-detect "
                    "when omitted)")
    ap.add_argument("--pr", default="", help="PR that produced the wheel")
    a = ap.parse_args()

    if a.wheel:
        make_venv_and_reexec(a)
        return

    check_guard()
    a.out.mkdir(parents=True, exist_ok=True)
    result = run_cell(a.model, a.ctx, a.out, a.local_files_only)
    stem = f"cardrow-{a.model}-ctx{a.ctx}"
    (a.out / f"{stem}.json").write_text(json.dumps(result, indent=2,
                                                    default=str) + "\n")
    print_row(a.machine, a.model, a.ctx, result,
              a.macos_decode, a.macos_prefill,
              a.mesa or mesa_pin(), a.pr)


if __name__ == "__main__":
    main()
