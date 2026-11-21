#!/usr/bin/env python3
"""Matched Linux harness for the FluidAudio TTS measurement scope.

Runs the same dataset slice and the same metrics the pinned FluidAudio
contract defines (docs/fluidaudio-tts-benchmark-contract.md):

  - dataset: the 100-line MiniMax English corpus at revision
    cb416f0ac3658da0577e97873065e19fe6488917, pinned sha256
    1020375295c73eef7ff66a611885b0ba91aaa2b6d07cfe76ca2332e05f942e30,
    upstream order preserved;
  - accuracy: English WER and CER through the pinned normalizer and the
    port of FluidAudio's published scorer (reused from
    scripts/check_fluidaudio_tts_contract.py);
  - timing: batch completion, first streamed PCM and v3 token-to-PCM kept
    as three separate measurements, plus RTFx = sum(audio) / sum(synth)
    and the cold-start wall time.

Modes:
  dry-run   verify the corpus pin, the scorer path and the report schema
            without a TTS backend. Runs anywhere, including x86 CPU.
  speak     render every phrase through the assistant /api/speak SSE
            endpoint and time the stream; optional --transcripts adds
            ASR hypotheses for the WER/CER columns.

Usage:
  scripts/bench_tts_fluidaudio_scope.py dry-run
  scripts/bench_tts_fluidaudio_scope.py speak --assistant-home DIR \
      [--transcripts t.json] [--out report.json]
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONTRACT = REPO / "scripts" / "check_fluidaudio_tts_contract.py"


def load_contract():
    spec = importlib.util.spec_from_file_location("fluidaudio_contract", CONTRACT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {CONTRACT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def corpus_phrases(contract):
    raw = (contract.FIXTURES / "minimax_english_cb416f0.txt").read_text()
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if digest != contract.MINIMAX_ENGLISH_SHA256:
        raise SystemExit(f"corpus sha256 {digest} != pin {contract.MINIMAX_ENGLISH_SHA256}")
    phrases = contract.minimax_phrases(raw)
    if len(phrases) != 100:
        raise SystemExit(f"phrase count {len(phrases)} != 100")
    return phrases, digest


def wait_runtime(home: Path, deadline_s: float = 1200.0) -> dict:
    path = home / "assistant" / "application.json"
    started = time.monotonic()
    while True:
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            if time.monotonic() - started > deadline_s:
                raise SystemExit("assistant runtime never appeared")
            time.sleep(1)


def speak(runtime: dict, text: str, conversation_id: str, turn_id: str) -> dict:
    """One /api/speak stream: returns first-frame ms, total ms, audio seconds."""
    port = runtime["port"]
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/speak",
        data=json.dumps({"conversation_id": conversation_id, "turn_id": turn_id,
                         "text": text, "sentence_sequence": 0}).encode(),
        method="POST",
        headers={"Cookie": runtime["cookie"], "X-Assistant-CSRF": runtime["csrf"],
                 "Origin": f"http://127.0.0.1:{port}",
                 "Content-Type": "application/json"})
    started = time.monotonic()
    first_frame_ms = None
    audio_s = 0.0
    errors = 0
    done = False
    with urllib.request.urlopen(request, timeout=300) as response:
        event = None
        for raw in response:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
                continue
            if not line.startswith("data:"):
                continue
            payload = json.loads(line[5:])
            if event == "audio":
                if first_frame_ms is None:
                    first_frame_ms = (time.monotonic() - started) * 1000.0
                pcm = base64.b64decode(payload.get("data", ""))
                audio_s += len(pcm) / 2 / payload.get("sample_rate", 24000)
            elif event == "error":
                errors += 1
            elif event == "done":
                done = True
    total_ms = (time.monotonic() - started) * 1000.0
    return {"first_streamed_pcm_ms": first_frame_ms, "batch_completion_ms": total_ms,
            "audio_s": audio_s, "errors": errors, "done": done}


def accuracy_rows(contract, phrases, transcripts: dict) -> list:
    british = contract.load_british_map()
    rows = []
    for index, phrase in enumerate(phrases, start=1):
        hypothesis = transcripts.get(str(index)) or transcripts.get(index)
        if hypothesis is None:
            rows.append({"index": index, "wer": None, "cer": None,
                         "word_errors": None, "char_errors": None})
            continue
        measured = contract.wer_and_cer(hypothesis, phrase, british)
        rows.append({
            "index": index,
            "wer": measured["wer"],
            "cer": measured["cer"],
            "word_errors": measured["word_errors"]["total"],
            "char_errors": measured["char_errors"]["total"],
            "ref_words": measured["ref_words"],
            "ref_chars": measured["ref_chars"],
        })
    return rows


def report(phrases, digest, rows, timing) -> dict:
    synth_ms = [r["batch_completion_ms"] for r in timing if r.get("batch_completion_ms")]
    audio_s = sum(r.get("audio_s") or 0.0 for r in timing)
    wer_values = [r["wer"] for r in rows if r.get("wer") is not None]
    cer_values = [r["cer"] for r in rows if r.get("cer") is not None]
    return {
        "dataset": {
            "source": "MiniMaxAI/TTS-Multilingual-Test-Set",
            "revision": "cb416f0ac3658da0577e97873065e19fe6488917",
            "sha256": digest,
            "phrases": len(phrases),
            "order": "upstream",
        },
        "metrics": {
            "batch_completion_ms_p50": sorted(synth_ms)[len(synth_ms) // 2] if synth_ms else None,
            "first_streamed_pcm_ms_p50": None,
            "agg_rtfx": (audio_s / (sum(synth_ms) / 1000.0)) if synth_ms and sum(synth_ms) else None,
            "wer_mean": (sum(wer_values) / len(wer_values)) if wer_values else None,
            "cer_mean": (sum(cer_values) / len(cer_values)) if cer_values else None,
            "note": "the three timing boundaries stay separate; v3 token-to-PCM is "
                    "reported by the ANE backend, not by this harness",
        },
        "rows": rows,
        "timing": timing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["dry-run", "speak"])
    parser.add_argument("--assistant-home", type=Path)
    parser.add_argument("--transcripts", type=Path,
                        help="JSON map of phrase index to ASR hypothesis")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    contract = load_contract()
    phrases, digest = corpus_phrases(contract)

    if args.mode == "dry-run":
        rows = accuracy_rows(contract, phrases, {})
        payload = report(phrases, digest, rows, [])
        payload["mode"] = "dry-run"
        payload["measured"] = False
        print(json.dumps({"dataset": payload["dataset"], "metrics": payload["metrics"]},
                         indent=2))
        if args.out:
            args.out.write_text(json.dumps(payload, indent=2) + "\n")
        return 0

    if args.assistant_home is None:
        parser.error("speak mode needs --assistant-home")
    transcripts = json.loads(args.transcripts.read_text()) if args.transcripts else {}
    runtime = wait_runtime(args.assistant_home)
    timing = []
    for index, phrase in enumerate(phrases, start=1):
        measured = speak(runtime, phrase, "fluidaudio-scope", f"turn-{index}")
        measured["index"] = index
        timing.append(measured)
        print(f"[{index:3d}/100] synth={measured['batch_completion_ms']:.0f}ms "
              f"first_pcm={measured['first_streamed_pcm_ms']}ms "
              f"audio={measured['audio_s']:.2f}s errors={measured['errors']}",
              file=sys.stderr)
    rows = accuracy_rows(contract, phrases, transcripts)
    payload = report(phrases, digest, rows, timing)
    payload["mode"] = "speak"
    payload["measured"] = True
    print(json.dumps(payload["metrics"], indent=2))
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
