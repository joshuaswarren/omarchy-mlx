#!/usr/bin/env python3
"""TTS round-trip scorer: speak sentences, transcribe them, report WER
per sentence.

Fills the V15 "round-trip WER/CER" gap: rendered Kokoro WAVs exist
unscored. The scorer is the audited port of FluidAudio's
TextNormalizer + WERCalculator in
scripts/check_fluidaudio_tts_contract.py, so WER/CER are comparable with
the published benchmark numbers.

Modes:
  score-only   --sentences FILE --hypotheses FILE (JSONL {index, text})
  wav ingest   --wav-dir DIR --sentences FILE
               WAVs named <index>.wav (1-based, any zero padding);
               each is transcribed then scored
  render       --render --wav-dir DIR   synthesize sentences to DIR
               first (mlx_audio Kokoro), then transcribe + score

Output: --out DIR gets per-sentence.jsonl and summary.json; summary is
also printed to stdout.

Hardware modes need mlx/mlx_audio and defer those imports so the
score-only path runs on a stdlib interpreter.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check_fluidaudio_tts_contract as contract  # noqa: E402

MINIMAX_FIXTURE = (
    HERE.parent / "tests" / "fixtures" / "fluidaudio_tts"
    / "minimax_english_cb416f0.txt"
)
PARAKEET_REPO = "mlx-community/parakeet-tdt-0.6b-v3"
PARAKEET_REVISION = "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15"


def load_sentences(path) -> list:
    """One sentence per line; MiniMax `<file>|` form and blank lines
    handled by the contract's minimax_phrases. None -> vendored
    MiniMax English 100-phrase fixture."""
    raw = Path(path).read_text() if path else MINIMAX_FIXTURE.read_text()
    return contract.minimax_phrases(raw)


def load_hypotheses(path) -> list:
    """JSONL rows {"index": <1-based int>, "text": <hypothesis>}."""
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def resolve_wavs(wav_dir, n_sentences: int) -> list:
    """Map wav files to sentence indices by the leading integer of the
    stem (1.wav, 001.wav, english_042.wav all index by their digits).
    Returns a list of length n_sentences with Path or None."""
    out = [None] * n_sentences
    for wav in sorted(Path(wav_dir).glob("*.wav")):
        m = re.search(r"(\d+)", wav.stem)
        if not m:
            continue
        idx = int(m.group(1))
        if 1 <= idx <= n_sentences and out[idx - 1] is None:
            out[idx - 1] = wav
    return out


def score_sentences(sentences: list, hypotheses: list, british_map: dict) -> dict:
    """Score hypotheses rows {index, text} against sentences. Every
    sentence index 1..N must appear exactly once."""
    by_index = {r["index"]: r["text"] for r in hypotheses}
    rows = []
    total_we = total_ce = total_rw = total_rc = scored = empty = 0
    for i, ref in enumerate(sentences, 1):
        if i not in by_index:
            raise ValueError(f"no hypothesis for index {i}")
        hyp = by_index[i]
        s = contract.wer_and_cer(hyp, ref, british_map)
        rows.append({
            "index": i, "reference": ref, "hypothesis": hyp,
            "normalized_reference": s["normalized_reference"],
            "normalized_hypothesis": s["normalized_hypothesis"],
            "wer": s["wer"], "cer": s["cer"],
            "word_errors": s["word_errors"], "char_errors": s["char_errors"],
            "ref_words": s["ref_words"], "ref_chars": s["ref_chars"],
            "empty": not s["normalized_hypothesis"],
        })
        total_we += s["word_errors"]["total"]
        total_ce += s["char_errors"]["total"]
        total_rw += s["ref_words"]
        total_rc += s["ref_chars"]
        scored += 1
        empty += not s["normalized_hypothesis"]
    wers = [r["wer"] for r in rows]
    cers = [r["cer"] for r in rows]
    return {
        "rows": rows,
        "summary": {
            "sentences": len(sentences),
            "scored": scored,
            "errors": 0,
            "empty_hypotheses": empty,
            "corpus_wer": total_we / total_rw if total_rw else None,
            "corpus_cer": total_ce / total_rc if total_rc else None,
            "mean_per_sentence_wer": statistics.mean(wers) if wers else None,
            "median_per_sentence_wer": statistics.median(wers) if wers else None,
            "mean_per_sentence_cer": statistics.mean(cers) if cers else None,
        },
    }


def roundtrip(sentences: list, wavs: list, transcribe, british_map: dict) -> dict:
    """Score a wav-per-sentence set. `wavs` is the resolved list
    (resolve_wavs output or a raw list to resolve against len(sentences));
    `transcribe(path) -> str`."""
    wavs = _resolve_from_list(wavs, len(sentences))
    rows = []
    total_we = total_ce = total_rw = total_rc = scored = errors = empty = 0
    wers, cers = [], []
    for i, (ref, wav) in enumerate(zip(sentences, wavs), 1):
        if wav is None:
            rows.append({"index": i, "reference": ref,
                         "error": f"no wav for index {i}"})
            errors += 1
            continue
        try:
            hyp = transcribe(wav)
        except Exception as exc:  # keep the row, mark the error
            rows.append({"index": i, "reference": ref, "wav": str(wav),
                         "error": f"transcribe failed: {exc}"})
            errors += 1
            continue
        s = contract.wer_and_cer(hyp, ref, british_map)
        rows.append({
            "index": i, "reference": ref, "hypothesis": hyp,
            "wav": str(wav),
            "normalized_reference": s["normalized_reference"],
            "normalized_hypothesis": s["normalized_hypothesis"],
            "wer": s["wer"], "cer": s["cer"],
            "word_errors": s["word_errors"], "char_errors": s["char_errors"],
            "ref_words": s["ref_words"], "ref_chars": s["ref_chars"],
            "empty": not s["normalized_hypothesis"],
        })
        total_we += s["word_errors"]["total"]
        total_ce += s["char_errors"]["total"]
        total_rw += s["ref_words"]
        total_rc += s["ref_chars"]
        scored += 1
        empty += not s["normalized_hypothesis"]
        wers.append(s["wer"])
        cers.append(s["cer"])
    return {
        "rows": rows,
        "summary": {
            "sentences": len(sentences),
            "scored": scored,
            "errors": errors,
            "empty_hypotheses": empty,
            "corpus_wer": total_we / total_rw if total_rw else None,
            "corpus_cer": total_ce / total_rc if total_rc else None,
            "mean_per_sentence_wer": statistics.mean(wers) if wers else None,
            "median_per_sentence_wer": statistics.median(wers) if wers else None,
            "mean_per_sentence_cer": statistics.mean(cers) if cers else None,
        },
    }


def _resolve_from_list(wavs: list, n: int) -> list:
    out = [None] * n
    for wav in wavs:
        if wav is None:
            continue
        m = re.search(r"(\d+)", Path(wav).stem)
        if m:
            idx = int(m.group(1))
            if 1 <= idx <= n and out[idx - 1] is None:
                out[idx - 1] = Path(wav)
    return out


def _mlx_transcriber(model_path: str | None):
    """Parakeet TDT via mlx_audio (hardware path; lazy import).

    Same call shape as serve/mlx_omarchy_assistant/gpu_stt_worker.py:
    float32 16 kHz mono samples to ``model.generate(audio,
    verbose=False).text``, a silence prime (the first generate after
    load returns ""), and one noise-padded retry on an empty transcript.
    ffmpeg decodes the wav, matching the ASR harness decode."""
    import subprocess

    import mlx.core as mx
    from mlx_audio.stt.utils import load_model

    model = load_model(model_path or PARAKEET_REPO)
    model.generate(mx.zeros((16_000,), dtype=mx.float32), verbose=False)

    import numpy as np

    rng = np.random.default_rng(0)
    noise = mx.array((rng.standard_normal(2400) * 1e-4).astype(np.float32))

    def transcribe(wav_path: Path) -> str:
        pcm = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(wav_path),
             "-ar", "16000", "-ac", "1", "-f", "f32le", "pipe:1"],
            capture_output=True, check=True).stdout
        audio = mx.array(np.frombuffer(pcm, dtype="<f4"))
        text = str(getattr(model.generate(audio, verbose=False), "text", "") or "")
        if not text:
            n = noise.shape[0] // 2
            audio = mx.concatenate([noise[:n], audio, noise[-n:]])
            text = str(getattr(model.generate(audio, verbose=False), "text", "") or "")
        return text

    return transcribe


def _render_wavs(sentences: list, wav_dir: Path, voice: str, lang_code: str) -> None:
    """mlx_audio Kokoro one-shot render (hardware path; lazy import)."""
    from mlx_audio.tts.utils import load_model

    model = load_model("mlx-community/Kokoro-82M-bf16")
    wav_dir.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(sentences, 1):
        chunks = [r.audio for r in model.generate(text=text, voice=voice,
                                                  lang_code=lang_code)]
        import numpy as np

        audio = np.concatenate([np.asarray(c).reshape(-1) for c in chunks])
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
        import wave as _wave

        with _wave.open(str(wav_dir / f"{i:03d}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(pcm.tobytes())


def _write_output(out_dir: Path, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per-sentence.jsonl").open("w") as f:
        for row in result["rows"]:
            f.write(json.dumps(row) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(result["summary"], indent=1) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sentences", type=Path, default=None,
                    help="one sentence per line (default: vendored MiniMax English 100)")
    ap.add_argument("--hypotheses", type=Path, default=None,
                    help="JSONL {index, text} — score-only mode, no audio")
    ap.add_argument("--wav-dir", type=Path, default=None,
                    help="directory of <index>.wav files to transcribe + score")
    ap.add_argument("--render", action="store_true",
                    help="synthesize sentences into --wav-dir first (mlx_audio Kokoro)")
    ap.add_argument("--model", default=None,
                    help=f"ASR model repo or path (default {PARAKEET_REPO})")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--lang-code", default="a")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sentences = load_sentences(args.sentences)
    british = contract.load_british_map()

    if args.hypotheses:
        result = score_sentences(sentences, load_hypotheses(args.hypotheses), british)
    else:
        if args.wav_dir is None and args.render:
            args.wav_dir = args.out / "wavs"
        if args.render:
            _render_wavs(sentences, args.wav_dir, args.voice, args.lang_code)
        if not args.wav_dir:
            ap.error("--wav-dir or --hypotheses is required")
        wavs = resolve_wavs(args.wav_dir, len(sentences))
        result = roundtrip(sentences, wavs, _mlx_transcriber(args.model), british)

    result["summary"]["scorer"] = "check_fluidaudio_tts_contract.wer_and_cer"
    result["summary"]["asr_model"] = args.model or PARAKEET_REPO
    _write_output(args.out, result)
    json.dump(result["summary"], sys.stdout, indent=1)
    print()
    return 0 if result["summary"]["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
