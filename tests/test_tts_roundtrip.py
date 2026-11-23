#!/usr/bin/env python3
"""Tests for scripts/tts_roundtrip.py — the Kokoro→Parakeet round-trip
scorer (V15 gap row: "rendered WAVs exist, unscored").

Expected numbers are derived by hand from the pinned fixture pairs
(tests/fixtures/tts_roundtrip/) through the published scorer contract
(scripts/check_fluidaudio_tts_contract.py): no test recomputes an
expectation by calling the code under test.

Fixture derivation (verified once against the contract scorer):

| # | pair | per-sentence WER / CER |
|---|------|------------------------|
| 1 | punctuation+case only        | 0 / 0 |
| 2 | "2024" vs spelled-out year   | 0 / 0 (number-word fold) |
| 3 | Colour/harbour vs US spelling| 0 / 0 (british_map fold) |
| 4 | "jumps"→"jumped"             | 1/9 ≈ 0.1111 / 2/35 ≈ 0.0571 |
| 5 | empty hypothesis             | 6/6 = 1.0 / 19/19 = 1.0 |

Corpus: word edits 7 / ref words 55 → 0.12727;
        char edits 21 / ref chars 217 → 0.09677;
        mean per-sentence WER (0+0+0+1/9+1)/5 ≈ 0.22222; median 0.0;
        empty hypotheses 1.
"""
import json
import subprocess
import sys
import wave
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import check_fluidaudio_tts_contract as contract  # noqa: E402
import tts_roundtrip as rt  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures" / "tts_roundtrip"
SENTENCES = FIXTURES / "sentences.txt"
HYPOTHESES = FIXTURES / "hypotheses.jsonl"

EXPECTED_WER = [0.0, 0.0, 0.0, 1 / 9, 1.0]
EXPECTED_CER = [0.0, 0.0, 0.0, 2 / 35, 1.0]


def _load():
    sentences = rt.load_sentences(SENTENCES)
    hypotheses = rt.load_hypotheses(HYPOTHESES)
    return rt.score_sentences(sentences, hypotheses, contract.load_british_map())


def test_load_sentences_plain_lines():
    sentences = rt.load_sentences(SENTENCES)
    assert len(sentences) == 5
    assert sentences[0].startswith("I guess it comes down")


def test_load_sentences_minimax_pipe_form(tmp_path):
    p = tmp_path / "mini.txt"
    p.write_text("english_1|First phrase.\n\nenglish_2|Second phrase.\n")
    assert rt.load_sentences(p) == ["First phrase.", "Second phrase."]


def test_load_sentences_default_is_minimax_fixture():
    sentences = rt.load_sentences(None)
    assert len(sentences) == 100
    assert sentences[0] == contract.minimax_phrases(
        (REPO / "tests" / "fixtures" / "fluidaudio_tts" / "minimax_english_cb416f0.txt")
        .read_text())[0]


def test_load_hypotheses_missing_index_fails():
    bad = [{"index": 1, "text": "a"}, {"index": 3, "text": "c"}]
    with pytest.raises(ValueError, match="index 2"):
        rt.score_sentences(["a", "b", "c"], bad, {})


def test_duplicate_hypothesis_index_fails():
    dup = [{"index": 1, "text": "a"}, {"index": 2, "text": "b"},
           {"index": 1, "text": "wrong"}]
    with pytest.raises(ValueError, match="duplicate hypothesis for index 1"):
        rt.score_sentences(["a", "b"], dup, {})


@pytest.mark.parametrize("index", [0, -1, 3, True, 1.0, "1"])
def test_hypothesis_index_outside_1_to_n_fails(index):
    rows = [{"index": 1, "text": "a"}, {"index": 2, "text": "b"},
            {"index": index, "text": "x"}]
    with pytest.raises(ValueError, match=r"is not an integer in 1\.\.2"):
        rt.score_sentences(["a", "b"], rows, {})


def test_per_sentence_wer_and_cer():
    result = _load()
    assert [r["index"] for r in result["rows"]] == [1, 2, 3, 4, 5]
    for row, wer, cer in zip(result["rows"], EXPECTED_WER, EXPECTED_CER):
        assert row["wer"] == pytest.approx(wer)
        assert row["cer"] == pytest.approx(cer)
    assert result["rows"][3]["word_errors"] == {
        "total": 1, "insertions": 0, "deletions": 0, "substitutions": 1}
    assert result["rows"][4]["empty"] is True
    assert result["rows"][0]["empty"] is False


def test_corpus_aggregate_is_edits_over_words_not_mean():
    result = _load()["summary"]
    assert result["sentences"] == 5
    assert result["scored"] == 5
    assert result["corpus_wer"] == pytest.approx(7 / 55)
    assert result["corpus_cer"] == pytest.approx(21 / 217)
    assert result["mean_per_sentence_wer"] == pytest.approx((0 + 0 + 0 + 1 / 9 + 1) / 5)
    assert result["median_per_sentence_wer"] == pytest.approx(0.0)
    assert result["empty_hypotheses"] == 1


def test_resolve_wavs_index_aligned(tmp_path):
    for name in ("001.wav", "002.wav", "3.wav"):
        _write_wav(tmp_path / name)
    wavs = rt.resolve_wavs(tmp_path, 3)
    assert [w.name for w in wavs] == ["001.wav", "002.wav", "3.wav"]


def test_resolve_wavs_missing_index_reports_none(tmp_path):
    _write_wav(tmp_path / "001.wav")
    _write_wav(tmp_path / "003.wav")
    wavs = rt.resolve_wavs(tmp_path, 3)
    assert wavs[0].name == "001.wav"
    assert wavs[1] is None
    assert wavs[2].name == "003.wav"


def test_roundtrip_over_wavs_with_stub_transcriber(tmp_path):
    sentences = rt.load_sentences(SENTENCES)
    for i, _ in enumerate(sentences, 1):
        _write_wav(tmp_path / f"{i:03d}.wav")
    stub_map = {r["index"]: r["text"] for r in rt.load_hypotheses(HYPOTHESES)}

    def fake_transcribe(wav_path: Path) -> str:
        return stub_map.get(int(wav_path.stem.lstrip("0")), "")

    result = rt.roundtrip(sentences, list(tmp_path.glob("*.wav")),
                          fake_transcribe, contract.load_british_map())
    assert result["summary"]["corpus_wer"] == pytest.approx(7 / 55)
    assert result["rows"][3]["wer"] == pytest.approx(1 / 9)


def test_missing_wav_counts_as_error_not_empty(tmp_path):
    _write_wav(tmp_path / "001.wav")
    result = rt.roundtrip(["alpha", "beta"], [tmp_path / "001.wav"],
                          lambda p: "alpha", contract.load_british_map())
    row = result["rows"][1]
    assert row["error"] == "no wav for index 2"
    assert result["summary"]["scored"] == 1
    assert result["summary"]["errors"] == 1


def test_cli_hypotheses_mode_writes_json(tmp_path):
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "tts_roundtrip.py"),
         "--sentences", str(SENTENCES), "--hypotheses", str(HYPOTHESES),
         "--out", str(out)],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert summary["corpus_wer"] == pytest.approx(7 / 55)
    rows = [json.loads(l) for l in (out / "per-sentence.jsonl").read_text().splitlines()]
    assert len(rows) == 5
    assert rows[4]["empty"] is True


def test_cli_hypotheses_mode_rejects_duplicate_index(tmp_path):
    sentences = tmp_path / "sentences.txt"
    sentences.write_text("alpha\nbeta\n")
    hyps = tmp_path / "hyps.jsonl"
    hyps.write_text('{"index": 1, "text": "alpha"}\n{"index": 2, "text": "beta"}\n'
                    '{"index": 1, "text": "wrong"}\n')
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "tts_roundtrip.py"),
         "--sentences", str(sentences), "--hypotheses", str(hyps),
         "--out", str(out)],
        capture_output=True, text=True)
    assert proc.returncode != 0
    assert "duplicate hypothesis for index 1" in proc.stderr
    assert not out.exists()


def _write_wav(path: Path, frames: int = 1600, rate: int = 16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00" * 2 * frames)
