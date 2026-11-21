#!/usr/bin/env python3
"""Tests for scripts/check_fluidaudio_tts_contract.py — the FluidAudio TTS
benchmark contract audit (docs/fluidaudio-tts-benchmark-contract.md).

The corpus expectations are derived from the pinned upstream file
(MiniMaxAI/TTS-Multilingual-Test-Set @ cb416f0ac3658da0577e97873065e19fe6488917),
the scorer expectations from the published English scorer at FluidAudio
commit e0c6103bc87a88d25276cad491c86e757f58c865
(WERCalculator.calculateWERAndCER + TextNormalizer.normalize +
editDistance). No test in this file recomputes expectations by calling the
code under test.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import check_fluidaudio_tts_contract as contract  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures" / "fluidaudio_tts"


def test_corpus_fixture_is_pinned_upstream_file():
    raw = (FIXTURES / "minimax_english_cb416f0.txt").read_text()
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    assert digest == contract.MINIMAX_ENGLISH_SHA256


def test_corpus_100_phrases_in_upstream_order():
    raw = (FIXTURES / "minimax_english_cb416f0.txt").read_text()
    phrases = contract.minimax_phrases(raw)
    assert len(phrases) == 100
    # Order guard: first/last phrases pinned by hand from the upstream file.
    assert phrases[0] == (
        "I guess it comes down a simple choice. Get busy living or get busy dying."
    )
    assert phrases[12].endswith("in the 2024 election.")
    assert phrases[-1] == (
        "The culinary tour introduces visitors to hidden neighborhood "
        "restaurants where local chefs prepare authentic regional specialties."
    )
    # Conversion guard: no cloning-audio filename survives into any phrase.
    assert all("english_" not in p for p in phrases)


def test_check_corpus_mode_passes_on_vendored_file():
    rc = contract.check_corpus(FIXTURES / "minimax_english_cb416f0.txt", fetch=False)
    assert rc == 0


# --- Expected counts derived from the published scorer --------------------
#
# Derived at FluidAudio commit e0c6103 on the MiniMax English corpus:
# - numeric phrase ("2024" vs "two thousand twenty four"): number-word
#   folding absorbs the surface difference → 0 word / 0 char errors.
# - punctuation/case stripped by normalize → 0 / 0.
# - empty transcript: all reference words become deletions → 8/8 words,
#   30/30 characters → 100% / 100%.
# - digit mismatch ("2024" vs "2025"): 1 word substitution, 1 char
#   substitution → 1/19 = 5.26% WER, 1/79 = 1.27% CER.

EXPECTED = {
    "numeric phrase": {
        "wer": 0.0, "cer": 0.0,
        "word_errors": {"total": 0, "insertions": 0, "deletions": 0, "substitutions": 0},
        "char_errors": {"total": 0, "insertions": 0, "deletions": 0, "substitutions": 0},
        "ref_words": 19, "ref_chars": 79,
    },
    "numeric mismatch": {
        "wer": 1 / 19, "cer": 1 / 79,
        "word_errors": {"total": 1, "insertions": 0, "deletions": 0, "substitutions": 1},
        "char_errors": {"total": 1, "insertions": 0, "deletions": 0, "substitutions": 1},
        "ref_words": 19, "ref_chars": 79,
    },
    "punctuation": {
        "wer": 0.0, "cer": 0.0,
        "word_errors": {"total": 0, "insertions": 0, "deletions": 0, "substitutions": 0},
        "char_errors": {"total": 0, "insertions": 0, "deletions": 0, "substitutions": 0},
        "ref_words": 15, "ref_chars": 57,
    },
    "empty transcript": {
        "wer": 1.0, "cer": 1.0,
        "word_errors": {"total": 8, "insertions": 0, "deletions": 8, "substitutions": 0},
        "char_errors": {"total": 30, "insertions": 0, "deletions": 30, "substitutions": 0},
        "ref_words": 8, "ref_chars": 30,
    },
}


@pytest.fixture(scope="module")
def british():
    return json.loads((FIXTURES / "english.json").read_text())


@pytest.mark.parametrize("probe", contract.PROBES, ids=[p["name"] for p in contract.PROBES])
def test_scorer_derived_counts(probe, british):
    want = EXPECTED[probe["name"]]
    got = contract.wer_and_cer(probe["hypothesis"], probe["reference"], british)
    assert got["word_errors"] == want["word_errors"]
    assert got["char_errors"] == want["char_errors"]
    assert got["ref_words"] == want["ref_words"]
    assert got["ref_chars"] == want["ref_chars"]
    assert got["wer"] == pytest.approx(want["wer"])
    assert got["cer"] == pytest.approx(want["cer"])


def test_normalizer_folds_spelled_out_year(british):
    # "two thousand twenty four" must fold to 2024 inside the phrase —
    # the mechanism behind the numeric probe's zero-error result.
    got = contract.normalize("in the two thousand twenty four election", british)
    assert "2024" in got.split()


def test_empty_reference_yields_zero_wer_not_crash(british):
    # Published scorer edge: empty reference → wer/cer defined as 0.0
    # (WERCalculator.swift: wer = refWords.isEmpty ? 0.0 : total/refWords).
    got = contract.wer_and_cer("some words", "", british)
    assert got["wer"] == 0.0 and got["cer"] == 0.0
    assert got["word_errors"]["insertions"] == 2


def test_cli_all_mode():
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "check_fluidaudio_tts_contract.py")],
        capture_output=True, text=True, check=False,
    )
    assert out.returncode == 0
    assert "PASS corpus: 100 MiniMax English phrases in pinned order" in out.stdout
    assert "probe[empty transcript]: wer=100.00%" in out.stdout
