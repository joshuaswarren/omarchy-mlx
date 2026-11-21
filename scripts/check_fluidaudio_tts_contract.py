#!/usr/bin/env python3
"""Check the FluidAudio TTS benchmark contract recorded in
docs/fluidaudio-tts-benchmark-contract.md against the pinned upstream
sources.

FluidAudio pin: e0c6103bc87a88d25276cad491c86e757f58c865
MiniMax corpus pin: MiniMaxAI/TTS-Multilingual-Test-Set @
cb416f0ac3658da0577e97873065e19fe6488917, text/english.txt

Modes:
  corpus   fetch the pinned upstream English file (or read --corpus-file),
           verify its sha256, apply the FluidAudio harness conversion, and
           check the 100 phrases in order against the vendored copy.
  scorer   run the three documented probes through a Python port of the
           published English scorer (TextNormalizer.normalize +
           WERCalculator.calculateWERAndCER + editDistance) and print the
           derived word/character error counts.
  anchors  grep the named timer/scorer anchors out of a FluidAudio
           checkout (--fluidaudio-src).
  all      corpus + scorer (default; offline-safe).
"""
import argparse
import hashlib
import json
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures" / "fluidaudio_tts"

FLUIDAUDIO_COMMIT = "e0c6103bc87a88d25276cad491c86e757f58c865"
MINIMAX_REVISION = "cb416f0ac3658da0577e97873065e19fe6488917"
MINIMAX_ENGLISH_URL = (
    "https://huggingface.co/datasets/MiniMaxAI/TTS-Multilingual-Test-Set"
    f"/resolve/{MINIMAX_REVISION}/text/english.txt"
)
MINIMAX_ENGLISH_SHA256 = "1020375295c73eef7ff66a611885b0ba91aaa2b6d07cfe76ca2332e05f942e30"

# ---------------------------------------------------------------------------
# Corpus conversion — mirrors MinimaxCorpusCommand.convert() (Swift,
# Sources/FluidAudioCLI/Commands/MinimaxCorpusCommand.swift:136-155): trim,
# skip empty, strip "<file>|" at the FIRST pipe, drop empty texts. Order is
# the upstream file order.
# ---------------------------------------------------------------------------


def minimax_phrases(raw: str) -> list:
    out = []
    for raw_line in raw.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if "|" in line:
            text = line.split("|", 1)[1].strip()
        else:
            text = line
        if text:
            out.append(text)
    return out


# ---------------------------------------------------------------------------
# Scorer port — TextNormalizer.normalize (English leaderboard path),
# Sources/FluidAudioCLI/Utils/TextNormalizer.swift:397-606, plus
# convertNumbers (:636-712) and WERCalculator.editDistance (WERCalculator.
# swift:186-248) and calculateWERAndCER (:36-96).
# ---------------------------------------------------------------------------

ADDITIONAL_DIACRITICS = {
    "œ": "oe", "Œ": "OE", "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE",
    "ß": "ss", "ẞ": "SS", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D",
    "þ": "th", "Þ": "th", "ł": "l", "Ł": "L",
}

ABBREVIATIONS = {
    "mr": "mister", "mrs": "missus", "ms": "miss", "dr": "doctor",
    "prof": "professor", "st": "saint", "jr": "junior", "sr": "senior",
    "esq": "esquire",
    "capt": "captain", "gov": "governor", "ald": "alderman", "gen": "general",
    "sen": "senator", "rep": "representative", "pres": "president",
    "rev": "reverend", "hon": "honorable", "asst": "assistant",
    "assoc": "associate", "lt": "lieutenant", "col": "colonel",
    "vs": "versus", "inc": "incorporated", "ltd": "limited", "co": "company",
    "am": "a m", "pm": "p m", "ad": "ad", "bc": "bc",
}

CONTRACTIONS = {
    "can't": "can not", "won't": "will not", "ain't": "aint",
    "let's": "let us", "n't": " not", "'re": " are", "'ve": " have",
    "'ll": " will", "'d": " would", "'m": " am", "'t": " not", "'s": " is",
    "y'all": "you all", "wanna": "want to", "gonna": "going to",
    "gotta": "got to", "i'ma": "i am going to", "imma": "i am going to",
    "woulda": "would have", "coulda": "could have", "shoulda": "should have",
    "ma'am": "madam",
    "'d been": " had been", "'s been": " has been", "'d gone": " had gone",
    "'s gone": " has gone", "'d done": " had done", "'s got": " has got",
    "it's": "it is", "that's": "that is", "there's": "there is",
    "here's": "here is", "what's": "what is", "where's": "where is",
    "who's": "who is", "how's": "how is", "i'm": "i am", "you're": "you are",
    "we're": "we are", "they're": "they are", "you've": "you have",
    "we've": "we have", "they've": "they have", "i've": "i have",
    "you'll": "you will", "we'll": "we will", "they'll": "they will",
    "i'll": "i will", "you'd": "you would", "we'd": "we would",
    "they'd": "they would", "i'd": "i would", "she's": "she is",
    "he's": "he is", "she'll": "she will", "he'll": "he will",
    "she'd": "she would", "he'd": "he would",
}

# The shared numberWords table the English path applies (TextNormalizer.swift
# :132-185) — it includes the Italian and French entries; the English
# normalizer runs the whole map.
NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
    "billion": "1000000000",
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
    "fifth": "5th", "sixth": "6th", "seventh": "7th", "eighth": "8th",
    "ninth": "9th", "tenth": "10th", "eleventh": "11th", "twelfth": "12th",
    "thirteenth": "13th", "fourteenth": "14th", "fifteenth": "15th",
    "sixteenth": "16th", "seventeenth": "17th", "eighteenth": "18th",
    "nineteenth": "19th", "twentieth": "20th", "thirtieth": "30th",
    "fortieth": "40th", "fiftieth": "50th", "sixtieth": "60th",
    "seventieth": "70th", "eightieth": "80th", "ninetieth": "90th",
    "hundredth": "100th", "thousandth": "1000th",
    "uno": "1", "due": "2", "tre": "3", "quattro": "4", "cinque": "5",
    "sei": "6", "sette": "7", "otto": "8", "nove": "9", "dieci": "10",
    "undici": "11", "dodici": "12", "tredici": "13", "quattordici": "14",
    "quindici": "15", "sedici": "16", "diciassette": "17", "diciotto": "18",
    "diciannove": "19", "venti": "20", "trenta": "30", "quaranta": "40",
    "cinquanta": "50", "sessanta": "60", "settanta": "70", "ottanta": "80",
    "novanta": "90", "cento": "100", "mila": "1000", "milione": "1000000",
    "milioni": "1000000", "miliardo": "1000000000", "miliardi": "1000000000",
    "primo": "1st", "secondo": "2nd", "terzo": "3rd", "quarto": "4th",
    "quinto": "5th", "sesto": "6th", "settimo": "7th", "ottavo": "8th",
    "nono": "9th", "decimo": "10th", "undicesimo": "11th", "dodicesimo": "12th",
    "tredicesimo": "13th", "quattordicesimo": "14th", "quindicesimo": "15th",
    "sedicesimo": "16th", "diciassettesimo": "17th", "diciottesimo": "18th",
    "diciannovesimo": "19th", "ventesimo": "20th", "trentesimo": "30th",
    "centesimo": "100th",
    "zéro": "0", "un": "1", "deux": "2", "trois": "3", "quatre": "4",
    "cinq": "5", "six": "6", "sept": "7", "huit": "8", "neuf": "9",
    "dix": "10", "onze": "11", "douze": "12", "treize": "13", "quatorze": "14",
    "quinze": "15", "seize": "16", "dix-sept": "17", "dix-huit": "18",
    "dix-neuf": "19", "vingt": "20", "trente": "30", "quarante": "40",
    "cinquante": "50", "soixante": "60", "soixante-dix": "70",
    "quatre-vingts": "80", "quatre-vingt-dix": "90", "cent": "100",
    "mille": "1000", "million": "1000000", "millions": "1000000",
    "milliard": "1000000000", "milliards": "1000000000",
    "premier": "1st", "première": "1st", "deuxième": "2nd", "troisième": "3rd",
    "quatrième": "4th", "cinquième": "5th", "sixième": "6th",
    "septième": "7th", "huitième": "8th", "neuvième": "9th",
    "dixième": "10th", "onzième": "11th", "douzième": "12th",
    "treizième": "13th", "quatorzième": "14th", "quinzième": "15th",
    "seizième": "16th", "dix-septième": "17th", "dix-huitième": "18th",
    "dix-neuvième": "19th", "vingtième": "20th", "trentième": "30th",
    "centième": "100th",
}

NUMBER_VALUES = {
    "zero": 0, "oh": 0,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

MULTIPLIERS = {
    "hundred": 100,
    "thousand": 1000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}


def _longest_key_first(table: dict) -> list:
    # Swift: sorted(by: longestKeyFirst) — key length desc, lexical asc.
    return sorted(table.items(), key=lambda kv: (-len(kv[0]), kv[0]))


def _is_number_word(word: str) -> bool:
    return word in NUMBER_VALUES or word in MULTIPLIERS


def _parse_number_sequence(words: list) -> str:
    results = []
    current_sum = 0
    last_scale = 0
    for word in words:
        val = NUMBER_VALUES.get(word, MULTIPLIERS.get(word, 0))
        is_multiplier = word in MULTIPLIERS
        if is_multiplier:
            if current_sum == 0:
                current_sum = 1
            current_sum *= val
            last_scale = val
        else:
            if current_sum == 0:
                current_sum = val
                last_scale = 1
            else:
                can_merge = False
                if last_scale >= 100 and val < last_scale:
                    can_merge = True
                elif (
                    last_scale == 1
                    and current_sum % 100 >= 20
                    and current_sum % 10 == 0
                    and val < 10
                ):
                    can_merge = True
                if can_merge:
                    current_sum += val
                    last_scale = 1
                else:
                    results.append(str(current_sum))
                    current_sum = val
                    last_scale = 1
    if current_sum > 0:
        results.append(str(current_sum))
    return " ".join(results)


def convert_numbers(text: str) -> str:
    words = text.split()
    result = []
    current = []
    for word in words:
        if _is_number_word(word):
            current.append(word)
        else:
            if current:
                result.append(_parse_number_sequence(current))
                current = []
            result.append(word)
    if current:
        result.append(_parse_number_sequence(current))
    return " ".join(result)


def normalize(text: str, british_map: dict) -> str:
    """Port of TextNormalizer.normalize (English path)."""
    t = text.lower()
    for british, american in _longest_key_first(british_map):
        t = re.sub(r"\b" + re.escape(british) + r"\b", american, t)
    for abbrev, expansion in _longest_key_first(ABBREVIATIONS):
        t = re.sub(r"\b" + abbrev + r"\b", expansion, t)
    t = re.sub(r"[<\[].*?[>\]]", "", t)
    t = re.sub(r"\([^)]+?\)", "", t)
    t = re.sub(r"\b(hmm|mm|mhm|mmm|uh|um)\b", "", t)
    t = re.sub(r"\b[a-z]{1,2}-\s+", "", t)
    t = t.replace(" '", "'")
    t = t.replace(" and a half", " point five")
    t = re.sub(r"([a-z])([0-9])", r"\1 \2", t)
    t = re.sub(r"([0-9])([a-z])", r"\1 \2", t)
    t = re.sub(r"([0-9])\s+(st|nd|rd|th|s)\b", r"\1\2", t)
    t = "".join(ADDITIONAL_DIACRITICS.get(ch, ch) for ch in t)
    t = t.replace("$", " dollar ").replace("&", " and ").replace("%", " percent ")
    t = re.sub(r"[^\w\s']", " ", t)
    for contraction, expansion in _longest_key_first(CONTRACTIONS):
        t = t.replace(contraction, expansion)
    t = convert_numbers(t)
    for word, digit in _longest_key_first(NUMBER_WORDS):
        t = re.sub(r"\b" + word + r"\b", digit, t)
    t = re.sub(r"(\d),(\d)", r"\1\2", t)
    t = re.sub(r"\.([^0-9]|$)", r" \1", t)
    t = t.replace("a d", "ad")
    t = re.sub(r"\b(\d{1,2})\s+(\d{2})\s+(am|pm)\b", r"\1 \2 \3", t)
    t = t.replace("€", " euro ").replace("£", " pound ").replace("¥", " yen ")
    t = t.replace("©", " copyright ").replace("®", " registered ").replace("™", " trademark ")
    t = re.sub(r"[.$¢€£]([^0-9])", r" \1", t)
    t = re.sub(r"([^0-9])%", r"\1 ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def edit_distance(seq1: list, seq2: list) -> dict:
    """Levenshtein with the scorer's S/I/D backtrace (WERCalculator.swift:186)."""
    m, n = len(seq1), len(seq2)
    if m == 0:
        return {"total": n, "insertions": 0, "deletions": n, "substitutions": 0}
    if n == 0:
        return {"total": m, "insertions": m, "deletions": 0, "substitutions": 0}
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    i, j = m, n
    insertions = deletions = substitutions = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and seq1[i - 1] == seq2[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            insertions += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            deletions += 1
            j -= 1
        else:
            break
    return {
        "total": dp[m][n],
        "insertions": insertions,
        "deletions": deletions,
        "substitutions": substitutions,
    }


def wer_and_cer(hypothesis: str, reference: str, british_map: dict) -> dict:
    """Port of WERCalculator.calculateWERAndCER (:36-96)."""
    hyp = normalize(hypothesis, british_map)
    ref = normalize(reference, british_map)
    hyp_words = [w for w in hyp.split() if w]
    ref_words = [w for w in ref.split() if w]
    word_dist = edit_distance(hyp_words, ref_words)
    wer = 0.0 if not ref_words else word_dist["total"] / len(ref_words)
    hyp_chars = list(hyp.replace(" ", ""))
    ref_chars = list(ref.replace(" ", ""))
    char_dist = edit_distance(hyp_chars, ref_chars)
    cer = 0.0 if not ref_chars else char_dist["total"] / len(ref_chars)
    return {
        "wer": wer,
        "cer": cer,
        "word_errors": word_dist,
        "char_errors": char_dist,
        "ref_words": len(ref_words),
        "ref_chars": len(ref_chars),
        "normalized_hypothesis": hyp,
        "normalized_reference": ref,
    }


# ---------------------------------------------------------------------------
# Probes from the audit: a numeric phrase, punctuation, an empty transcript.
# ---------------------------------------------------------------------------

PROBES = [
    {
        "name": "numeric phrase",
        # Corpus line 13 (digit-bearing MiniMax English line) against a
        # hypothesis that spells the year out as words.
        "reference": (
            "The Harris campaign has said this “is not what she is "
            "proposing or running on” in the 2024 election."
        ),
        "hypothesis": (
            "The Harris campaign has said this is not what she is "
            "proposing or running on in the two thousand twenty four election."
        ),
    },
    {
        "name": "numeric mismatch",
        # Same corpus line 13, hypothesis reads the wrong year — pins the
        # substitution path after identical normalization.
        "reference": (
            "The Harris campaign has said this “is not what she is "
            "proposing or running on” in the 2024 election."
        ),
        "hypothesis": (
            "The Harris campaign has said this is not what she is "
            "proposing or running on in the 2025 election."
        ),
    },
    {
        "name": "punctuation",
        "reference": "I guess it comes down a simple choice. Get busy living or get busy dying.",
        "hypothesis": "i guess it comes down a simple choice get busy living or get busy dying",
    },
    {
        "name": "empty transcript",
        "reference": "I guess it comes down a simple choice.",
        "hypothesis": "",
    },
]


def load_british_map() -> dict:
    return json.loads((FIXTURES / "english.json").read_text())


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def check_corpus(corpus_file: Path | None, fetch: bool) -> int:
    if corpus_file is not None:
        raw = corpus_file.read_text()
        origin = str(corpus_file)
    elif fetch:
        with urllib.request.urlopen(MINIMAX_ENGLISH_URL, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
        origin = MINIMAX_ENGLISH_URL
    else:
        raw = (FIXTURES / "minimax_english_cb416f0.txt").read_text()
        origin = "vendored fixture"

    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    ok = True
    if digest != MINIMAX_ENGLISH_SHA256:
        print(f"FAIL sha256 {digest} != pin {MINIMAX_ENGLISH_SHA256} ({origin})")
        ok = False
    phrases = minimax_phrases(raw)
    if len(phrases) != 100:
        print(f"FAIL phrase count {len(phrases)} != 100 ({origin})")
        ok = False
    expected = minimax_phrases((FIXTURES / "minimax_english_cb416f0.txt").read_text())
    for idx, (want, got) in enumerate(zip(expected, phrases), start=1):
        if want != got:
            print(f"FAIL phrase {idx} out of order/different:\n  want: {want}\n  got:  {got}")
            ok = False
            break
    if len(phrases) != len(expected):
        ok = False
    print(
        ("PASS" if ok else "FAIL")
        + f" corpus: {len(phrases)} MiniMax English phrases in pinned order"
        + f" (sha256 {digest[:12]}…, {origin})"
    )
    return 0 if ok else 1


def run_scorer() -> int:
    british = load_british_map()
    for probe in PROBES:
        m = wer_and_cer(probe["hypothesis"], probe["reference"], british)
        we, ce = m["word_errors"], m["char_errors"]
        print(
            f"probe[{probe['name']}]: wer={m['wer'] * 100:.2f}% "
            f"(total={we['total']} S={we['substitutions']} I={we['insertions']} "
            f"D={we['deletions']} / ref_words={m['ref_words']}) "
            f"cer={m['cer'] * 100:.2f}% "
            f"(total={ce['total']} S={ce['substitutions']} I={ce['insertions']} "
            f"D={ce['deletions']} / ref_chars={m['ref_chars']})"
        )
        print(f"  ref: {m['normalized_reference']}")
        print(f"  hyp: {m['normalized_hypothesis']}")
    return 0


# Named source anchors at the pinned FluidAudio commit — every metric in the
# doc links to one of these.
ANCHORS = [
    ("Sources/FluidAudioCLI/Commands/MinimaxCorpusCommand.swift", "cb416f0ac3658da0577e97873065e19fe6488917"),
    ("Sources/FluidAudioCLI/Commands/TtsBenchmarkCommand.swift", "let t0 = Date()"),
    ("Sources/FluidAudioCLI/Commands/TtsBenchmarkCommand.swift", "synthesizeStreaming"),
    ("Sources/FluidAudioCLI/Commands/TtsBenchmarkCommand.swift", "ttftMs = Date().timeIntervalSince(t0)"),
    ("Sources/FluidAudioCLI/Commands/TtsBenchmarkCommand.swift", "WERCalculator.calculateWERAndCER"),
    ("Sources/FluidAudioCLI/Utils/WERCalculator.swift", "static func calculateWERAndCER"),
    ("Sources/FluidAudioCLI/Utils/WERCalculator.swift", "private static func editDistance"),
    ("Sources/FluidAudioCLI/Utils/TextNormalizer.swift", "static func normalize(_ text: String)"),
    ("Sources/FluidAudio/TTS/KokoroAne/Pipeline/KokoroAneSynthesizer.swift", "let clock = ContinuousClock()"),
    ("Sources/FluidAudioCLI/Commands/ASR/Parakeet/SlidingWindow/FleursBenchmark.swift", "tree/main"),
    ("Documentation/TTS/Measurements/KokoroV3/comparison.json", "pipeline.synthesize through complete PCM"),
]


def check_anchors(src: Path) -> int:
    ok = True
    for rel, needle in ANCHORS:
        path = src / rel
        if not path.is_file():
            print(f"FAIL anchor missing file: {rel}")
            ok = False
            continue
        if needle not in path.read_text(errors="replace"):
            print(f"FAIL anchor drifted: {rel}: {needle!r}")
            ok = False
    print(("PASS" if ok else "FAIL") + f" anchors: {len(ANCHORS)} source anchors")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="all", choices=["all", "corpus", "scorer", "anchors"])
    ap.add_argument("--corpus-file", type=Path, help="local copy of upstream text/english.txt")
    ap.add_argument("--fetch", action="store_true", help="fetch the pinned upstream corpus over the network")
    ap.add_argument("--fluidaudio-src", type=Path, help="FluidAudio checkout for the anchors mode")
    args = ap.parse_args()

    rc = 0
    if args.mode in ("all", "corpus"):
        rc |= check_corpus(args.corpus_file, args.fetch)
    if args.mode in ("all", "scorer"):
        rc |= run_scorer()
    if args.mode == "anchors":
        if args.fluidaudio_src is None:
            print("anchors mode needs --fluidaudio-src <FluidAudio checkout>")
            return 2
        rc |= check_anchors(args.fluidaudio_src)
    return rc


if __name__ == "__main__":
    sys.exit(main())
