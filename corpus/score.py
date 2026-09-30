"""Score raw transcripts with one normalizer for every system.

python3 score.py manifest.json clips.jsonl[+overlimit.jsonl] [whisper_full_results.json] > scores.json

Normalization (identical for references and every hypothesis): '&' -> AND,
'%' -> PERCENT, digit groups (thousands commas, decimals, ordinals 1st/2nd/
3rd/Nth) -> English words, uppercase, every non-alphanumeric character ->
space, whitespace collapsed. WER per subset = total word edits / total
reference words (Levenshtein over words). A clip with no transcript (refused
or failed) counts every reference word as a deletion; nothing is skipped.

"parakeet_gpu" merges the '+'-joined files: a later row replaces an earlier
one only when the earlier has no transcript (the over-30 s clip, re-run as
the browser uploads it: first 30.0 s). "parakeet_gpu_raw_submission" scores
the first file alone.
"""
import json, re, sys

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
       "nine": "ninth", "twelve": "twelfth"}


def words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + words(n % 100))
    for size, name in ((10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")):
        if n >= size:
            return words(n // size) + " " + name + ("" if n % size == 0 else " " + words(n % size))


def ordinal(text: str) -> str:
    head, _, last = text.rpartition(" ")
    last = ORD.get(last, (last[:-1] + "ieth") if last.endswith("y") else last + "th")
    return (head + " " + last).strip()


def number(match) -> str:
    raw, suffix = match.group(1).replace(",", ""), (match.group(2) or "").lower()
    whole, _, frac = raw.partition(".")
    text = words(int(whole))
    if suffix in ("st", "nd", "rd", "th"):
        text = ordinal(text)
    if frac:
        text += " point " + " ".join(ONES[int(d)] for d in frac)
    return " " + text + " "


def normalize(text: str) -> str:
    text = (text or "").replace("&", " and ").replace("%", " percent ")
    text = re.sub(r"(\d[\d,]*(?:\.\d+)?)(st|nd|rd|th)?\b", number, text, flags=re.I)
    return " ".join("".join(c if c.isalnum() else " " for c in text.upper()).split())


def edits(ref: list, hyp: list) -> int:
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1]


def score(manifest, rows):
    refs = {s["utt_id"]: (s["subset"], s["reference"]) for s in manifest["samples"]}
    out = {}
    for utt, (subset, ref) in refs.items():
        b = out.setdefault(subset, {"n": 0, "scored": 0, "ref_words": 0, "edits": 0,
                                    "empty": 0, "errors": 0, "latency_ms": []})
        b["n"] += 1
        row = rows.get(utt)
        r = normalize(ref).split()
        if row is None or row.get("hypothesis") is None:
            b["errors"] += 1
            b["ref_words"] += len(r)
            b["edits"] += len(r)
            continue
        h = normalize(row["hypothesis"]).split()
        b["scored"] += 1
        b["ref_words"] += len(r)
        b["edits"] += edits(r, h)
        b["empty"] += not h
        if row.get("latency_ms") is not None:
            b["latency_ms"].append(row["latency_ms"])
    for b in out.values():
        b["wer"] = round(b["edits"] / b["ref_words"], 4) if b["ref_words"] else None
        b["empty_rate"] = round(b["empty"] / b["scored"], 4) if b["scored"] else None
        lat = sorted(b.pop("latency_ms"))
        if lat:
            b["latency_p50_ms"] = lat[(len(lat) + 1) // 2 - 1]
            b["latency_p95_ms"] = lat[-(-len(lat) * 95 // 100) - 1]
    return out


if __name__ == "__main__":
    manifest = json.load(open(sys.argv[1]))
    files = [list(map(json.loads, open(p).read().splitlines())) for p in sys.argv[2].split("+")]
    raw = {r["utt_id"]: r for r in files[0]}
    merged = dict(raw)
    for extra in files[1:]:
        for r in extra:
            if merged.get(r["utt_id"], {}).get("hypothesis") is None:
                merged[r["utt_id"]] = r
    result = {"parakeet_gpu": score(manifest, merged)}
    if len(files) > 1:
        result["parakeet_gpu_raw_submission"] = score(manifest, raw)
    if len(sys.argv) > 3:
        whisper = {r["utt_id"]: dict(r, hypothesis=r["hyp"], latency_ms=r["latency_s"] * 1000)
                   for r in json.load(open(sys.argv[3]))["per_clip"]}
        result["whisper_large_v3_turbo_mac"] = score(manifest, whisper)
    json.dump(result, sys.stdout, indent=1)
