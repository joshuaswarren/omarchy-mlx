#!/usr/bin/env python3
"""WER for the five fixed sentences: digit normalization + word Lev."""
import re
import sys

SENTENCES = [
    "The local assistant is ready to help.",
    "Your meeting starts at nine, and the review follows at eleven.",
    "I chose the shorter word, because it has fewer letters.",
    "Please check the camping list before you leave on Friday.",
    "Everything ran on this laptop, with no network connection.",
]
NUMBERS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
           "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
           "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
           "14": "fourteen", "15": "fifteen", "16": "sixteen",
           "17": "seventeen", "18": "eighteen", "19": "nineteen",
           "20": "twenty", "30": "thirty", "40": "forty", "50": "fifty"}


def norm(text):
    text = text.lower().strip()
    text = re.sub(
        r"\d+",
        lambda m: NUMBERS.get(m.group(0), " ".join(
            NUMBERS[c] for c in m.group(0))),
        text)
    text = re.sub(r"[^a-z ]", " ", text)
    return [w for w in text.split() if w]


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (r[i - 1] != h[j - 1]))
        prev = cur
    return prev[len(h)], len(r)


if __name__ == "__main__":
    # per-file mode: python wer_check.py t1.txt t2.txt t3.txt t4.txt t5.txt
    errors = words = 0
    for i, ref in enumerate(SENTENCES, 1):
        hyp = open(sys.argv[i]).read().strip()
        e, n = wer(ref, hyp)
        errors += e
        words += n
        print(f"sentence-{i}: {e}/{n} = {e / n:.1%}  hyp: {hyp!r}")
    print(f"overall: {errors}/{words} = {errors / words:.1%}")
