"""Only the same test ID and failure class can match a known baseline."""


def comparison(baseline, failures, skips):
    old_failures = baseline['failures']
    old_skips = set(baseline['skips'])
    matched = {k: v for k, v in failures.items() if old_failures.get(k) == v}
    return {
        'matched_failures': matched,
        'new_failures': {k: v for k, v in failures.items() if k not in matched},
        'matched_skips': sorted(old_skips.intersection(skips)),
        'new_skips': sorted(set(skips) - old_skips),
    }
