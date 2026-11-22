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


def reference_from_receipt(receipt, release, wheel_sha256, capture_sha256):
    """Build the M1 Max reference from a Linux receipt of the pinned wheel."""
    suite = receipt.get('suite') or {}
    files = suite.get('files') or {}
    if receipt.get('wheel', {}).get('sha256') != wheel_sha256:
        raise ValueError('receipt is not a run of the pinned wheel')
    if len(files) != suite.get('expected_files') or not all(
            f.get('ids_complete') for f in files.values()):
        raise ValueError('receipt suite is incomplete')
    classes = {}
    for cls in suite['failures'].values():
        classes[cls] = classes.get(cls, 0) + 1
    return {
        'schema': 'omarchy-mlx-m1-max-reference-v1',
        'capture_sha256': capture_sha256,
        'counts': {k: suite[k] for k in ('failed', 'passed', 'skipped')},
        'device': receipt['benchmark']['device'],
        'failure_classes': classes,
        'failures': suite['failures'],
        'icd_manifest_sha256': receipt.get('icd_manifest_sha256'),
        'release': release,
        'seconds_per_file': {name: f['seconds'] for name, f in files.items()},
        'skips': sorted(set(suite['skips'])),
        'suite_source': receipt['suite_source'],
        'utc': receipt['utc'],
        'versions': receipt['versions'],
        'wheel_sha256': wheel_sha256,
    }


if __name__ == '__main__':
    import hashlib
    import json
    import sys
    from pathlib import Path
    import qualify
    src = Path(sys.argv[1])
    ref = reference_from_receipt(json.loads(src.read_text()), qualify.WHEEL_TAG,
                                 qualify.ASSETS[qualify.WHEEL],
                                 hashlib.sha256(src.read_bytes()).hexdigest())
    out = Path(__file__).resolve().parent / 'm1-max-reference.json'
    out.write_text(json.dumps(ref, indent=2, sort_keys=True) + '\n')
    print(f"{out}: {ref['counts']}")
