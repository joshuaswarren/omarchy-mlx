#!/usr/bin/env python3
"""Compile the production scalar reduction on CPU; never model the old GPU tree."""
import ctypes
import hashlib
import json
import struct
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

from test_q4_affine_precision import compile_model

ROOT = Path(__file__).resolve().parent.parent
SHADER = ROOT / 'overlay/mlx/backend/omarchy/shaders/qmm_vec.comp'
FIXTURE = ROOT / 'tests/fixtures/qmv-batch-reduction.json'


def bf16(bits):
    return ((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffffffff) >> 16


def scalar(bits):
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def nearest(record):
    reference = Fraction(float.fromhex(record['reference_hex']))
    choices = [int(record[k], 16) for k in ('base_bits', 'cand_bits')]
    return min(choices, key=lambda b: (abs(Fraction(scalar(b << 16)) - reference), b & 1))


def compile_reduce(directory):
    loaded = compile_model(directory)
    return loaded, loaded.reduce


def check_records(records, totals):
    failures = [(r['t'], r['seed_index'], r['index'], f'{bf16(total):04x}', f'{nearest(r):04x}')
                for r, total in zip(records, totals, strict=True) if bf16(total) != nearest(r)]
    assert not failures, f'BF16 differs from nearer independent reference: {failures}'


def main():
    fixture = json.loads(FIXTURE.read_text())
    records = fixture['records']
    assert len(records) == 6 and len({(r['t'], r['seed_index']) for r in records}) == 6
    if sys.argv[1:] == ['--observed-original']:
        assert hashlib.sha256(SHADER.read_bytes()).hexdigest() == fixture['original_shader_sha256']
        print('Observed original native totals only; no host subgroup-tree emulation.', flush=True)
        check_records(records, [int(r['observed_original_total_bits'], 16) for r in records])
        return
    assert not sys.argv[1:], 'only --observed-original is supported'
    with tempfile.TemporaryDirectory() as directory:
        _library, reduce = compile_reduce(directory)
        for base in (0, 32, 64, 96):
            totals = [reduce((ctypes.c_uint32 * 32)(*[int(b, 16) for b in r['lane_bits']]), base)
                      for r in records]
            check_records(records, totals)
            broadcast = [_library.reduce_broadcast(
                (ctypes.c_uint32 * 32)(*[int(b, 16) for b in r['lane_bits']]), base)
                for r in records]
            assert broadcast == totals, 'scratch/broadcast source fold mismatch'
        assert bf16(0x3f808000) == 0x3f80 and bf16(0x3f818000) == 0x3f82
        for bits, expected in (([0] * 32, 0), ([0x3f800000] + [0] * 31, 0x3f800000),
                               ([0x7f800000] + [0] * 31, 0x7f800000)):
            assert reduce((ctypes.c_uint32 * 32)(*bits), 0) == expected
        nan = reduce((ctypes.c_uint32 * 32)(0x7f800000, 0xff800000, *([0] * 30)), 0)
        assert nan & 0x7f800000 == 0x7f800000 and nan & 0x7fffff
    print('PASS: production scaled fold; six native regressions in all four slots; scratch/broadcast agree; RNE/zero/Inf/NaN.')


if __name__ == '__main__':
    main()
