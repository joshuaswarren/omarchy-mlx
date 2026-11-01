#!/usr/bin/env python3
"""Focused Q4 bf16 affine precision regression.

Extracts the production scaled-accumulator helpers (q4_normalize,
q4_scaled_add, q4_cast, q4_product, q4_scaled_store) VERBATIM from
qmm_vec.comp into C++ and drives the retained real witnesses plus the
range/cast/RNE counterexamples through the production walk shape
(per-element weight cast, exact 8x24 significand product, exponent-aligned
accumulation, sequential lane fold, exponent-aware bf16 RNE store).
The old production interpretation (fma chunk epilogues, sequential float
lane adds, compensated float fold, fp32-only store) runs beside it as the
honest RED. No MLX/backend/provider/target/native use; pure CPU source math.
"""
import ctypes
import hashlib
import json
import struct
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SHADER = ROOT / 'overlay/mlx/backend/omarchy/shaders/qmm_vec.comp'
RAW = Path('/home/joshuawarren/scratch/w7N/a615-two-max-seed-generation-20261101T121517Z/output-copy/raw')
OBS = Path('/home/joshuawarren/scratch/w7N/a615-max-seed-observation-20261101T115117Z/output-copy/raw')
OLD48 = Path('/home/joshuawarren/scratch/w7N/issue305-diagnostic-controls-r2-20261031/output-copy/out/base.json')
OLD48_OPERANDS = Path('/home/joshuawarren/scratch/w7N/issue305-operand-custody-admission-20261101/output-copy/out/operands')
RAW_MANIFEST_SHA = '6a69ca4a99df0e8955d6480ffeeddc4ce471fa32f599654bfc45e95f1668c40e'
OBS_MANIFEST_SHA = '7ef71c909e1e07e6362f3a82440c819290032d44651f76e3a2b64a13917d121b'
WITNESSES = [
    ((2560, 9216, 7, 13), 0, 452, 'b8cf'),
    ((2560, 9216, 7, 13), 0, 509, '3c6c'),
    ((2560, 9216, 7, 13), 3, 8046, 'ba1b'),
    ((2560, 9216, 7, 13), 4, 6331, 'b7f2'),
    ((2560, 9216, 7, 13), 4, 7950, 'b92e'),
    ((2560, 9216, 7, 13), 4, 7972, '38ef'),
    ((2560, 9216, 7, 13), 5, 2871, 'bab6'),
    ((2048, 520, 5, 15), 4, 487, 'bcc3'),
]
OLD_RED = {(0, 509): '3c6b', (3, 8046): 'ba1a', (4, 6331): 'b7f1',
           (5, 2871): 'bab7', (4, 487, 15): 'bcc4'}


def scalar(bits):
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def bits_of(value):
    return struct.unpack('<I', struct.pack('<f', float(value)))[0]


def want_bits(reference_float):
    reference = Fraction(float(reference_float))
    center = bits_of(reference) >> 16
    choices = range(max(0, center - 2), min(65536, center + 3))
    return min(choices, key=lambda b: (abs(Fraction(scalar(b << 16)) - reference), b & 1))


def pinned(path, digest, count=None):
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == digest, str(path)
    assert count is None or len(data) == count
    return data


def extract_function(source, marker):
    start = source.index(marker)
    brace = source.index('{', start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == '{':
            depth += 1
        elif source[index] == '}':
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError(marker)


def extract_macro(source, marker):
    lines = source[source.index(marker):].splitlines()
    body = []
    for line in lines:
        body.append(line)
        if not line.endswith('\\'):
            return '\n'.join(body)
    raise AssertionError(marker)


def compile_model(directory):
    source = SHADER.read_text()
    depth = 0
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith('#if'):
            depth += 1
        elif stripped.startswith('#endif'):
            depth -= 1
        assert depth >= 0, 'unbalanced #endif'
    assert depth == 0, 'unbalanced #if'
    parts = [extract_function(
        source, 'struct Q4Scaled { float hi; float lo; int exponent; };') + ';']
    for marker in ('Q4Scaled q4_scaled_zero() {',
                   'Q4Scaled q4_scaled_nan() {',
                   'Q4Scaled q4_normalize(float hi, float lo, int exponent) {',
                   'Q4Scaled q4_scaled_add(Q4Scaled a, Q4Scaled b) {',
                   'float q4_cast(float scale, float q, float bias) {',
                   'Q4Scaled q4_product(float x, float weight) {',
                   'uint16_t q4_scaled_store(Q4Scaled pair) {'):
        parts.append(extract_function(source, marker))
    helpers = '\n'.join(parts)
    helpers = helpers.replace('frexp(', 'glsl_frexp(').replace('ldexp(', 'std::ldexp(')
    scratch_fold = extract_macro(source, '#define Q4_SCALED_FOLD_SCRATCH(total, base)')
    broadcast_fold = extract_macro(source, '#define Q4_SCALED_FOLD_BROADCAST(total, mine)')
    terms = extract_macro(source, '#define Q4_NIB(p, shift)') + '\n' + \
        extract_macro(source, '#define Q4_SCALED_TERMS(dst, p, lo, hi, sc, bi)')
    # One nonzero lane + zeros is an exact copy for any ARITHMETIC reduction order.
    for field in ('hi', 'lo'):
        expression = f'subgroupAdd(lane_in_slot == i ? q4_fold_source.{field} : 0.0)'
        assert broadcast_fold.count(expression) == 1
        broadcast_fold = broadcast_fold.replace(expression, f'cpu_lanes[i].{field}')
    expression = 'subgroupAdd(lane_in_slot == i ? uint(q4_fold_source.exponent) : 0u)'
    assert broadcast_fold.count(expression) == 1
    broadcast_fold = broadcast_fold.replace(expression, 'uint(cpu_lanes[i].exponent)')
    cpp = Path(directory) / 'model.cpp'
    library = Path(directory) / 'model.so'
    cpp.write_text('''#include <cmath>
#include <cstdint>
#include <cstring>
#include <cfenv>
#include <algorithm>
#include <vector>
using uint = unsigned;
using std::isnan;
using std::isinf;
using std::max;
#define floatBitsToUint fbits
#define uintBitsToFloat u2f
static float glsl_frexp(float value, int& exponent) {
  return std::frexp(value, &exponent);
}
#define precise
static float u2f(uint v) { float f; memcpy(&f, &v, 4); return f; }
static float fma32(float a, float b, float c) { return std::fma(a, b, c); }
static float add32(float a, float b) { return a + b; }
static float mul32(float a, float b) { return a * b; }
// Q4_NIB production macro body (exact float(n) for n in [0,15]).
static float q4_nib(uint p, uint shift) {
  return u2f((((p) >> (shift)) & 0xfu) | 0x4B000000u) - 8388608.0f;
}
static uint fbits(float f) { uint v; memcpy(&v, &f, 4); return v; }
''' + helpers + '''
const uint LANES = 32;
float reduce_scratch[256];
float reduce_scratch_lo[256];
int reduce_scratch_exponent[256];
Q4Scaled cpu_lanes[32];
''' + scratch_fold + '\n' + broadcast_fold + '\n' + terms + '''

// ---- production walk shape: 32 lanes, word step 1 or 2, element order
// identical to the shader macros (words strided per lane, nibbles low
// first, scale/bias per group of 64 elements), then the one sequential
// lane fold every consumer runs ----
static Q4Scaled walk(const float* x, const uint* wq, const float* s,
                     const float* b, uint k, uint step, bool broadcast = false) {
  const uint words_per_row = k / 8u;
  Q4Scaled lanes[32];
  for (uint lane = 0u; lane < 32u; ++lane) {
    lanes[lane] = Q4Scaled{0.0f, 0.0f, 0};
    for (uint w = lane * step; w < words_per_row; w += 32u * step) {
      for (uint cw = 0u; cw < step && w + cw < words_per_row; ++cw) {
        const uint word = w + cw;
        const uint group = word / 8u;
        const uint p = wq[word];
        for (uint j = 0u; j < 8u; j += 4u) {
          const uint offset = word * 8u + j;
          const uint lo = (fbits(x[offset]) >> 16) | (fbits(x[offset + 1]) & 0xffff0000u);
          const uint hi = (fbits(x[offset + 2]) >> 16) | (fbits(x[offset + 3]) & 0xffff0000u);
          Q4_SCALED_TERMS(lanes[lane], p >> (j * 4u), lo, hi, s[group], b[group])
        }
      }
    }
  }
  for (uint i = 0; i < 32; ++i) {
    reduce_scratch[i] = lanes[i].hi;
    reduce_scratch_lo[i] = lanes[i].lo;
    reduce_scratch_exponent[i] = lanes[i].exponent;
    cpu_lanes[i] = lanes[i];
  }
  Q4Scaled total;
  if (broadcast) {
    Q4_SCALED_FOLD_BROADCAST(total, lanes[0])
  } else {
    Q4_SCALED_FOLD_SCRATCH(total, 0)
  }
  return total;
}

extern "C" uint32_t run_new(const float* x, const uint* wq, const float* s,
                            const float* b, uint k, uint step) {
  if (std::fegetround() != FE_TONEAREST) __builtin_trap();
  return q4_scaled_store(walk(x, wq, s, b, k, step));
}
extern "C" uint32_t run_new_broadcast(const float* x, const uint* wq, const float* s,
                                      const float* b, uint k, uint step) {
  if (std::fegetround() != FE_TONEAREST) __builtin_trap();
  return q4_scaled_store(walk(x, wq, s, b, k, step, true));
}

// ---- old production interpretation (a615-reproduced source model) ----
static uint32_t old_model(const float* x, const uint* wq, const float* s,
                          const float* b, uint k, uint step) {
  if (std::fegetround() != FE_TONEAREST) __builtin_trap();
  const uint words_per_row = k / 8u;
  const uint chunk_words = step;
  const uint chunk_elements = 8u * chunk_words;
  const uint chunk_count = k / chunk_elements;
  std::vector<float> chunks(chunk_count);
  for (uint c = 0u; c < chunk_count; ++c) {
    const uint offset = c * chunk_elements;
    float dot = 0.0f;
    float input_sum = 0.0f;
    for (uint off = offset; off < offset + chunk_elements; off += 4u) {
      const float a = x[off], cc = x[off + 1u], d = x[off + 2u], e = x[off + 3u];
      const uint p = wq[off / 8u];
      const float qa = q4_nib(p, 4u * (off % 8u));
      const float qc = q4_nib(p, 4u * ((off + 1u) % 8u));
      const float qd = q4_nib(p, 4u * ((off + 2u) % 8u));
      const float qe = q4_nib(p, 4u * ((off + 3u) % 8u));
      input_sum = add32(input_sum, add32(add32(add32(a, cc), d), e));
      dot = add32(dot, fma32(e, qe, fma32(d, qd, fma32(cc, qc, mul32(a, qa)))));
    }
    const float scale = s[offset / 64u];
    const float bias = b[offset / 64u];
    chunks[c] = fma32(scale, dot, mul32(input_sum, bias));
  }
  float lanes[32];
  for (uint lane = 0u; lane < 32u; ++lane) {
    float value = 0.0f;
    for (uint c = lane; c < chunk_count; c += 32u) {
      value = add32(value, chunks[c]);
    }
    lanes[lane] = value;
  }
  float total = 0.0f;
  float correction = 0.0f;
  for (uint i = 0u; i < 32u; ++i) {
    const float value = lanes[i];
    const float next = add32(total, value);
    if (!std::isinf(next)) {
      const float residual = fabsf(total) >= fabsf(value)
          ? add32(add32(total, -next), value)
          : add32(add32(value, -next), total);
      correction = add32(correction, residual);
    }
    total = next;
  }
  return fbits(add32(total, correction));
}

extern "C" uint32_t run_old(const float* x, const uint* wq, const float* s,
                            const float* b, uint k, uint step) {
  uint32_t raw = old_model(x, wq, s, b, k, step);
  return ((raw + 0x7fffu + ((raw >> 16) & 1u)) >> 16) & 0xffffu;
}

extern "C" uint32_t run_products(const float* x, const float* weights, uint count) {
  Q4Scaled total = Q4Scaled{0.0f, 0.0f, 0};
  for (uint i = 0u; i < count; ++i) {
    total = q4_scaled_add(total, q4_product(x[i], weights[i]));
  }
  return q4_scaled_store(total);
}

// Adapter returns stored BF16 widened exactly to the old test's f32-bit API.
extern "C" uint32_t reduce(const uint32_t* words, uint base) {
  if (std::fegetround() != FE_TONEAREST || base > 96 || base % 32) __builtin_trap();
  for (uint i = 0; i < 32; ++i) {
    Q4Scaled value = q4_product(1.0f, u2f(words[i]));
    reduce_scratch[base + i] = value.hi;
    reduce_scratch_lo[base + i] = value.lo;
    reduce_scratch_exponent[base + i] = value.exponent;
    cpu_lanes[i] = value;
  }
  Q4Scaled total;
  Q4_SCALED_FOLD_SCRATCH(total, base)
  return uint32_t(q4_scaled_store(total)) << 16;
}
extern "C" uint32_t reduce_broadcast(const uint32_t* words, uint base) {
  reduce(words, base);
  Q4Scaled total;
  Q4_SCALED_FOLD_BROADCAST(total, cpu_lanes[0])
  return uint32_t(q4_scaled_store(total)) << 16;
}
''')
    subprocess.run(['c++', '-std=c++20', '-O2', '-shared', '-fPIC',
                    '-fno-fast-math', '-ffp-contract=off', '-frounding-math',
                    str(cpp), '-o', str(library)], check=True)
    loaded = ctypes.CDLL(str(library))
    loaded.run_new.restype = ctypes.c_uint32
    loaded.run_new.argtypes = [ctypes.POINTER(ctypes.c_float),
                               ctypes.POINTER(ctypes.c_uint32),
                               ctypes.POINTER(ctypes.c_float),
                               ctypes.POINTER(ctypes.c_float),
                               ctypes.c_uint32, ctypes.c_uint32]
    loaded.run_old.restype = ctypes.c_uint32
    loaded.run_old.argtypes = loaded.run_new.argtypes
    loaded.run_new_broadcast.restype = ctypes.c_uint32
    loaded.run_new_broadcast.argtypes = loaded.run_new.argtypes
    loaded.run_products.restype = ctypes.c_uint32
    loaded.run_products.argtypes = [ctypes.POINTER(ctypes.c_float),
                                   ctypes.POINTER(ctypes.c_float), ctypes.c_uint32]
    for name in ('reduce', 'reduce_broadcast'):
        function = getattr(loaded, name)
        function.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint]
        function.restype = ctypes.c_uint32
    return loaded


def arrays_of(record, observed):
    arrays = {}
    for name, desc in record['operands'].items():
        assert {key: desc[key] for key in ('shape', 'dtype', 'sha256')} == \
            observed['operands'][name]
        data = pinned(RAW / desc['path'], desc['sha256'], desc['bytes'])
        array = np.frombuffer(data, dtype='<u4' if desc['dtype'] == 'uint32' else '<u2')
        if desc['dtype'] == 'bf16':
            array = (array.astype('<u4') << 16).view('<f4')
        arrays[name] = array.reshape(desc['shape'])
    return arrays


def outputs_of(observed):
    data = {}
    for name, desc in {**observed['outputs'], 'reference': observed['reference']}.items():
        raw = pinned(OBS / desc['path'], desc['sha256'], desc['bytes'])
        data[name] = np.frombuffer(raw, dtype='<f8' if name == 'reference' else '<u2').reshape(desc['shape'])
    return data


def call(function, x, wq, s, b, step):
    k = len(x)
    return int(function(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        wq.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        k, step))


def check_witnesses(library):
    captured = json.loads(pinned(RAW / 'manifest.json', RAW_MANIFEST_SHA))
    observed = json.loads(pinned(OBS / 'manifest.json', OBS_MANIFEST_SHA))
    checked = 0
    for key, row, column, reference_hex in WITNESSES:
        record = next(r for r in captured['records'] if r['key'] == list(key))
        old = next(r for r in observed['records'] if r['key'] == list(key))
        arrays = arrays_of(record, old)
        outputs = outputs_of(old)
        x = arrays['x'][0, row]
        wq_column = arrays['wq'][column]
        s = arrays['s'][column]
        b = arrays['b'][column]
        reference = float(outputs['reference'][row, column])
        want = want_bits(reference)
        native = int(outputs['short_pass'][row, column])
        for step in (1, 2):
            got = call(library.run_new, x, wq_column, s, b, step)
            label = f'witness {key} r{row} c{column} step{step}'
            assert got == want == int(reference_hex, 16), \
                f'{label}: got {got:04x} want {want:04x} parent {reference_hex}'
            assert call(library.run_new_broadcast, x, wq_column, s, b, step) == got, \
                f'{label}: scratch/subgroup consumer mismatch'
            checked += 1
        old_two = call(library.run_old, x, wq_column, s, b, 2)
        assert old_two == native, \
            f'old model must reproduce native bits at {key} r{row} c{column}: ' \
            f'{old_two:04x} vs native {native:04x}'
        if (row, column) in OLD_RED or (row, column, key[3]) in OLD_RED:
            assert old_two != want, f'RED witness {key} r{row} c{column} no longer fails'
    assert checked == 16
    print(f'PASS actual witnesses: 16/16 new-model checks match retained '
          f'references (both word walks); old model reproduces native bits and '
          f'remains RED on {len(OLD_RED)} retained witnesses')


def check_fixtures(library):
    def run(x, wq, s, b, step):
        x = np.array(x, dtype='<f4')
        wq = np.array(wq, dtype='<u4')
        s = np.array(s, dtype='<f4')
        b = np.array(b, dtype='<f4')
        return call(library.run_new, x, wq, s, b, step), \
            call(library.run_old, x, wq, s, b, step)

    # Range fixture: K*2^-150 = 2^-133, the smallest bf16 subnormal.
    k = 131072
    x = np.full(k, 2.0 ** -75, dtype='<f4')
    q = np.full(k // 8, 0x11111111, dtype='<u4')
    s = np.full(k // 64, 2.0 ** -75, dtype='<f4')
    b = np.zeros(k // 64, dtype='<f4')
    for step in (1, 2):
        got = call(library.run_new, x, q, s, b, step)
        assert got == 0x0001, f'range fixture step{step}: {got:04x}'
    # Information-loss pair: identical (dot, input_sum), different cast.
    base_x = [1.0, 1.0] + [0.0] * 62 + [1.0] + [0.0] * 63
    s_ab = [1.0, 2.0 ** -7]
    b_ab = [2.0 ** -24, 0.0]
    x_ab = np.array(base_x, dtype='<f4')

    def wq_of(qs):
        words = []
        for w in range(16):
            value = 0
            for j in range(8):
                value |= int(qs[w * 8 + j]) << (4 * j)
            words.append(value)
        return np.array(words, dtype='<u4')

    q_a = [1, 1] + [0] * 62 + [1] + [0] * 63
    q_b = [0, 2] + [0] * 62 + [1] + [0] * 63
    for step in (1, 2):
        got_a, _ = run(x_ab, wq_of(q_a), s_ab, b_ab, step)
        got_b, _ = run(x_ab, wq_of(q_b), s_ab, b_ab, step)
        assert got_a == 0x4000, f'cast A step{step}: {got_a:04x}'
        assert got_b == 0x4001, f'cast B step{step}: {got_b:04x}'
    def products(xs, ws):
        xs, ws = np.array(xs, dtype='<f4'), np.array(ws, dtype='<f4')
        return int(library.run_products(
            xs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ws.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(xs)))

    # Signed low-part RNE through the store.
    for sign, expected in ((1, 0x3f81), (-1, 0xbf81)):
        assert products([sign] * 3, [1.0, 2.0**-8, 2.0**-30]) == expected
    assert products([2.0**120, -2.0**120, 1.0],
                    [2.0**120, 2.0**120, 1.0]) == 0x3f80
    # float(P) rounds up here: the product residual is -63, not uint wrap.
    assert products([scalar(0x3f810000)], [scalar(0x3f800041)]) == 0x3f81
    assert products([1.0], [float('inf')]) == 0x7f80
    assert products([-1.0], [float('inf')]) == 0xff80
    for result in (products([0.0], [float('inf')]),
                   products([1.0, -1.0], [float('inf'), float('inf')]),
                   products([float('nan')], [1.0])):
        assert result & 0x7f80 == 0x7f80 and result & 0x7f
    print('PASS fixtures: finite range and cast pair both walks; signed RNE; '
          'wide finite cancellation; negative product residual; Inf/NaN.')


def check_complete_max_seeds(library):
    captured = json.loads(pinned(RAW / 'manifest.json', RAW_MANIFEST_SHA))
    observed = json.loads(pinned(OBS / 'manifest.json', OBS_MANIFEST_SHA))
    checked = 0
    for record in captured['records']:
        old = next(r for r in observed['records'] if r['key'] == record['key'])
        arrays, outputs = arrays_of(record, old), outputs_of(old)
        k, n, t, _ = record['key']
        step = 2 if k % 512 == 0 and n % 8 == 0 else 1
        for row in range(t):
            for column in range(n):
                want = want_bits(outputs['reference'][row, column])
                got = call(library.run_new, arrays['x'][0, row], arrays['wq'][column],
                           arrays['s'][column], arrays['b'][column], step)
                assert got == want, \
                    f'complete max-seed {record["key"]} row{row} col{column}: {got:04x} vs {want:04x}'
                checked += 1
    assert checked == 67112
    print(f'PASS complete two max seeds: all {checked} outputs nearest retained references')


def check_old48(library):
    fixture = json.loads(pinned(OLD48, '38abd566dd79dd1fbca3ec5c026f9f6d3d465453f09e3b00ded94cfc2bb50c73'))
    manifest = json.loads(pinned(OLD48_OPERANDS / 'manifest.json',
                                '3cd540d34960b9f8cdcd40c8acdff30bc3b40171c61428063b5431f9432bf6ac'))
    assert len(fixture['records']) == len(manifest['records']) == 48
    old_bad = new_bad = checked = 0
    for record, captured in zip(fixture['records'], manifest['records'], strict=True):
        assert (record['t'], record['seed_index'], record['random_seed']) == \
            (captured['t'], captured['seed_index'], captured['random_seed'])
        arrays = {}
        for name, desc in captured['operands'].items():
            assert {key: desc[key] for key in ('shape', 'dtype', 'sha256')} == record['operands'][name]
            raw = pinned(OLD48_OPERANDS / desc['path'], desc['sha256'], desc['bytes'])
            array = np.frombuffer(raw, dtype='<u4' if desc['dtype'] == 'uint32' else '<u2')
            if desc['dtype'] == 'bf16':
                array = (array.astype('<u4') << 16).view('<f4')
            arrays[name] = array.reshape(desc['shape'])
        reference = np.array(record['reference'], dtype='<f8')
        assert hashlib.sha256(reference.tobytes()).hexdigest() == record['reference_sha256']
        for row in range(record['t']):
            for column in range(fixture['n']):
                index = row * fixture['n'] + column
                want = want_bits(reference[index])
                got = call(library.run_new, arrays['x'][0, row], arrays['wq'][column],
                           arrays['s'][column], arrays['b'][column], 1)
                bad_new = got != want
                bad_old = record['output_bf16'][index] != want
                assert not bad_new or bad_old, \
                    f'old48 regression T{record["t"]} seed{record["seed_index"]} row{row} col{column}'
                new_bad += bad_new
                old_bad += bad_old
                checked += 1
    assert checked == 31200 and new_bad < old_bad
    print(f'PASS old48 sweep: {checked} outputs, mismatches {old_bad} -> {new_bad} '
          f'vs nearest retained references (no regressions)')


def main():
    with tempfile.TemporaryDirectory() as directory:
        library = compile_model(directory)
        check_witnesses(library)
        check_fixtures(library)
        check_complete_max_seeds(library)
        check_old48(library)
    print('PASS: production scaled Q4 bf16 affine extraction; witnesses, '
          'counterexamples and all 50 captured cases green; old interpretation RED retained.')


if __name__ == '__main__':
    main()
