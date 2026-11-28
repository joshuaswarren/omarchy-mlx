#!/usr/bin/env python3
"""M3 parity-card comparator: verifies a macOS Metal baseline cell set and
a Linux PAR-45 cell set measure the same thing, then emits the parity-card
rows for docs/bench-rows.tsv (the format gen_readme_tables.py turns into
the README/BENCHMARKS parity tables).

Inputs are the per-cell JSONs written by the PAR-45 cell runners on both
sides (Linux par45_final_cell.py, macOS m3_macos_baseline.py):
decode-{model}-{ctx}.json / prefill-{model}-{ctx}.json.

Verifies, per (repo, ctx, mode) cell and across sides:
  - same model repo + pinned revision (snapshot parity)
  - same prompt string and same tiled prompt (input_sha256)
  - same decode_tokens budget on decode cells
  - warmups_discarded equal on both sides (warmup parity)
  - every rep's output_sha256 identical within a cell AND across sides
    (SHA parity: identical greedy output tokens)

Any violation refuses the whole comparison with a nonzero exit; nothing
is emitted. A cell present on one side only is PENDING-skipped with a
note on stderr, not silently dropped.

Emission (stdout): one TSV line per (ctx, metric), the bench-rows.tsv
schema, with both the linux and macos columns filled so the generated
table shows "linux / macos = share%".

usage: m3_compare_card.py --linux J... --macos J... --chip "14-inch M2 Max (G14C)"
       [--date YYYY-MM-DD] [--governor G] [--driver-name N] [--driver-commit C]
       [--wheel-commit W] [--header]
"""
import argparse
import datetime
import json
from statistics import median
import sys

REPO_NAMES = {
    'SiddhJagani/Qwen3.8-2B-mlx-4Bit': 'Qwen3.8-2B',
    'mlx-community/Qwen3-4B-Instruct-2507-4bit': 'Qwen3-4B-Instruct-2507',
    'mlx-community/Qwen3.5-9B-MLX-4bit': 'Qwen3.5-9B',
}


def refuse(why):
    raise SystemExit(f'REFUSE: {why}')


def load_cell(path):
    try:
        with open(path, encoding='utf-8') as fh:
            d = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        refuse(f'{path}: unreadable cell ({exc})')
    for field in ('repo', 'revision', 'prompt', 'mode', 'ctx',
                  'input_sha256', 'output_sha256', 'timed_tok_s',
                  'warmups_discarded'):
        if field not in d:
            refuse(f'{path}: missing field {field!r}')
    if d['mode'] not in ('decode', 'prefill'):
        refuse(f'{path}: unknown mode {d["mode"]!r}')
    return d


def key_of(d):
    return (d['repo'], d['ctx'], d['mode'])


def model_name(repo):
    return REPO_NAMES.get(repo, repo.rsplit('/', 1)[-1])


def verify_pair(lx, mac, where):
    """Cross-side checks for one (repo, ctx, mode) cell."""
    for field in ('revision', 'prompt', 'input_sha256'):
        if lx[field] != mac[field]:
            refuse(f'{where}: {field} differs '
                   f'({lx[field]!r} vs {mac[field]!r})')
    if lx['warmups_discarded'] != mac['warmups_discarded']:
        refuse(f'{where}: warmups_discarded differs '
               f'({lx["warmups_discarded"]} vs {mac["warmups_discarded"]})')
    if lx['mode'] == 'decode' and lx.get('decode_tokens') != mac.get('decode_tokens'):
        refuse(f'{where}: decode_tokens differs '
               f'({lx.get("decode_tokens")} vs {mac.get("decode_tokens")})')
    for side, d in (('linux', lx), ('macos', mac)):
        shas = set(d['output_sha256'])
        if len(shas) != 1:
            refuse(f'{where}: {side} side nondeterministic, '
                   f'{len(shas)} distinct output_sha256 over {len(d["output_sha256"])} reps')
    if lx['output_sha256'][0] != mac['output_sha256'][0]:
        refuse(f'{where}: output SHA parity broken '
               f'({lx["output_sha256"][0][:16]} vs {mac["output_sha256"][0][:16]})')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--linux', nargs='+', required=True)
    p.add_argument('--macos', nargs='+', required=True)
    p.add_argument('--chip', required=True)
    p.add_argument('--date', default=datetime.date.today().isoformat())
    p.add_argument('--governor', default='performance (filler on)')
    p.add_argument('--driver-name', default='package')
    p.add_argument('--driver-commit', default='')
    p.add_argument('--wheel-commit', default='')
    p.add_argument('--header', action='store_true')
    a = p.parse_args()

    sides = {}
    for side, paths in (('linux', a.linux), ('macos', a.macos)):
        cells = {}
        for path in paths:
            d = load_cell(path)
            k = key_of(d)
            if k in cells:
                refuse(f'{path}: duplicate cell {k} on the {side} side')
            cells[k] = d
        sides[side] = cells

    rows = []
    for k in sorted(set(sides['linux']) | set(sides['macos']),
                    key=lambda k: (k[0], k[1], k[2])):
        repo, ctx, mode = k
        where = f'{repo} ctx={ctx} {mode}'
        lx, mac = sides['linux'].get(k), sides['macos'].get(k)
        if lx is None or mac is None:
            missing = 'linux' if lx is None else 'macos'
            print(f'PENDING {where}: {missing} side missing', file=sys.stderr)
            continue
        verify_pair(lx, mac, where)
        med = {
            'linux': median(lx['timed_tok_s']),
            'macos': median(mac['timed_tok_s']),
        }
        runs = len(lx['timed_tok_s']) + len(mac['timed_tok_s'])
        rows.append([
            a.chip, model_name(repo), mode,
            f'{med["linux"]:.3f}', f'{med["macos"]:.3f}',
            a.governor, a.driver_name, a.driver_commit, a.wheel_commit,
            str(runs),
            '', '', '', '', '0', '0', a.date, '', '', '',
        ])

    header = ('chip\tmodel\tmetric\tlinux\tmacos\tgovernor\tdriver_name\t'
              'driver_commit\twheel_commit\truns\tstart_load_max\tane_busy\t'
              'control_spread_pct\tp_spread_pct\tnoisy\tinterleaved\tdate\t'
              'caption\tpv_decode\tpv_prefill')
    if a.header:
        print(header)
    for r in rows:
        line = '\t'.join(r)
        if len(r) != len(header.split('\t')):
            refuse(f'internal error: emitted {len(r)} columns')
        print(line)


if __name__ == '__main__':
    main()
