#!/usr/bin/env python3
"""M3 macOS baseline: the macOS/Metal twin of the PAR-45 combined
decode + prefill single-run cell.

Same recipe, same digest, same output schema as the Linux PAR-45 runner,
so m3_compare_card.py can verify snapshot/prompt/SHA parity between the
two sides and emit the parity-card rows:

- 1 warmup + 5 timed reps of decode (tokens 2..N timed; TTFT prefill
  recorded) and of pure prefill (model(tokens[None], cache) + mx.eval),
  from the same process / model load.
- digest(ids) = sha256(','.join(ids)); input_sha256 over the tiled
  prompt; output_sha256 per rep.

Linux-only provenance (boot_id, /proc/self/maps, DEFER_TOKENS, VK ICD)
is replaced by the macOS equivalents: the controller window nonce
(~/.m2-window-<nonce> dir) for --nonce and mx.device_info() for the
device block.

usage: m3_macos_baseline.py OUTDIR MODEL [--nonce HEX32] [--ctx 512]
       [--decode-tokens 64] [--fetch]
"""
import argparse
import hashlib
from importlib import metadata
from itertools import islice
import json
import os
from pathlib import Path
import platform
import re
from statistics import median
from time import perf_counter

MODELS = {
    '2b': ('SiddhJagani/Qwen3.8-2B-mlx-4Bit', '0867d98bfb174b042d88461c0e7c97b86b34b381'),
    '4b': ('mlx-community/Qwen3-4B-Instruct-2507-4bit', '50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b'),
    '9b': ('mlx-community/Qwen3.5-9B-MLX-4bit', '938d8919941c6e7efd3c7150eff7fe9d12afa631'),
}
PROMPT = 'The history of France begins in antiquity.'
SAMPLES = 5


def digest(ids):
    return hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest()


def tile_ids(base, ctx):
    """Prompt ids tiled to ctx, the PAR-45 shape."""
    return (base * (ctx // len(base) + 1))[:ctx]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def summarize(samples):
    assert len(samples) == SAMPLES and all(s > 0 for s in samples)
    middle = median(samples)
    return {
        'timed_tok_s': samples,
        'median_tok_s': middle,
        'first_vs_median_pct': 100.0 * (samples[0] / middle - 1.0),
        'last_vs_median_pct': 100.0 * (samples[-1] / middle - 1.0),
    }


def check_window(nonce):
    """The macOS controller window: a run is admitted only while the
    controller's nonce marker dir exists (~/.m2-window-<hex32>), the same
    gate the PAR-45 macOS reruns used."""
    if not re.fullmatch(r'[0-9a-f]{32}', nonce or '') or \
            not (Path.home() / f'.m2-window-{nonce}').is_dir():
        raise SystemExit('REFUSED: no active controller nonce '
                         '(expected ~/.m2-window-<hex32>)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out', type=Path)
    p.add_argument('model', choices=MODELS)
    p.add_argument('--nonce', required=True)
    p.add_argument('--ctx', type=int, default=512)
    p.add_argument('--decode-tokens', type=int, default=64)
    p.add_argument('--fetch', action='store_true')
    a = p.parse_args()

    assert platform.system() == 'Darwin', 'the macOS baseline runs under macOS/Metal'
    check_window(a.nonce)

    import inspect
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache

    mx.set_default_device(mx.gpu)
    repo, revision = MODELS[a.model]
    snap = Path(snapshot_download(repo, revision=revision,
                                  local_files_only=not a.fetch))
    model, tokenizer = load(str(snap))
    base = tokenizer.encode(PROMPT)
    assert base
    a.out.mkdir(parents=True, exist_ok=True)

    ids = tile_ids(base, a.ctx)
    tokens = mx.array(ids)

    try:
        dev_info = mx.device_info()
    except Exception:
        dev_info = {}

    # 1) PAR-45 decode: 1 warmup + 5 timed, tokens 2..N timed; TTFT prefill.
    dec_samples, ttft_samples, dec_outputs = [], [], []
    for rep in range(SAMPLES + 1):
        cache = make_prompt_cache(model)
        t_pf0 = perf_counter()
        with_stream = generate_step(
            tokens, model, max_tokens=a.decode_tokens, prompt_cache=cache
        )
        first, _ = next(with_stream)
        mx.synchronize()
        ttft_rate = a.ctx / (perf_counter() - t_pf0)
        seq = [int(first)]
        t0 = perf_counter()
        seq.extend(int(token) for token, _ in islice(with_stream, a.decode_tokens - 1))
        assert len(seq) == a.decode_tokens
        mx.synchronize()
        rate = (a.decode_tokens - 1) / (perf_counter() - t0)
        if rep:
            dec_samples.append(rate)
            ttft_samples.append(ttft_rate)
            dec_outputs.append(seq)

    # 2) PAR-45 pure prefill: 1 warmup + 5 timed, model(tokens[None], cache) + mx.eval.
    pf_samples, pf_outputs = [], []
    for rep in range(SAMPLES + 1):
        cache = make_prompt_cache(model)
        t0 = perf_counter()
        y = model(tokens[None], cache=cache)
        mx.eval(y)
        rate = a.ctx / (perf_counter() - t0)
        seq = [int(mx.argmax(y[0, -1]))]
        if rep:
            pf_samples.append(rate)
            pf_outputs.append(seq)

    core_so = os.path.realpath(mx.__file__)
    gen_py = os.path.realpath(inspect.getsourcefile(generate_step))
    meta = {
        'repo': repo,
        'revision': revision,
        'prompt': PROMPT,
        'device': dev_info,
        'os': platform.platform(),
        'nonce': a.nonce,
        'warmups_discarded': 1,
        'decode_tokens': a.decode_tokens,
        'defer_tokens': os.environ.get('MLX_OMARCHY_DEFER_TOKENS'),
        'vk_icd_filenames': os.environ.get('VK_ICD_FILENAMES'),
        'core_so_sha256': sha256_file(core_so),
        'mlx_lm': metadata.version('mlx-lm'),
        'mlx_lm_generate_path': gen_py,
        'mlx_lm_generate_sha256': sha256_file(gen_py),
        'runner_sha256': sha256_file(__file__),
        'side': 'macos',
    }
    for dist in ('mlx-omarchy', 'mlx'):
        try:
            meta['mlx'] = metadata.version(dist)
            break
        except metadata.PackageNotFoundError:
            continue

    dec_res = {
        **meta,
        **summarize(dec_samples),
        'ttft_prefill': summarize(ttft_samples),
        'mode': 'decode',
        'ctx': a.ctx,
        'input_sha256': digest(ids),
        'output_sha256': [digest(s) for s in dec_outputs],
        'output_ids': dec_outputs,
    }
    pf_res = {
        **meta,
        **summarize(pf_samples),
        'mode': 'prefill',
        'ctx': a.ctx,
        'input_sha256': digest(ids),
        'output_sha256': [digest(s) for s in pf_outputs],
        'output_ids': pf_outputs,
    }
    (a.out / f'decode-{a.model}-{a.ctx}.json').write_text(json.dumps(dec_res, indent=2, default=str) + '\n')
    (a.out / f'prefill-{a.model}-{a.ctx}.json').write_text(json.dumps(pf_res, indent=2, default=str) + '\n')
    print(
        f"FINAL {a.model} ctx={a.ctx}: dec={dec_res['median_tok_s']:.3f} tok/s "
        f"pf={pf_res['median_tok_s']:.2f} tok/s ttft_pf={dec_res['ttft_prefill']['median_tok_s']:.2f} "
        f"dec_sha={dec_res['output_sha256'][0][:16]} pf_sha={pf_res['output_sha256'][0][:16]}",
        flush=True,
    )


if __name__ == '__main__':
    main()
