# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Behavior regression for scripts/patch-mlx-lm-batch-greedy-head.py.

The patch sends greedy GenerationBatch steps (mlx-lm BatchGenerator, the
oMLX serve path) through the pruned greedy head and leaves the logprobs
lazy. These tests run the real patchers (mlx-lm-greedy-prune, then
batch-greedy-head) on a copy of the installed stock mlx_lm, imported as
bgh_mlx_lm, with a tiny tied-embedding llama. The backend greedy kernel is
not on CPU, so _greedy_head is replaced by a head whose token is the argmax
of the same tied projection. They require:

  1. greedy batches take the head and produce the same tokens and (when
     evaluated) the same logprobs as the full-logits step,
  2. the step never evaluates the lazy logprobs,
  3. sampled batches, batches with logits processors, and
     MLX_OMARCHY_BATCH_GREEDY=0 keep the full-logits step, and the patcher
     stays wired after the greedy-prune patch.
"""

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

REPO = Path(__file__).resolve().parent.parent
PATCHER = REPO / "scripts" / "patch-mlx-lm-batch-greedy-head.py"
PRUNE = REPO / "patches" / "mlx-lm-0.32" / "mlx-lm-greedy-prune.patch"
PROMPTS = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [3, 1, 4, 1, 5]]


@pytest.fixture(scope="module")
def pkg(tmp_path_factory):
    import mlx_lm

    src = Path(mlx_lm.__file__).parent
    text = (src / "generate.py").read_text()
    if "class GenerationBatch:" not in text:
        pytest.skip("installed mlx_lm predates GenerationBatch (0.31 line)")
    if "MLX_OMARCHY_BATCH_GREEDY" in text:
        pytest.skip(f"installed mlx_lm is already batch-greedy patched: {src}")
    root = tmp_path_factory.mktemp("bgh")
    site = root / "lib" / "python3" / "site-packages"
    shutil.copytree(src, site / "mlx_lm", ignore=shutil.ignore_patterns("__pycache__"))
    if "def _greedy_head(model):" not in text:
        subprocess.run(["patch", f"--directory={site}", "--strip=1", "--forward", "--fuzz=0", "--input",
                        str(PRUNE)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, str(PATCHER), str(root)], check=True)
    (site / "mlx_lm").rename(site / "bgh_mlx_lm")
    sys.path.insert(0, str(site))
    yield str(site), importlib.import_module("bgh_mlx_lm.generate")
    sys.path.remove(str(site))


@pytest.fixture
def model(pkg):
    _, gen = pkg
    llama = importlib.import_module("bgh_mlx_lm.models.llama")
    args = llama.ModelArgs(model_type="llama", hidden_size=64, num_hidden_layers=2, intermediate_size=128,
                           num_attention_heads=4, rms_norm_eps=1e-5, vocab_size=97, num_key_value_heads=2,
                           tie_word_embeddings=True)
    mx.random.seed(0)
    m = llama.Model(args)
    mx.eval(m.parameters())
    return m


@pytest.fixture
def head_calls(pkg, model, monkeypatch):
    """Replace _greedy_head with the tied projection's argmax; count its token calls."""
    _, gen = pkg
    emb = model.model.embed_tokens
    calls = []

    def fake(_model):
        def head(hidden):
            raise AssertionError("the batched step must not call the 1-row head")

        def token(hidden):
            calls.append(hidden.shape)
            return mx.argmax(emb.as_linear(hidden), axis=-1)

        head.body, head.token, head.logits = model.model, token, emb.as_linear
        return head

    monkeypatch.setattr(gen, "_greedy_head", fake)
    return calls


def run(gen, model, sampler, processors=None):
    """Two requests, two steps, then two more (continuous batching merges them into the running batch)."""
    bg = gen.BatchGenerator(model, max_tokens=10, sampler=sampler)
    tokens, logprobs, uids = {}, {}, []

    def insert(prompts):
        procs = processors[: len(prompts)] if processors else None
        for u in bg.insert(prompts, max_tokens=[10] * len(prompts), logits_processors=procs):
            uids.append(u)
            tokens[u], logprobs[u] = [], []

    def step():
        responses = bg.next_generated()
        for r in responses:
            tokens[r.uid].append(r.token)
            logprobs[r.uid].append(r.logprobs)
        return responses

    insert(PROMPTS[:2])
    step()
    step()
    insert(PROMPTS[2:])
    while step():
        pass
    bg.close()
    return [tokens[u] for u in uids], [logprobs[u] for u in uids]


def test_greedy_batch_takes_the_head_with_identical_output(pkg, model, head_calls, monkeypatch):
    _, gen = pkg
    monkeypatch.setattr(gen, "_BATCH_GREEDY", False)
    want_tokens, want_logprobs = run(gen, model, gen.greedy_sampler)
    assert not head_calls
    monkeypatch.setattr(gen, "_BATCH_GREEDY", True)

    # Keep every evaluated object alive so no id() is reused by a later array.
    kept, evaluated = [], set()
    real_eval, real_async = mx.eval, mx.async_eval

    def record(fn):
        def wrapper(*args):
            stack = list(args)
            while stack:
                a = stack.pop()
                if isinstance(a, (list, tuple)):
                    stack.extend(a)
                else:
                    kept.append(a)
                    evaluated.add(id(a))
            return fn(*args)
        return wrapper

    monkeypatch.setattr(mx, "eval", record(real_eval))
    monkeypatch.setattr(mx, "async_eval", record(real_async))
    got_tokens, got_logprobs = run(gen, model, gen.greedy_sampler)
    monkeypatch.setattr(mx, "eval", real_eval)
    monkeypatch.setattr(mx, "async_eval", real_async)

    assert head_calls and all(s == (1, 64) for s in head_calls)
    assert got_tokens == want_tokens
    lazy = [lp for row in got_logprobs for lp in row[1:]]
    assert lazy and not any(id(lp) in evaluated for lp in lazy), "the greedy step evaluated its logprobs"
    for got_row, want_row in zip(got_logprobs, want_logprobs):
        for got, want in zip(got_row, want_row):
            assert bool(mx.allclose(got, want, atol=1e-5).item())


def test_sampled_and_processed_batches_keep_the_full_logits_step(pkg, model, head_calls):
    _, gen = pkg
    sample_utils = importlib.import_module("bgh_mlx_lm.sample_utils")
    run(gen, model, sample_utils.make_sampler(temp=0.7))
    run(gen, model, lambda x: mx.argmax(x, axis=-1))
    penalty = sample_utils.make_logits_processors(repetition_penalty=1.1)
    run(gen, model, gen.greedy_sampler, processors=[penalty] * len(PROMPTS))
    assert not head_calls


def test_kill_switch_and_wiring(pkg):
    site, _ = pkg
    env = dict(os.environ, MLX_OMARCHY_BATCH_GREEDY="0", PYTHONPATH=site)
    code = ("import importlib; g = importlib.import_module('bgh_mlx_lm.generate'); "
            "assert g._BATCH_GREEDY is False")
    assert subprocess.run([sys.executable, "-c", code], env=env).returncode == 0
    apply = (REPO / "scripts" / "apply-mlx-lm-patches.sh").read_text()
    assert apply.index("apply mlx-lm-greedy-prune.patch") < apply.index("patch-mlx-lm-batch-greedy-head.py")
