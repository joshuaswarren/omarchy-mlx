"""Server decode loops skip the per-token logprob device read (0.32 series).

mlx_lm/server.py builds a Response for every generated token with
``r.logprobs[r.token].item()``: a gather graph plus a blocking device read
per token (3.48 ms/token of the 19% serve-loop decode gap measured on a
13-inch M1 with gemma-4-E4B) even when the request asked for neither
``logprobs`` nor ``top_logprobs > 0``. The handler reads Response.logprob
only when ``args.logprobs`` is set and Response.top_tokens only when
``args.top_logprobs > 0``, so the patch gates the read on a per-request
flag (``batch_results[uid]`` in the batched loop, a hoisted local in
``_serve_single``) and ``MLX_OMARCHY_SERVER_LOGPROB_READ=1`` restores the
upstream always-read.

Pinned here on the REAL ResponseGenerator driven in-process on the CPU
tiny model (no HTTP), against the pristine 0.32.0 wheel:

- generated token ids and text are equal with and without the patch;
- no ``mx.array.item()`` call happens per token when logprobs are not
  requested (steady-state count is exactly 0);
- the read happens per token and returns the real logprob when
  ``logprobs=True`` and when ``top_logprobs > 0``;
- the kill switch restores the read;
- the patch applies with fuzz 0 and the apply script wires it: the 0.32
  series applies it, the 0.31.3 series skips it and keeps upstream bytes.

Needs network (or a pip cache) for ``pip download mlx-lm==0.32.0`` /
``mlx-lm==0.31.3``; skips those cases when pip cannot provide the wheels,
and needs the CPU tiny model at /dev/shm/w7k-tinymodel.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import zipfile

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
PATCH = REPO / "patches" / "mlx-lm-0.32" / "mlx-lm-server-skip-logprob-read.patch"
APPLY = REPO / "scripts" / "apply-mlx-lm-patches.sh"
MARKER = "MLX_OMARCHY_SERVER_LOGPROB_READ"
TINY_MODEL = pathlib.Path("/dev/shm/w7k-tinymodel")
N_TOKENS = 24

DRIVER = """
import inspect, json, os, sys

model_dir, tree, n_tokens, want_logprobs, top_logprobs = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1", int(sys.argv[5]),
)
sys.path.insert(0, tree)
import mlx.core as mx

CALLS = {"item": 0}
_orig_item = mx.array.item


def _counting_item(self, *a, **k):
    CALLS["item"] += 1
    return _orig_item(self, *a, **k)


mx.array.item = _counting_item

import mlx_lm.server as sm  # noqa: E402

PROMPT = "Write a short story about a lighthouse keeper who finds a message in a bottle."


def request_args(n):
    return sm.GenerationArguments(
        model=sm.ModelDescription(model="default_model", draft="default_model", adapter=None),
        sampling=sm.SamplingArguments(temperature=0.0, top_p=1.0, top_k=0, min_p=0.0,
                                      xtc_probability=0.0, xtc_threshold=0.1),
        logits=sm.LogitsProcessorArguments(logit_bias=None, repetition_penalty=0.0,
                                           repetition_context_size=20, presence_penalty=0.0,
                                           presence_context_size=20, frequency_penalty=0.0,
                                           frequency_context_size=20),
        stop_words=[], max_tokens=n, num_draft_tokens=0,
        logprobs=want_logprobs, top_logprobs=top_logprobs, seed=None,
        chat_template_kwargs=None,
    )


def one(rg, n):
    req = sm.CompletionRequest("chat", "", [{"role": "user", "content": PROMPT}], None, None)
    CALLS["item"] = 0
    _ctx, it = rg.generate(req, request_args(n))
    raw_ids, texts, raw_lps, tops, finish = [], [], [], [], None
    for gen in it:
        raw_ids.append(gen.token)
        texts.append(gen.text)
        raw_lps.append(gen.logprob)
        tops.append(gen.top_tokens)
        finish = gen.finish_reason or finish
    calls = CALLS["item"]
    ids = [int(t) for t in raw_ids]
    lps = [None if v is None else float(v) for v in raw_lps]
    tops_first = [dict(t) for t in tops[0]] if tops and tops[0] is not None else None
    return {"n": len(ids), "finish": finish, "calls": calls, "ids": ids,
            "text": "".join(texts), "logprobs": lps, "top_first": tops_first}


def driver(host, port, response_generator, server_class=None, handler_class=None):
    rg = response_generator
    try:
        warm = one(rg, 8)
        steady = one(rg, n_tokens)
        print("RESULT " + json.dumps({"warm_calls": warm["calls"], "steady": steady}))
    finally:
        rg.stop_and_join()
    os._exit(0)


driver.__signature__ = inspect.signature(sm._run_http_server)
sm._run_http_server = driver
sys.argv = ["mlx_lm.server", "--model", model_dir, "--host", "127.0.0.1", "--port", "0",
            "--max-tokens", str(n_tokens), "--decode-concurrency", "1",
            "--prompt-concurrency", "1"]
sm.main()
"""


def _wheel(root: pathlib.Path, version: str) -> pathlib.Path:
    """A venv-shaped tree whose mlx_lm comes straight from the pinned wheel."""
    dl = root / "dl"
    r = subprocess.run(
        [sys.executable, "-m", "pip", "download", "--no-deps", "--quiet",
         "-d", str(dl), f"mlx-lm=={version}"],
        capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        pytest.skip(f"pip download mlx-lm=={version} failed, "
                    f"no pristine tree: {r.stderr.strip()[-160:]}")
    (wheel,) = dl.glob("mlx_lm-*.whl")
    site = root / "lib" / "python3.11" / "site-packages"
    with zipfile.ZipFile(wheel) as z:
        z.extractall(site)
    return site


def _apply_mypatch(site: pathlib.Path) -> None:
    """The patch must apply with the installer's exact fuzz=0 semantics."""
    dry = subprocess.run(["patch", "--dry-run", "--directory", str(site),
                          "--strip=1", "--forward", "--fuzz=0"],
                         stdin=PATCH.open("rb"), capture_output=True, text=True)
    assert dry.returncode == 0, f"fuzz-0 dry run failed:\n{dry.stdout}\n{dry.stderr}"
    r = subprocess.run(["patch", "--directory", str(site), "--strip=1",
                        "--forward", "--fuzz=0"],
                       stdin=PATCH.open("rb"), capture_output=True, text=True)
    assert r.returncode == 0, f"patch application failed:\n{r.stdout}\n{r.stderr}"


def _run_apply_script(tree: pathlib.Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("MLX_OMARCHY_CONV_RING", None)
    return subprocess.run(["bash", str(APPLY), str(tree)],
                          capture_output=True, text=True, env=env, timeout=600)


def _drive(tree_site: pathlib.Path, tmp_path: pathlib.Path,
           logprobs: bool, top_logprobs: int,
           forced: bool = False) -> dict:
    driver_py = tmp_path / "driver.py"
    driver_py.write_text(DRIVER)
    env = dict(os.environ)
    env.pop(MARKER, None)
    if forced:
        env[MARKER] = "1"
    r = subprocess.run(
        [sys.executable, str(driver_py), str(TINY_MODEL), str(tree_site),
         str(N_TOKENS), "1" if logprobs else "0", str(top_logprobs)],
        capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, f"driver failed:\n{r.stdout[-400:]}\n{r.stderr[-800:]}"
    line = next(l for l in r.stdout.splitlines() if l.startswith("RESULT "))
    return json.loads(line[len("RESULT "):])["steady"]


@pytest.fixture(scope="module")
def trees(tmp_path_factory):
    """stock: pristine 0.32.0 wheel; patched: the same tree plus this patch."""
    root = tmp_path_factory.mktemp("logprob-skip")
    stock = _wheel(root / "stock", "0.32.0")
    patched = root / "patched"
    shutil.copytree(stock, patched)
    _apply_mypatch(patched)
    return stock, patched


@pytest.mark.skipif(not TINY_MODEL.exists(), reason="no CPU tiny model")
class TestGeneratedTokensUnchanged:
    def test_ids_and_text_equal_stock(self, trees, tmp_path):
        stock, patched = trees
        s = _drive(stock, tmp_path, logprobs=False, top_logprobs=-1)
        p = _drive(patched, tmp_path, logprobs=False, top_logprobs=-1)
        assert s["finish"] == p["finish"] == "length"
        assert p["ids"] == s["ids"]
        assert p["text"] == s["text"]

    def test_logprobs_mode_ids_and_values_equal_stock(self, trees, tmp_path):
        stock, patched = trees
        s = _drive(stock, tmp_path, logprobs=True, top_logprobs=0)
        p = _drive(patched, tmp_path, logprobs=True, top_logprobs=0)
        assert p["ids"] == s["ids"]
        assert p["logprobs"] == s["logprobs"]
        assert all(v is not None for v in p["logprobs"])


@pytest.mark.skipif(not TINY_MODEL.exists(), reason="no CPU tiny model")
class TestItemCallCounts:
    def test_no_read_per_token_when_not_requested(self, trees, tmp_path):
        _, patched = trees
        p = _drive(patched, tmp_path, logprobs=False, top_logprobs=-1)
        assert p["n"] == N_TOKENS
        # Steady state: nothing evaluates the sampled token's logprob when no
        # consumer asked for it (upstream reads it exactly once per token).
        assert p["calls"] == 0, (
            f"expected no mx.array.item() calls, got {p['calls']} "
            f"for {p['n']} tokens")

    def test_read_per_token_when_logprobs_requested(self, trees, tmp_path):
        _, patched = trees
        p = _drive(patched, tmp_path, logprobs=True, top_logprobs=0)
        assert p["calls"] >= p["n"]
        assert all(v is not None for v in p["logprobs"])

    def test_read_per_token_when_top_logprobs_requested(self, trees, tmp_path):
        _, patched = trees
        p = _drive(patched, tmp_path, logprobs=False, top_logprobs=3)
        assert p["calls"] >= p["n"]
        assert p["top_first"] is not None and len(p["top_first"]) == 3
        assert set(p["top_first"][0]) == {"id", "token", "logprob"}

    def test_kill_switch_restores_the_read(self, trees, tmp_path):
        stock, patched = trees
        s = _drive(stock, tmp_path, logprobs=False, top_logprobs=-1)
        k = _drive(patched, tmp_path, logprobs=False, top_logprobs=-1, forced=True)
        assert k["calls"] >= k["n"]
        assert k["ids"] == s["ids"]
        assert k["logprobs"] == s["logprobs"]


class TestWiring:
    def test_apply_script_applies_on_032(self, tmp_path):
        site = _wheel(tmp_path / "w32", "0.32.0")
        r = _run_apply_script(tmp_path / "w32")
        detail = r.stdout[-800:] + r.stderr[-400:]
        assert r.returncode == 0, detail
        assert "applied: mlx-lm-server-skip-logprob-read.patch" in r.stdout
        assert (site / "mlx_lm" / "server.py").read_text().count(MARKER) >= 2

    def test_apply_script_skips_on_0313(self, tmp_path):
        site = _wheel(tmp_path / "w31", "0.31.3")
        r = _run_apply_script(tmp_path / "w31")
        detail = r.stdout[-800:] + r.stderr[-400:]
        assert r.returncode == 0, detail
        assert "server-skip-logprob-read: 0.32 series only; skipped" in r.stdout
        # The 0.31.3 series keeps upstream behavior: no skip flag anywhere in
        # the served tree (its series still patches server.py for other
        # changes, so byte identity is not the contract).
        assert MARKER not in (site / "mlx_lm" / "server.py").read_text()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
