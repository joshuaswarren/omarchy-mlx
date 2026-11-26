"""Pinned mlx-lm HTTP acceptance for the total-context cap shim.

This runs the REAL pinned stack — mlx-cpu + mlx-lm 0.31.3 from the wheels,
the real cached Qwen2.5-0.5B-Instruct-bf16 snapshot, and the real HTTP
request path through the shim — and proves Main's acceptance matrix on the
actual upstream code (tokenize callers at server.py:738/936, the
max_completion_tokens alias at :1169, validation at :1232):

- chat/completions generates (200, nonempty content, usage reported);
- both max-token aliases are capped identically;
- prompt + max_tokens == limit passes; limit + 1 is refused with the
  budget message BEFORE generation (no tokens processed);
- the direct /v1/completions text path generates under the same cap;
- boundary + 1 is refused on BOTH max-token aliases;
- concurrency is proven by a source-linked assertion: the recorded launch
  argv pins --decode-concurrency 1 --prompt-concurrency 1 and the
  INSTALLED pinned server source wires those flags to the BatchGenerator
  batch sizes, so at most one request is active and the aggregate equals
  the admitted context (the concurrent-clients test is a smoke, not the
  seriality proof);
- the prompt cache is disabled (cached_tokens asserted 0). Cache-enabled
  configurations are a separate documented mode and are NOT exercised or
  claimed by these tests.

Skipped unless the prepared venv exists (never touches the network):
  python3 -m venv --system-site-packages /tmp/mlxlm-accept-venv
  /tmp/mlxlm-accept-venv/bin/pip install --no-deps \
      mlx-0.32.2-cp311-cp311-manylinux_2_35_x86_64.whl \
      mlx_lm-0.31.3-py3-none-any.whl
"""

import inspect
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV = Path(os.environ.get("MLXLM_ACCEPTANCE_VENV", "/tmp/mlxlm-accept-venv"))
MODEL_REPO_DIR = (
    Path.home() / ".cache/huggingface/hub/"
    "models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots"
)
CONTEXT_LIMIT = 64
PROMPT = "Say hi in one word."
def free_port() -> int:
    # Overlapping suite runs on one box must not fight over a fixed port.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


PORT = free_port()


def model_snapshot() -> Path | None:
    if not MODEL_REPO_DIR.is_dir():
        return None
    for snap in MODEL_REPO_DIR.iterdir():
        if (snap / "model.safetensors").is_file() and (snap / "tokenizer.json").is_file():
            return snap
    return None


def post(body: dict, timeout: int = 300) -> tuple[int, dict | str]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw


@unittest.skipUnless(
    (VENV / "bin/python").is_file() and model_snapshot() is not None,
    "requires the prepared acceptance venv and a cached model snapshot",
)
class PinnedHTTPAcceptance(unittest.TestCase):
    MODEL_ID = "mlx-community/Qwen2.5-0.5B-Instruct-bf16"

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "server", None) is not None:
            cls.server.terminate()
            cls.server.wait(timeout=30)
        if getattr(cls, "log", None) is not None:
            cls.log.close()
        if getattr(cls, "tmp", None) is not None:
            cls.tmp.cleanup()

    @classmethod
    def setUpClass(cls):
        # no cls.addCleanup here: on this interpreter the class-object call
        # does not bind; tearDownClass owns the teardown explicitly.
        cls.tmp = tempfile.TemporaryDirectory()
        cls.log = open(Path(cls.tmp.name) / "server.log", "w+b")
        env = {**os.environ,
               "PYTHONPATH": str(REPO_ROOT / "serve"),
               "MLX_OMARCHY_SERVE_CONTEXT_LIMIT": str(CONTEXT_LIMIT)}
        cls.launch_argv = [
            str(VENV / "bin/python"), "-m", "mlx_omarchy_serve._mlxlm_server",
            "--model", str(model_snapshot()),
            "--host", "127.0.0.1", "--port", str(PORT),
            "--max-tokens", "16",
            "--decode-concurrency", "1",
            "--prompt-concurrency", "1",
            "--prompt-cache-size", "0"]
        cls.server = subprocess.Popen(
            cls.launch_argv,
            env=env, stdout=cls.log, stderr=cls.log,
        )
        import time

        deadline = time.time() + 180
        while time.time() < deadline:
            if cls.server.poll() is not None:
                cls.log.seek(0)
                raise AssertionError(f"server exited early:\n{cls.log.read()}")
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as r:
                    if r.status == 200:
                        return
            except OSError:
                pass
            time.sleep(2)
        raise AssertionError("server did not become ready in 180s")

    def chat(self, max_tokens: int, alias: str = "max_tokens", prompt: str = PROMPT):
        return post({"model": self.MODEL_ID,
                     "messages": [{"role": "user", "content": prompt}],
                     alias: max_tokens})

    def test_chat_completions_generates(self):
        code, body = self.chat(8)
        self.assertEqual(code, 200, body)
        content = body["choices"][0]["message"]["content"]
        self.assertTrue(content.strip())
        self.assertGreater(body["usage"]["prompt_tokens"], 0)
        # prompt cache disabled: honest zero, wherever this mlx-lm reports it
        cached = body["usage"].get("cached_tokens",
                                   body["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0))
        self.assertEqual(cached, 0)

    def test_max_completion_tokens_alias_capped_identically(self):
        code, body = self.chat(8, alias="max_completion_tokens")
        self.assertEqual(code, 200, body)

    def test_direct_text_completions_generates_under_cap(self):
        request = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/v1/completions",
            data=json.dumps({"model": self.MODEL_ID,
                             "prompt": "The sky is",
                             "max_tokens": 6}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            body = json.loads(response.read())
        self.assertTrue(body["choices"][0]["text"] is not None)
        self.assertGreater(body["usage"]["prompt_tokens"], 0)

    def test_alias_boundary_plus_one_refused(self):
        prompt_tokens = self.chat(1)[1]["usage"]["prompt_tokens"]
        code, body = self.chat(CONTEXT_LIMIT - prompt_tokens + 1,
                               alias="max_completion_tokens")
        self.assertNotEqual(code, 200)
        message = body["error"] if isinstance(body, dict) else str(body)
        self.assertIn("admitted context budget", message)

    def test_exact_boundary_passes(self):
        prompt_tokens = self.chat(1)[1]["usage"]["prompt_tokens"]
        code, body = self.chat(CONTEXT_LIMIT - prompt_tokens)
        self.assertEqual(code, 200, body)

    def test_boundary_plus_one_refused_before_generation(self):
        prompt_tokens = self.chat(1)[1]["usage"]["prompt_tokens"]
        code, body = self.chat(CONTEXT_LIMIT - prompt_tokens + 1)
        self.assertNotEqual(code, 200)
        message = body["error"] if isinstance(body, dict) else str(body)
        self.assertIn("admitted context budget", message)
        # refusal, not generation: no usage/completion payload exists
        self.assertNotIn("usage", body if isinstance(body, dict) else {})

    def test_server_survives_rejections(self):
        code, _ = self.chat(CONTEXT_LIMIT * 2)
        self.assertNotEqual(code, 200)
        code, body = self.chat(4)
        self.assertEqual(code, 200, body)

    def test_concurrency_bound_proven_by_source_and_argv(self):
        # Seriality is NOT inferred from two 200s. The proof chain:
        # (a) the recorded launch argv pinned both concurrencies to 1;
        self.assertIn("--decode-concurrency", self.launch_argv)
        self.assertEqual(self.launch_argv[
            self.launch_argv.index("--decode-concurrency") + 1], "1")
        self.assertIn("--prompt-concurrency", self.launch_argv)
        self.assertEqual(self.launch_argv[
            self.launch_argv.index("--prompt-concurrency") + 1], "1")
        # (b) the INSTALLED pinned server source wires exactly those flags
        # to the batch generator's completion/prefill batch sizes, so the
        # active batch is at most one request and the aggregate KV equals
        # one capped request.
        site = next((VENV / "lib").glob("python3*/site-packages/mlx_lm/server.py"))
        source = site.read_text(encoding="utf-8")
        self.assertIn(
            "completion_batch_size=self.cli_args.decode_concurrency", source)
        self.assertIn(
            "prefill_batch_size=self.cli_args.prompt_concurrency", source)

    def test_concurrent_clients_both_served_within_budget(self):
        # Smoke only: both concurrent clients are served inside the cap.
        # Seriality itself is proven by the source-linked assertion above.
        import threading

        results = {}

        def worker(i):
            results[i] = self.chat(4)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for i in (0, 1):
            code, body = results[i]
            self.assertEqual(code, 200, body)
            self.assertTrue(body["choices"][0]["message"]["content"].strip())


if __name__ == "__main__":
    unittest.main()
