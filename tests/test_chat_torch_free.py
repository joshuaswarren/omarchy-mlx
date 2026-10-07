"""The chat context-counter must run its whole transformers path torch-free.

Issue #33, second report: the reporter says chat did not work at all until
torch was installed (web and terminal), while the pinned stack (mlx-lm 0.31.3
+ transformers 5.16.1; default everyday chat model
mlx-community/Qwen3.5-9B-MLX-4bit, a ``TokenizersBackend`` tokenizer) has no
torch-dependent surface on the chat path — verified by an A/B battery in two
identical venvs differing only by torch. This
test pins that contract on any machine, with or without torch installed:
``LocalModels.count`` — the coordinator's transformers touchpoint, run once
per turn — must complete end to end in an interpreter where importing torch
raises.

The probe subprocess puts a meta-path finder in front of the import system:
while ``transformers`` imports (availability probes are ``lru_cache``d) torch
reads as absent, so transformers runs its real torch-free behavior; after
priming, any ``import torch`` anywhere in the counted path raises and fails
the test.

The fixture mirrors the default model's tokenizer shape: a local
transformers-format directory (``tokenizer.json`` + ``tokenizer_config.json``
declaring ``TokenizersBackend``) with a chat template that takes
``enable_thinking``, the kwarg the coordinator passes. No network, no GPU,
no model weights, no torch.

If this test fails, a torch dependency entered the chat counting path and
every torch-free install fails its first turn. Fix the path; do not add
torch (a multi-gigabyte dependency the product never uses).
"""

import importlib.util
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

SERVE = Path(__file__).resolve().parents[1] / "serve"

PROBE = textwrap.dedent("""
    import importlib.abc, importlib.machinery, sys

    state = {"priming": True}

    class NoTorchPathFinder(importlib.abc.MetaPathFinder):
        '''Replaces the path finder: while transformers imports (its
        availability probes are lru_cached) torch reads as uninstalled on
        this machine; afterwards any torch import fails the probe.'''

        def find_spec(self, fullname, path=None, target=None):
            if fullname == "torch" or fullname.startswith("torch."):
                if state["priming"]:
                    return None
                raise ImportError("torch import after priming (torch-free contract)")
            return importlib.machinery.PathFinder.find_spec(fullname, path, target)

        def find_distributions(self, context=None):
            # Package metadata (importlib.metadata) discovers through
            # meta_path too; hiding it breaks transformers' dependency
            # checks. Metadata stays visible; only the torch MODULE hides.
            return importlib.machinery.PathFinder.find_distributions(context)

    sys.meta_path[:] = [NoTorchPathFinder() if f is importlib.machinery.PathFinder
                        else f for f in sys.meta_path]

    import transformers  # availability probes cached here
    state["priming"] = False

    import sys as _sys
    _sys.path.insert(0, "@@SERVE@@")
    import mlx_omarchy_paths  # noqa: F401  (launcher bootstrap order)
    from mlx_omarchy_assistant.coordinator import LocalModels

    models = LocalModels(None)
    n = models.count("@@FIXTURE@@", [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Say hi."},
    ])
    assert isinstance(n, int) and n > 0, n
    print("count", n)
""")

CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message.role == 'system' %}<|im_start|>system\n{{ message.content }}<|im_end|>\n"
    "{%- elif message.role == 'user' %}<|im_start|>user\n"
    "{{- '' if enable_thinking else '' }}{{ message.content }}<|im_end|>\n"
    "{%- else %}<|im_start|>assistant\n{{ message.content }}<|im_end|>\n"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}"
)


def _fixture(directory: Path) -> Path:
    """Minimal local repo shaped like the default chat model's tokenizer.

    The default model declares ``tokenizer_class: TokenizersBackend``
    (transformers 5.x, the pinned line). On a 4.x interpreter that class
    does not exist, so the fixture falls back to ``PreTrainedTokenizerFast``:
    the contract under test is a torch-free ``count()``, not one class name.
    """
    import importlib.util

    if importlib.util.find_spec("transformers.tokenization_utils_tokenizers"):
        import transformers.tokenization_utils_tokenizers as tok_utils
        tokenizer_class = ("TokenizersBackend"
                           if hasattr(tok_utils, "TokenizersBackend")
                           else "PreTrainedTokenizerFast")
    else:
        tokenizer_class = "PreTrainedTokenizerFast"  # transformers 4.x
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    vocab = {t: i for i, t in enumerate(
        ["<|im_start|>", "<|im_end|>", "You", " are", " helpful", ".",
         "Say", " hi", " hello", " world", "!"])}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<|im_end|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoders = decoders.WordPiece(prefix="")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "tokenizer.json").open("w") as fh:
        json.dump(json.loads(tokenizer.to_str()), fh)
    (directory / "chat_template.jinja").write_text(CHAT_TEMPLATE)
    (directory / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": tokenizer_class,
        "model_max_length": 512,
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",
        # 4.x reads the template only from here; 5.x reads chat_template.jinja
        # and tolerates the duplicate key.
        "chat_template": CHAT_TEMPLATE,
    }))
    return directory


@unittest.skipUnless(importlib.util.find_spec("transformers"), "needs transformers (the path under test)")
class ChatTorchFreeTests(unittest.TestCase):
    def test_localmodels_count_runs_with_torch_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _fixture(Path(tmp) / "tokenizer-fixture")
            code = (PROBE.replace("@@SERVE@@", str(SERVE))
                         .replace("@@FIXTURE@@", str(fixture)))
            probe = subprocess.run([sys.executable, "-c", code],
                                   capture_output=True, text=True, timeout=120)
        self.assertEqual(
            probe.returncode, 0,
            "count() failed on the torch-free path:\n"
            + probe.stdout + probe.stderr)
        self.assertIn("count", probe.stdout)


if __name__ == "__main__":
    unittest.main()
