"""mlx_omarchy_muse: load the published Muse Spark Open community checkpoints through mlx-lm.

The mlx-community Muse Spark Open quantizations (3bit, 4bit) carry model_type "muse_spark" and the
Hugging Face text_config layout. mlx-lm main has no loader for that name: its muse_spark_open class
takes the native checkpoint layout, and it calls eight fused Metal kernels. This package adds the
text-only class of the closed mlx-lm pull request 1975 (head 117b84cae44a, MIT, Copyright Apple Inc.).
That class uses only mx operations, so it needs no Metal kernel.

The class file is copied unchanged: muse_spark.py, sha256 9b507590febd23224368b794d73baf0b2f2003e9b78cabf9200a781f7f1dc749.
The licence text is LICENSES/mlx-lm.txt.

    import mlx_omarchy_muse
    mlx_omarchy_muse.register()
    # mlx_lm.load(...) now resolves model_type "muse_spark"

register() installs this class as mlx_lm.models.muse_spark, also when the installed mlx-lm ships its own module of that name. mlx-lm 0.32.2 and later do: that module
reads the native checkpoint layout and imports the eight fused Metal kernels, so it does not load the community checkpoints on a Vulkan backend. Set
MLX_OMARCHY_MUSE_UPSTREAM=1 to keep the mlx-lm module instead; register() then does nothing.
This package changes no mlx-lm file. It does not change the memory a model needs: the 3bit file is 359 GB.
"""

import importlib
import importlib.util
import os
import sys
import weakref
from pathlib import Path

__version__ = "0.1.0"

MODEL_NAME = "mlx_lm.models.muse_spark"
MODEL_PATH = Path(__file__).with_name("muse_spark.py")
SOURCE_SHA256 = "9b507590febd23224368b794d73baf0b2f2003e9b78cabf9200a781f7f1dc749"
PRUNE_ENV = "MLX_OMARCHY_PIPELINE_PRUNE"


def _install_prune(module) -> None:
    """Opt in with MLX_OMARCHY_PIPELINE_PRUNE=1: after the pipeline split, a rank drops the shared tensors it never reads.

    Rank size-1 runs the first layers and is the only rank whose embedding values are used (the other ranks take the
    hidden state from a recv and read only the shape and dtype). Rank 0 runs the last layers and is the only rank that
    needs lm_head, provided one rank samples and shares the token (omarchy-cluster does). A stub with the same shape or
    dtype takes their place, so the load never reads the real tensors. Without that sharing, leave this off.
    A model with tie_word_embeddings keeps everything: its ranks all use embed_tokens for the logits."""
    import mlx.core as mx
    import mlx.nn as nn

    class _ZeroEmbed(nn.Module):
        def __init__(self, dim, dtype):
            super().__init__()
            self._dim, self._dtype = dim, dtype

        def __call__(self, x):
            return mx.zeros((*x.shape, self._dim), self._dtype)

    class _NoHead(nn.Module):
        def __init__(self, vocab):
            super().__init__()
            self.weight = mx.zeros((vocab, 1), mx.uint8)  # keeps weight.shape[0] for the token-sharing swap

        def __call__(self, x):
            raise RuntimeError("lm_head was pruned: only pipeline rank 0 computes logits")

    init = module.Model.__init__

    def model_init(self, args):
        init(self, args)
        self.model._owner = weakref.ref(self)

    def pipeline(self, group, *rest):
        module.PipelineMixin.pipeline(self, group, *rest)  # looked up late: omarchy-cluster replaces it
        rank, size = self.pipeline_rank, self.pipeline_size
        if self.args.tie_word_embeddings:
            return  # tied: every rank computes logits with embed_tokens.as_linear, so every rank keeps it
        if size > 1 and rank != size - 1 and not isinstance(self.embed_tokens, _ZeroEmbed):
            emb = self.embed_tokens
            dtype = emb.scales.dtype if hasattr(emb, "scales") else emb.weight.dtype
            self.embed_tokens = _ZeroEmbed(self.args.hidden_size, dtype)
        owner = self._owner() if hasattr(self, "_owner") else None
        if size > 1 and rank != 0 and owner is not None and hasattr(owner, "lm_head") and not isinstance(owner.lm_head, _NoHead):
            owner.lm_head = _NoHead(self.args.vocab_size)

    module.Model.__init__ = model_init
    module.MuseSparkModel.pipeline = pipeline




def is_registered() -> bool:
    module = sys.modules.get(MODEL_NAME)
    return module is not None and getattr(module, "__file__", None) == str(MODEL_PATH)


def register() -> bool:
    """Install the vendored class as mlx_lm.models.muse_spark. Return True when it was installed or replaced an mlx-lm module."""
    if is_registered() or os.environ.get("MLX_OMARCHY_MUSE_UPSTREAM") == "1":
        return False
    spec = importlib.util.spec_from_file_location(MODEL_NAME, MODEL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(MODEL_NAME)
    sys.modules[MODEL_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            del sys.modules[MODEL_NAME]
        else:
            sys.modules[MODEL_NAME] = previous
        raise
    if os.environ.get(PRUNE_ENV) == "1":
        _install_prune(module)
    setattr(importlib.import_module("mlx_lm.models"), "muse_spark", module)
    return True


__all__ = ["MODEL_NAME", "MODEL_PATH", "SOURCE_SHA256", "is_registered", "register"]
