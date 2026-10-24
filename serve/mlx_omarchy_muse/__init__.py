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
MLX_OMARCHY_MUSE_FUSE_GATE_UP=1 (off by default) fuses the routed experts' gate and up projections after the weights load (fuse_gate_up). It is bit-identical to the
separate calls on the CPU test shapes; its speed on the Vulkan backend is not measured, and it needs one layer's expert weights of extra memory while it runs.
It does nothing in a ring run (MLX_RANK or MLX_HOSTFILE set): pipeline ranks load unfused until a hook exists that fuses after the layers are pruned.
"""

import copy
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
FUSE_ENV = "MLX_OMARCHY_MUSE_FUSE_GATE_UP"


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


def fuse_gate_up(model) -> int:
    """Fuse the routed experts' gate and up projections into one gather_qmm per MoE layer. Return the number of layers fused.

    Affine quantization packs every output row on its own, so concatenating the up and gate expert weights along the output axis and
    running one gather_qmm over [E, 2 x intermediate, hidden] computes the same per-column sums as two calls and removes one launch per MoE layer
    per token. The idea is oMLX's gate+up fusion (jundot/omlx pull request 4573); the code here is ours and uses only mx operations.
    Only quantized experts with equal bits, group size and mode and no bias are fused; a layer that is not (or is not owned by this pipeline
    rank) is left alone. The fused weights are a copy: free the originals with the references this function drops, and expect one layer's expert
    weights of extra memory while it runs."""
    import mlx.core as mx

    switch = sys.modules["mlx_lm.models.switch_layers"]

    class FusedGateUpSwitchGLU(switch.SwitchGLU):
        def __call__(self, x, indices):
            x = mx.expand_dims(x, (-2, -3))
            indices = mx.stop_gradient(indices)
            do_sort = indices.size >= 64
            idx = indices
            inv_order = None
            if do_sort:
                x, idx, inv_order = switch._gather_sort(x, indices)
            up, gate = mx.split(self.gate_up_proj(x, idx, sorted_indices=do_sort), 2, axis=-1)
            x = self.down_proj(self.activation(up, gate), idx, sorted_indices=do_sort)
            if do_sort:
                x = switch._scatter_unsort(x, inv_order, indices.shape)
            return x.squeeze(-2)

    fused_layers = 0
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        glu = getattr(mlp, "switch_mlp", None)
        if glu is None or "gate_proj" not in glu or "up_proj" not in glu:
            continue
        gate, up = glu.gate_proj, glu.up_proj
        if not (isinstance(gate, switch.QuantizedSwitchLinear) and isinstance(up, switch.QuantizedSwitchLinear)):
            continue
        if (gate.bits, gate.group_size, gate.mode) != (up.bits, up.group_size, up.mode) or "bias" in gate or "bias" in up:
            continue
        if (gate.get("biases") is None) != (up.get("biases") is None):
            continue
        fused = copy.copy(gate)
        fused.weight = mx.concatenate([up.weight, gate.weight], axis=1)
        fused.scales = mx.concatenate([up.scales, gate.scales], axis=1)
        if gate.get("biases") is not None:
            fused.biases = mx.concatenate([up.biases, gate.biases], axis=1)
        mx.eval(fused.parameters())
        glu.gate_up_proj = fused
        del glu["gate_proj"], glu["up_proj"]
        glu.__class__ = FusedGateUpSwitchGLU
        fused_layers += 1
    return fused_layers


def _install_fuse(module) -> None:
    """Opt in with MLX_OMARCHY_MUSE_FUSE_GATE_UP=1: fuse_gate_up runs once after the weights are loaded.

    Not in a ring run (MLX_RANK or MLX_HOSTFILE set): mlx-lm's sharded_load calls load_weights on the whole lazy model BEFORE it prunes the layers this rank does
    not own, so fusing there would read every layer's experts from disk, and its later load_weights would find no gate_proj to fill. Those ranks run unfused."""
    load_weights = module.Model.load_weights

    def load_and_fuse(self, file_or_weights, strict=True):
        result = load_weights(self, file_or_weights, strict=strict)
        if "MLX_RANK" not in os.environ and "MLX_HOSTFILE" not in os.environ:
            fuse_gate_up(self)
        return result

    module.Model.load_weights = load_and_fuse


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
    if os.environ.get(FUSE_ENV) == "1":
        _install_fuse(module)
    setattr(importlib.import_module("mlx_lm.models"), "muse_spark", module)
    return True


__all__ = ["MODEL_NAME", "MODEL_PATH", "SOURCE_SHA256", "fuse_gate_up", "is_registered", "register"]
