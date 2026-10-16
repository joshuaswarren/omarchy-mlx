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
from pathlib import Path

__version__ = "0.1.0"

MODEL_NAME = "mlx_lm.models.muse_spark"
MODEL_PATH = Path(__file__).with_name("muse_spark.py")
SOURCE_SHA256 = "9b507590febd23224368b794d73baf0b2f2003e9b78cabf9200a781f7f1dc749"


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
    setattr(importlib.import_module("mlx_lm.models"), "muse_spark", module)
    return True


__all__ = ["MODEL_NAME", "MODEL_PATH", "SOURCE_SHA256", "is_registered", "register"]
