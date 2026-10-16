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

register() does nothing when mlx_lm already provides a muse_spark module.
This package changes no mlx-lm file. It does not change the memory a model needs: the 3bit file is 359 GB.
"""

import importlib
import importlib.util
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
    """Expose the vendored class as mlx_lm.models.muse_spark. Return True when it was added."""
    if MODEL_NAME in sys.modules:
        return False
    try:
        if importlib.util.find_spec(MODEL_NAME) is not None:
            return False
    except (ImportError, ValueError):
        pass
    spec = importlib.util.spec_from_file_location(MODEL_NAME, MODEL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODEL_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[MODEL_NAME]
        raise
    setattr(importlib.import_module("mlx_lm.models"), "muse_spark", module)
    return True


__all__ = ["MODEL_NAME", "MODEL_PATH", "SOURCE_SHA256", "is_registered", "register"]
