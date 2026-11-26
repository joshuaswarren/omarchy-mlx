#!/usr/bin/env python3
import json
import os
import runpy
from pathlib import Path

import mlx.core as mx

info = mx.device_info()
print(json.dumps({
    "cooperative_matrix_f32_8": info.get("cooperative_matrix_f32_8"),
    "device_name": info.get("device_name"),
    "mlx_version": mx.__version__,
    "VK_DRIVER_FILES": os.environ.get("VK_DRIVER_FILES"),
}, sort_keys=True), flush=True)
runpy.run_path(str(Path(__file__).with_name("bench_decode.py")), run_name="__main__")
