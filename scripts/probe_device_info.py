#!/usr/bin/env python3
# HwProbe probe: dump mx.metal.device_info / mx.device_info / set_wired_limit / set_cache_limit /
# set_memory_limit / get_active_memory / get_peak_memory / get_cache_memory against the live
# omarchy wheel on T6021. Single-shot, no build required. Writes JSON to stdout for receipt.
import json
import os
import sys
import traceback


def _variant(v):
    if isinstance(v, (int, str, float, bool)) or v is None:
        return v
    return repr(v)


def main():
    out = {}
    try:
        import mlx.core as mx  # noqa: F401
    except Exception as exc:
        out["import_error"] = repr(exc)
        print(json.dumps(out, indent=2, sort_keys=True))
        return 1

    try:
        out["metal_is_available"] = bool(mx.metal.is_available())
    except Exception as exc:
        out["metal_is_available_error"] = repr(exc)

    try:
        info_metal = mx.metal.device_info()
        out["mx_metal_device_info"] = {k: _variant(v) for k, v in info_metal.items()}
    except Exception as exc:
        out["mx_metal_device_info_error"] = repr(exc)

    try:
        info = mx.device_info()
        out["mx_device_info"] = {k: _variant(v) for k, v in info.items()}
    except Exception as exc:
        out["mx_device_info_error"] = repr(exc)

    # Wired / cache / memory limits — record previous value (returns prev on set).
    try:
        out["set_wired_limit_prev_0"] = mx.set_wired_limit(0)
        out["set_wired_limit_prev_1GiB"] = mx.set_wired_limit(1 << 30)
        out["set_wired_limit_prev_8GiB"] = mx.set_wired_limit(8 << 30)
    except Exception as exc:
        out["set_wired_limit_error"] = repr(exc)

    try:
        out["set_cache_limit_prev_64MiB"] = mx.set_cache_limit(64 << 20)
        out["set_cache_limit_prev_128MiB"] = mx.set_cache_limit(128 << 20)
    except Exception as exc:
        out["set_cache_limit_error"] = repr(exc)

    try:
        out["set_memory_limit_prev_64GiB"] = mx.set_memory_limit(64 << 30)
        out["set_memory_limit_prev_32GiB"] = mx.set_memory_limit(32 << 30)
    except Exception as exc:
        out["set_memory_limit_error"] = repr(exc)

    try:
        # Touch the allocator via a known-size allocation so get_cache_memory is non-zero.
        import mlx.core as mx
        a = mx.zeros((1024, 4096), mx.float32)
        mx.eval(a)
        out["get_active_memory_after_alloc"] = int(mx.get_active_memory())
        out["get_peak_memory_after_alloc"] = int(mx.get_peak_memory())
        out["get_cache_memory_after_alloc"] = int(mx.get_cache_memory())
        out["get_memory_limit"] = int(mx.get_memory_limit())
        mx.reset_peak_memory()
        out["get_peak_memory_after_reset"] = int(mx.get_peak_memory())
        del a
    except Exception as exc:
        out["memory_api_error"] = repr(exc)
        out["memory_api_trace"] = traceback.format_exc()

    out["provenance"] = {
        "wheel_dist": sorted(
            f for f in os.listdir(
                os.path.dirname(mx.__file__) + "/../../../"
            ) if f.startswith("mlx_omarchy-") and f.endswith(".dist-info")
        ) if hasattr(mx, "__file__") else [],
        "python": sys.version.split()[0],
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())