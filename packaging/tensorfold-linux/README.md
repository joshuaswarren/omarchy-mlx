# TensorFold platform-gate patch (omarchy-mlx port)

This directory contains the patch series that wires TensorFold's
`mx.metal.is_available()` family gates to take the omarchy path.

## What it does

- Adds a small helper module `src/tensorfold/_compat_gate.py`
  with two public functions:
  - `custom_kernels()` — answers True on macOS Metal (passes
    through `mx.metal.is_available()`) and True on omarchy-mlx
    (one-time canary `mx.fast.metal_kernel` build/run).
  - `device_info()` — best-effort device info that drops back to
    `mx.metal.device_info()` only when the omarchy
    `mx.device_info()` is missing (matches the existing
    `hasattr(mx, "device_info")` pattern).
- Rewires 8 gate sites:
  - 4× GLM flash `metal()` helpers (`fused.py`, `kda.py`,
    `kernels.py`, `sparse_attention.py`).
  - 2× qwen/dense `lane_gdn.py` `_step_kernel[_kh]` build-time
    gates.
  - 2× device-info chains (`qwen/flash_next/prefill_mm.py`,
    `kernels/threads.py`).

Pattern origin: TensorFoldPort's `device.custom_kernels()`
canary-probe commit `f5d1111` on their `tf-drowz` fork clone,
generalized into a standalone module so the family kernels don't
all need to import `device` (which would create a dependency cycle
in some files).

The matrix row B2 lists 9 gate sites; we wire 8 here and
intentionally leave `kernels/device.py:19` (`generation()` for M5
tensor-unit detection) untouched because that helper is
Metal-specific by design per the audit: an M5-only GPU attribute
that has no analog on M1/M2 hosts. Documented in the patches'
header comment.

## Files

- `apply-platform-gate.sh` — installer step; pins to TensorFold
  `609ca419abecebdc5a059498a613680bd3aa847f` (== v0.6.5).
- `patches/01-add-compat-gate.patch` — adds
  `src/tensorfold/_compat_gate.py`.
- `patches/02-gate-sites.patch` — rewires 8 sites.
- `patches/_build_patch.py` — generator that produces the two
  patches against a fresh pinned clone.
- `patches/tensorfold-compat-gate-source.py` — source embedded
  into patch 01.
- `LICENSE/patches/tensorfold/LICENSE` — Apache-2.0 license
  preserved from upstream TensorFold.

## How to use

```sh
git clone --depth 1 https://github.com/ashhart/TensorFold /tmp/tf-pin
git -C /tmp/tf-pin fetch --depth 1 origin 609ca419abecebdc5a059498a613680bd3aa847f
git -C /tmp/tf-pin checkout 609ca419abecebdc5a059498a613680bd3aa847f

packaging/tensorfold-linux/apply-platform-gate.sh /tmp/tf-pin --verify-only
packaging/tensorfold-linux/apply-platform-gate.sh /tmp/tf-pin

python3 tests/test_platform_gate.py --tf-src /tmp/tf-pin
```

## Pinned commit

```
$ git -C /tmp/tf-pin log -1 --format='%H %s'
609ca419abecebdc5a059498a613680bd3aa847f release: TensorFold 0.6.5
```

## License

TensorFold (Apache-2.0; relicensed from MIT at v0.6.0). The
fork carries the same license; this patch series preserves the
SPDX headers in all upstream files. The new
`src/tensorfold/_compat_gate.py` is also Apache-2.0. No upstream
source is copied into the omarchy-mlx repo.