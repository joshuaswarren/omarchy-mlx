# omlx platform-gate patch (omarchy-mlx port)

This directory contains the patch series that wires omlx's
`mx.metal.is_available()` gate sites to take the omarchy path.

## What it does

- Adds a small helper module `omlx/_compat_gate.py` with three
  public functions:
  - `custom_kernels_available()` — answers True on macOS Metal
    (passes through `mx.metal.is_available()`) and True on
    omarchy-mlx (one-time canary `mx.fast.metal_kernel` build/run).
  - `set_wired_limit_enabled()` — whether the backend accepts
    `mx.set_wired_limit` (short-circuits to False on a CPU default
    device).
  - `device_info_keys()` — best-effort device info dict with
    `max_recommended_working_set_size` defaulted to 0 on Linux
    (the runtime probe that would populate it is owned by the
    OmlxLinux lane, #7).
- Rewires 16 gate sites (A7 DFlash wired-limit, A15 memory
  monitor baseline, A26/A27 JIT-mx.fast.metal_kernel patches +
  SDPA256 native force-fused route) from
  `mx.metal.is_available()` to `custom_kernels_available()`.

The macOS path is unchanged: on macOS, `custom_kernels_available()`
returns `mx.metal.is_available()` because that's checked first.
M5-specific gates (NAX variant, applegpu_g17+ tensor units) are NOT
re-routed through here — those decisions are Metal-specific by
design and stay `n/a` on M1/M2 hosts per the parity matrix.

## Files

- `apply-platform-gate.sh` — installer step; pins to omlx
  `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40` (v0.7.0); refuses
  non-pinned HEAD; idempotent.
- `patches/01-add-compat-gate.patch` — adds
  `omlx/_compat_gate.py`.
- `patches/02-gate-sites.patch` — rewires 16 gate sites across
  `omlx/{engine/memory_monitor/custom_kernels/...}/*.py` and
  `omlx/patches/*`.
- `patches/_build_patch.py` — generator that produces the two
  patches against a fresh pinned clone. Re-run with the pinned
  clone as argv[1] when omlx ships a new tag.
- `patches/omlx-compat-gate-source.py` — source embedded into
  patch 01.
- `LICENSE/patches/omlx/LICENSE` — Apache-2.0 license preserved
  from upstream omlx.

## How to use

```sh
# 1. Clone pinned source (only if /tmp/omlx-tf-parity/omlx is not present)
git clone --depth 1 https://github.com/jundot/omlx /tmp/omlx-pin
git -C /tmp/omlx-pin fetch --depth 1 origin tag v0.7.0
git -C /tmp/omlx-pin checkout 4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40

# 2. Verify it would apply cleanly
packaging/omlx-linux/apply-platform-gate.sh /tmp/omlx-pin --verify-only

# 3. Apply
packaging/omlx-linux/apply-platform-gate.sh /tmp/omlx-pin

# 4. (Re-run integration test)
python3 tests/test_platform_gate.py --omlx-src /tmp/omlx-pin
```

## Acceptance

The patch series applies clean against the pinned commit (verified
by `apply-platform-gate.sh --verify-only`). Tests live under
`tests/test_platform_gate.py`: stub-module unit tests that import
the gated modules with a fake `mlx.core` and assert each gate
takes the omarchy path on a Linux fake and the Metal path on a Mac
fake.

## Pinned commit

```
$ git -C /tmp/omlx-pin log -1 --format='%H %s'
4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40 release: omlx v0.7.0
```

If upstream moves the commit, regenerate the patches:
```sh
python3 packaging/omlx-linux/patches/_build_patch.py /path/to/clean/pinned/clone \
    packaging/omlx-linux/patches
```

## License

omlx (Apache-2.0). This patch series preserves the SPDX headers in
all upstream files. The new `omlx/_compat_gate.py` is also
Apache-2.0 (carries the SPDX header). No upstream source is copied
into the omarchy-mlx repo; only patches-as-diffs land here.