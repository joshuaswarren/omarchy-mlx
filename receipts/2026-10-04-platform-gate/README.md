# Platform-gate patch (omlx / TensorFold → omarchy-mlx)

Lane: PlatformGate (matrix lane #1). Matrix rows A7, A15, A26-A27 gates, B2, B9 gates.

Pinned sources:
- omlx (jundot/omlx) `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40` (tag v0.7.0).
- TensorFold (ashhart/TensorFold) `609ca419abecebdc5a059498a613680bd3aa847f` (== v0.6.5; main at audit time).

Both Apache-2.0; their LICENSE files are preserved in
`packaging/omlx-linux/LICENSE/patches/omlx/LICENSE` and
`packaging/tensorfold-linux/LICENSE/patches/tensorfold/LICENSE`.
No upstream source is copied into this repo. Patches-as-diffs only.

## What the patch series does

Adds a small helper module to each repo (one new file, Apache-2.0,
preserves upstream SPDX headers):

- `omlx/_compat_gate.py` — `custom_kernels_available()`,
  `set_wired_limit_enabled()`, `device_info_keys()`.
- `src/tensorfold/_compat_gate.py` — `custom_kernels()`,
  `device_info()`.

The macOS path answers identically (the helper checks
`mx.metal.is_available()` first and returns its value). The
omarchy path runs a one-time canary `mx.fast.metal_kernel`
build/run on a 4-element zero buffer; any failure short-circuits
to False. Module-level `@functools.cache` keeps per-call cost to
one dict lookup after the first invocation.

The M5 tensor-unit detection in `kernels/device.generation()` and
the NAX-variant gate in `qwen35_prefill/fast.py` are NOT rewired:
they are Metal-specific by design and stay `n/a` on M1/M2 hosts.

## Sites rewired (matrix row → file:line)

### A7 DFlash speculative decoding (omlx)
- `omlx/engine/dflash.py:529` — wired-limit gate `_acquire_wired_limit`
  (`mx.metal.is_available()` → `_wire_enabled()`).
- `omlx/engine/dflash.py:532-533` — `mx.device_info()` →
  `_device_info_keys()`.
- `omlx/engine/dflash.py` header — `try: from omlx._compat_gate
  import ... except ImportError:` with fallback lambdas (preserve
  standalone importability when the patch isn't applied).

### A15 Memory guard / process memory enforcement + monitor
- `omlx/memory_monitor.py:36` — `HAS_MLX_METAL = mx.metal.is_available()`
  → `HAS_MLX_METAL = _cka()`.

### A26 JIT mx.fast.metal_kernel patch kernels (12 of 25 sites)
- `omlx/patches/bailing_hybrid/bailing_hybrid_model.py:411`
- `omlx/patches/qwen35_moe_weighted_sum.py:60`
- `omlx/patches/qwen35_verify_sdpa_split.py:914`
- `omlx/patches/qwen35_moe_routed_decode.py:816`
- `omlx/patches/glm_moe_dsa/sparse_mla.py:19,232`
- `omlx/patches/qwen35_gdn_chunked.py:51`
- `omlx/patches/qwen35_fa256_attention.py:161`
- `omlx/patches/gemma4_verify_kernel.py:347`
- `omlx/patches/qwen35_moe_router.py:660`
- `omlx/patches/deepseek_v4/hyper_connection.py:24,244`
- `omlx/patches/m5_gather_qmm.py:112`
- `omlx/patches/glm53_kda_prework.py:233`
- `omlx/patches/qwen35_gdn_prework.py:1738`
- (`omlx/patches/mlx_vlm_*/compat/vendor/...` sites and the
  `qwen35_prefill/fast.py` NAX mac_ver site are intentionally left
  intact — Linux already short-circuits the mac_ver check via
  `platform.mac_ver()[0]` returning ''.)

### A27 SDPA head-dim-256 O(L) tiled fast route + memory accounting
- `omlx/patches/sdpa256_attention.py:209` — `_should_route` native
  Metal gate → `_cka()`.
- The SDPA104 memory accounting (memory_monitor.py:43-66) is gated
  through `HAS_MLX_METAL` and is therefore already rewired via A15.

### B2 TensorFold family `metal()` gates + `_step_kernel` build-time gates
- `src/tensorfold/kernels/glm/flash/v1/fused.py:43`
- `src/tensorfold/kernels/glm/flash/v1/kda.py:286`
- `src/tensorfold/kernels/glm/flash/v1/kernels.py:352`
- `src/tensorfold/kernels/glm/flash/v1/sparse_attention.py:83`
- `src/tensorfold/kernels/qwen/dense/v1/lane_gdn.py:192,222`
- (`src/tensorfold/kernels/device.py:19` `generation()` is
  intentionally NOT rewired: M5 tensor-unit gate, n/a on M1/M2.)

### B9 device/memory APIs
- `src/tensorfold/kernels/qwen/flash_next/v1/prefill_mm.py:215`
- `src/tensorfold/kernels/threads.py:39`
- (Each routes `mx.device_info() if hasattr(mx, "device_info") else
  mx.metal.device_info()` through `_compat_gate.device_info()`.)

## Files in this receipt

```
receipts/2026-10-04-platform-gate/
├── README.md                       # this file
├── (private artifacts go to ~/.local/share/apple-silicon-lab/
│    artifacts/PlatformGate/platform-gate/)
```

## Patches and installer

```
packaging/omlx-linux/
├── README.md
├── apply-platform-gate.sh          # pins to 4d4f5a28; idempotent
├── LICENSE/patches/omlx/LICENSE    # Apache-2.0 preserved
└── patches/
    ├── 01-add-compat-gate.patch    # adds omlx/_compat_gate.py
    ├── 02-gate-sites.patch         # rewires 16 sites
    ├── _build_patch.py             # generator
    └── omlx-compat-gate-source.py  # source embedded in patch 01

packaging/tensorfold-linux/
├── README.md
├── apply-platform-gate.sh          # pins to 609ca419; idempotent
├── LICENSE/patches/tensorfold/LICENSE  # Apache-2.0 preserved
└── patches/
    ├── 01-add-compat-gate.patch    # adds src/tensorfold/_compat_gate.py
    ├── 02-gate-sites.patch         # rewires 8 sites
    ├── _build_patch.py             # generator
    └── tensorfold-compat-gate-source.py
```

The installer scripts pin to the exact commits, refuse non-pinned
HEAD, and refuse already-patched trees. They produce 0 exit code on
success and print a per-file diff summary.

## Tests

```
tests/test_platform_gate.py
```

Runs as a stdlib `unittest.TestCase` suite. Two layers:

1. Unit tests against the helper modules in isolation, with
   `_make_fake_mlx_factory` producing a fake `mlx.core` that
   behaves like an omarchy host (`mx.metal.is_available() ==
   False`, canary probe controlled by parameter) and a Mac host
   (`mx.metal.is_available() == True`). Asserts:
   - omarchy fake + canary True → `custom_kernels_available() == True`.
   - omarchy fake + canary False → `custom_kernels_available() == False`.
   - Mac fake → `custom_kernels_available() == True` (regardless of canary).
   - TensorFold helper mirrors the same logic.
2. Integration tests that apply the patch series to a fresh pinned
   clone (default: `/tmp/omlx-apply-test`, `/tmp/tf-apply-test`)
   and assert each wired gate-site file has the expected
   `_cka()` / `_wire_enabled()` / `_device_info_keys()` /
   `custom_kernels()` substitution and no live `mx.metal.is_available()`
   calls remain (except inside the documented fallback lambdas).

Receipt log: `~/.local/share/apple-silicon-lab/artifacts/PlatformGate/platform-gate/test-run-1.txt`
(`Ran 10 tests in 0.012s OK`).

## Acceptance

- [x] Patch series applies clean against the pinned commits
      (`apply-platform-gate.sh --verify-only` returns 0 on both
      repos; full apply returns 0).
- [x] Stub-module unit tests green on Linux fake (canary True/False)
      and Mac fake.
- [x] Integration tests confirm 16 omlx sites + 8 TensorFold sites
      are rewired and no live `mx.metal.is_available()` calls remain
      in the gate predicates.
- [x] Real M2 run: gated kernels engage on omarchy hardware — canary True
      both helpers, all family gates open, real gated completion correct,
      memory baseline nonzero (see "M2 real run" above; full model-level
      decode = OmlxLinux/KernelBattery follow-on).

## M2 real run (2026-10-05, closed)

Shared venv `/var/tmp/shared-omarchy-venv` (mlx 0.32.4.dev202610042317+b8af62c,
read-only), private copy `/tmp/pgate-venv`, pinned clones shipped from the dev
box, patch series applied ON the M2 by `apply-platform-gate.sh` (rc=0 both).
Two gpu-turn tickets (15 min + 8 min caps, HOME explicit). Artifacts:
`artifacts/PlatformGate/platform-gate/m2-run/` (private notebook; SHA256SUMS there).

- (a) canary on M2: `mx.metal.is_available()` = False (contract),
  `omlx._compat_gate.custom_kernels_available()` = **True**,
  `tensorfold._compat_gate.custom_kernels()` = **True** (B-canary.log).
- (b) family engagement: all four GLM flash `metal()` = True; lane_gdn
  `_step_kernel` and `_step_kernel_kh` build through the translator;
  `device_info` real (architecture=honeykrisp); `prefill_identity()`
  = `architecture=honeykrisp;matmul=pending` (C-family-gates.log).
- (c) real gated completion per project: canary kernel build+launch+eval
  through both gates on the M2 GPU, output [7.0, 7.0, 7.0, 7.0],
  COMPLETION_CORRECT=True (D2-baseline.log). Full model-level decode stays
  with KernelBattery: lane_gdn step LAUNCH raises the exact contract error
  `unsupported MSL feature 'device pointer arithmetic'` (D-completion.log) —
  the gate opened; the kernel body is the per-kernel battery's work (#4).
- (d) memory baseline: `HAS_MLX_METAL` (patched gate) = True,
  `set_baseline_memory()` = 262144 bytes, BASELINE_NONZERO=True
  (D2-baseline.log).

oMLX server completion note: the full oMLX server stack (mlx-lm,
transformers, …) is the OmlxLinux install lane; the oMLX-side gated-path
completion here is the patched gate + real kernel run through
`omlx._compat_gate`; riding OmlxLinux's smoke ticket for a model-level
oMLX completion remains open.

Matrix promotion: B2 and B9 → DONE (the rows' named features ARE the gates
and device-info paths; all verified on M2 hardware). A15 → DONE for the
named memory_monitor gate+baseline lines (process_memory_enforcer and
cluster/memory_guard surfaces remain untested, noted in the row). A7, A26,
A27 stay IN PROGRESS (gate landed + M2-verified; the DFlash engine run and
per-kernel MSL bodies are the remaining work, owned per the lane table).

## M2 run coordination

Coordinate through Main via lane-send; the OmlxLinux lane owns the
oMLX-on-omarchy install. Once oMLX is patched and installed on the
M2, the `omlx/_compat_gate.custom_kernels_available()` probe runs
on each request and the patched gate predicates take the omarchy
path. The receipt for "gated JIT kernels engage" is a successful
`omlx serve` startup with the model loaded (and a `native_kernel_status()`
-equivalent print, since omlx `custom_kernels/__init__.py:17` exposes
that). Status before M2 run: this row remains `tested: dev-box unit
tests`; the matrix cell flips to `tested: M2-real-completion` after
OmlxLinux reports the M2 completion.

## Reference pattern origin

Pattern origin: TensorFoldPort's `device.custom_kernels` canary
probe (`f5d1111` on the `tf-drowz` fork clone at
`/tmp/tfport/tf-drowz`). We generalize it into a standalone
module (avoids a dependency cycle from family kernels into
`kernels/device`) and add the matching `omlx/_compat_gate.py`
with the wired-limit probe.