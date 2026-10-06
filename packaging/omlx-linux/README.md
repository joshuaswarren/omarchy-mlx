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

## Linux compat layer (OmlxLinux lane #7)

A second patch series owned by the OmlxLinux lane handles the sites
the platform-gate does not touch — every macOS-only *helper* call
that the upstream tree still makes after the platform gate lands:

- `omlx/utils/hardware.py` — `get_total_memory_bytes` reads
  `/proc/meminfo` `MemTotal` when `mx.device_info()` lacks
  `memory_size` (the `no_metal.cpp` stub omits Metal keys on
  older wheels; the runtime probe that populates them is
  HwProbe). `get_chip_name` reads `/proc/device-tree/model`.
  `parse_chip_info` no-fake-M1 fallback. `get_os_version`
  returns `Linux <release>`. New `device_total_memory(info)`
  helper accepts either key.
- `omlx/cli.py` — server-start cache-limit uses
  `hardware.device_total_memory(mx.device_info())`.
- `omlx/process_memory_enforcer.py` — when both sysctl
  `iogpu.wired_limit_mb` and the omarchy metal cap are 0, emit
  one INFO log on non-darwin explaining the guard budgets
  against unified system RAM and `mx.set_wired_limit` is a no-op
  on this backend; the darwin branch is untouched.

These targets were deliberately *not* included in the platform-
gate series — the gate handles "is this fast path Metal?"; this
layer handles "what honest value does this helper return when
Metal APIs are absent?". Patches:

- `patches/0001-linux-hardware-proc-meminfo.patch`
- `patches/0002-linux-cli-cache-limit-total-memory.patch`
- `patches/0003-linux-enforcer-no-wired-limit-log.patch`
- `patches/0006-omlx-q35-decode-conv-fuse.patch` — GDN decode fast route
  for the oMLX MTP model. The server's own `qwen35_model.py` routes every
  chunk through `_process_chunk`'s composed chain (conv1d + silu +
  `normalize_qk` as separate dispatches) even for single-token decode,
  where the mlx-lm series folds all three into one
  `mx.fast.gdn_conv_update` dispatch: 18 layers × ~4 extra dispatches
  per decode step. The patch adds the series' fused branch — same gates
  (S==1, bf16, GPU device, `hasattr`), same qk-norm fold conditions
  (head_k_dim 128, widths %256), and a kill switch
  `MLX_OMARCHY_OMLX_CONV_FUSE=0` — to `_process_chunk`. Contract tests:
  `test_omlx_q35_conv_fuse.py` (fused route engages on decode; fused
  output bit-equal to composed on the omarchy wheel; kill switch and
  multi-token chunks stay composed).
- `patches/0007-omlx-laguna-rope-normalize.patch` — Laguna tokenizer
  preflight fix. Laguna checkpoints ship `rope_parameters` as a
  per-layer-type map (`full_attention` / `sliding_attention` dicts) plus
  non-dict top-level scalars (`original_max_position_embeddings`);
  Transformers 5.17 `validate_rope` (modeling_rope_utils.py:850) calls
  `.get` on every value and aborts tokenizer config validation with
  `'int' object has no attribute 'get'` before inference. The patch adds
  `omlx/utils/laguna_rope.py`: `coerce_nested_rope_parameters` narrows such
  mixed maps to the per-layer-type dicts (the shape the upstream
  remote-code `LagunaConfig.__post_init__` produces), and
  `install_rope_parameters_guard` wraps
  `transformers.modeling_rope_utils.RotaryEmbeddingConfigMixin.validate_rope`
  to apply that coercion first; `lm_load_compat` installs the guard before
  every load, so both the preflight and mlx-lm's real tokenizer load pass
  without enabling `trust_remote_code`. Flat rope maps and non-dict values
  pass through untouched. Contract tests: `test_omlx_laguna_rope.py`
  (coercion on the captured `fixtures/laguna-hf-config.json`, flat/non-dict
  pass-through, guard delegation and idempotence — the transformers-wiring
  tests skip on dev boxes without transformers and run on the target
  venv).
- `patches/0008-omlx-audio-sts-discovery.patch` — STS discovery for
  model_type-less checkpoints. `mlx-community/DeepFilterNet-mlx` (v3) and
  `mlx-community/sam-audio-small` ship complete checkpoints whose
  `config.json` has neither `model_type` nor `architectures`; discovery
  classified them as `llm` and the LLM loader aborted with
  `KeyError: 'model_type'` before the STS engine ever ran. The patch adds
  `_looks_like_deepfilternet_config` (df_order/nb_erb/nb_df + a
  `model_version` string starting with `DeepFilterNet`),
  `_looks_like_sam_audio_config` (`audio_codec` dict + `span_predictor` —
  the codec key alone is too common), and `_audio_sts_name_hint`
  (mirrors `omlx.engine.sts._detect_sts_family` directory rules:
  deepfilter/mossformer/sam-audio), all returning `audio_sts` from
  `detect_model_type`. Contract tests:
  `test_omlx_audio_sts_discovery.py` on the captured checkpoint config
  fixtures (`fixtures/deepfilternet3-config.json`,
  `fixtures/sam-audio-config.json`): both shapes plus directory-name hints
  classify `audio_sts`; plain LLM configs stay `llm`; Kokoro stays
  `audio_tts`.

Tools in this layer:

- `install.sh` — user-level venv installer that consumes the
  shared-omarchy-venv wheel, applies *both* series in order
  (platform-gate first, then these three), installs mlx-lm @
  `94cdcae` with `--no-deps` so its `mlx` dep does not pull the
  upstream Metal wheel, installs mlx-vlm @
  `ea79808` with `--no-deps`, runs
  `OMARCHY_ROOT/scripts/apply-mlx-lm-patches.sh` (which
  auto-selects the 0.32 series via the `StopSequences`
  marker), and prints the server smoke command.
- `ast_import_scan.py` — gates unguarded module-level imports
  of packages the install does not provide (`--allow patches`
  for the model_type-gated dflash_mlx imports).
- `test_compat_guards.py` — dev-box pytest, 9 tests, no mlx
  needed.
- `omlx-LICENSE-Apache-2.0.txt` — full upstream Apache-2.0
  text recorded.
- `apply-platform-gate.sh` — kept verbatim from the gate lane;
  install.sh calls it.

## How to use

```sh
# 1. Clone pinned source (only if /tmp/omlx-pin is not present)
git clone --depth 1 https://github.com/jundot/omlx /tmp/omlx-pin
git -C /tmp/omlx-pin fetch --depth 1 origin tag v0.7.0
git -C /tmp/omlx-pin checkout 4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40

# 2. Verify the platform-gate series would apply cleanly
packaging/omlx-linux/apply-platform-gate.sh /tmp/omlx-pin --verify-only

# 3. Apply the platform-gate series
packaging/omlx-linux/apply-platform-gate.sh /tmp/omlx-pin

# 4. Verify the Linux compat series would apply cleanly
packaging/omlx-linux/install.sh --mlx-wheel <shared-omarchy-venv-wheel> \
    --omlx-dir /tmp/omlx-pin --venv /tmp/omlx-venv --home /tmp/omlx-home

# 5. Run dev-box checks
python3 packaging/omlx-linux/ast_import_scan.py /tmp/omlx-pin/omlx --allow patches
python3 -m pytest packaging/omlx-linux/test_compat_guards.py

# 6. Platform-gate integration test
python3 tests/test_platform_gate.py --omlx-src /tmp/omlx-pin
```

## Acceptance

The platform-gate series applies clean against the pinned commit
(verified by `apply-platform-gate.sh --verify-only`). Tests live
under `tests/test_platform_gate.py`: stub-module unit tests that
import the gated modules with a fake `mlx.core` and assert each
gate takes the omarchy path on a Linux fake and the Metal path on a
Mac fake.

The Linux compat series applies clean against the same pinned commit
(verified by `install.sh` step 2, which is just `git apply` with
the per-patch `--check` predicate). The dev-box tests are 9/9
green on this machine today.

## Pinned commit

```
$ git -C /tmp/omlx-pin log -1 --format='%H %s'
4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40 release: omlx v0.7.0
```

If upstream moves the commit, regenerate both patch series
(platform-gate uses `_build_patch.py`; the Linux compat series is
hand-written, regenerate with the same `git diff` flow as the
platform gate).

## License

omlx (Apache-2.0). This patch series preserves the SPDX headers in
all upstream files. The new `omlx/_compat_gate.py` and the helper
function `device_total_memory` in `omlx/utils/hardware.py` are
also Apache-2.0 (carries the SPDX header). No upstream source is
copied into the omarchy-mlx repo; only patches-as-diffs land here.
